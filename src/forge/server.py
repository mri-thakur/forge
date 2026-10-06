"""Optional local completions/SSE server, one serialized engine worker.

Worker steps execute in a thread so HTTP arrivals can enqueue during model
execution. The worker owns engine state exclusively; submit/cancel are commands.
"""

import asyncio
import codecs
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi import Request as HTTPRequest
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from forge.engine import Engine
from forge.model import ByteTokenizer


class CompletionInput(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=32, ge=1)
    temperature: float = Field(default=0, ge=0)
    top_p: float = Field(default=1, gt=0, le=1)
    stream: bool = False


def create_app(model):
    engine = Engine(model)
    commands = asyncio.Queue()
    clients = {}

    async def worker():
        while True:
            while not commands.empty():
                kind, payload = commands.get_nowait()
                if kind == "submit":
                    request_id, body, accepted = payload
                    try:
                        request = engine.submit(
                            request_id,
                            ByteTokenizer.encode(body.prompt),
                            body.max_tokens,
                            temperature=body.temperature,
                            top_p=body.top_p,
                        )
                        if not accepted.done():
                            accepted.set_result(request.status)
                        else:
                            engine.cancel(request_id)
                            engine.requests.pop(request_id, None)
                        if request.status == "rejected":
                            engine.requests.pop(request_id, None)
                    except ValueError as error:
                        if not accepted.done():
                            accepted.set_exception(error)
                elif payload in engine.requests:
                    engine.cancel(payload)
                    engine.requests.pop(payload, None)
            if engine.busy:
                try:
                    step_task = asyncio.create_task(asyncio.to_thread(engine.step))
                    try:
                        events = await asyncio.shield(step_task)
                    except asyncio.CancelledError:
                        await step_task
                        raise
                    for request_id, token in events:
                        if request_id in clients:
                            await clients[request_id].put(token)
                except Exception as error:
                    for request_id in list(engine.active) + list(engine.waiting):
                        engine.cancel(request_id)
                        if request_id in clients:
                            await clients[request_id].put(error)
                for request_id in list(clients):
                    request = engine.requests.get(request_id)
                    if request and request.status in {"completed", "cancelled", "rejected"}:
                        await clients.pop(request_id).put(None)
                        engine.requests.pop(request_id, None)
            else:
                await asyncio.sleep(0.002)

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(worker())
        yield
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        for request_id in list(engine.active) + list(engine.waiting):
            engine.cancel(request_id)
        engine.pool.clear_prefixes()

    app = FastAPI(title="Forge", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "backend": model.backend, "device": str(model.device)}

    @app.post("/v1/completions")
    async def complete(body: CompletionInput, connection: HTTPRequest):
        request_id = uuid.uuid4().hex
        accepted = asyncio.get_running_loop().create_future()
        queue = asyncio.Queue()
        clients[request_id] = queue
        await commands.put(("submit", (request_id, body, accepted)))
        try:
            status = await accepted
        except ValueError as error:
            clients.pop(request_id, None)
            raise HTTPException(400, str(error)) from error
        if status == "rejected":
            clients.pop(request_id, None)
            raise HTTPException(429, "capacity exhausted")

        async def tokens():
            try:
                while True:
                    if await connection.is_disconnected():
                        break
                    try:
                        token = await asyncio.wait_for(queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                    if token is None:
                        break
                    if isinstance(token, Exception):
                        raise RuntimeError("engine failed") from token
                    yield token
            finally:
                await commands.put(("cancel", request_id))
                clients.pop(request_id, None)

        if body.stream:

            async def stream():
                decoder = codecs.getincrementaldecoder("utf-8")("replace")
                async for token in tokens():
                    text = decoder.decode(bytes([token]))
                    if text:
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "id": request_id,
                                    "choices": [{"text": text, "index": 0, "finish_reason": None}],
                                }
                            )
                            + "\n\n"
                        )
                tail = decoder.decode(b"", final=True)
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": request_id,
                            "choices": [{"text": tail, "index": 0, "finish_reason": "length"}],
                        }
                    )
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")
        generated = [token async for token in tokens()]
        return {
            "id": request_id,
            "object": "text_completion",
            "choices": [
                {"text": ByteTokenizer.decode(generated), "index": 0, "finish_reason": "length"}
            ],
            "usage": {
                "prompt_tokens": len(ByteTokenizer.encode(body.prompt)),
                "completion_tokens": len(generated),
            },
        }

    return app
