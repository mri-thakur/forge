import argparse
import json
from pathlib import Path

from forge.model import ByteTokenizer, ModelConfig, NumpyModel


def load_model(args):
    if args.checkpoint:
        model = NumpyModel.load(args.checkpoint)
    else:
        model = NumpyModel(
            ModelConfig(
                dim=args.dim,
                layers=args.layers,
                hidden=args.dim * 3,
                context=args.context,
                seed=args.seed,
            )
        )
    if args.backend == "torch":
        from forge.torch_backend import TorchModel

        model = TorchModel(model, args.device, args.attention)
    elif args.device != "cpu":
        raise ValueError("NumPy backend requires --device cpu; use --backend torch for GPUs")
    elif args.attention != "manual":
        raise ValueError("SDPA requires the torch backend")
    return model


def model_options(parser):
    parser.add_argument("--checkpoint")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--attention", choices=["manual", "sdpa"], default="manual")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)


def main():
    parser = argparse.ArgumentParser(
        description="Forge: tokenize, pretrain, fine-tune, run RL on, evaluate, and serve "
        "a language model built from scratch"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="create checksummed byte-token train/val shards")
    prepare.add_argument("--text")
    prepare.add_argument("--synthetic", action="store_true")
    prepare.add_argument("--out", required=True)
    wiki = sub.add_parser("prepare-wikitext", help="import official WikiText-103 parquet shards")
    wiki.add_argument("--train", nargs="+", required=True)
    wiki.add_argument("--validation", required=True)
    wiki.add_argument("--out", required=True)
    tiny = sub.add_parser(
        "prepare-tinystories", help="train the BPE tokenizer on TinyStories and encode it"
    )
    tiny.add_argument("--train", required=True)
    tiny.add_argument("--valid", required=True)
    tiny.add_argument("--out", required=True)
    tiny.add_argument("--vocab", type=int, default=4096)
    tiny.add_argument("--compare", type=int, nargs="+", default=[2048, 4096, 8192])
    tiny.add_argument("--workers", type=int)
    tiny.add_argument("--results", help="folder for the tokenizer comparison JSON")
    instruct = sub.add_parser(
        "prepare-instruct", help="parse TinyStories-Instruct and score its gold stories"
    )
    instruct.add_argument("--train", required=True)
    instruct.add_argument("--valid", required=True)
    instruct.add_argument("--out", required=True)
    instruct.add_argument("--results", help="folder for gold_check.json")
    instruct_eval = sub.add_parser(
        "instruct-eval", help="constraint satisfaction of generated stories"
    )
    instruct_eval.add_argument("--samples", required=True, help='JSONL of {"id", "story"}')
    instruct_eval.add_argument("--prompts", required=True)
    instruct_eval.add_argument("--out", required=True)
    sft = sub.add_parser("prepare-sft", help="tokenize instruction examples with a loss mask")
    sft.add_argument("--instruct", required=True, help="folder written by prepare-instruct")
    sft.add_argument("--tokenizer", required=True)
    sft.add_argument("--out", required=True)
    sft.add_argument("--train-examples", type=int, default=300_000)
    sft.add_argument("--val-examples", type=int, default=5_000)
    generate_instruct = sub.add_parser(
        "generate-instruct", help="sample stories for instruction prompts"
    )
    model_options(generate_instruct)
    generate_instruct.add_argument("--tokenizer", required=True)
    generate_instruct.add_argument("--prompts", required=True)
    generate_instruct.add_argument("--out", required=True, help="JSONL of generated stories")
    generate_instruct.add_argument("--samples", type=int, default=1, help="per prompt")
    generate_instruct.add_argument("--temperature", type=float, default=0.8)
    generate_instruct.add_argument("--top-p", type=float, default=0.95)
    generate_instruct.add_argument("--max-batch", type=int, default=32)
    story_loss = sub.add_parser(
        "story-loss", help="fluency: nats/token of generated stories under a model"
    )
    model_options(story_loss)
    story_loss.add_argument("--tokenizer", required=True)
    story_loss.add_argument(
        "--samples", nargs="+", required=True, help='JSONL files of {"story"}, scored separately'
    )
    story_loss.add_argument("--out", required=True)
    story_loss.add_argument("--batch", type=int, default=32)
    grpo = sub.add_parser("grpo", help="GRPO with verifiable rewards from an SFT model")
    grpo.add_argument("--init", required=True, help="SFT model.npz: the policy and KL reference")
    grpo.add_argument("--prompts", required=True, help="instruction records (training split)")
    grpo.add_argument("--exclude", help="JSONL of records never to train on (eval prompts)")
    grpo.add_argument("--tokenizer", required=True)
    grpo.add_argument("--out", required=True)
    grpo.add_argument("--steps", type=int, default=200)
    grpo.add_argument("--prompts-per-step", type=int, default=16)
    grpo.add_argument("--group", type=int, default=8, help="samples per prompt")
    grpo.add_argument("--lr", type=float, default=3e-5)
    grpo.add_argument("--kl", type=float, default=0.1, help="KL penalty coefficient")
    grpo.add_argument("--max-new-tokens", type=int, default=384)
    grpo.add_argument("--warmup", type=int, default=10)
    grpo.add_argument("--micro-batch", type=int, default=16)
    grpo.add_argument(
        "--max-batch", type=int, default=128, help="concurrent rollouts (128: 1.6x faster than 64)"
    )
    grpo.add_argument("--pool-size", type=int, default=20_000)
    grpo.add_argument("--save-every", type=int, default=25)
    grpo.add_argument(
        "--snapshot-every", type=int, default=0, help="also save model_step<N>.npz this often"
    )
    grpo.add_argument("--seed", type=int, default=17)
    grpo.add_argument("--device", default="cuda")
    grpo.add_argument("--attention", choices=["manual", "sdpa"], default="sdpa")
    grpo.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    grpo.add_argument("--resume", action="store_true")
    train_parser = sub.add_parser("train", help="train using Forge's NumPy reverse-mode engine")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--out", required=True)
    train_parser.add_argument("--steps", type=int, default=100)
    train_parser.add_argument("--dim", type=int, default=32)
    train_parser.add_argument("--layers", type=int, default=2)
    train_parser.add_argument("--context", type=int, default=32)
    train_parser.add_argument("--batch", type=int, default=4)
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--attention", choices=["manual", "sdpa"], default="manual")
    train_parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    torch_only = train_parser.add_argument_group("torch backend only")
    torch_only.add_argument("--heads", type=int, default=4)
    torch_only.add_argument("--kv-heads", type=int, default=2)
    torch_only.add_argument("--hidden", type=int, help="FFN width (default 3 x dim)")
    torch_only.add_argument("--accumulate", type=int, default=1, help="micro-batches per step")
    torch_only.add_argument("--lr", type=float, default=0.002)
    torch_only.add_argument("--warmup", type=int, help="warmup steps (default min(20, steps/10))")
    torch_only.add_argument("--min-lr-ratio", type=float, default=0.1)
    torch_only.add_argument("--eval-batches", type=int, default=8)
    torch_only.add_argument("--eval-batch", type=int, default=2)
    torch_only.add_argument("--init", help="model.npz to fine-tune instead of random weights")
    torch_only.add_argument(
        "--rope-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="bfloat16 reproduces the original bf16 RoPE angles (ablation only)",
    )
    bench = sub.add_parser("bench", help="open-loop scheduling experiment")
    model_options(bench)
    bench.add_argument("--out", required=True)
    bench.add_argument("--requests", type=int, default=24)
    bench.add_argument("--rates", type=float, nargs="+", default=[10, 30])
    bench.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    bench.add_argument(
        "--policies",
        nargs="+",
        choices=["static", "continuous", "chunked"],
        default=["static", "continuous", "chunked"],
    )
    bench.add_argument(
        "--workload", choices=["short", "mixed", "burst", "prefix", "head_of_line"], default="mixed"
    )
    bench.add_argument("--max-batch", type=int, default=8)
    bench.add_argument("--token-budget", type=int, default=64)
    bench.add_argument("--blocks", type=int, default=128)
    bench.add_argument("--block-size", type=int, default=16)
    bench.add_argument("--prefix-entries", type=int, default=0)
    bench.add_argument("--ttft-ms", type=float, default=500)
    bench.add_argument("--tpot-ms", type=float, default=50)
    ladder = sub.add_parser("ladder", help="reference versus paged-KV microbenchmark")
    model_options(ladder)
    ladder.add_argument("--out", required=True)
    ladder.add_argument("--repeats", type=int, default=3)
    generate = sub.add_parser("generate")
    model_options(generate)
    generate.add_argument("--prompt", default="The ")
    generate.add_argument("--tokens", type=int, default=32)
    serve = sub.add_parser("serve")
    model_options(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    evaluate = sub.add_parser("evaluate", help="score a finished run on its whole held-out split")
    model_options(evaluate)
    evaluate.add_argument("--run", required=True, help="training output folder")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--out", required=True, help="folder for evaluation.json")
    evaluate.add_argument("--report", help="folder for RESULTS.md and loss_curve.png")
    evaluate.add_argument("--batch", type=int, default=32)
    report = sub.add_parser("report", help="create measured Markdown table and plots")
    report.add_argument("--input", required=True)
    report.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        from forge.training import prepare_data

        print(json.dumps(prepare_data(args.out, args.text, args.synthetic), indent=2))
    elif args.command == "prepare-wikitext":
        from forge.datasets import prepare_wikitext

        print(json.dumps(prepare_wikitext(args.train, args.validation, args.out), indent=2))
    elif args.command == "prepare-tinystories":
        from forge.tinystories import prepare_tinystories

        manifest, comparison = prepare_tinystories(
            args.train, args.valid, args.out, args.vocab, tuple(args.compare), args.workers
        )
        if args.results:
            results = Path(args.results)
            results.mkdir(parents=True, exist_ok=True)
            payload = {"comparison": comparison, "manifest": manifest}
            (results / "compare.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        print(json.dumps(comparison, indent=2))
    elif args.command == "prepare-instruct":
        from forge.instruct import gold_check, load_jsonl, prepare

        manifest = prepare(args.train, args.valid, args.out)
        if args.results:
            report = gold_check(load_jsonl(Path(args.out) / "valid.jsonl"))
            report["failures"] = report["failures"][:200]  # examples, not all
            results = Path(args.results)
            results.mkdir(parents=True, exist_ok=True)
            (results / "gold_check.json").write_text(
                json.dumps({"manifest": manifest, **report}, indent=2) + "\n", encoding="utf-8"
            )
        print(json.dumps(manifest["splits"], indent=2))
    elif args.command == "instruct-eval":
        from forge.instruct import evaluate, load_jsonl

        # Each (prompt, sample) pair is scored on its own; rates average over all.
        prompts = {record["id"]: record for record in load_jsonl(args.prompts)}
        rows = load_jsonl(args.samples)
        keys = [f"{row['id']}#{row.get('sample', 0)}" for row in rows]
        records = [{**prompts[row["id"]], "id": key} for row, key in zip(rows, keys)]
        report = evaluate(records, {key: row["story"] for row, key in zip(rows, keys)})
        report["prompts"] = len({row["id"] for row in rows})
        report["samples_per_prompt"] = len(rows) / max(1, report["prompts"])
        report["finished_rate"] = sum(row.get("finished", True) for row in rows) / len(rows)
        missing = set(prompts) - {row["id"] for row in rows}
        report["missing_ids"] = sorted(missing)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: report[key] for key in ("satisfied", "mean_reward")}))
    elif args.command == "prepare-sft":
        from forge.sft import prepare_sft

        manifest = prepare_sft(
            args.instruct, args.tokenizer, args.out, args.train_examples, args.val_examples
        )
        print(json.dumps({k: v for k, v in manifest.items() if k.startswith(("train", "val"))}))
    elif args.command == "generate-instruct":
        from forge.bpe import Tokenizer
        from forge.instruct import load_jsonl
        from forge.sft import generate

        rows = generate(
            load_model(args),
            Tokenizer.load(args.tokenizer),
            load_jsonl(args.prompts),
            args.samples,
            args.temperature,
            args.top_p,
            args.seed,
            args.max_batch,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        finished = sum(row["finished"] for row in rows)
        print(json.dumps({"stories": len(rows), "finished": finished}))
    elif args.command == "story-loss":
        from forge.bpe import Tokenizer
        from forge.instruct import load_jsonl
        from forge.sft import story_loss

        if args.backend != "torch":
            raise ValueError("story-loss requires --backend torch")
        model, tokenizer = load_model(args), Tokenizer.load(args.tokenizer)
        report = {"checkpoint": args.checkpoint, "files": {}}
        for path in args.samples:
            rows = load_jsonl(path)
            result = story_loss(model, tokenizer, [row["story"] for row in rows], args.batch)
            report["files"][Path(path).name] = {
                **result,
                "ids": [f"{row['id']}#{row.get('sample', 0)}" for row in rows],
            }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report) + "\n", encoding="utf-8")
        print(json.dumps({name: r["nats_per_token"] for name, r in report["files"].items()}))
    elif args.command == "grpo":
        from forge.grpo import train_grpo

        summary = train_grpo(
            args.init,
            args.prompts,
            args.tokenizer,
            args.out,
            steps=args.steps,
            prompts_per_step=args.prompts_per_step,
            group=args.group,
            lr=args.lr,
            kl=args.kl,
            max_new_tokens=args.max_new_tokens,
            warmup=args.warmup,
            micro_batch=args.micro_batch,
            max_batch=args.max_batch,
            pool_size=args.pool_size,
            exclude=args.exclude,
            save_every=args.save_every,
            seed=args.seed,
            device=args.device,
            attention=args.attention,
            precision=args.precision,
            resume=args.resume,
            snapshot_every=args.snapshot_every,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "train":
        if args.backend == "torch":
            from forge.gpu_training import train_torch

            result = train_torch(
                args.data,
                args.out,
                args.steps,
                args.dim,
                args.layers,
                args.context,
                args.batch,
                args.seed,
                args.resume,
                args.device,
                args.attention,
                args.precision,
                heads=args.heads,
                kv_heads=args.kv_heads,
                hidden=args.hidden,
                accumulate=args.accumulate,
                lr=args.lr,
                warmup=args.warmup,
                min_lr_ratio=args.min_lr_ratio,
                eval_batches=args.eval_batches,
                eval_batch=args.eval_batch,
                rope_dtype=args.rope_dtype,
                init=args.init,
            )
        else:
            if args.device != "cpu":
                raise ValueError("NumPy training requires --device cpu")
            from forge.training import train

            result = train(
                args.data,
                args.out,
                args.steps,
                args.dim,
                args.layers,
                args.context,
                args.batch,
                args.seed,
                args.resume,
            )
        print(json.dumps(result, indent=2))
    elif args.command == "bench":
        from forge.bench import benchmark

        benchmark(
            load_model(args),
            args.out,
            args.requests,
            tuple(args.rates),
            args.workload,
            tuple(args.seeds),
            tuple(args.policies),
            args.max_batch,
            args.token_budget,
            args.blocks,
            args.block_size,
            args.prefix_entries,
            args.ttft_ms / 1000,
            args.tpot_ms / 1000,
        )
    elif args.command == "ladder":
        from forge.bench import cache_ladder

        print(json.dumps(cache_ladder(load_model(args), args.out, args.repeats), indent=2))
    elif args.command == "generate":
        from forge.engine import Engine

        engine = Engine(load_model(args), max_batch=1)
        request = engine.submit("cli", ByteTokenizer.encode(args.prompt), args.tokens)
        while engine.busy:
            engine.step()
        print(ByteTokenizer.decode(request.generated))
    elif args.command == "serve":
        import uvicorn

        from forge.server import create_app

        uvicorn.run(create_app(load_model(args)), host=args.host, port=args.port)
    elif args.command == "evaluate":
        from forge.evaluation import evaluate_run

        args.checkpoint = args.checkpoint or str(Path(args.run) / "model.npz")
        result = evaluate_run(
            load_model(args), args.run, args.data, args.out, args.report, args.batch
        )
        print(json.dumps({key: result[key] for key in ("heldout", "baselines_nats_per_token")}))
    elif args.command == "report":
        from forge.report import report

        print(report(args.input, args.out))


if __name__ == "__main__":
    main()
