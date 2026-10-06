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
    parser = argparse.ArgumentParser(description="Forge: train, verify, and measure serving")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="create checksummed byte-token train/val shards")
    prepare.add_argument("--text")
    prepare.add_argument("--synthetic", action="store_true")
    prepare.add_argument("--out", required=True)
    wiki = sub.add_parser("prepare-wikitext", help="import official WikiText-103 parquet shards")
    wiki.add_argument("--train", nargs="+", required=True)
    wiki.add_argument("--validation", required=True)
    wiki.add_argument("--out", required=True)
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
        print(json.dumps({key: result[key] for key in ("heldout", "baselines_nats_per_byte")}))
    elif args.command == "report":
        from forge.report import report

        print(report(args.input, args.out))


if __name__ == "__main__":
    main()
