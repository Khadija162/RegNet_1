from src.cli import build_parser
from src.training.train import run_training
from src.evaluation.evaluate import run_evaluation


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        run_training(args)
    elif args.command == "evaluate":
        run_evaluation(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
