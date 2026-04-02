import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAGNet multimodal runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for cmd in ["train", "evaluate"]:
        sub = subparsers.add_parser(cmd)
        sub.add_argument("--config", required=True, help="Path to experiment yaml")
        sub.add_argument("--dataset", default=None)
        sub.add_argument("--missing-ratio", type=float, default=None)
        sub.add_argument("--missing-scenario", default=None, choices=["mcar", "mar", "mnar", "random"])
        sub.add_argument("--checkpoint", default=None)
    return parser
