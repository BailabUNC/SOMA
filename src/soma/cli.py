"""Command index for the maintained second-revision implementation."""

import argparse
import importlib
import sys

COMMANDS = {
    "train": ("soma.training.main", "Train the three-stage main model"),
    "baseline": ("soma.training.raw", "Train the random-projection baseline"),
    "components": (
        "soma.experiments.components",
        "Run component/from-scratch experiments",
    ),
    "ae64": ("soma.experiments.ae64", "Run the matched AE64-only experiment"),
    "ae64-simsiam": (
        "soma.experiments.ae64_simsiam",
        "Refine matched AE64 with SimSiam",
    ),
    "augmentations": (
        "soma.experiments.augmentation",
        "Run augmentation/freeze conditions",
    ),
    "train-history": ("soma.training.history", "Train with per-epoch histories"),
    "evaluate": (
        "soma.evaluation.classification",
        "Evaluate held-out or external data",
    ),
    "reconstruct": ("soma.evaluation.reconstruction", "Evaluate Phase I denoising"),
    "predict": ("soma.evaluation.predict", "Export checkpoint predictions"),
    "plot-history": (
        "soma.plotting.learning_curves",
        "Plot aggregate histories as SVG",
    ),
}


def main():
    argv = sys.argv[1:]
    if argv and argv[0] in COMMANDS:
        name, *rest = argv
        sys.argv = [f"soma {name}", *rest]
        return importlib.import_module(COMMANDS[name][0]).main()
    parser = argparse.ArgumentParser(
        prog="soma",
        description="SOMA second-revision training and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n"
        + "\n".join(
            f"  {key:16} {description}" for key, (_, description) in COMMANDS.items()
        )
        + "\n\nUse: python -m soma <command> --help",
    )
    parser.add_argument("command", nargs="?", choices=COMMANDS)
    parser.parse_args(argv)
    parser.print_help()
