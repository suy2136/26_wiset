import argparse
import re


def safe_experiment_tag(value):
    """Return a path-safe experiment tag or raise an argparse error."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise argparse.ArgumentTypeError(
            "experiment tag must contain only letters, digits, underscores, "
            "or hyphens and must start with a letter or digit"
        )
    return value
