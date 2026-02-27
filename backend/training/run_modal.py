"""
Modal GPU training entry point.
Called as a subprocess by api.py so it gets a clean Python process
with its own event loop (Modal's SDK needs asyncio and doesn't
play well from inside threading.Thread).

Usage from api.py:
    python run_modal.py \\
        --x_train /tmp/x_train.csv \\
        --x_test  /tmp/x_test.csv  \\
        --y_train /tmp/y_train.csv \\
        --y_test  /tmp/y_test.csv

Prints JSON result to stdout on success, error to stderr on failure.
"""

import argparse
import json
import sys
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_train", required=True)
    parser.add_argument("--x_test",  required=True)
    parser.add_argument("--y_train", required=True)
    parser.add_argument("--y_test",  required=True)
    args = parser.parse_args()

    try:
        import modal

        def read_bytes(path):
            with open(path, "rb") as f:
                return f.read()

        x_train_bytes = read_bytes(args.x_train)
        x_test_bytes  = read_bytes(args.x_test)
        y_train_bytes = read_bytes(args.y_train)
        y_test_bytes  = read_bytes(args.y_test)

        # from_name connects to the already-deployed function on Modal's cloud
        train_on_gpu = modal.Function.from_name("heart-disease-training", "train_on_gpu")
        result = train_on_gpu.remote(
            x_train_bytes=x_train_bytes,
            x_test_bytes =x_test_bytes,
            y_train_bytes=y_train_bytes,
            y_test_bytes =y_test_bytes,
        )

        print(json.dumps(result))
        sys.exit(0)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
