#!/usr/bin/env python3
"""Subscribe to rk_studio Zenoh result topics and print JSON payloads."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time

try:
    import zenoh
except ImportError as exc:
    raise SystemExit(
        "Missing Python package 'zenoh' for this interpreter:\n"
        f"  {sys.executable}\n"
        "Install it with:\n"
        f"  {sys.executable} -m pip install eclipse-zenoh"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--key",
        default="rk_studio/**",
        help="Zenoh key expression to subscribe, default: rk_studio/**",
    )
    parser.add_argument(
        "--mode",
        default="peer",
        choices=("peer", "client"),
        help="Zenoh session mode, default: peer",
    )
    parser.add_argument(
        "--connect",
        action="append",
        default=[],
        help="Zenoh endpoint to connect to. Can be passed multiple times.",
    )
    parser.add_argument(
        "--listen",
        action="append",
        default=[],
        help="Zenoh endpoint to listen on. Can be passed multiple times.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw payload strings instead of pretty JSON.",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace):
    config = zenoh.Config()
    config.insert_json5("mode", json.dumps(args.mode))
    if args.connect:
        config.insert_json5("connect/endpoints", json.dumps(args.connect))
    if args.listen:
        config.insert_json5("listen/endpoints", json.dumps(args.listen))
    return config


def payload_to_text(payload) -> str:
    if hasattr(payload, "to_bytes"):
        return payload.to_bytes().decode("utf-8", errors="replace")
    return bytes(payload).decode("utf-8", errors="replace")


def main() -> int:
    args = parse_args()
    stopped = False

    def handle_signal(signum, frame) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    session = zenoh.open(make_config(args))

    def on_sample(sample) -> None:
        key = str(sample.key_expr)
        text = payload_to_text(sample.payload)
        print(f"\n[{time.strftime('%H:%M:%S')}] {key}", flush=False)
        if args.raw:
            print(text, flush=True)
            return
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            print(text, flush=True)
            return
        print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)

    subscriber = session.declare_subscriber(args.key, on_sample)
    print(f"subscribed: {args.key} mode={args.mode}")
    if args.connect:
        print(f"connect: {args.connect}")
    if args.listen:
        print(f"listen: {args.listen}")

    try:
        while not stopped:
            time.sleep(0.2)
    finally:
        subscriber.undeclare()
        session.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
