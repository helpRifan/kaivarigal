"""``python -m hwrobot.app [--host 0.0.0.0] [--port 8000]`` — start the web UI.

Use ``--host 0.0.0.0`` to reach it from a phone on the same Wi-Fi.
"""

import argparse

import uvicorn

from hwrobot.app.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m hwrobot.app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data", default="data", help="folder for samples, history and settings")
    args = parser.parse_args()
    print(f"Kaivarigal running at http://{'localhost' if args.host == '127.0.0.1' else args.host}:{args.port}")
    uvicorn.run(create_app(args.data), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
