"""python -m rubik.web [--agent agent] [--host 127.0.0.1] [--port 5000] [--device cuda|cpu]"""
from __future__ import annotations

import argparse
import logging
import os

import torch

from rubik.web.app import ROOT, create_app


def main():
    p = argparse.ArgumentParser(description="RPPO Cube: the web app")
    p.add_argument("--agent", default=os.path.join(ROOT, "agent"), help="agent folder (default: the trained agent of this repository)")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 opens it to other machines (there is no login)")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.backends.cuda.matmul.allow_tf32 = True
    if a.device == "cpu":
        torch.set_num_threads(4)
    app = create_app(dict(DEVICE=a.device, AGENT_DIR=a.agent))
    print(f"RPPO Cube on http://{a.host}:{a.port}  (device {a.device})", flush=True)
    app.run(host=a.host, port=a.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
