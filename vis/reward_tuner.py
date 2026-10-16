"""reward_tuner 入口。实际逻辑在 vis/tuner/。"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_prefix():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--prefix", default=None)
    args, _ = ap.parse_known_args()
    return args.prefix


if __name__ == "__main__":
    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        from vis.cli_tuner import main
        main()
        sys.exit(0)

    prefix = _parse_prefix()

    try:
        from vis.tuner.window import main
        main(prefix=prefix)
    except (ImportError, Exception) as e:
        print(f"[tuner] GUI (PyQt/pyqtgraph) unavailable: {e}")
        print("[tuner] Falling back to CLI mode...")
        from vis.cli_tuner import main
        main()