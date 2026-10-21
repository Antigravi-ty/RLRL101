"""reward_tuner 入口。实际逻辑在 vis/tuner/。"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--from", "--from-ckpt", dest="from_ckpt", default=None, choices=["best", "latest"])
    ap.add_argument("--dist", "--spawn-dist", dest="spawn_dist", default=None)
    ap.add_argument("--stage-level", type=int, default=None)
    ap.add_argument("--cam", "--cam-mode", dest="cam_mode", default=None, choices=["center", "attach"])
    ap.add_argument("--cam-params", dest="cam_params", default=None,
                    help="Camera params e.g. 'back=320,up=110,elev=12,azim=0,fov=75'")
    args, _ = ap.parse_known_args()
    return args


if __name__ == "__main__":
    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        from vis.cli_tuner import main
        main()
        sys.exit(0)

    args = _parse_args()

    try:
        import pyqtgraph  # noqa: F401
        from pyqtgraph.Qt import QtCore, QtWidgets  # noqa: F401
        from vis.tuner.window import main
    except ImportError as e:
        print(f"[tuner] GUI (PyQt/pyqtgraph) unavailable: {e}")
        print("[tuner] Falling back to CLI mode...")
        from vis.cli_tuner import main as cli_main
        cli_main()
    else:
        main(
            prefix=args.prefix,
            from_ckpt=args.from_ckpt,
            spawn_dist=args.spawn_dist,
            stage_level=args.stage_level,
            cam_mode=args.cam_mode,
            cam_params=args.cam_params,
        )

