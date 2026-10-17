import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.env import FocusRegionConfig, load_focus_config, save_focus_config, FOCUS_CONFIG_PATH


def main():
    cfg = load_focus_config()
    p = argparse.ArgumentParser(description="Live Focus Region Configurator for Tuner & Trainer")
    p.add_argument("--enable", "--on", dest="enable", action="store_true", help="Enable focus region sampling")
    p.add_argument("--disable", "--off", dest="disable", action="store_true", help="Disable focus region sampling")
    p.add_argument("--show", action="store_true", help="Show current focus region config without modifying")
    p.add_argument("--prob", type=float, default=None, help="Sampling probability (0.0 to 1.0, e.g. 0.5)")
    p.add_argument("--dist", nargs="+", default=None, help="Horizontal distance range min max (e.g. 150,800 or 150 800)")
    p.add_argument("--angle", nargs="+", default=None, help="Relative angle range deg_min deg_max (e.g. 135,180 or 135 180)")
    p.add_argument("--both-sides", type=int, choices=[0, 1], default=None, help="Mirror angle to +/- sides (1=yes, 0=no)")
    p.add_argument("--ball-z", nargs="+", default=None, help="Ball Z range min max (e.g. 93.15,500)")
    p.add_argument("--car-z", nargs="+", default=None, help="Car Z range min max (e.g. 17.0,400)")
    p.add_argument("--car-spd", nargs="+", default=None, help="Car initial speed range min max (e.g. 0,1400)")
    p.add_argument("--ball-spd", nargs="+", default=None, help="Ball initial speed range min max (e.g. 0,600)")
    p.add_argument("--preset", choices=["back", "front", "side", "aerial", "off"], default=None)
    args = p.parse_args()

    def _parse_pair(vals):
        if len(vals) == 1:
            parts = vals[0].split(",")
            return float(parts[0]), float(parts[1])
        return float(vals[0]), float(vals[1])

    if args.show:
        print(f"[focus] current ({FOCUS_CONFIG_PATH}):")
        print(f"  enabled={cfg.enabled} prob={cfg.prob:.0%} dist=[{cfg.dist_min:.0f}, {cfg.dist_max:.0f}] "
              f"angle=[{cfg.angle_deg_min:.0f}°, {cfg.angle_deg_max:.0f}°] (both_sides={cfg.both_sides}) "
              f"ball_z=[{cfg.ball_z_min:.0f}, {cfg.ball_z_max:.0f}] car_z=[{cfg.car_z_min:.0f}, {cfg.car_z_max:.0f}]")
        return

    if args.preset == "off":
        cfg.enabled = False
    elif args.preset == "back":
        cfg = FocusRegionConfig(enabled=True, prob=0.5, dist_min=120.0, dist_max=850.0, angle_deg_min=135.0, angle_deg_max=180.0, both_sides=True)
    elif args.preset == "front":
        cfg = FocusRegionConfig(enabled=True, prob=0.5, dist_min=150.0, dist_max=1200.0, angle_deg_min=0.0, angle_deg_max=30.0, both_sides=True)
    elif args.preset == "side":
        cfg = FocusRegionConfig(enabled=True, prob=0.5, dist_min=150.0, dist_max=1000.0, angle_deg_min=60.0, angle_deg_max=120.0, both_sides=True)
    elif args.preset == "aerial":
        cfg = FocusRegionConfig(enabled=True, prob=0.5, dist_min=300.0, dist_max=1800.0, angle_deg_min=0.0, angle_deg_max=180.0, both_sides=True, ball_z_min=250.0, ball_z_max=900.0)

    if args.enable:
        cfg.enabled = True
    if args.disable:
        cfg.enabled = False
    if args.prob is not None:
        cfg.prob = float(args.prob)
        cfg.enabled = cfg.prob > 0.0
    if args.dist:
        cfg.dist_min, cfg.dist_max = _parse_pair(args.dist)
    if args.angle:
        cfg.angle_deg_min, cfg.angle_deg_max = _parse_pair(args.angle)
    if args.both_sides is not None:
        cfg.both_sides = bool(args.both_sides)
    if args.ball_z:
        cfg.ball_z_min, cfg.ball_z_max = _parse_pair(args.ball_z)
    if args.car_z:
        cfg.car_z_min, cfg.car_z_max = _parse_pair(args.car_z)
    if args.car_spd:
        cfg.car_speed_min, cfg.car_speed_max = _parse_pair(args.car_spd)
    if args.ball_spd:
        cfg.ball_speed_min, cfg.ball_speed_max = _parse_pair(args.ball_spd)

    save_focus_config(cfg)
    print(f"[focus] saved to {FOCUS_CONFIG_PATH}:")
    print(f"  enabled={cfg.enabled} prob={cfg.prob:.0%} dist=[{cfg.dist_min:.0f}, {cfg.dist_max:.0f}] "
          f"angle=[{cfg.angle_deg_min:.0f}°, {cfg.angle_deg_max:.0f}°] (both_sides={cfg.both_sides}) "
          f"ball_z=[{cfg.ball_z_min:.0f}, {cfg.ball_z_max:.0f}] car_z=[{cfg.car_z_min:.0f}, {cfg.car_z_max:.0f}]")


if __name__ == "__main__":
    main()
