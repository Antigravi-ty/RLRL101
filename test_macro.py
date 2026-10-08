# test_macro.py
# 验证 action_abstraction.py 里各宏动作的实际物理效果
# 用法: python3 test_macro.py

import RocketSim as rs
from action_abstraction import ActionAbstractionLayer, Intent


def run_macro(name, intent):
    # 每个宏用独立的 arena，避免相互污染
    arena = rs.Arena(rs.GameMode.SOCCAR)
    car = arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)

    cs = rs.CarState()
    cs.pos = rs.Vec(3000.0, 0.0, 17.0)
    cs.vel = rs.Vec(0.0, 0.0, 0.0)
    car.set_state(cs)
    arena.step(1)

    layer = ActionAbstractionLayer()
    controls = layer.parse_intent(int(intent))

    p0 = car.get_state().pos
    max_speed = 0.0
    min_z = 1e9
    landed_at = None

    for i, ctrl in enumerate(controls):
        car.set_controls(ctrl)
        arena.step(1)
        s = car.get_state()
        sp = (s.vel.x**2 + s.vel.y**2 + s.vel.z**2) ** 0.5
        max_speed = max(max_speed, sp)
        min_z = min(min_z, s.pos.z)
        if s.is_on_ground and i > 5 and landed_at is None:
            landed_at = i

    s = car.get_state()
    sp_end = (s.vel.x**2 + s.vel.y**2 + s.vel.z**2) ** 0.5
    fwd = car.get_forward_dir()
    dx = s.pos.x - p0.x
    dy = s.pos.y - p0.y

    print(f"[{name:18s}] "
          f"ticks={len(controls):3d}  "
          f"disp=({dx:+7.1f},{dy:+6.1f})  "
          f"end_v={sp_end:6.1f}  max_v={max_speed:6.1f}  "
          f"min_z={min_z:5.1f}  "
          f"land_t={landed_at if landed_at is not None else '--':>3}  "
          f"og={int(s.is_on_ground)}  flip={int(s.is_flipping)}  "
          f"fwd=({fwd.x:+.2f},{fwd.y:+.2f},{fwd.z:+.2f})")


if __name__ == "__main__":
    macros = [
        ("JUMP_TAP",           Intent.JUMP_TAP),
        ("JUMP_HOLD",          Intent.JUMP_HOLD),
        ("DODGE_FRONT",        Intent.DODGE_FRONT),
        ("DODGE_BACK",         Intent.DODGE_BACK),
        ("DODGE_LEFT",         Intent.DODGE_LEFT),
        ("DODGE_RIGHT",        Intent.DODGE_RIGHT),
        ("DODGE_FRONT_LEFT",   Intent.DODGE_FRONT_LEFT),
        ("DODGE_FRONT_RIGHT",  Intent.DODGE_FRONT_RIGHT),
        ("DODGE_BACK_LEFT",    Intent.DODGE_BACK_LEFT),
        ("DODGE_BACK_RIGHT",   Intent.DODGE_BACK_RIGHT),
        ("FLIP_CANCEL_FRONT",  Intent.FLIP_CANCEL_FRONT),
        ("SPEED_FLIP_LEFT",    Intent.SPEED_FLIP_LEFT),
        ("SPEED_FLIP_RIGHT",   Intent.SPEED_FLIP_RIGHT),
        ("WAVEDASH_FRONT",     Intent.WAVEDASH_FRONT),
        ("HALF_FLIP",          Intent.HALF_FLIP),
    ]
    for name, intent in macros:
        run_macro(name, intent)
