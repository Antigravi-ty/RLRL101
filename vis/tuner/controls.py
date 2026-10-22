"""人类按键 → CarControls。"""
import RocketSim as rs


def build_car_controls(keys):
    """keys: {'W': bool, 'A': bool, ...}"""
    c = rs.CarControls()
    if keys.get('W', False): c.throttle = 1.0; c.pitch = -1.0
    if keys.get('S', False): c.throttle = -1.0; c.pitch = 1.0
    if keys.get('A', False): c.steer = -1.0; c.yaw = -1.0
    if keys.get('D', False): c.steer = 1.0; c.yaw = 1.0
    c.jump = bool(keys.get('Slash', False))
    c.handbrake = bool(keys.get('Shift', False))
    c.boost = bool(keys.get('Period', False))
    if keys.get('Q', False): c.roll = -1.0
    if keys.get('E', False): c.roll = 1.0
    return c


def ctrl_to_dict(c):
    from core.policy_net import controls_to_multi_discrete, multi_discrete_to_key_str
    d = {
        "thr": float(c.throttle), "str": float(c.steer),
        "ptc": float(c.pitch), "yaw": float(c.yaw), "roll": float(c.roll),
        "jump": bool(c.jump), "boost": bool(c.boost),
        "handbrake": bool(c.handbrake),
    }
    ch = controls_to_multi_discrete(d)
    d["ppo_channels"] = [int(x) for x in ch]
    d["keys_str"] = multi_discrete_to_key_str(ch)
    return d


def dict_to_car_controls(d):
    c = rs.CarControls()
    if not d:
        return c
    c.throttle = float(d.get("thr", 0.0))
    c.steer = float(d.get("str", 0.0))
    c.pitch = float(d.get("ptc", 0.0))
    c.yaw = float(d.get("yaw", 0.0))
    c.roll = float(d.get("roll", 0.0))
    c.jump = bool(d.get("jump", False))
    c.boost = bool(d.get("boost", False))
    c.handbrake = bool(d.get("handbrake", False))
    return c

