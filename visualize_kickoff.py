import os, time, struct, socket
import numpy as np
import torch
import RocketSim as rs

from kickoff_common import (
    OBS_DIM, N_ACTIONS, HIDDEN_DIM, ACTIONS,
    TICKS_PER_DECISION, MAX_DECISIONS, TOUCH_DIST,
    build_obs,
)

CKPT_LATEST = "./checkpoints/kickoff_latest.pt"
VIS_ADDR = ("127.0.0.1", 9999)

vis_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


class QNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(OBS_DIM, HIDDEN_DIM), torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_DIM, N_ACTIONS),
        )
    def forward(self, x):
        return self.net(x)


def load_latest():
    if not os.path.exists(CKPT_LATEST):
        return None, None
    try:
        ckpt = torch.load(CKPT_LATEST, map_location="cpu")
    except Exception:
        return None, None
    net = QNet()
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt.get("episode", 0)


def main():
    arena = rs.Arena(rs.GameMode.SOCCAR)
    car = arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)

    pads = arena.get_boost_pads()
    NUM_PADS = len(pads)

    net, cur_ep = None, -1
    last_mtime = 0.0
    frame_dt = 1.0 / 60.0

    print(f"[vis] NUM_PADS={NUM_PADS}")

    while True:
        try:
            mt = os.path.getmtime(CKPT_LATEST)
        except OSError:
            mt = 0.0
        if mt > last_mtime:
            last_mtime = mt
            new_net, ep = load_latest()
            if new_net is not None:
                net, cur_ep = new_net, ep
                print(f"[vis] loaded ckpt episode={cur_ep}")

        arena.reset_kickoff()
        arena.step(1)

        if net is not None:
            obs, _ = build_obs(car, arena.ball.get_state())
        else:
            obs = None

        for step in range(MAX_DECISIONS):
            if net is None:
                a_idx = 0
            else:
                with torch.no_grad():
                    q = net(torch.tensor(obs, dtype=torch.float32))
                    a_idx = int(q.argmax().item())

            thr, st = ACTIONS[a_idx]
            ctrl = rs.CarControls()
            ctrl.throttle = float(thr)
            ctrl.steer = float(st)
            cs_now = car.get_state()
            ctrl.boost = bool(thr > 0.0 and cs_now.boost > 0.5)
            car.set_controls(ctrl)

            arena.step(TICKS_PER_DECISION)

            if net is not None:
                obs, dist = build_obs(car, arena.ball.get_state())
            else:
                dist = 1e9

            # ---- 打包 UDP ----
            cs = car.get_state()
            bs = arena.ball.get_state()
            f = car.get_forward_dir(); u = car.get_up_dir(); r = car.get_right_dir()

            speed_uu = float((cs.vel.x**2 + cs.vel.y**2 + cs.vel.z**2) ** 0.5)
            is_boosting = 1.0 if ctrl.boost else 0.0
            is_on_ground = 1.0 if cs.is_on_ground else 0.0

            values = [
                float(-cs.pos.x), float(cs.pos.y), float(cs.pos.z),
                float(-f.x), float(f.y), float(f.z),
                float(-u.x), float(u.y), float(u.z),
                float(-r.x), float(r.y), float(r.z),
                float(1.0 if step == 0 else 0.0),
                float(-bs.pos.x), float(bs.pos.y), float(bs.pos.z),
                float(cs.boost),
                float(is_boosting),
                float(is_on_ground),
                float(speed_uu),
            ]
            pad_states = [1.0 if p.get_state().is_active else 0.0 for p in pads]

            pkt = struct.pack("f" * (20 + NUM_PADS), *(values + pad_states))
            vis_sock.sendto(pkt, VIS_ADDR)

            time.sleep(frame_dt)

            if dist < TOUCH_DIST:
                break


if __name__ == "__main__":
    main()