import os, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import RocketSim as rs

from kickoff_common import (
    OBS_DIM, N_ACTIONS, N_INTENTS, HIDDEN_DIM, ACTIONS,
    TICKS_PER_DECISION, MAX_DECISIONS, TOUCH_DIST,
    build_obs,
)
from action_abstraction import ActionAbstractionLayer, Intent, NUM_INTENTS

SEED = int(os.environ.get("SEED", "0"))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 256
BUFFER_CAP = 300_000
EPS_START_FRESH = 1.0
EPS_START_RESUME = 0.3
EPS_END = 0.05
EPS_DECAY_PER_EP = 0.9997
TARGET_UPDATE = 500
GRAD_CLIP = 10.0
TRAIN_EVERY = 8
TRAIN_EPISODES = int(os.environ.get("TRAIN_EPISODES", "200_000"))
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "200"))
USE_ABSTRACTION = os.environ.get("USE_ABSTRACTION", "1") == "1"

CKPT_DIR = "./checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_LATEST = os.path.join(CKPT_DIR, "kickoff_latest.pt")
CKPT_BEST   = os.path.join(CKPT_DIR, "kickoff_best.pt")

RESUME = os.environ.get("RESUME", "0") == "1"


class QNet(nn.Module):
    def __init__(self, action_dim=NUM_INTENTS if USE_ABSTRACTION else N_ACTIONS):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, action_dim),
        )
    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap)
    def push(self, *args):
        self.buf.append(args)
    def sample(self, bs):
        batch = random.sample(self.buf, bs)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.array(s2), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )
    def __len__(self):
        return len(self.buf)


def save_ckpt(path, q_net, episode, n_actions=NUM_INTENTS if USE_ABSTRACTION else N_ACTIONS, note=""):
    torch.save({
        "state_dict": q_net.state_dict(),
        "obs_dim": OBS_DIM,
        "n_actions": n_actions,
        "hidden_dim": HIDDEN_DIM,
        "episode": episode,
        "note": note,
    }, path)


def main():
    arena = rs.Arena(rs.GameMode.SOCCAR)
    car = arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)

    act_dim = NUM_INTENTS if USE_ABSTRACTION else N_ACTIONS
    action_layer = ActionAbstractionLayer(TICKS_PER_DECISION) if USE_ABSTRACTION else None

    q_net = QNet(act_dim)
    if RESUME and os.path.exists(CKPT_LATEST):
        ckpt = torch.load(CKPT_LATEST, map_location="cpu")
        q_net.load_state_dict(ckpt["state_dict"])
        print(f"[train] resumed from {CKPT_LATEST}, prev_episode={ckpt.get('episode')}")
    else:
        print("[train] fresh weights")

    target_net = QNet(act_dim)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_CAP)

    epsilon = EPS_START_RESUME if RESUME else EPS_START_FRESH
    total_steps = 0

    touch_history = deque(maxlen=1000)
    time_history  = deque(maxlen=1000)
    best_mean_time = 1e9

    fps_t0 = time.time()
    fps_dec = 0

    print(f"[train] obs={OBS_DIM} actions={act_dim} abstraction={USE_ABSTRACTION} seed={SEED} resume={RESUME} eps0={epsilon}")

    for episode in range(1, TRAIN_EPISODES + 1):
        arena.reset_kickoff()
        arena.step(1)
        if action_layer:
            action_layer.reset()

        obs, dist = build_obs(car, arena.ball.get_state())
        prev_dist = dist

        ep_reward = 0.0
        touched = False
        touched_at = MAX_DECISIONS
        done = False

        for step in range(MAX_DECISIONS):
            if USE_ABSTRACTION:
                if action_layer.is_executing_macro:
                    a_idx = int(action_layer.current_intent)
                    action_layer.step(arena, car)
                else:
                    if random.random() < epsilon:
                        a_idx = random.randrange(act_dim)
                    else:
                        with torch.no_grad():
                            q = q_net(torch.tensor(obs, dtype=torch.float32))
                            a_idx = int(q.argmax().item())
                    action_layer.step(arena, car, a_idx)
            else:
                if random.random() < epsilon:
                    a_idx = random.randrange(N_ACTIONS)
                else:
                    with torch.no_grad():
                        q = q_net(torch.tensor(obs, dtype=torch.float32))
                        a_idx = int(q.argmax().item())

                thr, st = ACTIONS[a_idx]
                ctrl = rs.CarControls()
                ctrl.throttle = float(thr)
                ctrl.steer = float(st)
                cs_now = car.get_state()
                ctrl.boost = bool(thr > 0.0 and cs_now.boost > 0.5)
                ctrl.handbrake = False
                ctrl.jump = False
                car.set_controls(ctrl)
                arena.step(TICKS_PER_DECISION)

            new_obs, new_dist = build_obs(car, arena.ball.get_state())

            r = 0.0
            r += 0.02 * (prev_dist - new_dist)
            r -= 0.01

            cs = car.get_state()
            fwd = car.get_forward_dir()
            bs = arena.ball.get_state()
            to_ball = np.array([bs.pos.x - cs.pos.x, bs.pos.y - cs.pos.y, bs.pos.z - cs.pos.z])
            n = np.linalg.norm(to_ball)
            if n > 1e-6:
                to_ball /= n
                align = to_ball[0]*fwd.x + to_ball[1]*fwd.y + to_ball[2]*fwd.z
                r += 0.005 * max(0.0, align)

            if cs.is_supersonic:
                r += 0.01

            up = car.get_up_dir()
            if up.z < 0.2:
                r -= 1.0
                done = True

            if new_dist < TOUCH_DIST:
                speed_bonus = max(0.0, (MAX_DECISIONS - step) * 0.1)
                r += 10.0 + speed_bonus
                done = True
                touched = True
                touched_at = step + 1

            if step == MAX_DECISIONS - 1:
                r -= 2.0
                done = True

            buffer.push(obs, a_idx, r, new_obs, float(done))
            obs = new_obs
            prev_dist = new_dist
            ep_reward += r
            total_steps += 1
            fps_dec += 1

            if len(buffer) >= BATCH_SIZE and total_steps % TRAIN_EVERY == 0:
                s, a, rr, s2, d = buffer.sample(BATCH_SIZE)
                qv = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_a = q_net(s2).argmax(1, keepdim=True)
                    next_q = target_net(s2).gather(1, next_a).squeeze(1)
                    target = rr + GAMMA * next_q * (1.0 - d)
                loss = nn.SmoothL1Loss()(qv, target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), GRAD_CLIP)
                optimizer.step()

                if total_steps % TARGET_UPDATE == 0:
                    target_net.load_state_dict(q_net.state_dict())

            if done:
                break

        epsilon = max(EPS_END, epsilon * EPS_DECAY_PER_EP)

        touch_history.append(1 if touched else 0)
        if touched:
            time_history.append(touched_at)

        if episode % PRINT_EVERY == 0:
            tr = sum(touch_history) / max(len(touch_history), 1)
            mean_t = (sum(time_history) / max(len(time_history), 1)) if time_history else 0.0
            now = time.time()
            dec_fps = fps_dec / max(now - fps_t0, 1e-6)
            print(f"Ep {episode:6d} | eps={epsilon:.3f} | "
                  f"touch={tr:6.2%} | mean_touch_step={mean_t:6.1f} | "
                  f"r={ep_reward:6.2f} | dec_fps={dec_fps:6.0f}")
            fps_t0 = time.time(); fps_dec = 0

            if tr > 0.9 and mean_t > 0 and mean_t < best_mean_time:
                best_mean_time = mean_t
                save_ckpt(CKPT_BEST, q_net, episode, note=f"best mean_t={mean_t:.1f}")
                print(f"  [ckpt] new best (mean_touch_step={mean_t:.1f})")

        if episode % 50 == 0:
            save_ckpt(CKPT_LATEST, q_net, episode)

    save_ckpt(CKPT_LATEST, q_net, TRAIN_EPISODES, note="final")


if __name__ == "__main__":
    main()