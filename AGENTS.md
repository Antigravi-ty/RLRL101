# RLRL101 Agent Guide — 1v1 Fastest Kickoff Branch (`kickoff`)

## 1. Core Architecture & Design (`kickoff` Orphan Branch)

1. **RocketSim Native Kickoff + 60-Frame Suspension Settling (`core/env.py`)**:
   - Uses RocketSim's native `arena.reset_kickoff(seed)` across all 5 standard kickoff positions (`0:DiagRight`, `1:DiagLeft`, `2:BackRight`, `3:BackLeft`, `4:Center`).
   - Steps the physics engine for **`SETTLE_TICKS = 60` frames (0.50s)** with zero controls (`rs.CarControls()`) before `tick = 0` so the Octane suspension settles cleanly onto the ground (`z = 17.0320`, `is_on_ground = True`) before allowing model or human inputs.
2. **1v1 Self-Play First-Touch Duel & Potential + Non-Linear Time Reward (`core/reward.py`)**:
   - Two cars (Blue `#1` vs. Orange `#2`) race to the center kickoff ball (`MAX_TICKS = 720`, 6.0s limit).
   - **Distance Potential (`r_dist`)**: `distance_delta_scale * (prev_dist - eff_new_dist)` — moving closer than the previous frame adds points (faster closing speed = higher per-step score); moving farther deducts points.
   - **Non-Linear Fast-Touch Bonus (`r_touch`)**: First car to touch the ball receives `touch_bonus + touch_time_bonus_max * (speed_ratio ** 3.0)` (cubic scaling strongly rewards saving every frame around ~1.95s–2.10s to incentivize Speed Flip emergence).
   - **Loser / Timeout Penalty (`r_timeout`)**: The slower car that loses the first touch is immediately penalized (`-lose_penalty - lose_dist_penalty_scale * rem_dist`).
   - **Long-Horizon Discounting (`core/constants.py`, `train/ppo_loss.py`)**: `GAMMA = 0.998`, `GAE_LAMBDA = 0.975` so terminal touch rewards propagate strongly back across 3+ seconds (90+ decisions) to initial takeoff frames.
3. **Minimal Parameter Count & Maximum Speed (`core/obs.py`, `core/policy_net.py`, `train/vec_env.py`)**:
   - **42-Dim Team & Left/Right Mirror-Symmetric Observation (`OBS_DIM = 42`)**: Ego-centric body-frame telemetry + opponent relative state, with exact Blue/Orange team symmetry and bilateral left/right mirror loss (`sym_coef = 0.02`).
   - **Pure 2-Layer MLP (`hidden=128`)**: Only **23,951 Actor inference parameters** (46,096 total with Critic), zero attention overhead, supporting fast NumPy CPU rollout in 8 parallel workers (`MultiCoreVecEnv`) + `mps` (Apple Silicon GPU) / `cuda` / `cpu` backprop.

---

## 2. Mandatory Headless Tuner Verification

```bash
python3 run.py verify_tuner
```

`tools/verify_tuner.py` verifies:
- All 5 standard kickoff slots (`DiagRight`, `DiagLeft`, `BackRight`, `BackLeft`, `Center`) + 60-frame suspension settling (`SETTLE_TICKS = 60`).
- 42D team/mirror observation symmetry and 46,096-param Pure-MLP `ActorCritic`.
- `RewardTuner` GUI startup, Human recording (`WASD`, `/` Jump, `.` Boost, `Shift` Slide, `Q/E` Roll), Mid-trajectory Takeover (rewind with `←/→`, `[/]`, `I/K`, `,/;` then press any driving key or `X` to take over from that frame and race the original trajectory), `R` (reset & replay current kickoff slot), `N` (switch to a new kickoff slot), `Shift+1..5` (lock kickoff slot `0..4`), `Enter` (export CSV to `data/*.csv`), `M` (toggle `HUMAN` ↔ `MODEL 1v1`), `9`/`0` (switch `#1 BLUE` / `#2 ORANGE`), and bottom charts (`1..6`, `OBS`, `FLAGS`, `G`).

---

## 3. Recommended Local Training & Tuner Commands

```bash
# 1. Start 8-Worker 1v1 Self-Play PPO Training (auto-selects Apple MPS / CUDA / CPU)
python3 run.py train \
  --save-prefix ko_v1 \
  --workers 8 \
  --rollout-steps 4096 \
  --batch-size 512 \
  --epochs 4 \
  --gamma 0.998 \
  --gae-lambda 0.975 \
  --lr 3e-4 \
  --lr-end 3e-5 \
  --ent-coef 0.005 \
  --sym-coef 0.02

# 2. Launch Interactive Kickoff Tuner (Human Recording + Model 1v1 Comparison + CSV Export)
python3 run.py tuner --prefix ko_v1 --from best

# 3. CLI 5-Slot Kickoff Evaluation & Human vs. Model CSV Comparison
python3 run.py cli --prefix ko_v1
python3 run.py cli --verify data/_S0_..._human.csv data/_S0_..._model.csv

# 4. Multi-Worker Throughput Benchmark
python3 run.py bench --steps 4096 --workers 1 4 8
```
