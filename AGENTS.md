# RLRL101 Agent Guide (`air_dribble` Branch — Freestyle Air-Dribble Showman)

## 1. Environment & Dependencies

- **Verification / Physics / Curriculum / Reward / Headless Tuner / Checkpoint Evaluation**:
  Only `rocketsim` and `numpy` are required (~5 MB total, prebuilt wheel, no C++ compiler needed):
  ```bash
  pip install rocketsim numpy
  ```
  **Do NOT install PyTorch (`torch`) unless you are explicitly running PPO gradient training (`train/trainer.py`).** Even `.pt` checkpoint loading (`train/checkpoint.py`) and policy inference (`core/policy_net.py`, `vis/tuner/auto_session.py`, `vis/cli_tuner.py`) have a built-in pure-Python/NumPy forward engine and work 100% without PyTorch installed.

- **Training & Local GUI Visualization (Optional, only when needed)**:
  - PPO Training: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  - 3D GUI Tuner (`python3 run.py tuner`): `pip install pyqt5 pyqtgraph PyOpenGL`

---

## 2. Core Mechanics: Central Air-Box Chained Distance × Touch Multiplier

1. **Central Air-Box ("离开任何平面的中间盒子区")**:
   - The ball enters the scoring **Air-Box** when it detaches from all 6 arena planes (floor, ceiling, side walls, back walls, and 45° corners): `bz >= 220.0` and `surface_clearance >= 180.0`.
2. **Chained Touch-to-Touch Path Distance $\times$ Touch Multiplier**:
   - Every valid airborne touch inside the Air-Box ($k = 1, 2, \dots, K$) records the ball's 3D coordinates $\vec{P}_k$.
   - The chained path distance accumulates segment by segment:
     $$D_{\text{chain}} = \sum_{i=2}^{K} \|\vec{P}_i - \vec{P}_{i-1}\|_2$$
     weighted by segment altitude $H(\bar{z}_i)$ ("越高越好，全套距离越长越好").
   - The total combo score is multiplied by the touch multiplier $M_K \approx K - 1$:
     - 1 long kick + 1 catch ($K=2$) earns only $1\times$ distance;
     - 5 continuous air touches ($K=5$) earn $(d_{1\to 2} + d_{2\to 3} + d_{3\to 4} + d_{4\to 5}) \times 4$!
   - **Telescopic Step Delta ($\Delta S_k = S_k - S_{k-1}$)**: Paid immediately on each Air-Box touch so the sum of rewards over the chain equals the exact terminal formula while preventing PPO from intentionally dropping the ball early due to $\gamma$ discounting.
   - When the ball touches the floor or any surface plane (`bz <= 140` or `surface_clearance <= 138`), the current combo chain settles and resets to `0`, while the car continues running in the arena to pop/launch the ball again.
3. **No Goals Allowed (`-150.0`)**:
   - Scoring a goal (`|ball_y| > 5120`) triggers a massive `-150.0` penalty and terminates the episode.
4. **5 Freestyle Scenario Buckets (`C/A/B/W/G`) & 3 Boost Modes**:
   - Scenario buckets: `Carry` (mid-air carry), `Catch` (dynamic lobbed ball catch), `Bounce` (bouncing ball pop), `Wall` (side-wall rolling ball launch), and `Ground` (grounded/rolling ball autonomous pop).
   - Boost modes (`--boost-mode`): `unlimited` (default, locked at 100), `recharge` (regenerates in air & on air touches), `pads` (standard 34 boost pads).

---

## 3. Mandatory Headless Tuner Verification (MUST RUN Before Every Commit)

```bash
python3 run.py verify_tuner
# or directly:
python3 tools/verify_tuner.py
```

**Rule for Agents**: Whenever you modify `core/*`, `train/*`, or `vis/*`, you **MUST** run `python3 run.py verify_tuner` and confirm `[verify_tuner] ALL OK!` before committing.

---

## 4. Recommended Training & Tuner Commands

```bash
# Train Freestyle Air-Dribble Showman (Ctrl+C safely saves latest.pt at any time)
python3 run.py train \
  --stage auto \
  --from latest \
  --save-prefix air_dribble_v1 \
  --boost-mode unlimited \
  --workers 6 \
  --rollout-steps 2048 \
  --batch-size 256 \
  --epochs 4 \
  --lr 3e-4 \
  --lr-end 3e-5 \
  --ent-coef 0.005 \
  --sym-coef 0.02 \
  --episodes 0

# Live GUI Tuner (press M for MODEL, B to toggle latest/best, Shift+5..8 for Carry/Catch/Wall/Ground)
python3 run.py tuner --prefix air_dribble_v1 --from latest --cam ball

# CLI Evaluation (4-Phase matrix + 5-Bucket Carry/Catch/Bounce/Wall/Ground diagnostics)
python3 run.py eval --prefix air_dribble_v1 --dir5
```
