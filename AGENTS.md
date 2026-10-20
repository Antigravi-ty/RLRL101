# RLRL101 Agent Guide

## 1. Environment & Dependencies

- **Verification / Physics / Curriculum / Reward / Headless Tuner / Checkpoint Evaluation**:
  Only `rocketsim` and `numpy` are required (~5 MB total, prebuilt wheel, no C++ compiler needed):
  ```bash
  pip install rocketsim numpy
  ```
  **Do NOT install PyTorch (`torch`) unless you are explicitly running PPO gradient training (`train/trainer.py`).** Even `.pt` checkpoint loading (`train/checkpoint.py`) and policy inference (`core/policy_net.py`, `vis/tuner/auto_session.py`, `tools/speed_suite.py`, `vis/cli_tuner.py`) have a built-in pure-Python/NumPy forward engine and work 100% without PyTorch installed.

- **Training & Local GUI Visualization (Optional, only when needed)**:
  - PPO Training: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  - 3D GUI Tuner (`python3 run.py tuner`): `pip install pyqt5 pyqtgraph PyOpenGL`

---

## 2. Mandatory Headless Tuner Verification (MUST RUN Before Every Commit)

Even in a headless container without a display server (`DISPLAY`) or `PyQt5`/`pyqtgraph` installed, **you CAN and MUST verify that `python3 run.py tuner` starts and runs cleanly** by executing the headless Qt/pyqtgraph mock verifier:

```bash
python3 run.py verify_tuner
# or directly:
python3 tools/verify_tuner.py
```

### Why & What It Verifies
1. **Never Swallow Runtime Errors as "GUI Unavailable"**:
   `vis/reward_tuner.py` only catches `ImportError` when importing `pyqtgraph`/`Qt`. Any `AttributeError`, `KeyError`, or `TypeError` inside `vis/tuner/*` will raise a real traceback instead of falsely printing `GUI (PyQt/pyqtgraph) unavailable`.
2. **100% Real Execution of `vis/tuner/*` Without a Display**:
   `tools/verify_tuner.py` injects lightweight headless stubs for `pyqtgraph`, `pyqtgraph.opengl`, and `pyqtgraph.Qt` into `sys.modules`, then instantiates the real `RewardTuner` window and exercises:
   - `RewardTuner.__init__`, `_build_ui()`, `_sync_env_curriculum_from_ckpt()`, `_refresh_ui_state()`, `_refresh_hud()`, and `_render()`
   - `format_hud()` across all 41 `CurriculumStepSpec` steps (`.name`, `.code`, focus header, reward breakdown)
   - Full state machine transitions: `STATE_IDLE` $\to$ `STATE_RECORDING` (including boost pad pickup rewards) $\to$ `STATE_DONE` $\to$ `STATE_PLAYBACK` $\to$ `STATE_FREERUN`
   - All right-side panels: `plot_mode` (`1..6`), `OBS_GROUPS` (`A..G`), `flags` view, and `delta` toggle
   - `ScenarioEditor` (`[X]` toggle, `[Tab]` 3-way target cycle `CAR -> VEL -> BALL(camera focused on ball + forward along ball launch velocity) -> CAR`, keyboard/mouse drag & adjustments, and `_launch_from_editor()`)
   - `CarStatePanel` (`[Z]` shortcut toggle, `Boost Amount` slider + spinbox + presets, and `Has Jump` checkbox)
   - 3D `BALL-CAM` POV camera pitch (`vis/camera.py` tilts up/down according to ball elevation relative to car)
   - `MODEL` + `MANUAL-RESET` QoL: pressing `N` immediately simulates and draws the full 3D trajectory (`STATE_DONE`), and pressing `R` replays step-by-step live (`STATE_RECORDING`)

**Rule for Agents**: Whenever you modify `core/*`, `train/*`, or `vis/*`, you **MUST** run `python3 run.py verify_tuner` and confirm `[verify_tuner] ALL OK!` before committing.

---

## 3. Recommended Training Parameters & Dual-Format Logging

```bash
python3 run.py train \
  --stage auto \
  --from none \
  --save-prefix sbm_v1 \
  --workers 6 \
  --rollout-steps 2048 \
  --batch-size 512 \
  --epochs 4 \
  --lr 3e-4 \
  --lr-end 3e-5 \
  --lr-period 500000 \
  --ent-coef 0.005 \
  --sym-coef 0.02 \
  --episodes 0
```

- **Dual-Format Logging (`train/logger.py` & `tools/log_to_csv.py`)**:
  - `logs/<prefix>.log`: Human-readable console mirror with timestamps.
  - `logs/<prefix>_train.csv`: Compact numeric table of rollout & curriculum telemetry (~60-77% smaller than `.log`).
  - `logs/<prefix>_events.csv`: Sparse milestone table of curriculum step advances and best-checkpoint duels.

### Quick Commands
```bash
# Mandatory Headless Tuner Verification
python3 run.py verify_tuner

# GUI Tuner (X: 3-way Tab editor CAR/VEL/BALL | Z: Boost & HasJump panel | N in MANUAL-RESET: instant trajectory | R: step-by-step replay)
python3 run.py tuner --prefix sbm_v3 --from best --cam attach

# Convert existing .log to agent-friendly _train.csv and _events.csv
python3 run.py log_to_csv logs/sbm_v3.log

# CLI Evaluation (5-Phase matrix + 5-Direction F/B/R/L/U diagnostics)
python3 run.py eval --prefix sbm_v3 --dir5

# 40-Scenario Deterministic Skill Suite (SpeedFlip / HalfFlip / Powerslide / UpPyramid / LowAltJump)
python3 run.py suite --ckpt checkpoints/sbm_v3_best.pt
```
