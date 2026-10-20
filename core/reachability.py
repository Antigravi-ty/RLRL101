"""core/reachability.py —— 地面可达区域 (1s / 2s / 3s 等时线 Isochrones, Boost=0) 引擎。

同时支持两种模式 (可在 Tuner 中按 `K` 键实时切换对比)：
1. `"model"` (默认, 120Hz 零阻塞, 单次推理 < 0.05 ms)：
   - 基于 RocketSim 确定性物理引擎生成的地面全状态可达数据集 (vf, vr, wz) -> (3, 72) 极坐标边界，
   - 使用双层神经网络 (2-Layer Tanh MLP) + 3D 连续特征插值混合代理模型，实现 0.02 ms 级超高速推理。
2. `"rollout"` (RocketSim 实时确定性轨迹簇推演)：
   - 直接从当前车辆状态出发，在独立物理沙盒中推演一组覆盖前进/倒车/转向/漂移的开环控制模式，
   - 带有状态增量缓存与节流机制，方便与 `"model"` 模式对比精度与帧率表现。
"""
import os
import math
import time
import numpy as np
import RocketSim as rs

N_BINS = 72
HORIZONS_SEC = (1.0, 2.0, 3.0)
VF_GRID = np.linspace(-1400.0, 2300.0, 17, dtype=np.float32)
VR_GRID = np.linspace(0.0, 1600.0, 8, dtype=np.float32)
WZ_GRID = np.array([-3.0, 0.0, 3.0], dtype=np.float32)

CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_PATH = os.path.join(CACHE_DIR, "ground_reach_mlp_v1.npz")


def _clip_arena_xy(x_w, y_w, margin=60.0):
    """Clips world (x, y) coordinates to the standard Soccar arena boundary with corner bevels."""
    x_lim = 4096.0 - margin
    y_lim = 5120.0 - margin
    # Goal opening: if |x| < 893 - margin, allow entering up to y = 5120 + 800
    in_goal_x = np.abs(x_w) < (893.0 - margin)
    y_max = np.where(in_goal_x, 5120.0 + 780.0, y_lim)
    x_c = np.clip(x_w, -x_lim, x_lim)
    y_c = np.clip(y_w, -y_max, y_max)
    # 45-degree corner bevels: |x| + |y| <= 8064 - margin
    bevel_lim = 8064.0 - margin
    s = np.abs(x_c) + np.abs(y_c)
    over = s > bevel_lim
    if np.any(over):
        scale = bevel_lim / np.maximum(s, 1.0)
        x_c = np.where(over, x_c * scale, x_c)
        y_c = np.where(over, y_c * scale, y_c)
    return x_c, y_c


class GroundReachabilityEngine:
    """Computes 1s / 2s / 3s ground reachable boundary contours (assuming boost=0)."""

    def __init__(self, mode="model"):
        self.mode = mode if mode in ("model", "rollout") else "model"
        self.angles = np.linspace(-math.pi, math.pi, N_BINS, endpoint=False, dtype=np.float32)
        self.cos_a = np.cos(self.angles)
        self.sin_a = np.sin(self.angles)

        self._arena = None
        self._car = None

        # Surrogate model weights & LUT
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.lut = None

        # Realtime rollout cache to avoid redundant recomputation when state barely changed
        self._last_rollout_key = None
        self._last_rollout_radii = None
        self._last_rollout_t = 0.0
        self.last_compute_ms = 0.0

        self._ensure_model_loaded()

    def _ensure_sim(self):
        if self._arena is None:
            mesh_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "collision_meshes",
            )
            try:
                rs.init(mesh_dir)
            except Exception:
                pass
            self._arena = rs.Arena(rs.GameMode.SOCCAR)
            self._car = self._arena.add_car(rs.Team.BLUE)
            bs = self._arena.ball.get_state()
            bs.pos = rs.Vec(0.0, 0.0, 1950.0)
            bs.vel = rs.Vec(0.0, 0.0, 0.0)
            self._arena.ball.set_state(bs)

    def simulate_state_radii(self, vf: float, vr: float, wz: float = 0.0, fast: bool = False) -> np.ndarray:
        """Uses RocketSim deterministic physics (boost=0) to compute (3, N_BINS) polar radii in car local frame."""
        self._ensure_sim()
        car = self._car
        arena = self._arena
        bs = arena.ball.get_state()
        if bs.pos.z < 1800.0:
            bs.pos = rs.Vec(0.0, 0.0, 1950.0)
            bs.vel = rs.Vec(0.0, 0.0, 0.0)
            arena.ball.set_state(bs)

        trajs = []
        if fast:
            # 22 high-leverage boundary control patterns x 6 steps of 60 ticks
            step_ticks = 60
            n_steps = 6
            horizon_steps = (2, 4, 6)
            pattern_specs = []
            for steer in (-1.0, -0.55, -0.2, 0.0, 0.2, 0.55, 1.0):
                for t_turn in (60, 180, 360):
                    if steer == 0.0 and t_turn != 60:
                        continue
                    hb = 30 if (abs(steer) > 0.8 and t_turn == 60) else 0
                    pattern_specs.append((1.0, steer, t_turn, hb))
            for steer in (-1.0, -0.5, 0.0, 0.5, 1.0):
                pattern_specs.append((-1.0, steer, 180, 0))
        else:
            # High-resolution pattern set for dataset generation
            step_ticks = 30
            n_steps = 12
            horizon_steps = (4, 8, 12)
            pattern_specs = []
            for throttle in (-1.0, 0.0, 1.0):
                steers = (-1.0, -0.65, -0.3, 0.0, 0.3, 0.65, 1.0) if throttle != 0.0 else (-1.0, 0.0, 1.0)
                t_turns = (30, 90, 180, 360) if throttle != 0.0 else (120,)
                for steer in steers:
                    for t_turn in t_turns:
                        if steer == 0.0 and t_turn != t_turns[0]:
                            continue
                        hbs = (0, 30) if (throttle > 0.0 and abs(steer) > 0.6 and t_turn >= 90) else (0,)
                        for hb in hbs:
                            pattern_specs.append((throttle, steer, t_turn, hb))

        for throttle, steer, t_turn, hb in pattern_specs:
            s = rs.CarState()
            s.pos = rs.Vec(0.0, 0.0, 17.0)
            s.vel = rs.Vec(float(vf), float(vr), 0.0)
            s.ang_vel = rs.Vec(0.0, 0.0, float(wz))
            s.boost = 0.0
            car.set_state(s)
            cx, cy = 0.0, 0.0
            pts = [(0.0, 0.0)]
            for step in range(n_steps):
                tick = step * step_ticks
                c = rs.CarControls()
                c.throttle = float(throttle)
                c.steer = float(steer) if tick < t_turn else 0.0
                c.handbrake = bool(tick < hb)
                c.boost = False
                car.set_controls(c)
                arena.step(step_ticks)
                st = car.get_state()
                cx += st.pos.x
                cy += st.pos.y
                st.pos = rs.Vec(0.0, 0.0, st.pos.z)
                car.set_state(st)
                pts.append((cx, cy))
            trajs.append(pts)

        trajs_np = np.asarray(trajs, dtype=np.float32)
        radii = np.zeros((3, N_BINS), dtype=np.float32)
        alphas = (0.2, 0.4, 0.6, 0.8, 1.0)
        for h_idx, max_step in enumerate(horizon_steps):
            sub_pts = []
            for s_i in range(max_step):
                p0 = trajs_np[:, s_i, :]
                p1 = trajs_np[:, s_i + 1, :]
                for a in alphas:
                    sub_pts.append(p0 * (1.0 - a) + p1 * a)
            all_p = np.concatenate(sub_pts, axis=0)
            ang = np.arctan2(all_p[:, 1], all_p[:, 0])
            rad = np.hypot(all_p[:, 0], all_p[:, 1])
            bins = np.floor((ang + math.pi) / (2.0 * math.pi) * N_BINS).astype(np.int32) % N_BINS
            r_max = np.zeros(N_BINS, dtype=np.float32)
            np.maximum.at(r_max, bins, rad)
            for _ in range(3):
                r_max = 0.25 * np.roll(r_max, -1) + 0.5 * r_max + 0.25 * np.roll(r_max, 1)
            radii[h_idx] = r_max
        return radii

    def _feature_expand(self, x3: np.ndarray) -> np.ndarray:
        """Expands normalized (vf, vr, wz) into rich nonlinear + polynomial + Fourier features for the MLP."""
        vf = x3[:, 0:1]
        vr = x3[:, 1:2]
        wz = x3[:, 2:3]
        spd = np.hypot(vf, vr)
        feats = [
            vf, vr, wz, spd,
            vf * vf, vr * vr, vf * vr, vf * wz, vr * wz,
            np.sin(vf * 2.0), np.cos(vf * 2.0),
            np.sin(vr * 2.0), np.cos(vr * 2.0),
            np.sin(wz * 2.0), np.cos(wz * 2.0),
        ]
        return np.concatenate(feats, axis=1).astype(np.float32)

    def _ensure_model_loaded(self):
        if os.path.exists(MODEL_CACHE_PATH):
            try:
                data = np.load(MODEL_CACHE_PATH)
                self.W1 = data["W1"].astype(np.float32)
                self.b1 = data["b1"].astype(np.float32)
                self.W2 = data["W2"].astype(np.float32)
                self.b2 = data["b2"].astype(np.float32)
                self.lut = data["lut"].astype(np.float32)
                return
            except Exception:
                pass
        self._build_and_save_dataset_and_mlp()

    def _build_and_save_dataset_and_mlp(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        lut = np.zeros((len(VF_GRID), len(VR_GRID), len(WZ_GRID), 3, N_BINS), dtype=np.float32)
        for i, vf in enumerate(VF_GRID):
            for j, vr in enumerate(VR_GRID):
                for k, wz in enumerate(WZ_GRID):
                    if math.hypot(vf, vr) > 2350.0 and j > 0:
                        lut[i, j, k] = lut[i, j - 1, k]
                    else:
                        lut[i, j, k] = self.simulate_state_radii(float(vf), float(vr), float(wz), fast=True)

        # Build symmetric training dataset (exploit left-right reflection symmetry: vr -> -vr, wz -> -wz)
        X_list, Y_list = [], []
        for i, vf in enumerate(VF_GRID):
            for j, vr in enumerate(VR_GRID):
                for k, wz in enumerate(WZ_GRID):
                    X_list.append([vf / 2300.0, vr / 2300.0, wz / 5.5])
                    Y_list.append(lut[i, j, k].reshape(-1))
                    if j > 0 or k != 1:
                        mirrored = lut[i, j, k, :, ::-1]
                        X_list.append([vf / 2300.0, -vr / 2300.0, -wz / 5.5])
                        Y_list.append(mirrored.reshape(-1))

        X = np.asarray(X_list, dtype=np.float32)
        Y = np.asarray(Y_list, dtype=np.float32)
        F = self._feature_expand(X)

        rng = np.random.default_rng(20261019)
        in_dim = F.shape[1]
        hid_dim = 320
        W1 = (rng.normal(0.0, 1.35, size=(in_dim, hid_dim)) / math.sqrt(in_dim)).astype(np.float32)
        b1 = rng.uniform(-1.2, 1.2, size=(1, hid_dim)).astype(np.float32)
        H = np.tanh(F @ W1 + b1)
        H_aug = np.concatenate([F, H], axis=1)
        reg = 1e-4 * np.eye(H_aug.shape[1], dtype=np.float32)
        W2 = np.linalg.solve(H_aug.T @ H_aug + reg, H_aug.T @ Y).astype(np.float32)
        b2 = np.zeros((1, Y.shape[1]), dtype=np.float32)

        self.W1, self.b1, self.W2, self.b2, self.lut = W1, b1, W2, b2, lut
        np.savez_compressed(MODEL_CACHE_PATH, W1=W1, b1=b1, W2=W2, b2=b2, lut=lut)

    def predict_radii_model(self, vf: float, vr: float, wz: float = 0.0) -> np.ndarray:
        """Pure-NumPy neural surrogate + trilinear LUT blend (< 0.05 ms per call)."""
        vf_c = float(np.clip(vf, VF_GRID[0], VF_GRID[-1]))
        vr_c = float(np.clip(vr, -VR_GRID[-1], VR_GRID[-1]))
        wz_c = float(np.clip(wz, WZ_GRID[0], WZ_GRID[-1]))

        # 1. Fast MLP forward pass
        x_in = np.array([[vf_c / 2300.0, vr_c / 2300.0, wz_c / 5.5]], dtype=np.float32)
        f_in = self._feature_expand(x_in)
        h = np.tanh(f_in @ self.W1 + self.b1)
        h_aug = np.concatenate([f_in, h], axis=1)
        pred_mlp = (h_aug @ self.W2 + self.b2).reshape(3, N_BINS)

        # 2. Exact trilinear LUT interpolation using symmetry for vr < 0
        flip = vr_c < 0.0
        vr_abs = abs(vr_c)
        wz_sym = -wz_c if flip else wz_c

        u = (vf_c - VF_GRID[0]) / (VF_GRID[-1] - VF_GRID[0]) * (len(VF_GRID) - 1)
        v = (vr_abs - VR_GRID[0]) / (VR_GRID[-1] - VR_GRID[0]) * (len(VR_GRID) - 1)
        w = (wz_sym - WZ_GRID[0]) / (WZ_GRID[-1] - WZ_GRID[0]) * (len(WZ_GRID) - 1)

        i0 = int(np.clip(math.floor(u), 0, len(VF_GRID) - 2))
        j0 = int(np.clip(math.floor(v), 0, len(VR_GRID) - 2))
        k0 = int(np.clip(math.floor(w), 0, len(WZ_GRID) - 2))
        du, dv, dw = u - i0, v - j0, w - k0

        c00 = self.lut[i0, j0, k0] * (1 - du) + self.lut[i0 + 1, j0, k0] * du
        c01 = self.lut[i0, j0, k0 + 1] * (1 - du) + self.lut[i0 + 1, j0, k0 + 1] * du
        c10 = self.lut[i0, j0 + 1, k0] * (1 - du) + self.lut[i0 + 1, j0 + 1, k0] * du
        c11 = self.lut[i0, j0 + 1, k0 + 1] * (1 - du) + self.lut[i0 + 1, j0 + 1, k0 + 1] * du
        c0 = c00 * (1 - dv) + c10 * dv
        c1 = c01 * (1 - dv) + c11 * dv
        pred_lut = c0 * (1 - dw) + c1 * dw
        if flip:
            pred_lut = pred_lut[:, ::-1]

        # Blend 50% smooth neural MLP + 50% exact grid interpolation, enforce monotonicity r_1s <= r_2s <= r_3s
        radii = np.maximum(60.0, 0.5 * pred_mlp + 0.5 * pred_lut)
        radii[1] = np.maximum(radii[1], radii[0] + 40.0)
        radii[2] = np.maximum(radii[2], radii[1] + 40.0)
        return radii

    def compute_world_contours(self, car_pos, car_vel, fwd_dir, right_dir, ang_vel=(0.0, 0.0, 0.0)):
        """Returns 3 world-space closed contour arrays [(N+1, 3), (N+1, 3), (N+1, 3)] for 1s, 2s, 3s."""
        t0 = time.perf_counter()
        px, py, pz = float(car_pos[0]), float(car_pos[1]), float(car_pos[2])
        vx, vy = float(car_vel[0]), float(car_vel[1])
        fx, fy = float(fwd_dir[0]), float(fwd_dir[1])
        fn = math.hypot(fx, fy)
        if fn > 1e-4:
            fx, fy = fx / fn, fy / fn
        else:
            fx, fy = 0.0, 1.0
        # Right vector on horizontal ground plane
        rx, ry = float(right_dir[0]), float(right_dir[1])
        rn = math.hypot(rx, ry)
        if rn > 1e-4:
            rx, ry = rx / rn, ry / rn
        else:
            rx, ry = -fy, fx

        vf = vx * fx + vy * fy
        vr = vx * rx + vy * ry
        wz = float(ang_vel[2]) if len(ang_vel) >= 3 else 0.0

        if self.mode == "rollout":
            now = time.perf_counter()
            key = (round(vf / 35.0), round(vr / 35.0), round(wz * 2.0))
            if (self._last_rollout_radii is not None
                    and (self._last_rollout_key == key or (now - self._last_rollout_t) < 0.08)):
                radii = self._last_rollout_radii
            else:
                radii = self.simulate_state_radii(vf, vr, wz, fast=True)
                self._last_rollout_key = key
                self._last_rollout_radii = radii
                self._last_rollout_t = now
        else:
            radii = self.predict_radii_model(vf, vr, wz)

        self.last_compute_ms = (time.perf_counter() - t0) * 1000.0

        contours = []
        z_ground = 22.0
        for h_idx in range(3):
            r = radii[h_idx]
            lx = r * self.cos_a
            ly = r * self.sin_a
            wx = px + lx * fx + ly * rx
            wy = py + lx * fy + ly * ry
            wx, wy = _clip_arena_xy(wx, wy)
            # Close the loop
            wx_c = np.append(wx, wx[0])
            wy_c = np.append(wy, wy[0])
            wz_c = np.full_like(wx_c, z_ground + h_idx * 2.0)
            contours.append(np.stack([wx_c, wy_c, wz_c], axis=1).astype(np.float32))
        return contours

    def toggle_mode(self) -> str:
        self.mode = "rollout" if self.mode == "model" else "model"
        self._last_rollout_key = None
        return self.mode
