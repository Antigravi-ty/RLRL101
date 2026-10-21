"""vis/tuner/editor.py —— 交互式 3D 场景编辑器 (ScenarioEditor: Tab 键三态切换【车头方向 CAR】↔【初速度方向 VEL】↔【球发射方向 BALL】)。"""
import math
import numpy as np
from core.curriculum import clamp_safe_arena_xy
from core.env import euler_to_rot_vectors, get_adaptive_max_decisions
from core.spawner import apply_car_jump_state, dist_to_tier


class ScenarioEditor:
    TARGETS = ("car", "vel", "ball")
    TARGET_LABELS = {
        "car":  "CAR (车头方向)",
        "vel":  "VEL (Init速度方向)",
        "ball": "BALL (手动发球/球速矢量方向)",
    }

    def __init__(self, env):
        self.env = env
        self.active = False
        self.target = "car"  # "car" (车头方向) | "vel" (init速度方向) | "ball" (手动发球/球速矢量方向)

        # Editable Car parameters
        self.cx = 0.0
        self.cy = -1000.0
        self.cz = 17.0
        self.car_yaw = 90.0      # degrees [-180, 180]
        self.car_pitch = 0.0     # degrees [-85, 85]
        self.car_roll = 0.0      # degrees [-180, 180]

        # Editable Initial Velocity parameters (independent of car heading!)
        self.vel_yaw = 90.0      # degrees [-180, 180]
        self.vel_pitch = 0.0     # degrees [-90, 0] (horizontal or downward, never upward)
        self.car_speed = 0.0     # uu/s [0, 2300]

        # Editable Ball Position & Launch Velocity Vector (for BALL edit mode)
        self.ball_yaw = 90.0     # degrees [-180, 180] (Ball velocity / launch heading)
        self.ball_pitch = 0.0    # degrees [-85, 85]   (Ball velocity / launch pitch)
        self.ball_speed = 0.0    # uu/s [0, 4000]      (Ball initial launch speed)
        self.ball_dist = 1000.0  # uu [160, 6000]      (Car-to-Ball 3D distance)
        self.bx = 0.0
        self.by = 0.0
        self.bz = 93.15

        self.car_boost = 33.0    # [0, 100]
        self.is_jumping = False
        self.has_second_jump = True

        # Optional persistent manual overrides (can be toggled/adjusted anytime via [Z] panel or sliders)
        self.manual_override_enabled = False
        self.manual_override_locked = False

        self._held = set()
        self._last_t = None

    def _sync_ball_ray_from_positions(self):
        dx = self.bx - self.cx
        dy = self.by - self.cy
        dz = self.bz - self.cz
        dist = math.hypot(dx, dy, dz)
        self.ball_dist = float(max(160.0, dist))

    def _sync_ball_pos_from_ray(self):
        dx = self.bx - self.cx
        dy = self.by - self.cy
        dz = self.bz - self.cz
        dist = math.hypot(dx, dy, dz)
        if dist > 1e-4:
            ux, uy, uz = dx / dist, dy / dist, dz / dist
        else:
            ux, uy, uz = self.ball_dir_vec()
        self.bx, self.by = clamp_safe_arena_xy(
            self.cx + self.ball_dist * ux,
            self.cy + self.ball_dist * uy,
            margin=60.0,
        )
        self.bz = float(np.clip(self.cz + self.ball_dist * uz, 93.15, 1900.0))

    def load_from_current_env(self):
        """Initialize editor parameters from the environment's current spawn state."""
        sd = self.env.last_spawn_state
        if sd is not None:
            bp = sd["ball_pos"]
            bv = sd.get("ball_vel", (0.0, 0.0, 0.0))
            cp = sd["car_pos"]
            cv = sd["car_vel"]
            cf = sd["car_fwd"]
            cu = sd["car_up"]
            cr = sd["car_rgt"]
            self.bx, self.by = clamp_safe_arena_xy(float(bp[0]), float(bp[1]), margin=60.0)
            self.bz = float(bp[2])
            self.cx, self.cy = clamp_safe_arena_xy(float(cp[0]), float(cp[1]), margin=40.0)
            self.cz = float(cp[2])
            self.car_yaw = float(math.degrees(math.atan2(cf[1], cf[0])))
            self.car_pitch = float(math.degrees(math.asin(np.clip(cf[2], -0.999, 0.999))))
            self.car_roll = float(math.degrees(math.atan2(-cr[2], cu[2]))) if abs(cu[2]) > 1e-4 or abs(cr[2]) > 1e-4 else 0.0

            cspd = math.hypot(cv[0], cv[1], cv[2])
            self.car_speed = float(cspd)
            if cspd > 1.0:
                self.vel_yaw = float(math.degrees(math.atan2(cv[1], cv[0])))
                self.vel_pitch = float(np.clip(math.degrees(math.asin(np.clip(cv[2] / cspd, -1.0, 1.0))), -90.0, 0.0))
            else:
                self.vel_yaw = self.car_yaw
                self.vel_pitch = 0.0

            bspd = math.hypot(bv[0], bv[1], bv[2])
            self.ball_speed = float(bspd)
            if bspd > 1.0:
                self.ball_yaw = float(math.degrees(math.atan2(bv[1], bv[0])))
                self.ball_pitch = float(np.clip(math.degrees(math.asin(np.clip(bv[2] / bspd, -0.999, 0.999))), -85.0, 85.0))
            else:
                dx = self.cx - self.bx
                dy = self.cy - self.by
                if math.hypot(dx, dy) > 1e-3:
                    self.ball_yaw = float(math.degrees(math.atan2(dy, dx)))
                else:
                    self.ball_yaw = self.car_yaw
                self.ball_pitch = 0.0

            if not self.manual_override_enabled:
                self.car_boost = float(sd.get("car_boost", 33.0))
                self.is_jumping = bool(sd.get("is_jumping", False))
                self.has_second_jump = bool(sd.get("has_second_jump", True))
            self._sync_ball_ray_from_positions()
        self.sync_to_arena()

    def apply_manual_car_state(self, boost_amount: float = None, has_jump: bool = None, is_jumping: bool = None, lock_override: bool = None):
        """Manually sets Boost Amount (0..100) and Has Jump / Second Jump flag on the current scenario and environment."""
        if boost_amount is not None:
            self.car_boost = float(np.clip(float(boost_amount), 0.0, 100.0))
        if has_jump is not None:
            self.has_second_jump = bool(has_jump)
        if is_jumping is not None:
            self.is_jumping = bool(is_jumping)
        if lock_override is not None:
            self.manual_override_enabled = bool(lock_override)
            self.manual_override_locked = bool(lock_override)

        if self.active:
            self.sync_to_arena()
            return

        sd = self.env.last_spawn_state
        if sd is not None:
            sd["car_boost"] = float(self.car_boost)
            sd["has_second_jump"] = bool(self.has_second_jump)
            sd["is_jumping"] = bool(self.is_jumping)
            if not self.has_second_jump and sd.get("on_ground", True):
                # To test Has Jump = False, lift car slightly off ground (cz=22) so RocketSim preserves has_double_jumped=True
                cp = sd.get("car_pos", (0.0, 0.0, 17.0))
                sd["car_pos"] = (float(cp[0]), float(cp[1]), max(22.0, float(cp[2])))
                sd["on_ground"] = False
            self.env._apply_spawn_state(sd)
            self.env.get_obs()

    def toggle_target(self) -> str:
        idx = self.TARGETS.index(self.target) if self.target in self.TARGETS else 0
        self.target = self.TARGETS[(idx + 1) % len(self.TARGETS)]
        return self.target

    def ball_dir_vec(self):
        y = math.radians(self.ball_yaw)
        p = math.radians(self.ball_pitch)
        return (
            math.cos(p) * math.cos(y),
            math.cos(p) * math.sin(y),
            math.sin(p),
        )

    def vel_dir_vec(self):
        y = math.radians(self.vel_yaw)
        p = math.radians(self.vel_pitch)
        return (
            math.cos(p) * math.cos(y),
            math.cos(p) * math.sin(y),
            math.sin(p),
        )

    def car_axes_vecs(self):
        y = math.radians(self.car_yaw)
        p = math.radians(self.car_pitch)
        r = math.radians(self.car_roll)
        return euler_to_rot_vectors(y, p, r)

    def cycle_jump_variation(self) -> str:
        """Cycles through the jump states:
        (is_jumping=False, has_second_jump=True) -> (is_jumping=True, has_second_jump=True) -> (is_jumping=False, has_second_jump=False)
        """
        if not self.is_jumping and self.has_second_jump:
            self.is_jumping = True
            self.has_second_jump = True
            return "一段跳上升中 (is_jumping=True, HasJump=ON)"
        elif self.is_jumping:
            self.is_jumping = False
            self.has_second_jump = False
            return "无跳/二段跳已耗尽 (is_jumping=False, HasJump=OFF)"
        else:
            self.is_jumping = False
            self.has_second_jump = True
            return "拥有跳跃/二段跳 (is_jumping=False, HasJump=ON)"

    def on_mouse_drag(self, dx_px: float, dy_px: float):
        """Rotate active target's 3D direction via mouse left-drag."""
        if self.target == "ball":
            self.ball_yaw = float(((self.ball_yaw + dx_px * 0.45 + 180.0) % 360.0) - 180.0)
            self.ball_pitch = float(np.clip(self.ball_pitch - dy_px * 0.35, -85.0, 85.0))
        elif self.target == "vel":
            self.vel_yaw = float(((self.vel_yaw + dx_px * 0.45 + 180.0) % 360.0) - 180.0)
            # Velocity vertical pitch is strictly horizontal or downward in [-90°, 0°]
            self.vel_pitch = float(np.clip(self.vel_pitch - dy_px * 0.35, -90.0, 0.0))
            if self.car_speed < 50.0:
                self.car_speed = 600.0
        else:
            self.car_yaw = float(((self.car_yaw + dx_px * 0.45 + 180.0) % 360.0) - 180.0)
            self.car_pitch = float(np.clip(self.car_pitch - dy_px * 0.35, -85.0, 85.0))
        self.sync_to_arena()

    def step_held_keys(self, now: float, cam_forward_world=(0.0, 1.0)):
        if self._last_t is None:
            self._last_t = now
            return
        dt = min(0.05, max(0.001, now - self._last_t))
        self._last_t = now
        if not self._held:
            return

        fx_w, fy_w = float(cam_forward_world[0]), float(cam_forward_world[1])
        fn = math.hypot(fx_w, fy_w)
        if fn > 1e-4:
            fx_w, fy_w = fx_w / fn, fy_w / fn
        else:
            fx_w, fy_w = 0.0, 1.0
        rx_w, ry_w = -fy_w, fx_w

        move_xy = 1800.0 * dt
        move_z  = 600.0 * dt

        dx_w, dy_w, dz_w = 0.0, 0.0, 0.0
        if "W" in self._held:
            dx_w += fx_w * move_xy
            dy_w += fy_w * move_xy
        if "S" in self._held:
            dx_w -= fx_w * move_xy
            dy_w -= fy_w * move_xy
        if "D" in self._held:
            dx_w += rx_w * move_xy
            dy_w += ry_w * move_xy
        if "A" in self._held:
            dx_w -= rx_w * move_xy
            dy_w -= ry_w * move_xy
        if "UP" in self._held:
            dz_w += move_z
        if "DOWN" in self._held:
            dz_w -= move_z

        if self.target == "ball":
            if dx_w != 0.0 or dy_w != 0.0 or dz_w != 0.0:
                self.bx, self.by = clamp_safe_arena_xy(self.bx + dx_w, self.by + dy_w, margin=60.0)
                self.bz = float(np.clip(self.bz + dz_w, 93.15, 1900.0))
                self._sync_ball_ray_from_positions()
            if "LBRACKET" in self._held:
                self.ball_speed = float(max(0.0, self.ball_speed - 1200.0 * dt))
            if "RBRACKET" in self._held:
                self.ball_speed = float(min(4000.0, self.ball_speed + 1200.0 * dt))
        else:
            if dx_w != 0.0 or dy_w != 0.0 or dz_w != 0.0:
                self.cx, self.cy = clamp_safe_arena_xy(self.cx + dx_w, self.cy + dy_w, margin=40.0)
                self.cz = float(np.clip(self.cz + dz_w, 17.0, 75.0))
                self._sync_ball_ray_from_positions()
            if "Q" in self._held:
                self.car_roll = float(((self.car_roll - 130.0 * dt + 180.0) % 360.0) - 180.0)
            if "E" in self._held:
                self.car_roll = float(((self.car_roll + 130.0 * dt + 180.0) % 360.0) - 180.0)
            if "LBRACKET" in self._held:
                self.car_speed = float(max(0.0, self.car_speed - 1100.0 * dt))
            if "RBRACKET" in self._held:
                self.car_speed = float(min(2300.0, self.car_speed + 1100.0 * dt))
        if "9" in self._held:
            self.car_boost = float((self.car_boost - 60.0 * dt) % 101.0)

        self.sync_to_arena()

    def build_spawn_state(self) -> dict:
        self.cx, self.cy = clamp_safe_arena_xy(self.cx, self.cy, margin=40.0)
        self.bx, self.by = clamp_safe_arena_xy(self.bx, self.by, margin=60.0)
        fwd, rgt, up = self.car_axes_vecs()
        vx_u, vy_u, vz_u = self.vel_dir_vec()
        cvx = self.car_speed * vx_u
        cvy = self.car_speed * vy_u
        cvz = self.car_speed * vz_u

        bvx_u, bvy_u, bvz_u = self.ball_dir_vec()
        bvx = self.ball_speed * bvx_u
        bvy = self.ball_speed * bvy_u
        bvz = self.ball_speed * bvz_u

        eff_cz = float(self.cz)
        if not self.has_second_jump and eff_cz <= 20.0:
            eff_cz = 22.0
        on_ground = bool(eff_cz <= 20.0 and abs(self.car_pitch) < 5.0 and abs(self.car_roll) < 5.0 and self.has_second_jump and (not self.is_jumping))
        dist_3d = math.hypot(self.bx - self.cx, self.by - self.cy, self.bz - eff_cz)
        tier_idx = dist_to_tier(dist_3d)
        max_dec = get_adaptive_max_decisions(dist_3d, self.bz, car_spd=self.car_speed)

        sd = {
            "mode": "custom_edit",
            "stage_idx": 0,
            "dist_tier": int(tier_idx),
            "max_decisions": int(max_dec),
            "is_hover_ball": (self.ball_speed <= 1.0),
            "is_high_band_sample": True,
            "on_ground": on_ground,
            "is_jumping": bool(self.is_jumping) if not on_ground else False,
            "has_second_jump": bool(self.has_second_jump) if not on_ground else True,
            "air_time": 0.12 if self.is_jumping else 0.25,
            "car_pos": (float(self.cx), float(self.cy), float(eff_cz)),
            "car_vel": (float(cvx), float(cvy), float(cvz)),
            "car_ang": (0.0, 0.0, 0.0),
            "car_fwd": (float(fwd[0]), float(fwd[1]), float(fwd[2])),
            "car_rgt": (float(rgt[0]), float(rgt[1]), float(rgt[2])),
            "car_up":  (float(up[0]),  float(up[1]),  float(up[2])),
            "car_boost": float(self.car_boost),
            "ball_pos": (float(self.bx), float(self.by), float(self.bz)),
            "ball_vel": (float(bvx), float(bvy), float(bvz)),
            "ball_ang": (0.0, 0.0, 0.0),
        }
        sd["spawn_id"] = self.env._compute_spawn_id(sd)
        return sd

    def sync_to_arena(self):
        sd = self.build_spawn_state()
        self.env.last_spawn_state = sd
        self.env._apply_spawn_state(sd)
        self.env.get_obs()

    def format_hud_summary(self) -> str:
        t_lbl = self.TARGET_LABELS.get(self.target, self.target.upper())
        on_gnd = (self.cz <= 20.0 and self.has_second_jump and not self.is_jumping)
        j_str = "GROUND(满跳 HasJump=ON)" if on_gnd else (
            "JUMPING(一段跳中 HasJump=ON)" if self.is_jumping else ("AIR+2ND(可二段跳 HasJump=ON)" if self.has_second_jump else "NO_JUMP(二段跳已耗尽 HasJump=OFF)")
        )
        return (
            f"==== 场景编辑器 (EDIT MODE: Tab 切换 3 大目标) ====\n"
            f"当前激活 [Tab 切换]: 👉 {t_lbl}\n"
            f"1. [CAR 车头方向] : pos=({self.cx:+.0f}, {self.cy:+.0f}, {self.cz:.0f}) [高度17..75] | yaw={self.car_yaw:+.1f}° pitch={self.car_pitch:+.1f}°\n"
            f"2. [VEL 速度方向] : spd={self.car_speed:.0f} uu/s | vel_yaw={self.vel_yaw:+.1f}° vel_pitch={self.vel_pitch:+.1f}° (仅水平/向下[-90°,0°])\n"
            f"3. [BALL 手动发球]: pos=({self.bx:+.0f}, {self.by:+.0f}, {self.bz:.0f}) (dist={self.ball_dist:.0f}) | ball_spd={self.ball_speed:.0f} uu/s | yaw={self.ball_yaw:+.1f}° pitch={self.ball_pitch:+.1f}°\n"
            f"• 车辆状态标志     : Boost={self.car_boost:.0f} | HasJump={int(self.has_second_jump)} ({j_str}) [按 Z 展开/收起手动面板]\n"
            f"----------------------------------------\n"
            f"[操作指南]\n"
            f"  Tab       : 循环切换 【1.车头方向 CAR】 → 【2.初速度方向 VEL】 → 【3.手动发球 BALL(相机聚焦于球)】\n"
            f"  鼠标左键拖 : 旋转当前激活目标的方向 (车头朝向 / 车初速方向 / 发球速度矢量朝向)\n"
            f"  [ / ]     : CAR/VEL 模式下增减车初速 (0..2300) | BALL 模式下增减发球速度 (0..4000)\n"
            f"  W/A/S/D   : 水平平移车辆 (CAR/VEL) 或平移球 (BALL)\n"
            f"  / 或 .    : 垂直调整车高 (17..75 uu) 或球高 (93..1900 uu)\n"
            f"  Z (或 9/0): Z 展开/聚焦右侧 Boost Slider & Has Jump Checkbox 面板 | 9调氮气 | 0切 HasJump\n"
            f"  T (或回车): 🚀 立即发射测试! (按 R 重放，按 M 切模型对比)\n"
            f"  X         : 退出编辑模式"
        )
