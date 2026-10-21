"""vis/tuner/multi_car_renderer.py —— 1v1 Kickoff 双车 3D 场景渲染器 (蓝/橙双车碰撞盒、朝向坐标轴、尾迹与聚焦高亮)。"""
import numpy as np
import pyqtgraph.opengl as gl
from vis import scene as sc


class MultiCarSceneRenderer:
    """为 1v1 Kickoff 渲染蓝/橙双车的碰撞盒、朝向坐标轴、尾迹 (含 Jump/DoubleJump 触发点自动过期销毁) 与聚焦高亮。"""

    BLUE_COAST_COLOR   = (0.0, 0.80, 1.0, 0.62)
    ORANGE_COAST_COLOR = (1.0, 0.45, 0.15, 0.62)

    def __init__(self, view, car_teams, max_trail_len=600, max_trail_age_s=5.0):
        self.view = view
        self.car_teams = [int(t) for t in car_teams]
        self.n_cars = len(self.car_teams)
        self.max_trail_len = int(max_trail_len)
        self.max_trail_age_s = float(max_trail_age_s)

        self.hitboxes = []
        self.trails = []
        self.scatters = []
        self._prev_boosts = [None] * self.n_cars

        for t in self.car_teams:
            coast_col = self.BLUE_COAST_COLOR if t == 0 else self.ORANGE_COAST_COLOR
            self.trails.append(
                sc.TrailRenderer(
                    view,
                    max_len=self.max_trail_len,
                    max_age_s=self.max_trail_age_s,
                    coast_color=coast_col,
                )
            )
            self.hitboxes.append(sc.CarHitboxRenderer(view, team=t))
            sp = gl.GLScatterPlotItem()
            dot_col = (0.3, 0.85, 1.0, 1.0) if t == 0 else (1.0, 0.55, 0.2, 1.0)
            sp.setData(pos=np.zeros((1, 3), dtype=np.float32), color=dot_col, size=14)
            view.addItem(sp)
            self.scatters.append(sp)

    def clear_trails(self):
        for tr in self.trails:
            tr.clear()
        self._prev_boosts = [None] * self.n_cars

    def set_max_trail_len(self, max_trail_len: int, frames=None):
        self.max_trail_len = max(2, int(max_trail_len))
        self.max_trail_age_s = max(0.05, float(self.max_trail_len) / 120.0)
        for tr in self.trails:
            tr.set_max_len(self.max_trail_len, max_age_s=self.max_trail_age_s)
        if frames:
            frames_list = list(frames)[-self.max_trail_len:]
            for i in range(self.n_cars):
                self.trails[i].clear()
                for f_t in frames_list:
                    c_list = f_t.get("all_cars")
                    if not c_list or i >= len(c_list):
                        continue
                    cs_i = c_list[i]
                    self.trails[i].push(
                        sc.r2r(cs_i["car_pos"]),
                        boosting=cs_i.get("boosting", 0.0),
                        jumping=cs_i.get("is_jumping", False),
                        djump_or_flip=cs_i.get("djump_or_flip", False),
                        now=float(f_t.get("i", 0)) / 120.0,
                    )

    def render_live(self, env, focused_idx: int = 0, now: float = None):
        """渲染实时帧的所有车辆及其轨迹。"""
        for i, car in enumerate(env.cars):
            cs = car.get_state()
            pos_r = sc.r2r((cs.pos.x, cs.pos.y, cs.pos.z))
            fwd_r = sc.vec_r2r(car.get_forward_dir())
            up_r  = sc.vec_r2r(car.get_up_dir())
            rgt_r = sc.vec_r2r(car.get_right_dir())

            cur_boost = float(cs.boost)
            prev_b = self._prev_boosts[i]
            ctrl = car.get_controls()
            boosting = 1.0 if ((bool(ctrl.boost) and cur_boost > 0.1) or (prev_b is not None and cur_boost < prev_b - 0.005)) else 0.0
            self._prev_boosts[i] = cur_boost

            jumping = bool(cs.is_jumping)
            djump_or_flip = bool(cs.has_double_jumped or cs.is_flipping)
            self.trails[i].push(pos_r, boosting=boosting, jumping=jumping, djump_or_flip=djump_or_flip, now=now)

            is_focused = (i == focused_idx)
            team = self.car_teams[i]

            if is_focused:
                edge_col = (1.0, 1.0, 1.0, 0.98)
                edge_w = 3.2
                dot_col = (1.0, 1.0, 1.0, 1.0)
                dot_size = 20
            else:
                edge_col = None
                edge_w = 2.2
                dot_col = (0.25, 0.80, 1.0, 0.90) if team == 0 else (1.0, 0.55, 0.20, 0.90)
                dot_size = 15

            self.scatters[i].setData(pos=pos_r.reshape(1, 3), color=dot_col, size=dot_size)
            self.hitboxes[i].update(
                pos_r, fwd_r, up_r, rgt_r,
                length=250.0 if is_focused else 200.0,
                edge_color_override=edge_col,
                edge_width=edge_w,
            )

    def render_playback_frame(self, frames, playback_idx: int, focused_idx: int = 0):
        """在回放/暂停状态下重绘所有车辆在 playback_idx 时刻的位置与窗口内轨迹。"""
        if not frames or playback_idx < 0 or playback_idx >= len(frames):
            return
        fr = frames[playback_idx]
        cars_snaps = fr.get("all_cars")
        if not cars_snaps:
            return

        window_start = max(0, playback_idx - self.max_trail_len)
        for i in range(min(self.n_cars, len(cars_snaps))):
            self.trails[i].clear()
            for t_idx in range(window_start, playback_idx + 1):
                f_t = frames[t_idx]
                c_list = f_t.get("all_cars")
                if not c_list or i >= len(c_list):
                    continue
                cs_i = c_list[i]
                self.trails[i].push(
                    sc.r2r(cs_i["car_pos"]),
                    boosting=cs_i.get("boosting", 0.0),
                    jumping=cs_i.get("is_jumping", False),
                    djump_or_flip=cs_i.get("djump_or_flip", False),
                    now=float(t_idx) / 120.0,
                )

            cs_now = cars_snaps[i]
            pos_r = sc.r2r(cs_now["car_pos"])
            fwd_r = sc.vec_r2r_tuple(cs_now["fwd"])
            up_r  = sc.vec_r2r_tuple(cs_now["up"])
            rgt_r = sc.vec_r2r_tuple(cs_now["right"])
            is_focused = (i == focused_idx)
            team = self.car_teams[i]

            if is_focused:
                edge_col = (1.0, 1.0, 1.0, 0.98)
                edge_w = 3.2
                dot_col = (1.0, 1.0, 1.0, 1.0)
                dot_size = 20
            else:
                edge_col = None
                edge_w = 2.2
                dot_col = (0.25, 0.80, 1.0, 0.90) if team == 0 else (1.0, 0.55, 0.20, 0.90)
                dot_size = 15

            self.scatters[i].setData(pos=pos_r.reshape(1, 3), color=dot_col, size=dot_size)
            self.hitboxes[i].update(
                pos_r, fwd_r, up_r, rgt_r,
                length=250.0 if is_focused else 200.0,
                edge_color_override=edge_col,
                edge_width=edge_w,
            )
