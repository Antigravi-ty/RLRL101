from core.obs_v2 import GROUP_ORDER
from vis import scene as sc
from .ui_constants import STATE_PLAYBACK, STATE_DONE
from .hud import format_hud


class RenderMixin:
    def _render(self):
        sess = self._sess()
        if sess is None:
            return
        if sess.state in (STATE_PLAYBACK, STATE_DONE) and sess.frames:
            self._render_frame(sess.frames[sess.playback_idx])
            return

        snap = self.env.snapshot()
        car_pos, ball_pos = snap["car_pos"], snap["ball_pos"]

        cur_boost = snap["boost"]
        if sess._prev_boost is None:
            boosting = 0.0
        else:
            boosting = 1.0 if cur_boost < sess._prev_boost - 0.005 else 0.0
        sess._prev_boost = cur_boost

        self.trail.push(sc.r2r(car_pos), boosting)

        pos_r = sc.r2r(car_pos)
        self.car_scatter.setData(pos=pos_r.reshape(1, 3), color=(1, 1, 1, 1), size=18)
        fwd_r = sc.vec_r2r(self.env.car.get_forward_dir())
        up_r  = sc.vec_r2r(self.env.car.get_up_dir())
        rgt_r = sc.vec_r2r(self.env.car.get_right_dir())
        self.axes.update(pos_r, fwd_r, up_r, rgt_r)

        ball_r = sc.r2r(ball_pos)
        self.ball_scatter.setData(pos=ball_r.reshape(1, 3),
                                  color=(1.0, 0.2, 0.2, 0.9), size=6)
        if hasattr(self, "ball_mesh"):
            self.ball_mesh.resetTransform()
            self.ball_mesh.translate(float(ball_r[0]), float(ball_r[1]), float(ball_r[2]))

        if hasattr(self, "target_scatter"):
            t_pos = getattr(self.env, "target_pos", None)
            if t_pos is not None:
                tgt_r = sc.r2r(t_pos)
                self.target_scatter.setData(pos=tgt_r.reshape(1, 3), color=(0.1, 1.0, 0.4, 0.95), size=16)
            else:
                import numpy as np
                self.target_scatter.setData(pos=np.zeros((0, 3)))

        show_init_arrows = (len(sess.frames) == 0)
        if hasattr(self, "car_vel_arrow"):
            self.car_vel_arrow.update(
                pos_r, sc.r2r(snap["car_vel"]), fallback_dir_r=fwd_r,
                min_len=150.0, max_len=1150.0, scale=0.45, show=show_init_arrows,
            )
        if hasattr(self, "ball_vel_arrow"):
            self.ball_vel_arrow.update(
                ball_r, sc.r2r(snap["ball_vel"]), fallback_dir_r=None,
                min_len=93.0, max_len=1250.0, scale=1.5, show=show_init_arrows,
            )
        if hasattr(self, "focus_sector"):
            self.focus_sector.update(
                car_pos, snap["fwd"], snap["right"],
                getattr(self.env, "focus_cfg", None),
                show=show_init_arrows,
            )

        if hasattr(self, "speed_bar"):
            last_wasted = (float(sess.frames[-1].get("boost_wasted", 0.0)) > 0.05) if sess.frames else False
            self.speed_bar.update_speed(
                snap.get("car_speed", 0.0),
                is_supersonic=snap.get("is_supersonic", False),
                boost_wasted=last_wasted,
            )

        sc.update_pad_colors(self.big_pads, self.small_pads,
                             self.big_idx, self.small_idx, self.pads_ref)
        self._refresh_hud()

    def _render_frame(self, fr):
        sess = self._sess()
        if getattr(self, '_last_rendered_idx', None) == sess.playback_idx:
            return
        self._last_rendered_idx = sess.playback_idx
        pos_r = sc.r2r(fr["car_pos"])
        ball_r = sc.r2r(fr["ball_pos"])
        self.car_scatter.setData(pos=pos_r.reshape(1, 3), color=(1, 1, 1, 1), size=18)
        self.ball_scatter.setData(pos=ball_r.reshape(1, 3),
                                  color=(1.0, 0.2, 0.2, 0.9), size=6)
        if hasattr(self, "ball_mesh"):
            self.ball_mesh.resetTransform()
            self.ball_mesh.translate(float(ball_r[0]), float(ball_r[1]), float(ball_r[2]))

        if hasattr(self, "target_scatter"):
            sd = fr.get("spawn_state") or {}
            t_pos = sd.get("target_pos") or getattr(self.env, "target_pos", None)
            if t_pos is not None:
                tgt_r = sc.r2r(t_pos)
                self.target_scatter.setData(pos=tgt_r.reshape(1, 3), color=(0.1, 1.0, 0.4, 0.95), size=16)
            else:
                import numpy as np
                self.target_scatter.setData(pos=np.zeros((0, 3)))

        fwd_r = sc.vec_r2r_tuple(fr["fwd"])
        up_r  = sc.vec_r2r_tuple(fr["up"])
        rgt_r = sc.vec_r2r_tuple(fr["right"])
        show_init_arrows = (sess.playback_idx == 0)
        if hasattr(self, "car_vel_arrow"):
            self.car_vel_arrow.update(
                pos_r, sc.r2r(fr.get("car_vel", (0, 0, 0))), fallback_dir_r=fwd_r,
                min_len=150.0, max_len=1150.0, scale=0.45, show=show_init_arrows,
            )
        if hasattr(self, "ball_vel_arrow"):
            self.ball_vel_arrow.update(
                ball_r, sc.r2r(fr.get("ball_vel", (0, 0, 0))), fallback_dir_r=None,
                min_len=93.0, max_len=1250.0, scale=1.5, show=show_init_arrows,
            )
        if hasattr(self, "focus_sector"):
            self.focus_sector.update(
                fr["car_pos"], fr["fwd"], fr["right"],
                getattr(self.env, "focus_cfg", None),
                show=show_init_arrows,
            )

        if hasattr(self, "speed_bar"):
            cv = fr.get("car_vel", (0.0, 0.0, 0.0))
            spd = fr.get("car_speed")
            if spd is None:
                spd = float((cv[0]**2 + cv[1]**2 + cv[2]**2) ** 0.5)
            self.speed_bar.update_speed(
                spd,
                is_supersonic=fr.get("is_supersonic", spd >= 2200.0),
                boost_wasted=(float(fr.get("boost_wasted", 0.0)) > 0.05),
            )

        self.trail.clear()
        start = max(0, sess.playback_idx - 400)
        for i in range(start, sess.playback_idx + 1):
            f = sess.frames[i]
            prev_f = sess.frames[i - 1] if i > 0 else None
            boosting = 1.0 if (prev_f is not None and
                               f["boost"] < prev_f["boost"] - 0.005) else 0.0
            self.trail.push(sc.r2r(f["car_pos"]), boosting)

        self.axes.update(pos_r, fwd_r, up_r, rgt_r)
        sc.update_pad_colors(self.big_pads, self.small_pads,
                             self.big_idx, self.small_idx, self.pads_ref)
        self._refresh_hud()

    def _refresh_hud(self):
        sess = self._sess()
        cur_b = self.env.snapshot()["boost"]
        obs_group = None
        if self.view_mode == "obs":
            obs_group = GROUP_ORDER[self.obs_group_idx]
        boost_line, body = format_hud(sess, cur_b, obs_group=obs_group)
        if self.delta_mode:
            body = "== DELTA MODE（一阶差分）==\n" + body

        extra = getattr(self, "_hud_extra", "")
        if extra and sess.state in (STATE_DONE, STATE_PLAYBACK):
            body = extra + "\n" + body

        self.boost_label.setText(boost_line)
        self.hud.setPlainText(body)
