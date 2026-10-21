from core.obs import GROUP_ORDER
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

        cs = self.env.car.get_state()
        jumping = bool(cs.is_jumping)
        djump_or_flip = bool(cs.has_double_jumped or cs.is_flipping)

        pos_r = sc.r2r(car_pos)
        sim_now = (float(len(sess.frames)) / 120.0) if sess.frames else None
        self.trail.push(pos_r, boosting=boosting, jumping=jumping, djump_or_flip=djump_or_flip, now=sim_now)

        self.car_scatter.setData(pos=pos_r.reshape(1, 3), color=(1, 1, 1, 1), size=18)
        fwd_r = sc.vec_r2r(self.env.car.get_forward_dir())
        up_r  = sc.vec_r2r(self.env.car.get_up_dir())
        rgt_r = sc.vec_r2r(self.env.car.get_right_dir())
        self.axes.update(pos_r, fwd_r, up_r, rgt_r)

        ball_r = sc.r2r(ball_pos)
        is_edit = getattr(self, "editor", None) is not None and self.editor.active
        if hasattr(self.view, "edit_pov_active"):
            self.view.edit_pov_active = bool(is_edit)
        if is_edit and hasattr(self.view, "update_edit_pov_camera"):
            if self.editor.target == "ball":
                ball_dir_r = sc.r2r(self.editor.ball_dir_vec())
                self.view.update_edit_pov_camera("ball", ball_r, ball_dir_r)
            elif self.editor.target == "vel":
                vel_dir_r = sc.r2r(self.editor.vel_dir_vec())
                self.view.update_edit_pov_camera("vel", pos_r, vel_dir_r)
            else:
                self.view.update_edit_pov_camera("car", pos_r, fwd_r, up_r, rgt_r)
        elif hasattr(self.view, "update_attach_camera"):
            self.view.update_attach_camera(pos_r, fwd_r, up_r, rgt_r, ball_r)

        if hasattr(self, "ball_guides"):
            self.ball_guides.update(ball_r)

        in_box = bool(snap.get("in_air_box", False))
        ball_col = (0.1, 1.0, 0.5, 0.95) if in_box else (1.0, 0.2, 0.2, 0.9)
        self.ball_scatter.setData(pos=ball_r.reshape(1, 3), color=ball_col, size=7 if in_box else 6)
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

        show_init_arrows = is_edit or (len(sess.frames) == 0)
        if hasattr(self, "car_vel_arrow"):
            vel_fb = sc.r2r(self.editor.vel_dir_vec()) if is_edit else fwd_r
            self.car_vel_arrow.update(
                pos_r, sc.r2r(snap["car_vel"]), fallback_dir_r=vel_fb,
                min_len=240.0 if is_edit else 150.0, max_len=1250.0, scale=0.45, show=show_init_arrows,
            )
        if hasattr(self, "ball_vel_arrow"):
            ball_fb = sc.r2r(self.editor.ball_dir_vec()) if is_edit else None
            self.ball_vel_arrow.update(
                ball_r, sc.r2r(snap["ball_vel"]), fallback_dir_r=ball_fb,
                min_len=280.0 if is_edit else 93.0, max_len=1400.0, scale=1.5, show=show_init_arrows,
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
        if getattr(self, "_last_rendered_idx", None) == sess.playback_idx:
            return
        self._last_rendered_idx = sess.playback_idx
        pos_r = sc.r2r(fr["car_pos"])
        ball_r = sc.r2r(fr["ball_pos"])
        if hasattr(self, "ball_guides"):
            self.ball_guides.update(ball_r)

        in_box = bool(fr.get("in_air_box", False))
        ball_col = (0.1, 1.0, 0.5, 0.95) if in_box else (1.0, 0.2, 0.2, 0.9)
        self.car_scatter.setData(pos=pos_r.reshape(1, 3), color=(1, 1, 1, 1), size=18)
        self.ball_scatter.setData(pos=ball_r.reshape(1, 3), color=ball_col, size=7 if in_box else 6)
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
        if hasattr(self.view, "update_attach_camera"):
            self.view.update_attach_camera(pos_r, fwd_r, up_r, rgt_r, ball_r)

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
        trail_dur = max(10, int(getattr(self, "trail_duration", 600)))
        start = max(0, sess.playback_idx - trail_dur)
        for i in range(start, sess.playback_idx + 1):
            f = sess.frames[i]
            prev_f = sess.frames[i - 1] if i > 0 else None
            b_used = float(f.get("boost_used", 0.0))
            boosting = 1.0 if (b_used > 0.001 or (prev_f is not None and f["boost"] < prev_f["boost"] - 0.005)) else 0.0
            fl = f.get("flags") or {}
            jumping = bool(fl.get("is_jumping", False))
            djump_or_flip = bool(fl.get("has_djump", False) or fl.get("is_flipping", False) or f.get("is_flipping", False))
            self.trail.push(sc.r2r(f["car_pos"]), boosting=boosting, jumping=jumping, djump_or_flip=djump_or_flip, now=float(i) / 120.0)

        self.axes.update(pos_r, fwd_r, up_r, rgt_r)
        sc.update_pad_colors(self.big_pads, self.small_pads,
                             self.big_idx, self.small_idx, self.pads_ref)
        self._refresh_hud()

    def _refresh_hud(self):
        sess = self._sess()
        cur_b = self.env.snapshot()["boost"]
        if getattr(self, "editor", None) is not None and self.editor.active:
            self.boost_label.setText(f"BOOST: {cur_b:5.1f}")
            self.hud.setPlainText(self.editor.format_hud_summary())
            return
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
