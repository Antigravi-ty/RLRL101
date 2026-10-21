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

        f_idx = int(getattr(self.env, "focused_car_idx", 0))
        snap = self.env.snapshot(car_idx=f_idx)
        car_pos, ball_pos = snap["car_pos"], snap["ball_pos"]
        pos_r = sc.r2r(car_pos)
        ball_r = sc.r2r(ball_pos)
        sim_now = (float(len(sess.frames)) / 120.0) if sess.frames else None

        if getattr(self, "multi_car_renderer", None) is not None:
            self.multi_car_renderer.render_live(self.env, focused_idx=f_idx, now=sim_now)

        car_obj = self.env.cars[f_idx]
        fwd_r = sc.vec_r2r(car_obj.get_forward_dir())
        up_r  = sc.vec_r2r(car_obj.get_up_dir())
        rgt_r = sc.vec_r2r(car_obj.get_right_dir())
        if hasattr(self.view, "update_attach_camera"):
            self.view.update_attach_camera(pos_r, fwd_r, up_r, rgt_r, ball_r)

        if hasattr(self, "ball_guides"):
            self.ball_guides.update(ball_r)
        self.ball_scatter.setData(pos=ball_r.reshape(1, 3),
                                  color=(1.0, 0.2, 0.2, 0.9), size=6)
        if hasattr(self, "ball_mesh"):
            self.ball_mesh.resetTransform()
            self.ball_mesh.translate(float(ball_r[0]), float(ball_r[1]), float(ball_r[2]))

        show_init_arrows = (len(sess.frames) == 0)
        if hasattr(self, "car_vel_arrow"):
            self.car_vel_arrow.update(
                pos_r, sc.r2r(snap["car_vel"]), fallback_dir_r=fwd_r,
                min_len=150.0, max_len=1250.0, scale=0.45, show=show_init_arrows,
            )
        if hasattr(self, "ball_vel_arrow"):
            self.ball_vel_arrow.update(
                ball_r, sc.r2r(snap["ball_vel"]), fallback_dir_r=None,
                min_len=93.0, max_len=1400.0, scale=1.5, show=show_init_arrows,
            )

        if hasattr(self, "speed_bar"):
            self.speed_bar.update_speed(
                snap.get("car_speed", 0.0),
                is_supersonic=snap.get("is_supersonic", False),
                boost_wasted=False,
            )

        sc.update_pad_colors(self.big_pads, self.small_pads,
                             self.big_idx, self.small_idx, self.pads_ref)
        self._refresh_hud()

    def _render_frame(self, fr):
        sess = self._sess()
        if getattr(self, '_last_rendered_idx', None) == sess.playback_idx:
            return
        self._last_rendered_idx = sess.playback_idx
        f_idx = int(getattr(self.env, "focused_car_idx", 0))
        pos_r = sc.r2r(fr["car_pos"])
        ball_r = sc.r2r(fr["ball_pos"])
        if hasattr(self, "ball_guides"):
            self.ball_guides.update(ball_r)

        if getattr(self, "multi_car_renderer", None) is not None and fr.get("all_cars"):
            self.multi_car_renderer.render_playback_frame(
                sess.frames, sess.playback_idx, focused_idx=f_idx
            )
            all_cars = fr["all_cars"]
            if 0 <= f_idx < len(all_cars):
                c_focus = all_cars[f_idx]
                pos_r = sc.r2r(c_focus["car_pos"])
                fwd_r = sc.vec_r2r_tuple(c_focus["fwd"])
                up_r  = sc.vec_r2r_tuple(c_focus["up"])
                rgt_r = sc.vec_r2r_tuple(c_focus["right"])
            else:
                fwd_r = sc.vec_r2r_tuple(fr["fwd"])
                up_r  = sc.vec_r2r_tuple(fr["up"])
                rgt_r = sc.vec_r2r_tuple(fr["right"])
        else:
            fwd_r = sc.vec_r2r_tuple(fr["fwd"])
            up_r  = sc.vec_r2r_tuple(fr["up"])
            rgt_r = sc.vec_r2r_tuple(fr["right"])

        self.ball_scatter.setData(pos=ball_r.reshape(1, 3),
                                  color=(1.0, 0.2, 0.2, 0.9), size=6)
        if hasattr(self, "ball_mesh"):
            self.ball_mesh.resetTransform()
            self.ball_mesh.translate(float(ball_r[0]), float(ball_r[1]), float(ball_r[2]))

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
                boost_wasted=False,
            )

        sc.update_pad_colors(self.big_pads, self.small_pads,
                             self.big_idx, self.small_idx, self.pads_ref)
        self._refresh_hud()

    def _refresh_hud(self):
        sess = self._sess()
        f_idx = int(getattr(self.env, "focused_car_idx", 0))
        cur_b = self.env.snapshot(car_idx=f_idx)["boost"]
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
