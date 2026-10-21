"""vis/camera.py —— TunerGLViewWidget 3D 摄像机控制 (支持 CENTER 环绕视角与 POV 车体硬挂载 / 球视角)。"""
import math
import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui


class TunerGLViewWidget(gl.GLViewWidget):
    """GLViewWidget supporting Third-Party ('center') and POV ('pov': 'attach' or 'ball') camera modes."""

    VIRTUAL_LOOK_DIST = 200.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cam_mode = "center"      # "center" (Third-Party) or "pov"
        self.pov_submode = "attach"   # "attach" (Hard-Attach to Car 3D axes) or "ball" (Ball Camera)
        self.center_params = {
            "distance": 5500.0,
            "elevation": 30.0,
            "azimuth": 45.0,
            "fov": 60.0,
            "cx": 0.0,
            "cy": 0.0,
            "cz": 0.0,
        }
        # Unified POV camera parameters shared by both 'attach' (Hard-Attach) and 'ball' (Ball-Cam)
        self.pov_params = {
            "back": 1500.0,
            "up": 250.0,
            "elev": 20.0,
            "azim": 0.0,
            "fov": 130.0,
        }
        self.attach_params = self.pov_params
        self.ball_params = self.pov_params
        self._last_car_pos_r = np.array([0.0, 0.0, 17.0], dtype=np.float32)
        self._last_car_fwd_r = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._last_car_up_r  = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._last_car_rgt_r = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._last_ball_pos_r = np.array([0.0, 0.0, 93.15], dtype=np.float32)
        self._last_c2b_h = np.array([0.0, 1.0], dtype=np.float32)

        self._pov_eye = QtGui.QVector3D(0.0, -1500.0, 267.0)
        self._pov_target = QtGui.QVector3D(0.0, 0.0, 267.0)
        self._pov_up = QtGui.QVector3D(0.0, 0.0, 1.0)
        self._mouse_last_pos = None
        self.edit_drag_callback = None
        self.edit_pov_active = False
        self._temp_pov_azim = 0.0
        self._temp_pov_elev = 0.0
        self._last_edit_target = "ball"
        self._last_edit_pos_r = np.array([0.0, 1500.0, 600.0], dtype=np.float32)
        self._last_edit_fwd_r = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        self._last_edit_up_r  = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._last_edit_rgt_r = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def camera_forward_world_xy(self):
        """Return horizontal (fx_w, fy_w) unit vector in RocketSim world coords for W/A/S/D editor movement."""
        if self.cam_mode == "pov" or self.edit_pov_active:
            dx_r = float(self._pov_target.x() - self._pov_eye.x())
            dy_r = float(self._pov_target.y() - self._pov_eye.y())
        else:
            az_deg = float(self.opts.get("azimuth", 45.0))
            az_rad = math.radians(az_deg)
            dx_r = -math.cos(az_rad)
            dy_r = -math.sin(az_rad)
        n = math.hypot(dx_r, dy_r)
        if n < 1e-4:
            return (0.0, 1.0)
        return (-dx_r / n, dy_r / n)

    def viewMatrix(self):
        if self.cam_mode == "pov" or self.edit_pov_active:
            tr = QtGui.QMatrix4x4()
            tr.lookAt(self._pov_eye, self._pov_target, self._pov_up)
            return tr
        return super().viewMatrix()

    def sync_center_params_from_opts(self):
        if self.cam_mode != "center" or self.edit_pov_active:
            return
        c = self.opts.get("center", QtGui.QVector3D(0.0, 0.0, 0.0))
        self.center_params["distance"] = float(self.opts.get("distance", 5500.0))
        self.center_params["elevation"] = float(self.opts.get("elevation", 30.0))
        self.center_params["azimuth"] = float(self.opts.get("azimuth", 45.0))
        self.center_params["fov"] = float(self.opts.get("fov", 60.0))
        self.center_params["cx"] = float(c.x())
        self.center_params["cy"] = float(c.y())
        self.center_params["cz"] = float(c.z())

    def apply_center_params(self):
        cp = self.center_params
        self.opts["center"] = QtGui.QVector3D(float(cp["cx"]), float(cp["cy"]), float(cp["cz"]))
        self.opts["fov"] = float(np.clip(cp["fov"], 15.0, 130.0))
        self.setCameraPosition(
            distance=float(max(50.0, cp["distance"])),
            elevation=float(np.clip(cp["elevation"], -89.0, 89.0)),
            azimuth=float(cp["azimuth"]),
        )
        self.update()

    def _normalize3(self, v, fallback):
        arr = np.asarray(v, dtype=np.float64)
        n = float(np.linalg.norm(arr))
        if n > 1e-6:
            return arr / n
        return np.asarray(fallback, dtype=np.float64)

    def _apply_3d_basis_pov(self, p_origin, fwd_vec, up_vec, rgt_vec=None):
        pp = self.pov_params
        back = float(pp["back"])
        up_off = float(pp["up"])
        base_elev_deg = float(np.clip(float(pp["elev"]), -85.0, 85.0))
        orbit_elev_deg = float(np.clip(self._temp_pov_elev, -85.0, 85.0))
        orbit_azim_deg = float(pp["azim"]) + self._temp_pov_azim
        fov = float(np.clip(pp["fov"], 15.0, 130.0))

        fwd = self._normalize3(fwd_vec, (0.0, 1.0, 0.0))
        up  = self._normalize3(up_vec,  (0.0, 0.0, 1.0))
        rgt = np.cross(fwd, up)
        rn = float(np.linalg.norm(rgt))
        if rn > 1e-6:
            rgt = rgt / rn
        elif rgt_vec is not None:
            rgt = self._normalize3(rgt_vec, (1.0, 0.0, 0.0))
        else:
            rgt = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        up = self._normalize3(np.cross(rgt, fwd), (0.0, 0.0, 1.0))

        # Orbit around p_origin (car center of mass):
        # 1) Horizontal orbit: mouse left (dx < 0 -> orbit_azim_deg > 0) orbits CCW from rear (-fwd)
        #    toward right side (+rgt), so camera looks at the car's right side (-rgt).
        az_rad = math.radians(orbit_azim_deg)
        fwd_yaw = math.cos(az_rad) * fwd - math.sin(az_rad) * rgt

        # 2) Vertical orbit: mouse down (dy > 0 -> orbit_elev_deg > 0) rolls camera from rear (-fwd_yaw)
        #    upward over the car's roof (+up); mouse up rolls toward the bottom (-up).
        orb_el_rad = math.radians(orbit_elev_deg)
        fwd_orb = math.cos(orb_el_rad) * fwd_yaw - math.sin(orb_el_rad) * up
        up_orb  = math.sin(orb_el_rad) * fwd_yaw + math.cos(orb_el_rad) * up

        eye = np.asarray(p_origin, dtype=np.float64) - back * fwd_orb + up_off * up_orb

        base_el_rad = math.radians(base_elev_deg)
        look_dir = math.cos(base_el_rad) * fwd_orb - math.sin(base_el_rad) * up_orb
        cam_up   = math.sin(base_el_rad) * fwd_orb + math.cos(base_el_rad) * up_orb

        target = eye + self.VIRTUAL_LOOK_DIST * look_dir
        self._pov_eye = QtGui.QVector3D(float(eye[0]), float(eye[1]), float(eye[2]))
        self._pov_target = QtGui.QVector3D(float(target[0]), float(target[1]), float(target[2]))
        self._pov_up = QtGui.QVector3D(float(cam_up[0]), float(cam_up[1]), float(cam_up[2]))
        self.opts["center"] = self._pov_target
        self.opts["fov"] = fov
        self.update()

    def update_edit_pov_camera(self, target_type: str = None, pos_r=None, fwd_r=None, up_r=None, rgt_r=None):
        if target_type is not None:
            self._last_edit_target = target_type
        if pos_r is not None:
            self._last_edit_pos_r = np.asarray(pos_r, dtype=np.float32)
        if fwd_r is not None:
            self._last_edit_fwd_r = np.asarray(fwd_r, dtype=np.float32)
        if up_r is not None:
            self._last_edit_up_r = np.asarray(up_r, dtype=np.float32)
        if rgt_r is not None:
            self._last_edit_rgt_r = np.asarray(rgt_r, dtype=np.float32)

        if not self.edit_pov_active:
            return

        fwd = self._normalize3(self._last_edit_fwd_r, (0.0, 1.0, 0.0))
        if self._last_edit_target in ("ball", "vel"):
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            rgt = np.cross(fwd, world_up)
            rn = float(np.linalg.norm(rgt))
            if rn > 1e-6:
                rgt = rgt / rn
            else:
                rgt = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            up = self._normalize3(np.cross(rgt, fwd), (0.0, 0.0, 1.0))
            self._apply_3d_basis_pov(self._last_edit_pos_r, fwd, up, rgt)
        else:
            self._apply_3d_basis_pov(
                self._last_edit_pos_r,
                fwd,
                self._last_edit_up_r,
                self._last_edit_rgt_r,
            )

    def update_attach_camera(self, car_pos_r=None, car_fwd_r=None, car_up_r=None, car_rgt_r=None, ball_pos_r=None):
        if car_pos_r is not None:
            self._last_car_pos_r = np.asarray(car_pos_r, dtype=np.float32)
        if car_fwd_r is not None:
            self._last_car_fwd_r = np.asarray(car_fwd_r, dtype=np.float32)
        if car_up_r is not None:
            self._last_car_up_r = np.asarray(car_up_r, dtype=np.float32)
        if car_rgt_r is not None:
            self._last_car_rgt_r = np.asarray(car_rgt_r, dtype=np.float32)
        if ball_pos_r is not None:
            self._last_ball_pos_r = np.asarray(ball_pos_r, dtype=np.float32)

        if self.edit_pov_active:
            return
        if self.cam_mode != "pov":
            return

        p_car = np.asarray(self._last_car_pos_r, dtype=np.float64)

        if self.pov_submode == "attach":
            self._apply_3d_basis_pov(
                p_car,
                self._last_car_fwd_r,
                self._last_car_up_r,
                self._last_car_rgt_r,
            )
            return

        # Ball-Cam: 3D line-of-sight vector from car to ball (tilts up/down with ball height)
        p_ball = np.asarray(self._last_ball_pos_r, dtype=np.float64)
        rel_3d = p_ball - p_car
        dx = float(rel_3d[0])
        dy = float(rel_3d[1])
        dh = math.hypot(dx, dy)
        if dh > 1e-3:
            fh_x, fh_y = dx / dh, dy / dh
            self._last_c2b_h = np.array([fh_x, fh_y], dtype=np.float32)
        else:
            fh_x, fh_y = float(self._last_c2b_h[0]), float(self._last_c2b_h[1])

        dist_3d = float(np.linalg.norm(rel_3d))
        if dist_3d > 1e-3:
            fwd_ball_3d = rel_3d / dist_3d
        else:
            fwd_ball_3d = np.array([fh_x, fh_y, 0.0], dtype=np.float64)

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        rgt_ball = np.cross(fwd_ball_3d, world_up)
        rn = float(np.linalg.norm(rgt_ball))
        if rn > 1e-6:
            rgt_ball = rgt_ball / rn
        else:
            rgt_ball = np.array([fh_y, -fh_x, 0.0], dtype=np.float64)
        up_ball = self._normalize3(np.cross(rgt_ball, fwd_ball_3d), (0.0, 0.0, 1.0))
        self._apply_3d_basis_pov(p_car, fwd_ball_3d, up_ball, rgt_ball)

    def active_mode_key(self) -> str:
        if self.cam_mode == "center":
            return "center"
        return self.pov_submode

    def set_cam_mode(self, mode: str):
        m = (mode or "center").strip().lower()
        if m in ("center", "third_party", "third"):
            self.cam_mode = "center"
            self.apply_center_params()
            return
        if self.cam_mode == "center":
            self.sync_center_params_from_opts()
        self.cam_mode = "pov"
        if m in ("ball", "ball_cam", "ballcam"):
            self.pov_submode = "ball"
        elif m in ("attach", "hard_attach", "car", "pov"):
            self.pov_submode = "attach"
        self.update_attach_camera()

    def toggle_cam_mode(self) -> str:
        if self.cam_mode == "center":
            self.sync_center_params_from_opts()
            self.cam_mode = "pov"
            self.update_attach_camera()
        else:
            self.cam_mode = "center"
            self.apply_center_params()
        return self.active_mode_key()

    def toggle_pov_submode(self) -> str:
        if self.cam_mode != "pov":
            self.sync_center_params_from_opts()
            self.cam_mode = "pov"
        else:
            self.pov_submode = "ball" if self.pov_submode == "attach" else "attach"
        self.update_attach_camera()
        return self.pov_submode

    def format_current_params(self, mode: str = None) -> str:
        m = (mode or self.active_mode_key()).lower()
        if m in ("attach", "hard_attach", "ball", "ball_cam", "pov"):
            pp = self.pov_params
            return (
                f"back={pp['back']:.1f}, up={pp['up']:.1f}, "
                f"elev={pp['elev']:.1f}, azim={pp['azim']:.1f}, fov={pp['fov']:.1f}"
            )
        self.sync_center_params_from_opts()
        cp = self.center_params
        return (
            f"distance={cp['distance']:.1f}, elevation={cp['elevation']:.1f}, "
            f"azimuth={cp['azimuth']:.1f}, fov={cp['fov']:.1f}, "
            f"cx={cp['cx']:.1f}, cy={cp['cy']:.1f}, cz={cp['cz']:.1f}"
        )

    def apply_params_str(self, spec: str, mode: str = None):
        if not spec:
            return
        m = (mode or self.active_mode_key()).lower()
        tokens = [t.strip() for t in spec.replace(";", ",").split(",") if t.strip()]
        for tok in tokens:
            if "=" not in tok:
                continue
            k, v_str = [x.strip().lower() for x in tok.split("=", 1)]
            try:
                val = float(v_str)
            except ValueError:
                continue
            if m in ("attach", "hard_attach", "ball", "ball_cam", "pov"):
                if k in ("back", "dist", "d"):
                    self.pov_params["back"] = val
                elif k in ("up", "z", "height", "h"):
                    self.pov_params["up"] = val
                elif k in ("elev", "elevation", "pitch"):
                    self.pov_params["elev"] = float(np.clip(val, -85.0, 85.0))
                elif k in ("azim", "azimuth", "yaw"):
                    self.pov_params["azim"] = float(((val + 180.0) % 360.0) - 180.0)
                elif k == "fov":
                    self.pov_params["fov"] = float(np.clip(val, 15.0, 140.0))
            else:
                if k in ("distance", "dist", "d"):
                    self.center_params["distance"] = max(50.0, val)
                elif k in ("elevation", "elev", "pitch"):
                    self.center_params["elevation"] = float(np.clip(val, -89.0, 89.0))
                elif k in ("azimuth", "azim", "yaw"):
                    self.center_params["azimuth"] = val
                elif k == "fov":
                    self.center_params["fov"] = float(np.clip(val, 15.0, 140.0))
                elif k == "cx":
                    self.center_params["cx"] = val
                elif k == "cy":
                    self.center_params["cy"] = val
                elif k == "cz":
                    self.center_params["cz"] = val

        if self.edit_pov_active:
            self.update_edit_pov_camera()
        elif self.cam_mode == "pov":
            self.update_attach_camera()
        else:
            self.apply_center_params()

    def wheelEvent(self, ev):
        if self.cam_mode == "pov" or self.edit_pov_active:
            ev.accept()
            return
        super().wheelEvent(ev)
        self.sync_center_params_from_opts()

    def mousePressEvent(self, ev):
        self._mouse_last_pos = ev.pos()
        if self.edit_pov_active or self.cam_mode == "pov":
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._mouse_last_pos is not None:
            diff = ev.pos() - self._mouse_last_pos
            dx = float(diff.x())
            dy = float(diff.y())
            self._mouse_last_pos = ev.pos()

            if self.edit_pov_active:
                if (ev.buttons() & QtCore.Qt.MouseButton.LeftButton) and self.edit_drag_callback is not None:
                    self.edit_drag_callback(dx, dy)
                elif ev.buttons() & QtCore.Qt.MouseButton.RightButton:
                    self._temp_pov_azim = float(((self._temp_pov_azim - dx * 0.45 + 180.0) % 360.0) - 180.0)
                    self._temp_pov_elev = float(np.clip(self._temp_pov_elev + dy * 0.35, -85.0, 85.0))
                    self.update_edit_pov_camera()
                ev.accept()
                return

            if self.cam_mode == "pov":
                if ev.buttons() & QtCore.Qt.MouseButton.LeftButton:
                    self._temp_pov_azim = float(((self._temp_pov_azim - dx * 0.45 + 180.0) % 360.0) - 180.0)
                    self._temp_pov_elev = float(np.clip(self._temp_pov_elev + dy * 0.35, -85.0, 85.0))
                    self.update_attach_camera()
                ev.accept()
                return

        super().mouseMoveEvent(ev)
        if self.cam_mode == "center":
            self.sync_center_params_from_opts()

    def mouseReleaseEvent(self, ev):
        if self.edit_pov_active:
            if self._temp_pov_azim != 0.0 or self._temp_pov_elev != 0.0:
                self._temp_pov_azim = 0.0
                self._temp_pov_elev = 0.0
                self.update_edit_pov_camera()
            ev.accept()
            return
        if self.cam_mode == "pov":
            if self._temp_pov_azim != 0.0 or self._temp_pov_elev != 0.0:
                self._temp_pov_azim = 0.0
                self._temp_pov_elev = 0.0
                self.update_attach_camera()
            ev.accept()
            return
        super().mouseReleaseEvent(ev)
