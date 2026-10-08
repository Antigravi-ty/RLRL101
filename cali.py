"""
calib_runner.py — RocketSim 动作标定 + 人类监督可视化

按顺序执行 CASES 表里的测试，同步渲染 3D 视图。
每个 case 完成后自动暂停，等用户确认。

按键:
  SPACE   暂停/继续；case 完成时=下一个 case
  N       下一个 case
  R       重放当前 case
  X       标记 bad，跳到下一个
  Q/ESC   退出
"""
import sys, os, math
import numpy as np
import RocketSim as rs
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets


# =====================================================
# 坐标转换：RocketSim (x,y,z) → 渲染 (-x, y, z)
# =====================================================
def r2r(p):
    return np.array([-p[0], p[1], p[2]], dtype=np.float32)


def vec_r2r(v):
    return np.array([-v.x, v.y, v.z], dtype=np.float32)


# =====================================================
# 按键字符串 → CarControls
#   "jump+W" / "S" / "wait" / "boost+W+A" ...
# =====================================================
def decode_action(s):
    c = rs.CarControls()
    if s.strip() in ("wait", ""):
        return c
    for k in s.split("+"):
        k = k.strip()
        if k == "W":
            c.throttle = 1.0; c.pitch = -1.0
        elif k == "S":
            c.throttle = -1.0; c.pitch = 1.0
        elif k == "A":
            c.steer = -1.0; c.yaw = -1.0
        elif k == "D":
            c.steer = 1.0; c.yaw = 1.0
        elif k == "jump":
            c.jump = True
        elif k == "boost":
            c.boost = True
        elif k == "hb":
            c.handbrake = True
    return c


# =====================================================
# CASE 表 —— 顺序执行的测试清单
# =====================================================
def build_cases():
    C = []
    P = (3000, 0, 17)

    # --- Phase 0: 基线 ---
    C.append(dict(id="00_idle", phase="baseline",
                  init={"pos": P}, seq=[("wait", 60)],
                  desc="静止基线（校对坐标系）"))

    # --- Phase 1: jump tap vs hold ---
    for hold in [1, 2, 4, 8, 12, 16, 20]:
        C.append(dict(
            id=f"10_jump_h{hold}", phase="jump",
            init={"pos": P},
            seq=[("jump", hold), ("wait", 120)],
            desc=f"jump hold={hold}t"))

    # --- Phase 2: dodge 方向 × 速度 ---
    dirs = {"front": "jump+W", "back": "jump+S",
            "left": "jump+A", "right": "jump+D",
            "fl": "jump+W+A", "fr": "jump+W+D",
            "bl": "jump+S+A", "br": "jump+S+D"}
    for dname, dkeys in dirs.items():
        for v in [0, 800, 1500]:
            C.append(dict(
                id=f"20_dodge_{dname}_v{v}", phase="dodge_dir",
                init={"pos": P, "vel": (v, 0, 0)},
                seq=[("jump", 1), ("wait", 5), (dkeys, 1), ("wait", 60)],
                desc=f"{dname} dodge v={v}"))

    # --- Phase 3: flip cancel 延迟扫描 ---
    # 前翻起手 = jump → 5t → jump+W（触发前 dodge）
    # cancel = dodge 后 N 帧按 S 反向
    for N in [1, 2, 3, 4, 5, 6, 8]:
        C.append(dict(
            id=f"30_cancel_front_N{N}", phase="flip_cancel",
            init={"pos": P},
            seq=[("jump", 1), ("wait", 5), ("jump+W", 1),
                 ("wait", N), ("S", 40)],
            desc=f"前翻 cancel delay={N}t"))

    # --- Phase 4: cancel 的 cancel ---
    # 前翻 → N1 帧后 S cancel → N2 帧后再 W
    for tag, n1, n2 in [("A", 3, 4), ("B", 3, 8), ("C", 5, 4), ("D", 4, 6)]:
        C.append(dict(
            id=f"40_cancel2_{tag}", phase="cancel_cancel",
            init={"pos": P},
            seq=[("jump", 1), ("wait", 5), ("jump+W", 1),
                 ("wait", n1), ("S", n2), ("wait", 20), ("W", 30)],
            desc=f"cancel 的 cancel N1={n1} N2={n2}"))

    # --- Phase 5: 重力基线 ---
    C.append(dict(
        id="50_freefall", phase="gravity",
        init={"pos": (3000, 0, 2000)},
        seq=[("wait", 240)],
        desc="自由落体 z=2000（测重力）"))
    C.append(dict(
        id="51_freefall_vx500", phase="gravity",
        init={"pos": (3000, 0, 2000), "vel": (500, 0, 0)},
        seq=[("wait", 240)],
        desc="自由落体 + vx=500"))

    # --- Phase 6: boost 加速 + dodge ---
    C.append(dict(
        id="60_boost_then_dodge", phase="boost_dodge",
        init={"pos": P},
        seq=[("boost+W", 40), ("jump", 1), ("wait", 5),
             ("jump+W+boost", 1), ("wait", 60)],
        desc="boost 加速 → 前翻（含 boost）"))

    return C


# =====================================================
# 执行器
# =====================================================
class Runner:
    def __init__(self, cases):
        self.cases = cases
        self.case_idx = 0
        self.step_idx = 0
        self.tick_in_step = 0
        self.frames = []
        self.phase = "loading"
        self.marks = {}          # case_id -> "bad"
        self._rebuild()

    def _rebuild(self):
        self.arena = rs.Arena(rs.GameMode.SOCCAR)
        self.car = self.arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)

    def load_case(self, idx):
        if idx >= len(self.cases):
            self.phase = "done"
            return
        self.case_idx = idx
        self._rebuild()

        case = self.cases[idx]
        cs = rs.CarState()
        cs.pos = rs.Vec(*case["init"].get("pos", (3000, 0, 17)))
        cs.vel = rs.Vec(*case["init"].get("vel", (0, 0, 0)))
        self.car.set_state(cs)
        self.arena.step(1)

        self.step_idx = 0
        self.tick_in_step = 0
        self.frames = [self._snap(rs.CarControls())]
        self.phase = "running"

    def _snap(self, ctrl):
        cs = self.car.get_state()
        return dict(
            pos=(cs.pos.x, cs.pos.y, cs.pos.z),
            vel=(cs.vel.x, cs.vel.y, cs.vel.z),
            ang=(cs.ang_vel.x, cs.ang_vel.y, cs.ang_vel.z),
            boost=cs.boost, air=cs.air_time,
            on_ground=bool(cs.is_on_ground),
            flipping=bool(cs.is_flipping),
            jumping=bool(cs.is_jumping),
            thr=ctrl.throttle, str=ctrl.steer, ptc=ctrl.pitch,
            yaw=ctrl.yaw, roll=ctrl.roll,
            jump=bool(ctrl.jump), boost_in=bool(ctrl.boost),
        )

    def tick(self):
        if self.phase != "running":
            return
        case = self.cases[self.case_idx]
        seq = case["seq"]
        if self.step_idx >= len(seq):
            self.phase = "paused"
            self._save()
            return
        act, dur = seq[self.step_idx]
        ctrl = decode_action(act)
        self.car.set_controls(ctrl)
        self.arena.step(1)
        self.frames.append(self._snap(ctrl))
        self.tick_in_step += 1
        if self.tick_in_step >= dur:
            self.step_idx += 1
            self.tick_in_step = 0

    def _save(self):
        if not self.frames:
            return
        os.makedirs("calib_out", exist_ok=True)
        cid = self.cases[self.case_idx]["id"]
        fn = f"calib_out/{cid}.csv"
        with open(fn, "w") as f:
            f.write("# calib_runner v1\n")
            f.write(f"# case_id: {cid}\n")
            f.write(f"# desc: {self.cases[self.case_idx]['desc']}\n")
            f.write("i,px,py,pz,vx,vy,vz,wx,wy,wz,boost,air,"
                    "og,flip,jump,thr,str,ptc,yaw,roll,jin,bin\n")
            for i, fr in enumerate(self.frames):
                f.write(",".join(str(x) for x in [
                    i, *fr["pos"], *fr["vel"], *fr["ang"],
                    round(fr["boost"], 2), round(fr["air"], 3),
                    int(fr["on_ground"]), int(fr["flipping"]), int(fr["jumping"]),
                    fr["thr"], fr["str"], fr["ptc"], fr["yaw"], fr["roll"],
                    int(fr["jump"]), int(fr["boost_in"]),
                ]) + "\n")

    def next_case(self):
        self.load_case(self.case_idx + 1)

    def replay(self):
        self.load_case(self.case_idx)


# =====================================================
# 可视化
# =====================================================
class CalibView(gl.GLViewWidget):
    def __init__(self, runner):
        super().__init__()
        self.runner = runner
        self._last_case_id = None

        # 顶部 HUD
        self.hud = QtWidgets.QLabel(self)
        self.hud.setStyleSheet(
            "color:#00ffcc;font-size:14px;font-weight:bold;"
            "background:rgba(0,0,0,220);padding:10px;"
            "font-family:Consolas,monospace;")
        self.hud.setGeometry(10, 10, 720, 210)
        self.hud.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        # 右侧状态大标签
        self.status_label = QtWidgets.QLabel(self)
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self._set_status("⏳ LOADING", "#888888")

        # 底部提示
        self.help_label = QtWidgets.QLabel(self)
        self.help_label.setStyleSheet(
            "color:#ffcc00;font-size:13px;font-weight:bold;"
            "background:rgba(0,0,0,200);padding:8px;")
        self.help_label.setText(
            "SPACE 暂停/继续(NEXT)   N 下一个   R 重放   X 标记BAD   Q 退出")
        self.help_label.setGeometry(10, 730, 800, 32)

        # ---- 场景 ----
        self._setup_scene()

        # ---- 动态元素 ----
        self.car_scatter = gl.GLScatterPlotItem()
        self.car_scatter.setData(pos=np.zeros((1, 3)), color=(1, 1, 1, 1), size=18)
        self.addItem(self.car_scatter)

        self.ball_scatter = gl.GLScatterPlotItem()
        self.addItem(self.ball_scatter)

        # 轨迹（保留 view.py 的 boost 金色逻辑）
        self.trail = gl.GLLinePlotItem(antialias=True)
        self.addItem(self.trail)
        self.trail_pts = []   # (x, y, z, is_boosting)

        # 三轴指示器
        self.fwd_line = gl.GLLinePlotItem(antialias=True); self.addItem(self.fwd_line)
        self.up_line  = gl.GLLinePlotItem(antialias=True); self.addItem(self.up_line)
        self.rgt_line = gl.GLLinePlotItem(antialias=True); self.addItem(self.rgt_line)

    # ---------------- 场景静态元素 ----------------
    def _setup_scene(self):
        # 地板
        floor = gl.GLGridItem()
        floor.setSize(8000, 10240, 0)
        floor.setSpacing(500, 500, 0)
        floor.setColor((255, 255, 255, 35))
        self.addItem(floor)

        # 场地边界
        X_MIN, X_MAX = -4000, 4000
        Y_MIN, Y_MAX = -5120, 5120
        Z_MIN, Z_MAX = 0, 2000
        corners = np.array([
            [X_MIN, Y_MIN, Z_MIN], [X_MAX, Y_MIN, Z_MIN],
            [X_MAX, Y_MAX, Z_MIN], [X_MIN, Y_MAX, Z_MIN],
            [X_MIN, Y_MIN, Z_MAX], [X_MAX, Y_MIN, Z_MAX],
            [X_MAX, Y_MAX, Z_MAX], [X_MIN, Y_MAX, Z_MAX],
        ], dtype=np.float32)
        edges = [(0,1),(1,2),(2,3),(3,0),
                 (4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        lines = []
        for a, b in edges:
            lines.append(corners[a]); lines.append(corners[b])
        border = gl.GLLinePlotItem(antialias=True)
        border.setData(pos=np.array(lines), color=(255, 255, 255, 90),
                       width=2, mode='lines')
        self.addItem(border)

        # Boost pads（静态位置，直接从 arena 里读）
        pads = self.runner.arena.get_boost_pads()
        big_xyz, small_xyz = [], []
        for p in pads:
            pos = p.get_pos()
            xyz = [-pos.x, pos.y, pos.z]
            (big_xyz if p.is_big else small_xyz).append(xyz)
        self.big_xyz = np.array(big_xyz, dtype=np.float32) if big_xyz else np.zeros((0,3), np.float32)
        self.small_xyz = np.array(small_xyz, dtype=np.float32) if small_xyz else np.zeros((0,3), np.float32)

        big = gl.GLScatterPlotItem()
        big.setData(pos=self.big_xyz, color=(1.0, 0.95, 0.0, 1.0), size=22)
        self.addItem(big)
        small = gl.GLScatterPlotItem()
        small.setData(pos=self.small_xyz, color=(1.0, 0.85, 0.0, 0.5), size=10)
        self.addItem(small)

    # ---------------- 状态标签 ----------------
    def _set_status(self, text, color):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"color:white;font-size:30px;font-weight:bold;"
            f"background:rgba(0,0,0,220);padding:12px;"
            f"border: 3px solid {color};")
        self.status_label.setGeometry(self.width() - 330, 10, 310, 60)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.status_label.setGeometry(self.width() - 330, 10, 310, 60)
        self.help_label.setGeometry(10, self.height() - 40, 800, 32)

    # ---------------- 每帧刷新 ----------------
    def refresh(self):
        r = self.runner
        if r.phase == "done":
            self.hud.setText("全部 case 完成。按 Q 退出。")
            self._set_status("✓ DONE", "#00ff00")
            return

        case = r.cases[r.case_idx]

        # case 切换 → 清空轨迹
        if case["id"] != self._last_case_id:
            self._last_case_id = case["id"]
            self.trail_pts = []
            self.trail.setData(pos=np.zeros((0, 3)))

        cs = r.car.get_state()
        fr = r.frames[-1] if r.frames else None

        if r.phase == "running":
            self._set_status("▶ RUNNING", "#00ff00")
        elif r.phase == "paused":
            self._set_status("⏸ PAUSED", "#ffaa00")
        else:
            self._set_status("⏳ LOADING", "#888888")

        speed = math.sqrt(cs.vel.x**2 + cs.vel.y**2 + cs.vel.z**2)
        info = (
            f"[{r.case_idx+1}/{len(r.cases)}]  {case['id']}\n"
            f"phase: {case['phase']}\n"
            f"desc : {case['desc']}\n"
            f"step : {r.step_idx}/{len(case['seq'])}  "
            f"t_in_step: {r.tick_in_step}  tick: {len(r.frames)}\n"
            f"pos  : ({cs.pos.x:7.1f}, {cs.pos.y:7.1f}, {cs.pos.z:6.1f})\n"
            f"vel  : ({cs.vel.x:7.1f}, {cs.vel.y:7.1f}, {cs.vel.z:6.1f})"
            f"  |v|={speed:6.1f}\n"
            f"ang  : ({cs.ang_vel.x:6.2f}, {cs.ang_vel.y:6.2f}, {cs.ang_vel.z:6.2f})\n"
            f"state: on_ground={int(cs.is_on_ground)}  "
            f"flipping={int(cs.is_flipping)}  jumping={int(cs.is_jumping)}  "
            f"boost={cs.boost:5.1f}  air={cs.air_time:5.2f}s\n"
            f"keys : thr={fr['thr'] if fr else 0:+.0f}  "
            f"str={fr['str'] if fr else 0:+.0f}  "
            f"ptc={fr['ptc'] if fr else 0:+.0f}  "
            f"yaw={fr['yaw'] if fr else 0:+.0f}  "
            f"roll={fr['roll'] if fr else 0:+.0f}  "
            f"jump={int(fr['jump']) if fr else 0}  "
            f"boost_in={int(fr['boost_in']) if fr else 0}"
        )
        if case["id"] in r.marks:
            info += f"\n[MARKED BAD]"
        self.hud.setText(info)

        # ---------- 3D 更新 ----------
        pos = r2r([cs.pos.x, cs.pos.y, cs.pos.z])
        self.car_scatter.setData(pos=pos.reshape(1, 3), color=(1, 1, 1, 1), size=18)

        # 轨迹（金色=boost）
        is_boost = bool(fr and fr["boost_in"])
        self.trail_pts.append((float(pos[0]), float(pos[1]), float(pos[2]), float(is_boost)))
        if len(self.trail_pts) > 2000:
            self.trail_pts.pop(0)
        if len(self.trail_pts) > 1:
            tp = np.array(self.trail_pts, dtype=np.float32)
            col = np.zeros((len(tp), 4), dtype=np.float32)
            bm = tp[:, 3] > 0.5
            col[~bm] = (0.0, 0.8, 1.0, 0.55)   # 青蓝 = 普通
            col[bm]  = (1.0, 0.8, 0.0, 0.95)   # 金黄 = boost
            self.trail.setData(pos=tp[:, :3], color=col, width=4)

        # 三轴指示器
        L = 300.0
        fwd = vec_r2r(r.car.get_forward_dir())
        up  = vec_r2r(r.car.get_up_dir())
        rgt = vec_r2r(r.car.get_right_dir())
        self.fwd_line.setData(pos=np.array([pos, pos + L*fwd]),
                              color=(0, 1, 0, 1), width=4)
        self.up_line.setData(pos=np.array([pos, pos + L*up]),
                             color=(1, 0, 0, 1), width=4)
        self.rgt_line.setData(pos=np.array([pos, pos + 0.5*L*rgt]),
                              color=(0, 0.5, 1, 1), width=4)

        # 球（占位）
        bs = r.arena.ball.get_state()
        bpos = r2r([bs.pos.x, bs.pos.y, bs.pos.z])
        self.ball_scatter.setData(pos=bpos.reshape(1, 3),
                                  color=(1, 0.2, 0.2, 1), size=24)

    # ---------------- 键盘 ----------------
    def keyPressEvent(self, e):
        k = e.key()
        r = self.runner
        if k == QtCore.Qt.Key_Space:
            if r.phase == "running":
                r.phase = "paused"
            elif r.phase == "paused":
                # 若 seq 跑完 → 下一个；否则继续
                if r.step_idx >= len(r.cases[r.case_idx]["seq"]):
                    r.next_case()
                else:
                    r.phase = "running"
        elif k == QtCore.Qt.Key_N:
            r._save()
            r.next_case()
        elif k == QtCore.Qt.Key_R:
            r.replay()
        elif k == QtCore.Qt.Key_X:
            cid = r.cases[r.case_idx]["id"]
            r.marks[cid] = "bad"
            print(f"[MARK BAD] {cid}")
            r._save()
            r.next_case()
        elif k in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            QtWidgets.QApplication.quit()
        else:
            super().keyPressEvent(e)


# =====================================================
# 主程序
# =====================================================
def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    cases = build_cases()
    runner = Runner(cases)
    runner.load_case(0)

    view = CalibView(runner)
    view.setWindowTitle("RocketSim 动作标定 — Calib Runner")
    view.setGeometry(60, 60, 1280, 800)
    view.setCameraPosition(distance=3500, elevation=35, azimuth=45)
    view.show()

    def tick():
        runner.tick()
        view.refresh()

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(8)   # ~125Hz，与物理 120Hz 接近

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()