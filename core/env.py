"""环境封装。让 trainer 和可视化 (Tuner) 共享完全一致的 reset / settle / step 逻辑。

支持场景模式（mode）：
  - "ground" (alias "stage0") : 全场地面随机位置 + 随机偏航角 + 随机初速度球 + 30 帧悬挂稳定
  - "target" (alias "stage1") : 全场地面随机球与车 + 随机目标点与期望到达时间 (CPA 评估)
  - "static_aerial"           : 空中低速/慢抛高球与车辆生成
  - "aerial"                  : 全场随机空中/高空球与车辆生成
  - "mixed"                   : 50% 地面随机球 + 50% 空中随机球
"""
import math
import random
import numpy as np
import RocketSim as rs
from .constants import KICKOFF_SETTLE_TICKS
from .obs_v3 import build_obs_v3, OPP_GOAL_POS


class KickoffEnv:
    def __init__(self, mode="ground"):
        self.mode = "ground" if mode in ("stage0", "kickoff", None) else ("target" if mode == "stage1" else mode)
        self.arena = rs.Arena(rs.GameMode.SOCCAR)
        self.car = self.arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)
        self.pads = self.arena.get_boost_pads()
        self.num_pads = len(self.pads)

        self._touched = False
        def _on_touch(arena, car, data):
            self._touched = True
        self.arena.set_ball_touch_callback(_on_touch)

        self.last_spawn_state = None
        self.spawn_id = "0000"
        self.target_pos = None
        self.target_time = None
        self._shadow_arena = rs.Arena(rs.GameMode.SOCCAR)
        self._shadow_car = self._shadow_arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)
        # 无车纯球竞技场：用于在触球瞬间以 0.015ms 极速推演球后续飞行轨迹的时空最近点 (CPA)
        self._ball_sim_arena = rs.Arena(rs.GameMode.SOCCAR)

    def _settle(self, ticks=KICKOFF_SETTLE_TICKS):
        """空跑指定物理帧数（无任何控制输入），消除开局悬架压缩回弹与微小滑移。"""
        neutral = rs.CarControls()
        self.car.set_controls(neutral)
        if ticks > 0:
            self.arena.step(int(ticks))
        self._touched = False

    def _compute_spawn_id(self, sd):
        import hashlib
        key_str = (
            f"{sd.get('mode','g')}_{sd['car_pos'][0]:.1f}_{sd['car_pos'][1]:.1f}_"
            f"{sd['car_fwd'][0]:.3f}_{sd['car_fwd'][1]:.3f}_"
            f"{sd['ball_pos'][0]:.1f}_{sd['ball_pos'][1]:.1f}_"
            f"{sd['ball_vel'][0]:.1f}_{sd['ball_vel'][1]:.1f}"
        )
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()[:4]

    def _capture_pre_settle_state(self, use_mode, settle_ticks):
        cs = self.car.get_state()
        bs = self.arena.ball.get_state()
        sd = {
            "mode": str(use_mode),
            "settle_ticks": int(settle_ticks),
            "car_pos": (float(cs.pos.x), float(cs.pos.y), float(cs.pos.z)),
            "car_vel": (float(cs.vel.x), float(cs.vel.y), float(cs.vel.z)),
            "car_ang": (float(cs.ang_vel.x), float(cs.ang_vel.y), float(cs.ang_vel.z)),
            "car_fwd": (float(cs.rot_mat.forward.x), float(cs.rot_mat.forward.y), float(cs.rot_mat.forward.z)),
            "car_rgt": (float(cs.rot_mat.right.x),   float(cs.rot_mat.right.y),   float(cs.rot_mat.right.z)),
            "car_up":  (float(cs.rot_mat.up.x),      float(cs.rot_mat.up.y),      float(cs.rot_mat.up.z)),
            "car_boost": float(cs.boost),
            "ball_pos": (float(bs.pos.x), float(bs.pos.y), float(bs.pos.z)),
            "ball_vel": (float(bs.vel.x), float(bs.vel.y), float(bs.vel.z)),
            "ball_ang": (float(bs.ang_vel.x), float(bs.ang_vel.y), float(bs.ang_vel.z)),
            "target_pos": self.target_pos,
            "target_time": self.target_time,
        }
        sd["spawn_id"] = self._compute_spawn_id(sd)
        return sd

    def _apply_pre_settle_state(self, sd):
        # 重新挂载车辆实例以彻底清零 Bullet 悬架弹簧与轮胎残余阻尼，确保 100% 确定性复现
        self.arena.remove_car(self.car)
        self.car = self.arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)
        # 同步影子物理环境中的 car_id，以便后续用 clone_into 极速执行反事实 (Counterfactual) Boost Waste 检测
        while self._shadow_car.id < self.car.id:
            self._shadow_arena.remove_car(self._shadow_car)
            self._shadow_car = self._shadow_arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)
        self._touched = False
        self.arena.reset_kickoff()

        cs = self.car.get_state()
        cs.pos = rs.Vec(*sd["car_pos"])
        cs.vel = rs.Vec(*sd["car_vel"])
        cs.ang_vel = rs.Vec(*sd["car_ang"])
        cs.rot_mat.forward = rs.Vec(*sd["car_fwd"])
        cs.rot_mat.right   = rs.Vec(*sd["car_rgt"])
        cs.rot_mat.up      = rs.Vec(*sd["car_up"])
        cs.boost = float(sd["car_boost"])
        self.car.set_state(cs)

        bs = self.arena.ball.get_state()
        bs.pos = rs.Vec(*sd["ball_pos"])
        bs.vel = rs.Vec(*sd["ball_vel"])
        bs.ang_vel = rs.Vec(*sd["ball_ang"])
        bs.rot_mat.forward = rs.Vec(1.0, 0.0, 0.0)
        bs.rot_mat.right   = rs.Vec(0.0, 1.0, 0.0)
        bs.rot_mat.up      = rs.Vec(0.0, 0.0, 1.0)
        self.arena.ball.set_state(bs)

        self.target_pos = sd.get("target_pos")
        self.target_time = sd.get("target_time")

        settle_ticks = int(sd.get("settle_ticks", KICKOFF_SETTLE_TICKS))
        self._settle(settle_ticks)
        self.spawn_id = sd.get("spawn_id") or self._compute_spawn_id(sd)

    def step_with_waste_check(self, ctrl):
        """单物理帧步进并执行反事实 (Counterfactual) Boost Wasted 检测：
        当按下 Boost 且车体有氮气时，在影子环境中同步推演『假如本帧不按 Boost』的速度 v_no_boost：
        - 若实际消耗了氮气 (cur_boost < no_boost_boost)，且按下 Boost 后的速度相比不按 Boost 没有任何提升甚至未阻止掉速 (v_with - v_no < 0.1 uu/s)，
          则判定本帧为 Boost Wasted (返回 waste_ratio)，否则返回 0.0。
        """
        cs_before = self.car.get_state()
        prev_boost = float(cs_before.boost)
        v_no_vec = None
        v_no_mag = None
        b_no_boost = prev_boost
        was_on_ground = bool(cs_before.is_on_ground)

        if ctrl.boost and prev_boost > 0.01:
            self.arena.clone_into(self._shadow_arena)
            c_no = rs.CarControls()
            c_no.throttle  = ctrl.throttle
            c_no.steer     = ctrl.steer
            c_no.pitch     = ctrl.pitch
            c_no.yaw       = ctrl.yaw
            c_no.roll      = ctrl.roll
            c_no.jump      = ctrl.jump
            c_no.boost     = False
            c_no.handbrake = ctrl.handbrake
            self._shadow_car.set_controls(c_no)
            self._shadow_arena.step(1)
            cf_s = self._shadow_car.get_state()
            v_no_vec = (float(cf_s.vel.x), float(cf_s.vel.y), float(cf_s.vel.z))
            v_no_mag = float(math.sqrt(v_no_vec[0]**2 + v_no_vec[1]**2 + v_no_vec[2]**2))
            b_no_boost = float(cf_s.boost)

        self.car.set_controls(ctrl)
        self.arena.step(1)

        cur_s = self.car.get_state()
        cur_boost = float(cur_s.boost)
        # 仅统计实际消耗的氮气 (0..100 标度)，若捡到 Boost Pad 导致 cur_boost 上升则记为 0.0
        boost_used = max(0.0, prev_boost - cur_boost)

        waste_ratio = 0.0
        if v_no_vec is not None and cur_boost < b_no_boost - 1e-4:
            vx, vy, vz = float(cur_s.vel.x), float(cur_s.vel.y), float(cur_s.vel.z)
            v_with_mag = float(math.sqrt(vx*vx + vy*vy + vz*vz))
            # 仅当车速接近或达到 2300 uu/s 超音速上限 (>= 2290 uu/s) 时，才计算被速度上限截断的无效喷气比例；
            # 彻底避免地面转向/漂移时因速度矢量与车头夹角导致的标量速率差被误判为 Boost Wasted。
            if v_with_mag >= 2290.0:
                if was_on_ground:
                    # 地面单帧满额 Boost 净增速度常数为 991.667 / 120 = 8.2639 uu/s
                    dv = max(0.0, v_with_mag - v_no_mag)
                    full_dv = 8.2639
                else:
                    # 空中单帧满额 Boost 净增速度常数为 1063.33 / 120 = 8.8611 uu/s
                    dv = max(0.0, v_with_mag - v_no_mag)
                    full_dv = 8.8611
                if dv < full_dv - 0.05:
                    waste_ratio = max(0.0, min(1.0, 1.0 - (dv / full_dv)))

        self.last_boost_used = boost_used
        self.last_waste_ratio = waste_ratio
        return waste_ratio

    def reset(self, mode=None, settle_ticks=KICKOFF_SETTLE_TICKS, replay_last=False, spawn_state=None):
        """重置环境并执行前置物理稳定（默认丢弃前 30 个物理帧，从第 31 帧开始正式记录/控制）。
        - replay_last=True: 100% 确定性重放上一次生成的场景初始状态（不重新随机），供 R 键与 HUMAN/MODEL 同场景对比；
        - spawn_state=... : 从指定的场景字典恢复初始状态；
        - 默认 (replay_last=False): 随机生成新场景（供 N 键或训练采样）并记录到 self.last_spawn_state。
        返回 (obs_v2, dist) 兼容接口，同时更新 self.last_obs_v3 与 self.last_parts。
        """
        if spawn_state is not None:
            self.last_spawn_state = dict(spawn_state)
            if "spawn_id" not in self.last_spawn_state:
                self.last_spawn_state["spawn_id"] = self._compute_spawn_id(self.last_spawn_state)
            self._apply_pre_settle_state(self.last_spawn_state)
        elif replay_last and self.last_spawn_state is not None:
            self._apply_pre_settle_state(self.last_spawn_state)
        else:
            use_mode = mode or self.mode
            if use_mode in ("stage0", "kickoff"):
                use_mode = "ground"
            elif use_mode == "stage1":
                use_mode = "target"
            elif use_mode == "mixed":
                use_mode = "ground" if random.random() < 0.5 else "aerial"

            self.arena.remove_car(self.car)
            self.car = self.arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)
            self._touched = False
            self.arena.reset_kickoff()

            actual_settle = int(settle_ticks)
            self.target_pos = None
            self.target_time = None
            if use_mode in ("ground", "stage0"):
                self._randomize_stage0()
            elif use_mode in ("target", "stage1"):
                self._randomize_stage1()
            elif use_mode == "static_aerial":
                self._randomize_static_aerial()
                actual_settle = 1
            elif use_mode == "aerial":
                self._randomize_aerial()
                actual_settle = 1

            self.last_spawn_state = self._capture_pre_settle_state(use_mode, actual_settle)
            self._apply_pre_settle_state(self.last_spawn_state)

        obs_v3, obs_v2, parts = build_obs_v3(
            self.car, self.arena,
            target_pos=self.target_pos, target_time=self.target_time,
        )
        self.last_obs_v3 = obs_v3
        self.last_parts = parts
        dist = float(parts["D.ball_rel_b"][3])
        return obs_v2, dist

    def _randomize_stage0(self):
        bs = self.arena.ball.get_state()
        bx = random.uniform(-2800.0, 2800.0)
        by = random.uniform(-3200.0, 3200.0)
        bs.pos = rs.Vec(bx, by, 93.15)
        bs.vel = rs.Vec(random.uniform(-300.0, 300.0), random.uniform(-300.0, 300.0), 0.0)
        bs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        bs.rot_mat.forward = rs.Vec(1.0, 0.0, 0.0)
        bs.rot_mat.right   = rs.Vec(0.0, 1.0, 0.0)
        bs.rot_mat.up      = rs.Vec(0.0, 0.0, 1.0)
        self.arena.ball.set_state(bs)

        for _ in range(20):
            cx = random.uniform(-3200.0, 3200.0)
            cy = random.uniform(-4200.0, 3200.0)
            if math.hypot(cx - bx, cy - by) > 900.0:
                break

        yaw = random.uniform(-math.pi, math.pi)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        cs = self.car.get_state()
        cs.pos = rs.Vec(cx, cy, 17.0)
        cs.vel = rs.Vec(0.0, 0.0, 0.0)
        cs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        cs.rot_mat.forward = rs.Vec(cos_y, sin_y, 0.0)
        cs.rot_mat.right   = rs.Vec(-sin_y, cos_y, 0.0)
        cs.rot_mat.up      = rs.Vec(0.0, 0.0, 1.0)
        cs.boost = random.uniform(33.3, 100.0)
        self.car.set_state(cs)

    def _randomize_stage1(self):
        """Stage 1 (地面带意图定点定时击球)：
        在 stage0 地面球与车辆随机分布的基础上，额外随机生成一个目标点 target_pos (x, y, z)
        与期望到达时间 target_time (秒)，供触球后快速模拟评估时空最近点 (CPA)。
        """
        self._randomize_stage0()
        bs = self.arena.ball.get_state()
        bx, by = float(bs.pos.x), float(bs.pos.y)
        for _ in range(30):
            tx = random.uniform(-3000.0, 3000.0)
            ty = random.uniform(-3000.0, 4800.0)
            d_b2t = math.hypot(tx - bx, ty - by)
            if 1000.0 <= d_b2t <= 4200.0:
                break
        else:
            tx, ty = 0.0, 5120.0
            d_b2t = max(1000.0, math.hypot(tx - bx, ty - by))

        tz = random.uniform(93.15, 450.0)
        # 根据距离与随机期望球速 (900..2200 uu/s，涵盖慢吊传球与大力快抽) 设定合理的目标时间
        desired_avg_spd = random.uniform(900.0, 2200.0)
        tt = float(np.clip(d_b2t / desired_avg_spd, 0.5, 2.8))
        self.target_pos = (float(tx), float(ty), float(tz))
        self.target_time = round(tt, 3)

    def evaluate_shot_cpa(self, max_sim_ticks=360):
        """触球后快速时空评估 (Fast Post-Touch Spatio-Temporal CPA Simulation)：
        将当前球的状态放入无车纯球竞技场 `_ball_sim_arena` 中快速模拟最多 max_sim_ticks (3.0s)：
        追踪球与目标点 target_pos 的三维标量距离 d(t)：
          - 当 d(t) 持续缩小时，更新最小距离 min_d 与对应时刻 min_tick；
          - 一旦 d(t) 停止缩小并开始增大 (d_now > d_prev)，说明刚刚经过最近点 (CPA)，立即截断停止模拟；
          - 若触球后第 1 帧起球就在远离目标点 (not has_decreased)，第 1 帧即刻截断并返回 inf (得 0 分，防止反向击球混初始距离分)。
        返回 (cpa_dist, cpa_time, time_err)。
        """
        tgt = OPP_GOAL_POS if self.target_pos is None else np.asarray(self.target_pos, dtype=np.float32)
        tx, ty, tz = float(tgt[0]), float(tgt[1]), float(tgt[2])

        bs = self.arena.ball.get_state()
        self._ball_sim_arena.ball.set_state(bs)

        d_prev = math.sqrt((bs.pos.x - tx)**2 + (bs.pos.y - ty)**2 + (bs.pos.z - tz)**2)
        min_d = d_prev
        min_tick = 0
        has_decreased = False

        step_ticks = 2
        for tick in range(step_ticks, int(max_sim_ticks) + 1, step_ticks):
            self._ball_sim_arena.step(step_ticks)
            b_now = self._ball_sim_arena.ball.get_state()
            d_now = math.sqrt((b_now.pos.x - tx)**2 + (b_now.pos.y - ty)**2 + (bs.pos.z if False else b_now.pos.z - tz)**2)
            if d_now < d_prev - 1e-3:
                has_decreased = True
                if d_now < min_d:
                    min_d = d_now
                    min_tick = tick
            elif d_now > d_prev + 1e-3:
                # 距离停止下降并开始反升（或触球后直接反向飞离）：立即停止模拟
                break
            d_prev = d_now

        if not has_decreased:
            return float("inf"), 0.0, float(self.target_time or 0.0)

        cpa_time = float(min_tick) / 120.0
        time_err = abs(cpa_time - float(self.target_time)) if self.target_time is not None else 0.0
        return float(min_d), cpa_time, float(time_err)

    def generate_stratified_stage0_batch(self, settle_ticks=KICKOFF_SETTLE_TICKS, rng=None, include_target=None):
        """生成 24 个正交分层桶 (Stratified Archetypes) 的 stage0 / stage1 初始场景列表：
        - 3 档车球距离: 近距 [950, 1800]、中距 [1800, 3200]、远距 [3200, 4800]
        - 4 档车头相对球夹角: 正对 [0°, 30°]、斜前 [30°, 90°]、斜后 [90°, 135°]、正背对 [135°, 180°]
        - 2 档球初速度: 低速/静止 [0, 80]、高速移动 [200, 400]
        每个桶内的实际坐标、朝向符号与速度方向均为现场随机生成，兼具『性质全覆盖』与『防固定点过拟合』。
        若为 stage1，则每个桶同时随机生成对应的 target_pos 与 target_time。
        """
        r = rng if rng is not None else random
        if include_target is None:
            include_target = (self.mode in ("target", "stage1"))
        dist_bins = [(950.0, 1800.0), (1800.0, 3200.0), (3200.0, 4800.0)]
        angle_bins = [
            (0.0, math.pi / 6.0),
            (math.pi / 6.0, math.pi / 2.0),
            (math.pi / 2.0, 3.0 * math.pi / 4.0),
            (3.0 * math.pi / 4.0, math.pi),
        ]
        bvel_bins = [(0.0, 80.0), (200.0, 400.0)]

        batch = []
        for d_lo, d_hi in dist_bins:
            for a_lo, a_hi in angle_bins:
                for v_lo, v_hi in bvel_bins:
                    for _ in range(50):
                        bx = r.uniform(-2400.0, 2400.0)
                        by = r.uniform(-2800.0, 2800.0)
                        dist = r.uniform(d_lo, d_hi)
                        bearing = r.uniform(-math.pi, math.pi)
                        cx = bx - dist * math.cos(bearing)
                        cy = by - dist * math.sin(bearing)
                        if -3200.0 <= cx <= 3200.0 and -4200.0 <= cy <= 3200.0:
                            break
                    else:
                        bx, by = 0.0, 500.0
                        dist = 0.5 * (d_lo + d_hi)
                        bearing = math.pi * 0.5
                        cx, cy = 0.0, 500.0 - dist

                    rel_ang = r.uniform(a_lo, a_hi) * (1.0 if r.random() < 0.5 else -1.0)
                    yaw = bearing + rel_ang
                    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

                    bspd = r.uniform(v_lo, v_hi)
                    bdir = r.uniform(-math.pi, math.pi)
                    bvx = bspd * math.cos(bdir)
                    bvy = bspd * math.sin(bdir)

                    t_pos = None
                    t_time = None
                    if include_target:
                        for _ in range(30):
                            tx = r.uniform(-3000.0, 3000.0)
                            ty = r.uniform(-3000.0, 4800.0)
                            d_b2t = math.hypot(tx - bx, ty - by)
                            if 1000.0 <= d_b2t <= 4200.0:
                                break
                        else:
                            tx, ty = 0.0, 5120.0
                            d_b2t = max(1000.0, math.hypot(tx - bx, ty - by))
                        tz = r.uniform(93.15, 450.0)
                        desired_avg_spd = r.uniform(900.0, 2200.0)
                        t_time = round(float(np.clip(d_b2t / desired_avg_spd, 0.5, 2.8)), 3)
                        t_pos = (float(tx), float(ty), float(tz))

                    sd = {
                        "mode": "target" if include_target else "ground",
                        "settle_ticks": int(settle_ticks),
                        "car_pos": (float(cx), float(cy), 17.0),
                        "car_vel": (0.0, 0.0, 0.0),
                        "car_ang": (0.0, 0.0, 0.0),
                        "car_fwd": (float(cos_y), float(sin_y), 0.0),
                        "car_rgt": (float(-sin_y), float(cos_y), 0.0),
                        "car_up":  (0.0, 0.0, 1.0),
                        "car_boost": float(r.uniform(35.0, 100.0)),
                        "ball_pos": (float(bx), float(by), 93.15),
                        "ball_vel": (float(bvx), float(bvy), 0.0),
                        "ball_ang": (0.0, 0.0, 0.0),
                        "target_pos": t_pos,
                        "target_time": t_time,
                    }
                    sd["spawn_id"] = self._compute_spawn_id(sd)
                    batch.append(sd)
        return batch

    def _randomize_static_aerial(self):
        """Static Aerial (空中低速/慢抛高球)：
        球在水平方向近似静止或微动，带有向上的初始速度使球在空中停留较长时间，供训练初级起跳飞球。
        """
        bs = self.arena.ball.get_state()
        bx = random.uniform(-2000.0, 2000.0)
        by = random.uniform(-2000.0, 2500.0)
        bz = random.uniform(220.0, 750.0)
        bs.pos = rs.Vec(bx, by, bz)
        bs.vel = rs.Vec(random.uniform(-80.0, 80.0), random.uniform(-80.0, 80.0), random.uniform(350.0, 650.0))
        bs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        bs.rot_mat.forward = rs.Vec(1.0, 0.0, 0.0)
        bs.rot_mat.right   = rs.Vec(0.0, 1.0, 0.0)
        bs.rot_mat.up      = rs.Vec(0.0, 0.0, 1.0)
        self.arena.ball.set_state(bs)

        for _ in range(20):
            cx = random.uniform(-2500.0, 2500.0)
            cy = random.uniform(-3500.0, 2000.0)
            if 900.0 <= math.hypot(cx - bx, cy - by) <= 2800.0:
                break
        yaw = random.uniform(-math.pi, math.pi)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        cs = self.car.get_state()
        cs.pos = rs.Vec(cx, cy, 17.0)
        cs.vel = rs.Vec(0.0, 0.0, 0.0)
        cs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        cs.rot_mat.forward = rs.Vec(cos_y, sin_y, 0.0)
        cs.rot_mat.right   = rs.Vec(-sin_y, cos_y, 0.0)
        cs.rot_mat.up      = rs.Vec(0.0, 0.0, 1.0)
        cs.boost = random.uniform(60.0, 100.0)
        self.car.set_state(cs)

    def _randomize_aerial(self):
        bs = self.arena.ball.get_state()
        bx = random.uniform(-2000.0, 2000.0)
        by = random.uniform(-2000.0, 2500.0)
        bz = random.uniform(300.0, 1100.0)
        bs.pos = rs.Vec(bx, by, bz)
        bs.vel = rs.Vec(random.uniform(-400.0, 400.0), random.uniform(-400.0, 400.0), random.uniform(100.0, 500.0))
        bs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        bs.rot_mat.forward = rs.Vec(1.0, 0.0, 0.0)
        bs.rot_mat.right   = rs.Vec(0.0, 1.0, 0.0)
        bs.rot_mat.up      = rs.Vec(0.0, 0.0, 1.0)
        self.arena.ball.set_state(bs)

        cx = random.uniform(-2500.0, 2500.0)
        cy = random.uniform(-3800.0, 1000.0)
        yaw = random.uniform(-math.pi, math.pi)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        cs = self.car.get_state()
        cs.pos = rs.Vec(cx, cy, 17.0)
        cs.vel = rs.Vec(0.0, 0.0, 0.0)
        cs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        cs.rot_mat.forward = rs.Vec(cos_y, sin_y, 0.0)
        cs.rot_mat.right   = rs.Vec(-sin_y, cos_y, 0.0)
        cs.rot_mat.up      = rs.Vec(0.0, 0.0, 1.0)
        cs.boost = random.uniform(60.0, 100.0)
        self.car.set_state(cs)

    def get_obs_v3(self, intent_override=None):
        obs_v3, obs_v2, parts = build_obs_v3(
            self.car, self.arena,
            intent_override=intent_override,
            target_pos=self.target_pos, target_time=self.target_time,
        )
        self.last_obs_v3 = obs_v3
        self.last_parts = parts
        return obs_v3, obs_v2, parts

    def snapshot(self):
        cs = self.car.get_state()
        bs = self.arena.ball.get_state()
        up = self.car.get_up_dir()
        fwd = self.car.get_forward_dir()
        right = self.car.get_right_dir()
        rel = np.array([bs.pos.x - cs.pos.x, bs.pos.y - cs.pos.y, bs.pos.z - cs.pos.z])
        dist = float(np.linalg.norm(rel))
        dist_h = float(np.hypot(rel[0], rel[1]))
        rel_vel = np.array([bs.vel.x - cs.vel.x, bs.vel.y - cs.vel.y, bs.vel.z - cs.vel.z])
        closing_speed = -float(np.dot(rel, rel_vel)) / max(dist, 1e-6)
        fwd_h = np.array([fwd.x, fwd.y])
        d_h = np.array([rel[0], rel[1]])
        n = np.linalg.norm(d_h) + 1e-6
        align = float(np.dot(fwd_h, d_h / n))

        car_speed = float(math.sqrt(cs.vel.x**2 + cs.vel.y**2 + cs.vel.z**2))
        ball_pos_np = np.array([bs.pos.x, bs.pos.y, bs.pos.z], dtype=np.float32)
        ball_vel_np = np.array([bs.vel.x, bs.vel.y, bs.vel.z], dtype=np.float32)
        tgt_np = OPP_GOAL_POS if self.target_pos is None else np.asarray(self.target_pos, dtype=np.float32)
        b2g = tgt_np - ball_pos_np
        b2g_dir = b2g / (float(np.linalg.norm(b2g)) + 1e-6)
        ball_vel_to_goal = float(np.dot(ball_vel_np, b2g_dir))
        ball_speed = float(np.linalg.norm(ball_vel_np))

        return {
            "car_pos": (cs.pos.x, cs.pos.y, cs.pos.z),
            "car_vel": (cs.vel.x, cs.vel.y, cs.vel.z),
            "car_ang": (cs.ang_vel.x, cs.ang_vel.y, cs.ang_vel.z),
            "car_speed": car_speed,
            "is_supersonic": bool(cs.is_supersonic),
            "ball_pos": (bs.pos.x, bs.pos.y, bs.pos.z),
            "ball_vel": (bs.vel.x, bs.vel.y, bs.vel.z),
            "ball_speed": ball_speed,
            "ball_vel_to_goal": ball_vel_to_goal,
            "target_pos": self.target_pos,
            "target_time": self.target_time,
            "up_z": up.z,
            "fwd":   (fwd.x,   fwd.y,   fwd.z),
            "up":    (up.x,    up.y,    up.z),
            "right": (right.x, right.y, right.z),
            "on_ground": bool(cs.is_on_ground),
            "is_flipping": bool(cs.is_flipping),
            "is_jumping": bool(cs.is_jumping),
            "air_time": float(cs.air_time),
            "boost": float(cs.boost),
            "dist": dist,
            "dist_h": dist_h,
            "align": align,
            "closing": closing_speed,
            "touched": self._touched,
        }
