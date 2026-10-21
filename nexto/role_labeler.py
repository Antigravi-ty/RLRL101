"""nexto/role_labeler.py —— 四角色战术体系 (Chaser / 乐观接应 / 悲观接应 / 自由人) 实时启发式标注器与势能场。

角色定义 (4-Dim One-Hot):
  0: CHSR [1, 0, 0, 0] — Chaser (1st Man / 主控逼抢): 本队预计到达球时间 (ETA) 最短的可用车辆
  1: OPTI [0, 1, 0, 0] — Optimistic Support (2nd Man / 乐观接应): 前场/侧翼 1500~2500 uu 黄金接应位
  2: PESS [0, 0, 1, 0] — Pessimistic Support (3rd Man / 悲观接应): 球与本方球门连线后方 Goal-side 防守保底位
  3: FREE [0, 0, 0, 1] — Freelancer / Recovery (Rotating Out / 自由人): 刚踢完球失控恢复、冲入对方球门或低氮气退防寻路
"""
import math
import numpy as np

ROLE_CHASER      = 0
ROLE_OPTIMISTIC  = 1
ROLE_PESSIMISTIC = 2
ROLE_FREELANCER  = 3
NUM_ROLES        = 4

ROLE_SHORT_NAMES = ("CHSR", "OPTI", "PESS", "FREE")
ROLE_FULL_NAMES  = (
    "Chaser(逼抢主控)",
    "Optimistic(乐观接应)",
    "Pessimistic(悲观保底)",
    "Freelancer(自由恢复)",
)

ROLE_ONE_HOTS = np.eye(NUM_ROLES, dtype=np.float32)

# 3D 渲染与 HUD 角色专属高亮配色 (RGBA)
ROLE_EDGE_COLORS = {
    ROLE_CHASER:      (1.00, 0.90, 0.10, 0.98),  # 明亮金黄 (1st Man 主控逼抢)
    ROLE_OPTIMISTIC:  (0.15, 0.95, 0.45, 0.92),  # 翠绿 (2nd Man 乐观前插接应)
    ROLE_PESSIMISTIC: (0.25, 0.70, 1.00, 0.92),  # 护盾蓝 (3rd Man 悲观拖后保底)
    ROLE_FREELANCER:  (0.80, 0.40, 0.95, 0.85),  # 紫罗兰 (Rotating Out 自由恢复/吃氮气)
}


class TeamRoleLabeler:
    """对场上蓝队 (team=0) 与橙队 (team=1) 独立执行四角色实时/事后隐式意图自动标注 (含迟滞防抖)。"""

    def __init__(self, hysteresis_eta_s: float = 0.14, post_touch_free_ticks: int = 72):
        self.hysteresis_eta_s = float(hysteresis_eta_s)
        self.post_touch_free_ticks = int(post_touch_free_ticks)
        self._last_touch_tick = {}
        self._prev_team_chaser = {0: None, 1: None}
        self._prev_roles = {}

    def reset(self):
        self._last_touch_tick.clear()
        self._prev_team_chaser = {0: None, 1: None}
        self._prev_roles.clear()

    def compute_roles(
        self,
        cars: list,
        car_teams: np.ndarray,
        arena,
        step_seq: int = 0,
        touched_by_car: list = None,
    ) -> list:
        """为场上所有车辆计算当前帧的四角色标签、One-Hot 向量、ETA 估计与角色势能 Φ_role。
        返回长度为 n_cars 的字典列表:
          {
            "role_id": int (0..3),
            "role_code": str ("CHSR"|"OPTI"|"PESS"|"FREE"),
            "role_name": str,
            "role_onehot": np.ndarray (4,),
            "eta_s": float,
            "potential": float,
            "reason": str,
          }
        """
        n_cars = len(cars)
        if n_cars == 0:
            return []

        bs = arena.ball.get_state()
        bx, by, bz = float(bs.pos.x), float(bs.pos.y), float(bs.pos.z)
        bvx, bvy, bvz = float(bs.vel.x), float(bs.vel.y), float(bs.vel.z)
        # 短期球轨迹线性外推锚点 (0.35s)
        pred_bx = float(np.clip(bx + 0.35 * bvx, -3950.0, 3950.0))
        pred_by = float(np.clip(by + 0.35 * bvy, -5050.0, 5050.0))
        pred_bz = float(np.clip(bz + 0.35 * bvz - 0.5 * 650.0 * (0.35 ** 2), 93.0, 1950.0))

        if touched_by_car is not None:
            for i, t_flag in enumerate(touched_by_car):
                if t_flag:
                    self._last_touch_tick[i] = int(step_seq)

        # 预提取每辆车的运动学快照与 ETA
        car_metrics = []
        for i, car in enumerate(cars):
            team = int(car_teams[i]) if i < len(car_teams) else 0
            cs = car.get_state()
            fwd = car.get_forward_dir()
            up = car.get_up_dir()

            cx, cy, cz = float(cs.pos.x), float(cs.pos.y), float(cs.pos.z)
            vx, vy, vz = float(cs.vel.x), float(cs.vel.y), float(cs.vel.z)
            spd = math.hypot(vx, vy, vz)
            boost = float(cs.boost)
            on_gnd = bool(cs.is_on_ground)
            is_demo = bool(cs.is_demoed)
            hf = cs.has_flip_or_jump() if callable(cs.has_flip_or_jump) else cs.has_flip_or_jump
            has_flip = bool(hf)

            dx = pred_bx - cx
            dy = pred_by - cy
            dz = pred_bz - cz
            dist_pred = math.hypot(dx, dy, dz)

            raw_dx = bx - cx
            raw_dy = by - cy
            raw_dz = bz - cz
            raw_dist = math.hypot(raw_dx, raw_dy, raw_dz)
            inv_d = 1.0 / max(raw_dist, 1e-6)
            closing_spd = (raw_dx * vx + raw_dy * vy + raw_dz * vz) * inv_d

            dist_h = math.hypot(raw_dx, raw_dy) + 1e-6
            align_ball = (float(fwd.x) * raw_dx + float(fwd.y) * raw_dy) / dist_h

            # 1. 计算预计到达时间 ETA (秒)
            eff_accel_spd = max(550.0, closing_spd + 9.5 * boost + (280.0 if align_ball > 0.4 else -220.0))
            eta_s = dist_pred / eff_accel_spd
            if align_ball < -0.25:
                eta_s += 0.40 * (abs(align_ball))
            if not on_gnd and cz > 160.0 and closing_spd < 150.0:
                eta_s += 0.65
            if pred_bz > 380.0 and boost < 15.0 and cz < 120.0:
                eta_s += 0.75
            if is_demo:
                eta_s = 99.0

            # 2. 判断是否属于 Freelancer (刚触完球失控远离 / 冲进对方球门 / 低氮气退防寻路 / 空中失控)
            ticks_since_touch = step_seq - self._last_touch_tick.get(i, -9999)
            opp_goal_y = 5120.0 if team == 0 else -5120.0
            own_goal_y = -5120.0 if team == 0 else 5120.0

            in_opp_goal_box = (cy > 4980.0) if team == 0 else (cy < -4980.0)
            post_touch_recoil = (
                0 < ticks_since_touch <= self.post_touch_free_ticks
                and (not on_gnd or closing_spd < -120.0 or align_ball < -0.15)
                and raw_dist > 260.0
            )
            air_helpless = (
                (not on_gnd)
                and cz > 180.0
                and (not has_flip)
                and boost < 10.0
                and closing_spd < 80.0
            )
            starved_retreat = (
                boost < 14.0
                and raw_dist > 1350.0
                and closing_spd < -80.0
                and align_ball < -0.20
            )
            is_free_candidate = bool(is_demo or in_opp_goal_box or post_touch_recoil or air_helpless or starved_retreat)
            free_reason = (
                "demoed" if is_demo else
                "in_opp_goal" if in_opp_goal_box else
                "post_touch_recovery" if post_touch_recoil else
                "air_helpless" if air_helpless else
                "low_boost_retreat" if starved_retreat else ""
            )

            # 计算与本方球门连线的保底距离 (用于区分 Pessimistic 3rd Man 与 Optimistic 2nd Man)
            dist_own_goal = math.hypot(cx - 0.0, cy - own_goal_y)
            dist_opp_goal = math.hypot(cx - 0.0, cy - opp_goal_y)

            car_metrics.append({
                "idx": i,
                "team": team,
                "pos": (cx, cy, cz),
                "vel": (vx, vy, vz),
                "fwd": (float(fwd.x), float(fwd.y), float(fwd.z)),
                "up_z": float(up.z),
                "spd": spd,
                "boost": boost,
                "on_ground": on_gnd,
                "raw_dist": raw_dist,
                "closing_spd": closing_spd,
                "align_ball": align_ball,
                "eta_s": float(eta_s),
                "is_free_cand": is_free_candidate,
                "free_reason": free_reason,
                "dist_own_goal": dist_own_goal,
                "dist_opp_goal": dist_opp_goal,
                "own_goal_y": own_goal_y,
                "opp_goal_y": opp_goal_y,
            })

        results = [None] * n_cars

        for team_id in (0, 1):
            team_members = [m for m in car_metrics if m["team"] == team_id]
            if not team_members:
                continue

            # 若整队只有 1 辆车 (如 1v1)，除非处于严重失控恢复，否则默认为 Chaser
            if len(team_members) == 1:
                m0 = team_members[0]
                r_id = ROLE_FREELANCER if (m0["is_free_cand"] and m0["raw_dist"] > 900.0) else ROLE_CHASER
                reason = m0["free_reason"] if r_id == ROLE_FREELANCER else "solo_chaser"
                results[m0["idx"]] = self._pack_result(m0, r_id, reason, (bx, by, bz))
                self._prev_team_chaser[team_id] = m0["idx"] if r_id == ROLE_CHASER else None
                continue

            # 1) 选出本队的 Chaser (1st Man):
            #    优先在非 Freelancer 候选车中选 ETA 最短的；若全队都是 Freelancer 候选，则选全局 ETA 最短的
            active_cands = [m for m in team_members if not m["is_free_cand"]]
            chaser_pool = active_cands if active_cands else team_members
            chaser_pool_sorted = sorted(chaser_pool, key=lambda m: m["eta_s"])
            best_chaser = chaser_pool_sorted[0]

            # 迟滞防抖 (Hysteresis): 若上一帧的 Chaser 仍在可用池中且 ETA 与当前最快者差距 < hysteresis_eta_s，则保持不跳变
            prev_c_idx = self._prev_team_chaser.get(team_id)
            if prev_c_idx is not None and prev_c_idx != best_chaser["idx"]:
                prev_m = next((m for m in chaser_pool if m["idx"] == prev_c_idx), None)
                if prev_m is not None and (prev_m["eta_s"] - best_chaser["eta_s"]) <= self.hysteresis_eta_s:
                    best_chaser = prev_m

            chaser_idx = best_chaser["idx"]
            self._prev_team_chaser[team_id] = chaser_idx
            results[chaser_idx] = self._pack_result(best_chaser, ROLE_CHASER, f"min_eta({best_chaser['eta_s']:.2f}s)", (bx, by, bz))

            # 2) 对剩余队友分配 Freelancer / Pessimistic (3rd Man) / Optimistic (2nd Man)
            remaining = [m for m in team_members if m["idx"] != chaser_idx]
            non_free_rem = []
            for m in remaining:
                if m["is_free_cand"]:
                    results[m["idx"]] = self._pack_result(m, ROLE_FREELANCER, m["free_reason"], (bx, by, bz))
                else:
                    non_free_rem.append(m)

            if len(non_free_rem) == 1:
                m_rem = non_free_rem[0]
                # 若仅剩 1 名健康队友：根据球在我方半场还是前场、以及该车是否处于 Goal-side 决定是 PESS 还是 OPTI
                ball_in_own_half = (by < -400.0) if team_id == 0 else (by > 400.0)
                car_behind_ball = (m_rem["pos"][1] < by + 200.0) if team_id == 0 else (m_rem["pos"][1] > by - 200.0)
                if ball_in_own_half or (car_behind_ball and m_rem["dist_own_goal"] < best_chaser["dist_own_goal"]):
                    results[m_rem["idx"]] = self._pack_result(m_rem, ROLE_PESSIMISTIC, "goal_side_anchor", (bx, by, bz))
                else:
                    results[m_rem["idx"]] = self._pack_result(m_rem, ROLE_OPTIMISTIC, "offensive_wing_2nd", (bx, by, bz))
            elif len(non_free_rem) >= 2:
                # 按距离本方球门的远近排序：离本方球门最近的是悲观保底 (PESS, 3rd Man)，其余靠前的是乐观接应 (OPTI, 2nd Man)
                rem_by_goal = sorted(non_free_rem, key=lambda m: m["dist_own_goal"])
                pess_m = rem_by_goal[0]
                results[pess_m["idx"]] = self._pack_result(pess_m, ROLE_PESSIMISTIC, "3rd_man_last_back", (bx, by, bz))
                for opt_m in rem_by_goal[1:]:
                    results[opt_m["idx"]] = self._pack_result(opt_m, ROLE_OPTIMISTIC, "2nd_man_support", (bx, by, bz))

        return results

    def _pack_result(self, m: dict, role_id: int, reason: str, ball_pos: tuple) -> dict:
        phi = compute_role_potential(
            role_id=role_id,
            car_pos=m["pos"],
            car_vel=m["vel"],
            car_fwd=m["fwd"],
            car_up_z=m["up_z"],
            on_ground=m["on_ground"],
            ball_pos=ball_pos,
            team=m["team"],
        )
        self._prev_roles[m["idx"]] = role_id
        return {
            "role_id": int(role_id),
            "role_code": ROLE_SHORT_NAMES[role_id],
            "role_name": ROLE_FULL_NAMES[role_id],
            "role_onehot": ROLE_ONE_HOTS[role_id].copy(),
            "eta_s": float(m["eta_s"]),
            "potential": float(phi),
            "reason": str(reason),
        }


def compute_role_potential(
    role_id: int,
    car_pos: tuple,
    car_vel: tuple,
    car_fwd: tuple,
    car_up_z: float,
    on_ground: bool,
    ball_pos: tuple,
    team: int = 0,
) -> float:
    """计算四角色专属无冲突位置势能函数 Φ_role(s) ∈ [0, 1]：
    - Chaser (0)     : exp(-||p_car - p_ball|| / 1410)
    - Optimistic (1) : 黄金接应环 exp(-(||p_car - p_ball|| - 1800)^2 / (2 * 800^2)) * (0.5 + 0.5 * cos_offensive)
    - Pessimistic (2): 球门后方 2500 uu 防线锚点 exp(-||p_car - p_anchor|| / 1500) * goal_side_alignment
    - Freelancer (3) : 远离球混乱区排斥势能 + 平稳落地高速恢复奖励
    """
    cx, cy, cz = float(car_pos[0]), float(car_pos[1]), float(car_pos[2])
    vx, vy, vz = float(car_vel[0]), float(car_vel[1]), float(car_vel[2])
    fx, fy, fz = float(car_fwd[0]), float(car_fwd[1]), float(car_fwd[2])
    bx, by, bz = float(ball_pos[0]), float(ball_pos[1]), float(ball_pos[2])

    dist_ball = math.hypot(cx - bx, cy - by, cz - bz)
    own_goal_y = -5120.0 if int(team) == 0 else 5120.0
    opp_goal_y = 5120.0 if int(team) == 0 else -5120.0

    if role_id == ROLE_CHASER:
        return float(math.exp(-dist_ball / 1410.0))

    if role_id == ROLE_OPTIMISTIC:
        # 黄金接应环: 距离球 1800 uu 处势能最高，贴太近 (<500) 势能下降，车头朝向球与对方球门中线
        ring = math.exp(-((dist_ball - 1800.0) ** 2) / (2.0 * (800.0 ** 2)))
        target_x = 0.5 * bx
        target_y = 0.5 * (by + opp_goal_y)
        tx, ty = target_x - cx, target_y - cy
        tn = math.hypot(tx, ty) + 1e-6
        cos_off = (fx * tx + fy * ty) / tn
        return float(ring * (0.5 + 0.5 * max(-1.0, min(1.0, cos_off))))

    if role_id == ROLE_PESSIMISTIC:
        # 黄金保底锚点: 球与自家球门中心连线往自家球门方向 2500 uu 处
        gx, gy = 0.0 - bx, own_goal_y - by
        gn = math.hypot(gx, gy) + 1e-6
        ux, uy = gx / gn, gy / gn
        anchor_x = float(np.clip(bx + 2500.0 * ux, -3400.0, 3400.0))
        anchor_y = float(np.clip(by + 2500.0 * uy, -4850.0, 4850.0))
        dist_anchor = math.hypot(cx - anchor_x, cy - anchor_y)
        # 车头朝向球方向准备迎球拦截
        bx_rel, by_rel = bx - cx, by - cy
        bn = math.hypot(bx_rel, by_rel) + 1e-6
        align_play = 0.5 + 0.5 * ((fx * bx_rel + fy * by_rel) / bn)
        return float(math.exp(-dist_anchor / 1500.0) * align_play)

    # ROLE_FREELANCER: 远离球 1200 uu 碰撞区 + 落地平稳高速恢复
    spacing_factor = 1.0 - 0.65 * math.exp(-dist_ball / 900.0)
    spd = math.hypot(vx, vy, vz)
    recovery_factor = (0.7 + 0.3 * min(1.0, spd / 1400.0)) if on_ground else (0.35 * max(0.0, float(car_up_z)))
    return float(spacing_factor * recovery_factor)
