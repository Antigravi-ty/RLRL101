"""core/reward.py —— 1v1 最快开球 (Fastest Kickoff) 极简势能差分 + 非线性极速时间大奖 + 胜负零和博弈奖励函数。

设计原则:
1. 纯欧氏距离势能差分 (`r_dist`):
   - `r_dist = distance_delta_scale * (prev_dist - eff_new_dist)`
   - 比上一帧更近则加分，且靠近速度越快单步加分越高；比上一帧更远则立刻扣分。
   - 触球瞬间将 `eff_new_dist` 锁定为球面接触距离 `touch_surface_dist = 150.0`，保证整条轨迹的距离总势能严格守恒 (`init_dist - 150.0`)，杜绝任何刷分可能。
2. 每帧固定时间税 (`r_time`):
   - `-time_penalty_per_tick * ticks_elapsed`
3. 终端非线性极速大奖 + 1v1 先触球胜负奖惩 (`r_touch`, `r_timeout`):
   - 获胜方 (先触球): 获得基础胜利分 `touch_bonus` + 三次方非线性剩余时间大奖 `touch_time_bonus_max * (speed_ratio ** time_bonus_power)`。
     触球时间越短，边际奖励呈指数/高次幂爆发，逼迫 PPO 跳出“纯直线喷气”局部最优，涌现出 Speed Flip！
   - 失败方 (被对手先触球): 立即终止回合，并扣除失败惩罚 `-lose_penalty` 以及落后剩余距离惩罚 `-lose_dist_penalty_scale * (new_dist - 150.0)`。
   - 超时 (6秒内双方均未触球): 扣除 `-timeout_penalty` 及剩余距离惩罚。
"""
from dataclasses import dataclass
from .constants import MAX_DECISIONS, TICKS_PER_DECISION, TOUCH_SURFACE_DIST


@dataclass
class RewardConfig:
    distance_delta_scale:    float = 0.015              # 每缩短 1000 uu 距离获得 +15.0 势能分
    touch_surface_dist:      float = TOUCH_SURFACE_DIST # 150.0 uu 触球中心距离守恒锚点
    time_penalty_per_tick:   float = 0.025              # 每物理帧时间税 (-3.0 / 秒)
    touch_bonus:             float = 45.0               # 先触球基础胜利大奖
    touch_time_bonus_max:    float = 95.0               # 非线性时间大奖上限 (越快越呈立方级暴涨)
    time_bonus_power:        float = 3.0                # 非线性时间奖励幂次 (p=3.0 强力拉开 1.90s SpeedFlip 与 2.20s 直线冲刺的差距)
    min_possible_ticks:      float = 180.0              # 理论极限最短帧数锚点 (1.50s = 180 ticks)
    ball_vy_bonus_scale:     float = 0.005              # 独自触球瞬间球纵向速度奖励系数 (同时触球平局时球初速为无效数据不计分)
    lose_penalty:            float = 35.0               # 1v1 慢于对手被抢先触球的基础惩罚
    lose_dist_penalty_scale: float = 0.010              # 失败/超时时未跑完的剩余距离追加惩罚系数
    timeout_penalty:         float = 45.0               # 6秒 (720 ticks) 未触球超时惩罚
    max_decisions:           int   = MAX_DECISIONS      # 180 决策步 (720 ticks = 6.0 秒)


def compute_step_reward(
    cfg: RewardConfig,
    prev_dist: float = 0.0,
    new_dist: float = 0.0,
    init_dist: float = None,
    touched: bool = False,
    opp_touched: bool = False,
    timeout: bool = False,
    ticks_elapsed: int = 1,
    ep_ticks: int = 0,
    max_decisions_ep: int = None,
    new_snap: dict = None,
    ball_vel_y: float = None,
    is_orange: bool = False,
    **kwargs,
):
    elapsed_ticks = float(max(1, int(ticks_elapsed)))

    # 1. 距离势能差分 (靠近加分且速度越快分越高，远离扣分；触球时锚定到 touch_surface_dist 保证势能守恒)
    if touched:
        eff_new_dist = min(float(init_dist), cfg.touch_surface_dist) if init_dist is not None else cfg.touch_surface_dist
    else:
        eff_new_dist = float(new_dist)
    eff_prev_dist = float(prev_dist)
    r_dist = cfg.distance_delta_scale * (eff_prev_dist - eff_new_dist)

    # 2. 每物理帧时间惩罚
    r_time = -cfg.time_penalty_per_tick * elapsed_ticks

    # 3. 终端非线性极速大奖 & 1v1 胜负零和惩罚 & 独自触球纵向球速加成
    eff_max_dec = int(max_decisions_ep) if (max_decisions_ep is not None and max_decisions_ep > 0) else int(cfg.max_decisions)
    max_ticks = float(max(1, eff_max_dec * TICKS_PER_DECISION))
    min_ticks = float(min(max_ticks - 1.0, max(0.0, cfg.min_possible_ticks)))

    r_touch = 0.0
    r_timeout = 0.0

    if touched:
        if ep_ticks > 0:
            speed_ratio = max(0.0, min(1.0, (max_ticks - float(ep_ticks)) / max(1.0, max_ticks - min_ticks)))
        else:
            speed_ratio = 0.5
        nonlinear_time_bonus = cfg.touch_time_bonus_max * (speed_ratio ** cfg.time_bonus_power)
        if opp_touched:
            # 同一物理帧双方同时触球 (平局): 基础胜利分折半，依然享有速度时间分；此时球第一瞬间速度属于对撞无效数据，不计球速分
            r_touch = 0.5 * cfg.touch_bonus + nonlinear_time_bonus
        else:
            # 独自率先触球获胜: 额外根据开球瞬间球的纵向速度 (朝对方球门方向 vy) 给予奖励
            vy_val = 0.0
            if ball_vel_y is not None:
                vy_val = (-1.0 if is_orange else 1.0) * float(ball_vel_y)
            elif isinstance(new_snap, dict) and "ball_vel" in new_snap:
                team_s = -1.0 if int(new_snap.get("team", 0)) == 1 else 1.0
                vy_val = team_s * float(new_snap["ball_vel"][1])
            vy_bonus = max(0.0, vy_val) * getattr(cfg, "ball_vy_bonus_scale", 0.0)
            r_touch = cfg.touch_bonus + nonlinear_time_bonus + vy_bonus
    elif opp_touched:
        # 被对手抢先触球 (1v1 战败惩罚 + 剩余未闭合距离惩罚)
        rem_dist = max(0.0, float(new_dist) - cfg.touch_surface_dist)
        r_timeout = -(cfg.lose_penalty + cfg.lose_dist_penalty_scale * rem_dist)
    elif timeout:
        # 6秒超时未触球
        rem_dist = max(0.0, float(new_dist) - cfg.touch_surface_dist)
        r_timeout = -(cfg.timeout_penalty + cfg.lose_dist_penalty_scale * rem_dist)

    r_total = r_dist + r_time + r_touch + r_timeout

    detail = {
        "r_dist":     r_dist,
        "r_time":     r_time,
        "r_touch":    r_touch,
        "r_timeout":  r_timeout,
        "r_slide":    0.0,
        "r_align":    0.0,
        "r_boost":    0.0,
        "r_pickup":   0.0,
        "r_jump":     0.0,
        "r_min_dist": 0.0,
        "r_total":    r_total,
    }
    return r_total, detail
