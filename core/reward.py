"""core/reward.py —— 静态球大王：大道至简欧几里得距离势能守恒奖励函数。

核心设计哲学 (大道至简):
1. 路径无关的欧几里得直线距离势能 (`r_dist`):
   - 逐步势能差分: `(prev_dist - new_dist) * distance_delta_scale`
   - 触球瞬间精准补齐碰撞残余距离 `(new_dist - touch_surface_dist)`，使得无论从任何路径、在任意时刻完成触球，
     整局在距离势能上的总得分恒定等于 `max(0, init_dist - touch_surface_dist) * distance_delta_scale`！
2. 时间惩罚 (`r_time`) + 触球剩余时间稀疏大奖 (`r_touch`):
   - 每帧时间惩罚 `-time_penalty_per_tick`，触球瞬间额外获得 `touch_bonus + touch_time_bonus_max * rem_ratio`。
3. 终局历史最小距离比例稀疏结算 (`r_min_dist` at touch or timeout):
   - 回合结束时按 `(init_dist - min_dist) / init_dist` 发放 `+min_dist_ratio_bonus * ratio`。
4. 极微弱动作微调项 (`r_jump`, `r_boost`, `r_slide`):
   - Jump / Double-Jump / Boost 在绝大多数情况下带来巨大正收益，因此施加极微小扣分，使模型在无需使用时自然松手；
   - Powerslide (`handbrake`) 在地面转向/横滑泄力时短期具有减速负效应，因此给予极微小正加分 (`+powerslide_bonus_per_tick`) 鼓励探索漂移恢复与急转。
"""
from dataclasses import dataclass
from .constants import MAX_DECISIONS, TICKS_PER_DECISION


@dataclass
class RewardConfig:
    distance_delta_scale:       float = 0.010     # 10.0 pts per 1000 uu Euclidean distance closed
    touch_surface_dist:         float = 150.0     # Contact center-to-center distance for exact path conservation
    time_penalty_per_tick:      float = 0.045     # Time penalty per tick (-5.4/s) to drive maximum speed
    boost_penalty_per_unit:     float = 0.004     # Very weak boost cost per unit used
    boost_pickup_bonus:         float = 0.045     # Small bonus per boost pad picked up (offsets 1 time_penalty_per_tick for BoostPadAttention)
    jump_hold_penalty_per_tick: float = 0.0008    # Very weak penalty while holding Jump button (~0.096/s)
    djump_trigger_penalty:      float = 0.015     # Very weak one-time penalty on triggering double-jump / flip
    powerslide_bonus_per_tick:  float = 0.0006    # Very weak bonus while holding Powerslide on ground (~0.072/s)
    touch_bonus:                float = 70.0      # Base sparse touch score
    touch_time_bonus_max:       float = 50.0      # Early-arrival time bonus (up to +50.0)
    min_dist_ratio_bonus:       float = 20.0      # Terminal score based on (init_dist - min_dist) / init_dist
    timeout_penalty:            float = 12.0      # Terminal timeout penalty
    max_decisions:              int   = MAX_DECISIONS


def compute_min_dist_ratio_reward(
    cfg: RewardConfig,
    init_dist: float,
    min_dist: float,
    touched: bool = False,
) -> float:
    if touched:
        return float(cfg.min_dist_ratio_bonus)
    d0 = float(init_dist)
    if d0 <= 1e-4:
        return 0.0
    d_min = max(0.0, min(d0, float(min_dist)))
    ratio = max(0.0, min(1.0, (d0 - d_min) / d0))
    return float(cfg.min_dist_ratio_bonus) * ratio


def compute_step_reward(
    cfg: RewardConfig,
    prev_dist: float = 0.0,
    new_dist: float = 0.0,
    init_dist: float = None,
    min_dist: float = None,
    on_ground: bool = True,
    boost_used: float = 0.0,
    boost_pickups: int = 0,
    jump_held_ticks: int = 0,
    handbrake_held_ticks: int = 0,
    djump_triggered: bool = False,
    flip_triggered: bool = False,
    touched: bool = False,
    timeout: bool = False,
    ticks_elapsed: int = 1,
    prev_snap=None,
    new_snap=None,
    ep_ticks: int = 0,
    max_decisions_ep: int = None,
    **kwargs,
):
    elapsed_ticks = float(max(1, int(ticks_elapsed)))
    if new_snap is not None:
        on_ground = bool(new_snap.get("on_ground", on_ground))
        if init_dist is None:
            init_dist = new_snap.get("init_dist")
        if min_dist is None:
            min_dist = new_snap.get("min_dist")

    # 1. Pure Euclidean straight-line distance potential (exact path conservation upon touch!)
    if touched:
        eff_new_dist = min(float(init_dist), cfg.touch_surface_dist) if init_dist is not None else cfg.touch_surface_dist
    else:
        eff_new_dist = float(new_dist)
    eff_prev_dist = float(prev_dist)
    r_dist = cfg.distance_delta_scale * (eff_prev_dist - eff_new_dist)

    # 2. Dense time penalty
    r_time = -cfg.time_penalty_per_tick * elapsed_ticks

    # 3. Very weak boost & jump/double-jump penalties, boost pad pickup bonus, and ground powerslide bonus
    r_pickup = cfg.boost_pickup_bonus * float(max(0, int(boost_pickups)))
    r_boost = -cfg.boost_penalty_per_unit * max(0.0, float(boost_used)) + r_pickup

    r_jump_hold = -cfg.jump_hold_penalty_per_tick * float(max(0, int(jump_held_ticks)))
    r_djump = -cfg.djump_trigger_penalty if (djump_triggered or flip_triggered) else 0.0
    r_jump = r_jump_hold + r_djump

    r_slide = cfg.powerslide_bonus_per_tick * float(max(0, int(handbrake_held_ticks)))

    # 4. Terminal sparse rewards: early-touch bonus + timeout penalty + historical min_dist ratio bonus
    eff_max_dec = int(max_decisions_ep) if (max_decisions_ep is not None and max_decisions_ep > 0) else int(cfg.max_decisions)
    if touched:
        max_ticks = float(max(1, eff_max_dec * TICKS_PER_DECISION))
        rem_ratio = max(0.0, min(1.0, (max_ticks - float(ep_ticks)) / max_ticks)) if ep_ticks > 0 else 0.5
        r_touch = cfg.touch_bonus + cfg.touch_time_bonus_max * rem_ratio
    else:
        r_touch = 0.0

    r_timeout = -cfg.timeout_penalty if timeout else 0.0

    is_terminal = bool(touched or timeout)
    if is_terminal and init_dist is not None and min_dist is not None:
        r_min_dist = compute_min_dist_ratio_reward(cfg, init_dist, min_dist, touched=touched)
    elif is_terminal and touched:
        r_min_dist = float(cfg.min_dist_ratio_bonus)
    else:
        r_min_dist = 0.0

    r_total = r_dist + r_time + r_boost + r_jump + r_slide + r_touch + r_timeout + r_min_dist

    detail = {
        "r_dist":     r_dist,
        "r_slide":    r_slide,
        "r_align":    r_slide,      # Alias for plot/HUD compatibility
        "r_time":     r_time,
        "r_boost":    r_boost,
        "r_pickup":   r_pickup,
        "r_jump":     r_jump,
        "r_touch":    r_touch,
        "r_timeout":  r_timeout,
        "r_min_dist": r_min_dist,
        "r_total":    r_total,
    }
    return r_total, detail

