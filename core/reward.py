"""
奖励函数单一真相来源。train_ppo (30Hz 决策) 和 reward_tuner (120Hz 录制/回放) 共用。

★ 120Hz 与 30Hz 完全对齐设计 (Tick-Accurate Scaling)：
  在 Tuner 中，每 1 个物理帧 (ticks_elapsed=1) 调一次 compute_step_reward；
  在 PPO Trainer 中，每 1 个决策步 (ticks_elapsed=ticks_run) 调一次 compute_step_reward，且 step 以精确帧折算步数 (ep_ticks / 4.0) 传入。
  使得无论在 30Hz 训练还是 120Hz 录制，同一条轨迹的：
    - `r_dist` 总和
    - `r_progress` 总和
    - `r_closing_bonus` (closing_accum) 总和
    - `r_time` 总和
    - `r_touch` 与 `r_cum` 终局总分
  都 100% 严格相等，且快 1 个物理帧的轨迹总分严格高于慢轨迹！
"""
from dataclasses import dataclass
from .constants import MAX_DECISIONS


@dataclass
class RewardConfig:
    close_bonus:      float = 0.01
    away_penalty:     float = 0.01
    use_horizontal:   bool  = False
    # 默认关闭 min_dist 单向水位线 (设为 False)，使 r_dist = k_d * (prev_dist - new_dist) 成为严格守恒的距离势函数：
    # 1. 任意起点到触球终点的总距离分恒为 0.01 * (d_0 - 135)，路径积分与轨迹形状无关，杜绝绕圈刷分；
    # 2. 当背对球出生调头或球向远处滚动 (dist > d_0) 时依然提供 100% 稠密的方向梯度，且漏球飞远时立刻按距离扣分。
    use_min_dist_watermark: bool = False
    air_scale:        float = 1.0
    air_time_thresh:  float = 0.01

    # 每物理 tick 的时间惩罚（在 120Hz 下每帧 -0.04，在 30Hz 下每决策步 4 帧合为 -0.16）
    time_penalty:     float = 0.04
    warmup_steps:     int   = 5
    time_penalty_growth: float = 0.0

    align_weight:     float = 0.0

    touch_bonus:      float = 80.0
    # 每剩余 1 个决策步 (4 ticks) 奖励 2.4 分（连同 time_penalty 共 2.56 分/步 = 0.64 分/物理帧），
    # 保证快 1 个物理帧（包括半翻转调头、低氮气前翻冲刺）的轨迹总分 100% 严格高于绕大圈的慢轨迹
    time_bonus_per_step: float = 2.4
    timeout_penalty:  float = 10.0

    # 触球瞬间球朝向球门速度的附加分（在纯 Reach 竞速阶段默认设为 0.0）
    ball_vel_to_goal_scale: float = 0.0

    # ---- closing speed 激励 ----
    # 设为 2300.0 uu/s 超音速上限，使 1600->2250 uu/s 的翻滚 (Flip) 爆发能直接获得全额 peak_closing 进度分；
    # 同时将 closing_accum_scale 设为 0.0，彻底杜绝背对球时不半翻转而故意画大圈延长超音速时间刷分的漏洞！
    closing_threshold:      float = 2300.0
    closing_progress_scale: float = 0.015
    closing_potential_mode: bool  = False
    closing_accum_scale:    float = 0.0

    immediate_closing_bonus: bool = True
    timeout_deduct_accum:    bool = True
    timeout_deduct_mult:     float = 1.0

    # 每物理 tick 满额无效喷气 (在 >=2290 uu/s 超音速上限处继续喷气浪费单帧加速度) 的惩罚权重
    boost_waste_penalty:     float = 0.01

    # 实际消耗氮气量惩罚接口 (0..100 标度，仅统计 prev_boost - cur_boost > 0 的净消耗，不受捡 Boost Pad 影响)
    boost_used_penalty:      float = 0.0

    # ---- Stage 1 (定点定时击球) 触球后快速模拟时空最近点 (CPA) 奖励 ----
    target_cpa_bonus:        float = 0.0
    target_cpa_max_dist:     float = 3000.0
    target_time_penalty:     float = 30.0

    max_decisions:           int   = MAX_DECISIONS


def compute_step_reward(
    cfg, prev_dist, new_dist, step,
    touched=False, timeout=False,
    air_time=0.0, align=0.0,
    prev_dist_h=None, new_dist_h=None,
    closing=0.0,
    peak_closing=0.0,
    closing_accum=0.0,
    ball_vel_to_goal=0.0,
    ticks_elapsed=1,
    boost_wasted_ticks=0.0,
    boost_used=0.0,
    min_dist=None,
    cpa_dist=None,
    cpa_time=0.0,
    cpa_time_err=0.0,
):
    """返回 (r_total, detail)。
    ticks_elapsed: 本次调用跨越的物理帧数（120Hz Tuner 传 1，30Hz Trainer 传实际执行的 ticks_run 1..4）。
    step: 支持浮点精确步数 (例如 active_ticks / 4.0)，使单步内提前 1~3 个物理帧触球也能精确获得帧级时间奖励。
    """
    d = {}
    max_dec = getattr(cfg, "max_decisions", MAX_DECISIONS)
    w_ticks = float(max(1, int(ticks_elapsed)))

    # ---- 距离势函数 ----
    eff_new_dist = min(float(new_dist), 135.0) if touched else float(new_dist)
    cur_min_dist = float(prev_dist) if min_dist is None else float(min_dist)

    if getattr(cfg, "use_min_dist_watermark", False) and min_dist is not None:
        if eff_new_dist < cur_min_dist:
            delta = cur_min_dist - eff_new_dist
            new_min_dist = eff_new_dist
        else:
            delta = 0.0
            new_min_dist = cur_min_dist
    else:
        if cfg.use_horizontal and prev_dist_h is not None and new_dist_h is not None:
            eff_new_dist_h = min(float(new_dist_h), 135.0) if touched else float(new_dist_h)
            delta = prev_dist_h - eff_new_dist_h
        else:
            delta = prev_dist - eff_new_dist
        new_min_dist = min(cur_min_dist, eff_new_dist)

    clip_lim = 300.0 * w_ticks
    delta = max(-clip_lim, min(clip_lim, delta))
    scale = cfg.air_scale if air_time > cfg.air_time_thresh else 1.0
    r_dist = (cfg.close_bonus if delta >= 0 else cfg.away_penalty) * delta * scale
    d["r_dist"]   = r_dist
    d["min_dist"] = new_min_dist

    # ---- 时间 (按实际流逝的物理 tick 数线性计费) ----
    if float(step) <= cfg.warmup_steps:
        d["r_time"] = 0.0
    else:
        base = -cfg.time_penalty
        if cfg.time_penalty_growth > 0:
            base -= cfg.time_penalty_growth * (float(step) / max_dec)
        d["r_time"] = base * w_ticks

    # ---- align ----
    r_align = (cfg.align_weight * align * w_ticks) if cfg.align_weight > 0 else 0.0
    d["r_align"] = r_align

    # ---- closing ----
    r_progress = 0.0
    cur_v_clamped = max(0.0, min(float(closing), cfg.closing_threshold))
    prev_v_clamped = max(0.0, min(float(peak_closing), cfg.closing_threshold))

    if getattr(cfg, "closing_potential_mode", False):
        # 触球帧锁定触球前速度，避免碰撞瞬间球反弹导致 closing 变负被误扣分
        eff_v_clamped = max(cur_v_clamped, prev_v_clamped) if touched else cur_v_clamped
        r_progress = (eff_v_clamped - prev_v_clamped) * cfg.closing_progress_scale
        new_peak = eff_v_clamped
    else:
        new_peak = peak_closing
        if closing > peak_closing:
            up_to = min(closing, cfg.closing_threshold)
            if up_to > peak_closing:
                r_progress = (up_to - peak_closing) * cfg.closing_progress_scale
            new_peak = closing

    new_accum = closing_accum
    r_closing_bonus = 0.0

    if closing > cfg.closing_threshold and cfg.closing_accum_scale > 0.0:
        speed_excess = (closing - cfg.closing_threshold) * w_ticks
        if cfg.immediate_closing_bonus:
            r_step_bonus = speed_excess * cfg.closing_accum_scale
            r_closing_bonus = r_step_bonus
            new_accum = closing_accum + r_step_bonus
        else:
            new_accum = closing_accum + speed_excess

    if not cfg.immediate_closing_bonus and cfg.closing_accum_scale > 0.0:
        if touched and new_accum > 0:
            r_closing_bonus = new_accum * cfg.closing_accum_scale

    d["r_progress"]      = r_progress
    d["r_closing_bonus"] = r_closing_bonus
    d["peak_closing"]    = new_peak
    d["closing_accum"]   = new_accum

    # ---- Boost Wasted 惩罚 (按损失的单帧加速度比例 waste_ratio 连续计费) ----
    waste_pen = getattr(cfg, "boost_waste_penalty", 0.0)
    wasted_val = float(boost_wasted_ticks)
    r_boost_waste = -wasted_val * waste_pen if (waste_pen > 0.0 and wasted_val > 0.0) else 0.0
    d["r_boost_waste"] = r_boost_waste
    d["boost_wasted"]  = wasted_val

    # ---- Boost 净消耗量惩罚接口 (默认权重 0.0，留好接口与统计) ----
    used_pen = getattr(cfg, "boost_used_penalty", 0.0)
    used_val = float(boost_used)
    r_boost_used = -used_val * used_pen if (used_pen > 0.0 and used_val > 0.0) else 0.0
    d["r_boost_used"] = r_boost_used
    d["boost_used"]   = used_val

    # ---- 触球 & Stage 1 时空最近点 (CPA) 结算 ----
    r_touch = 0.0
    r_goal_vel = 0.0
    r_target_cpa = 0.0
    if touched:
        rem_steps = max(0.0, float(max_dec) - float(step))
        if getattr(cfg, "ball_vel_to_goal_scale", 0.0) > 0.0:
            r_goal_vel = max(-10.0, float(ball_vel_to_goal) * cfg.ball_vel_to_goal_scale)
        cpa_bonus_max = getattr(cfg, "target_cpa_bonus", 0.0)
        if cpa_bonus_max > 0.0 and cpa_dist is not None:
            max_d = max(100.0, float(getattr(cfg, "target_cpa_max_dist", 3000.0)))
            spatial_ratio = max(0.0, 1.0 - float(cpa_dist) / max_d)
            spatial_score = spatial_ratio * cpa_bonus_max
            # 按到达最近点的时间误差扣分（仅在空间命中有效范围内扣减，最低扣至 0）
            t_pen = float(cpa_time_err or 0.0) * float(getattr(cfg, "target_time_penalty", 30.0))
            r_target_cpa = max(0.0, spatial_score - t_pen * spatial_ratio)
        r_touch = (cfg.touch_bonus
                   + rem_steps * cfg.time_bonus_per_step
                   + r_goal_vel
                   + r_target_cpa)
    d["r_touch"]      = r_touch
    d["r_goal_vel"]   = r_goal_vel
    d["r_target_cpa"] = r_target_cpa
    d["cpa_dist"]     = float(cpa_dist) if cpa_dist is not None else 0.0
    d["cpa_time"]     = float(cpa_time) if cpa_time is not None else 0.0
    d["cpa_time_err"] = float(cpa_time_err) if cpa_time_err is not None else 0.0

    # ---- 超时 ----
    r_timeout = 0.0
    if timeout:
        base_timeout = -cfg.timeout_penalty
        if cfg.timeout_deduct_accum and cfg.immediate_closing_bonus and new_accum > 0:
            deduct = -new_accum * cfg.timeout_deduct_mult
            r_timeout = base_timeout + deduct
        else:
            r_timeout = base_timeout
    d["r_timeout"] = r_timeout

    r_total = (r_dist + d["r_time"] + r_align
               + r_progress + r_closing_bonus + r_boost_waste + r_boost_used + r_touch + r_timeout)
    d["r_total"] = r_total
    return r_total, d
