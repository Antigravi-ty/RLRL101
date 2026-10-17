from dataclasses import dataclass
from .constants import MAX_DECISIONS


@dataclass
class RewardConfig:
    distance_delta_scale:   float = 0.01
    time_penalty_per_tick:  float = 0.04
    boost_penalty_per_unit: float = 0.01
    flip_base_penalty:      float = 0.001
    flip_zero_speed_extra:  float = 0.0018
    flip_ref_speed:         float = 1800.0
    touch_bonus:            float = 80.0
    timeout_penalty:        float = 10.0
    max_decisions:          int   = MAX_DECISIONS


def compute_flip_penalty(cfg: RewardConfig, flip_trigger_speed: float) -> float:
    speed_clamped = max(0.0, min(float(flip_trigger_speed), cfg.flip_ref_speed))
    speed_deficit_hundreds = (cfg.flip_ref_speed - speed_clamped) / 100.0
    return cfg.flip_base_penalty + 0.0001 * speed_deficit_hundreds


def compute_step_reward(
    cfg: RewardConfig,
    prev_dist: float,
    new_dist: float,
    boost_used: float = 0.0,
    flip_triggered: bool = False,
    flip_trigger_speed: float = 0.0,
    touched: bool = False,
    timeout: bool = False,
    ticks_elapsed: int = 1,
):
    elapsed_ticks = float(max(1, int(ticks_elapsed)))

    distance_delta = float(prev_dist) - float(new_dist)
    r_dist = cfg.distance_delta_scale * distance_delta

    r_time = -cfg.time_penalty_per_tick * elapsed_ticks

    r_boost = -cfg.boost_penalty_per_unit * max(0.0, float(boost_used))

    r_flip = -compute_flip_penalty(cfg, flip_trigger_speed) if flip_triggered else 0.0

    r_touch = cfg.touch_bonus if touched else 0.0

    r_timeout = -cfg.timeout_penalty if timeout else 0.0

    r_total = r_dist + r_time + r_boost + r_flip + r_touch + r_timeout

    detail = {
        "r_dist":    r_dist,
        "r_time":    r_time,
        "r_boost":   r_boost,
        "r_flip":    r_flip,
        "r_touch":   r_touch,
        "r_timeout": r_timeout,
        "r_total":   r_total,
    }
    return r_total, detail
