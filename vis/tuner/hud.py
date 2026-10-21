from core.constants import TICKS_PER_DECISION, MAX_DECISIONS, NUM_CURRICULUM_STAGES, DIR5_LABELS
from core.curriculum import get_step_spec
from core.obs import OBS_GROUPS
from .ui_constants import STATE_PLAYBACK, STATE_DONE


def _is_auto(session):
    return hasattr(session, "last_ppo_probs")


def _fmt_obs_group(fr, group_name):
    parts = fr.get("obs_parts") if isinstance(fr, dict) else None
    if not parts or group_name not in parts:
        return ""
    _, _, labels = OBS_GROUPS[group_name]
    vals = parts[group_name]
    lines = [f"==== {group_name} (raw) ===="]
    for l, v in zip(labels, vals):
        lines.append(f"  {l:>10} = {float(v):+10.3f}")
    return "\n".join(lines) + "\n"


def _fmt_flags(fr):
    if not isinstance(fr, dict):
        return ""
    fl = fr.get("flags")
    if not fl:
        return ""
    line1 = (f"on_ground={int(fl['on_ground'])}  "
             f"jumping={int(fl['is_jumping'])}  "
             f"flipping={int(fl['is_flipping'])}  "
             f"auto_flip={int(fl['auto_flip'])}  "
             f"has_flipped={int(fl['has_flipped'])}")
    line2 = (f"has_djump={int(fl['has_djump'])}  "
             f"flip_or_jump={int(fl['flip_or_jump'])}  "
             f"flip_reset={int(fl['flip_reset'])}  "
             f"world_cont={int(fl['world_cont'])}")
    line3 = (f"air_t={fl['air_time']:.3f}  "
             f"flip_t={fl['flip_time']:.3f}  "
             f"jump_t={fl['jump_time']:.3f}  "
             f"boost_t={fl['boost_time']:.2f}")
    return f"---- flags ----\n{line1}\n{line2}\n{line3}\n"


def _fmt_focus(session):
    env = session.env
    if getattr(session, "is_nexto_session", False) or getattr(env, "is_nexto_env", False):
        lbl = env.car_label() if hasattr(env, "car_label") else "#1"
        f_idx = getattr(env, "focused_car_idx", 0) + 1
        n_cars = getattr(env, "n_cars", 1)
        nb = getattr(env, "n_blue", 0)
        no = getattr(env, "n_orange", 0)
        sb = getattr(env, "score_blue", 0)
        so = getattr(env, "score_orange", 0)
        return f"[NEXTO {nb}v{no} | SCORE {sb}:{so} | 视角车 {lbl} ({f_idx}/{n_cars})]\n"
    from core.env import STAGE_SHORT_LABELS
    s_idx = getattr(env, "current_stage_idx", 0)
    s_lbl = STAGE_SHORT_LABELS[s_idx] if 0 <= s_idx < len(STAGE_SHORT_LABELS) else f"P{s_idx+1}"
    dir_idx = getattr(env, "current_dir_idx", 0)
    dir_lbl = DIR5_LABELS[max(0, min(len(DIR5_LABELS) - 1, dir_idx))]
    d_tier = getattr(env, "current_dist_tier", 1)
    tier_name = ("Near", "Mid", "Far")[max(0, min(2, d_tier))]
    c_step = getattr(env, "curriculum_step_idx", 0)
    spec = get_step_spec(c_step)
    dist_cfg = getattr(env, "spawn_dist_cfg", None)
    dist_info = f" | dist={dist_cfg.raw}" if (dist_cfg and dist_cfg.is_active) else f" | step={c_step}({spec.name})"
    return f"[{s_lbl} | Dir5={dir_lbl} | Tier={tier_name}{dist_info}]\n"


def format_hud(session, current_boost, obs_group=None):
    is_nexto = getattr(session, "is_nexto_session", False) or getattr(session.env, "is_nexto_env", False)
    if is_nexto:
        lbl = session.env.car_label() if hasattr(session.env, "car_label") else "#1"
        boost_line = f"{lbl}  BOOST: {current_boost:5.1f}   SCORE: {session.env.score_blue} - {session.env.score_orange}"
    else:
        boost_line = f"BOOST: {current_boost:5.1f}"
    frames = session.frames
    is_auto = _is_auto(session)
    focus_str = _fmt_focus(session)
    rec_limit = getattr(session.env, "current_max_decisions", MAX_DECISIONS) * TICKS_PER_DECISION

    if session.state in (STATE_PLAYBACK, STATE_DONE) and frames:
        fr = frames[session.playback_idx]
        d = fr["detail"]
        fwd = fr.get("fwd",   (0, 0, 0))
        up  = fr.get("up",    (0, 0, 0))
        rgt = fr.get("right", (0, 0, 0))
        closing = fr.get("closing", 0.0)
        car_spd = fr.get("car_speed", 0.0)
        keys_str = fr["input"].get("keys_str", "")
        ch_list  = fr["input"].get("ppo_channels", [])
        ball_tag = "[DYNAMIC MATCH]" if is_nexto else "[STATIC TARGET]"
        body = (
            f"{focus_str}"
            f"{'NEXTO' if is_nexto else ('AUTO' if is_auto else 'PLAYBACK')}  "
            f"i={fr['i']}/{len(frames)-1} (buf={len(frames)}f)  "
            f"速度 {int(session.speed*100)}%\n"
            f"keys:  [{keys_str}]  ch={ch_list}\n"
            f"input: thr={fr['input']['thr']:+.0f} str={fr['input']['str']:+.0f} "
            f"ptc={fr['input']['ptc']:+.0f} yaw={fr['input']['yaw']:+.0f}\n"
            f"       roll={fr['input']['roll']:+.0f} "
            f"jump={int(fr['input']['jump'])} boost={int(fr['input']['boost'])} "
            f"hb={int(fr['input'].get('handbrake', False))}\n"
            f"car:   pos=({fr['car_pos'][0]:.0f},{fr['car_pos'][1]:.0f},"
            f"{fr['car_pos'][2]:.0f}) spd={car_spd:.1f}\n"
            f"       up_z={fr['up_z']:.2f} air={fr['air_time']:.2f} "
            f"flip={int(fr['is_flipping'])} boost={fr['boost']:.1f}\n"
            f"axes:  fwd=({fwd[0]:+.2f},{fwd[1]:+.2f},{fwd[2]:+.2f})\n"
            f"       up =({up[0]:+.2f},{up[1]:+.2f},{up[2]:+.2f})\n"
            f"       rgt=({rgt[0]:+.2f},{rgt[1]:+.2f},{rgt[2]:+.2f})\n"
            f"ball:  pos=({fr['ball_pos'][0]:.0f},{fr['ball_pos'][1]:.0f},"
            f"{fr['ball_pos'][2]:.0f}) {ball_tag}\n"
            f"dist:  {fr['dist']:.1f}  (h={fr['dist_h']:.1f})  "
            f"closing={closing:+.1f}\n"
            f"---- reward (Tuner Diagnostic) ----\n"
            f"r_dist     = {d.get('r_dist', 0):+8.3f}\n"
            f"r_slide    = {d.get('r_slide', 0):+8.4f}\n"
            f"r_time     = {d.get('r_time', 0):+8.3f}\n"
            f"r_boost    = {d.get('r_boost', 0):+8.4f} (pickup={d.get('r_pickup', 0):+6.3f})\n"
            f"r_jump     = {d.get('r_jump', 0):+8.4f}\n"
            f"r_min_dist = {d.get('r_min_dist', 0):+8.3f}\n"
            f"r_touch    = {d.get('r_touch', 0):+8.3f}\n"
            f"r_timeout  = {d.get('r_timeout', 0):+8.3f}\n"
            f"r_total    = {fr['r_total']:+8.3f}\n"
            f"r_cum      = {fr['r_cum']:+8.3f}\n"
        )
        body += _fmt_flags(fr)
        if obs_group:
            body += _fmt_obs_group(fr, obs_group)
        if is_auto:
            body += _model_block(session, fr)
        return boost_line, body

    if frames:
        fr = frames[-1]
        closing = fr.get("closing", 0.0)
        car_spd = fr.get("car_speed", 0.0)
        keys_str = fr["input"].get("keys_str", "")
        body = (
            f"{focus_str}"
            f"{'NEXTO LIVE' if is_nexto else ('AUTO' if is_auto else 'RECORDING')}  "
            f"step={fr['i']} (buf={len(frames)})  r_cum={fr['r_cum']:+.3f}\n"
            f"keys: [{keys_str}]\n"
            f"dist={fr['dist']:.1f}  spd={car_spd:.1f}  closing={closing:+.1f}\n"
            f"boost={fr['boost']:.1f}  air={fr['air_time']:.2f}  flip={int(fr['is_flipping'])}\n"
        )
        body += _fmt_flags(fr)
        if obs_group:
            body += _fmt_obs_group(fr, obs_group)
        if is_auto:
            body += _model_block(session, fr)
        return boost_line, body

    return boost_line, f"{focus_str}按任一操作键开始录制  (M: 切换 MODEL/HUMAN)  (V: 切视角)  (F: FREERUN)"


def _model_block(session, fr=None):
    probs = fr.get("ppo_probs") if isinstance(fr, dict) and fr.get("ppo_probs") is not None else getattr(session, "last_ppo_probs", None)
    val = fr.get("ppo_value") if isinstance(fr, dict) and fr.get("ppo_value") is not None else getattr(session, "last_ppo_value", 0.0)
    pad_ent = fr.get("pad_ent") if isinstance(fr, dict) and fr.get("pad_ent") is not None else getattr(session, "last_pad_ent", 0.0)
    pad_max = fr.get("pad_max") if isinstance(fr, dict) and fr.get("pad_max") is not None else getattr(session, "last_pad_max", 0.0)
    act = fr["input"].get("ppo_channels") if isinstance(fr, dict) and "ppo_channels" in fr.get("input", {}) else getattr(session, "current_ppo_action", [1, 1, 1, 0, 0, 0])
    if probs is None:
        return "\n---- PPO model (waiting) ----\n"
    p_ws = probs[0]
    p_ad = probs[1]
    p_qe = probs[2]
    p_jp = probs[3][1]
    p_bs = probs[4][1]
    p_hb = probs[5][1]
    is_nexto = getattr(session, "is_nexto_session", False) or getattr(session.env, "is_nexto_env", False)
    if is_nexto:
        enum_idx = fr["input"].get("nexto_enum_idx", 0) if isinstance(fr, dict) else 0
        header = f"\n---- Nexto EARLPerceiver (V={val:+.2f} | Enum={enum_idx}/90) ----\n"
    else:
        header = f"\n---- PPO + PadAttn (V={val:+.2f}) ----\n"
    return (
        f"{header}"
        f"act={list(act)}  attn_ent={pad_ent:.2f}  attn_max={pad_max:.0%}\n"
        f"W/0/S: W={p_ws[2]:.0%} 0={p_ws[1]:.0%} S={p_ws[0]:.0%}\n"
        f"A/0/D: A={p_ad[0]:.0%} 0={p_ad[1]:.0%} D={p_ad[2]:.0%}\n"
        f"Q/0/E: Q={p_qe[0]:.0%} 0={p_qe[1]:.0%} E={p_qe[2]:.0%}\n"
        f"Jump={p_jp:.0%}  Boost={p_bs:.0%}  Slide={p_hb:.0%}\n"
    )

