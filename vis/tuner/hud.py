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
    from core.env import STAGE_SHORT_LABELS
    env = session.env
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


def _takeover_block(session, fr):
    ti = getattr(session, "takeover_info", None)
    if not ti or not isinstance(fr, dict):
        return ""
    t_idx = int(ti.get("takeover_idx", 0))
    b_len = int(ti.get("baseline_len", len(session.frames)))
    b_src = str(ti.get("baseline_source", "MODEL"))
    b_reason = str(ti.get("baseline_reason", "touch"))
    b_rcum = float(ti.get("baseline_r_cum", 0.0))
    b_frames = ti.get("baseline_frames") or []
    cur_i = int(fr.get("i", 0))
    cur_len = len(session.frames)
    seg_tag = f"{b_src}前段 (0..{t_idx})" if cur_i <= t_idx else f"HUMAN接管段 ({t_idx+1}..)"

    lines = [
        f"---- 中途接管对比 (Takeover @f={t_idx} vs {b_src}) ----",
        f"当前帧段: [{seg_tag}]  基准={b_len}f ({b_len/120.0:.2f}s, {b_reason})",
    ]
    if 0 <= cur_i < len(b_frames):
        bf = b_frames[cur_i]
        b_dist = float(bf.get("dist", 0.0))
        b_spd = float(bf.get("car_speed", 0.0))
        d_diff = float(fr.get("dist", 0.0)) - b_dist
        s_diff = float(fr.get("car_speed", 0.0)) - b_spd
        lead_tag = "领先(更近)" if d_diff < -0.5 else ("落后(更远)" if d_diff > 0.5 else "持平")
        lines.append(
            f"同帧对比(f={cur_i}): Δdist={d_diff:+.1f}uu ({lead_tag}) | Δspd={s_diff:+.1f}"
        )
    if session.state in (STATE_PLAYBACK, STATE_DONE):
        short = session.format_takeover_short() if hasattr(session, "format_takeover_short") else ""
        seg_h = max(0, cur_len - 1 - t_idx)
        seg_b = max(0, b_len - 1 - t_idx)
        lines.append(
            f"结算成绩: 总 {cur_len}f vs {b_len}f (接管后 {seg_h}f vs {seg_b}f) | r_cum={session.r_cum:+.2f} vs {b_rcum:+.2f}"
        )
        if short:
            lines.append(f"结论: {short}")
    return "\n".join(lines) + "\n"


def format_hud(session, current_boost, obs_group=None):
    boost_line = f"BOOST: {current_boost:5.1f}"
    frames = session.frames
    is_auto = _is_auto(session)
    focus_str = _fmt_focus(session)
    rec_limit = getattr(session.env, "current_max_decisions", MAX_DECISIONS) * TICKS_PER_DECISION
    ti = getattr(session, "takeover_info", None)

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
        src_tag = f" [{fr.get('source', 'human').upper()}]" if ti else ""
        body = (
            f"{focus_str}"
            f"{'AUTO' if is_auto else 'PLAYBACK'}{src_tag}  "
            f"i={fr['i']}/{len(frames)-1} (max={rec_limit}f)  "
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
            f"{fr['ball_pos'][2]:.0f}) [STATIC TARGET]\n"
            f"dist:  {fr['dist']:.1f}  (h={fr['dist_h']:.1f})  "
            f"closing={closing:+.1f}\n"
            f"---- reward (Static Ball Master) ----\n"
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
        if ti:
            body += _takeover_block(session, fr)
        body += _fmt_flags(fr)
        if obs_group:
            body += _fmt_obs_group(fr, obs_group)
        if is_auto or (isinstance(fr, dict) and fr.get("ppo_probs") is not None):
            body += _model_block(session, fr)
        return boost_line, body

    if frames:
        fr = frames[-1]
        closing = fr.get("closing", 0.0)
        car_spd = fr.get("car_speed", 0.0)
        keys_str = fr["input"].get("keys_str", "")
        state_hdr = "AUTO" if is_auto else ("TAKEOVER" if ti else "RECORDING")
        body = (
            f"{focus_str}"
            f"{state_hdr}  "
            f"i={fr['i']}/{rec_limit}  r_cum={fr['r_cum']:+.3f}\n"
            f"keys: [{keys_str}]\n"
            f"dist={fr['dist']:.1f}  spd={car_spd:.1f}  closing={closing:+.1f}\n"
            f"boost={fr['boost']:.1f}  air={fr['air_time']:.2f}  flip={int(fr['is_flipping'])}\n"
        )
        if ti:
            body += _takeover_block(session, fr)
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
    return (
        f"\n---- PPO + PadAttn (V={val:+.2f}) ----\n"
        f"act={list(act)}  pad_ent={pad_ent:.2f}  pad_max={pad_max:.0%}\n"
        f"W/0/S: W={p_ws[2]:.0%} 0={p_ws[1]:.0%} S={p_ws[0]:.0%}\n"
        f"A/0/D: A={p_ad[0]:.0%} 0={p_ad[1]:.0%} D={p_ad[2]:.0%}\n"
        f"Q/0/E: Q={p_qe[0]:.0%} 0={p_qe[1]:.0%} E={p_qe[2]:.0%}\n"
        f"Jump={p_jp:.0%}  Boost={p_bs:.0%}  Slide={p_hb:.0%}\n"
    )
