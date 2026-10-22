from core.constants import TICKS_PER_DECISION, MAX_DECISIONS, KICKOFF_SLOT_NAMES
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


def _fmt_duel_cars(session, fr):
    if not isinstance(fr, dict):
        return ""
    all_cars = fr.get("all_cars")
    if not all_cars:
        return ""
    f_idx = fr.get("car_idx", getattr(session.env, "focused_car_idx", 0))
    lines = ["---- 1v1 Kickoff Duel Cars ----"]
    for c_entry in all_cars:
        idx = c_entry.get("car_idx", 0)
        mark = "▶" if idx == f_idx else " "
        lbl = c_entry.get("label", f"#{idx+1}")
        dist = c_entry.get("dist", 0.0)
        spd = c_entry.get("car_speed", 0.0)
        bst = c_entry.get("boost", 0.0)
        lines.append(
            f"{mark}{lbl:<10} dist={dist:6.1f}  spd={spd:6.1f}  boost={bst:4.1f}"
        )
    return "\n".join(lines) + "\n"


def _fmt_focus(session):
    env = session.env
    slot = int(getattr(env, "current_slot_idx", 0))
    slot_name = KICKOFF_SLOT_NAMES[max(0, min(len(KICKOFF_SLOT_NAMES) - 1, slot))]
    lbl = env.car_label() if hasattr(env, "car_label") else "#1 BLUE"
    settle = int(getattr(env, "settle_ticks", 60))
    lock_slot = getattr(env, "fixed_slot_idx", None)
    lock_tag = f" | 锁定KO{lock_slot}" if lock_slot is not None else ""
    return f"[KO{slot}:{slot_name} | 悬挂稳定={settle}f | 视角车 {lbl}{lock_tag}]\n"


def format_hud(session, current_boost, obs_group=None):
    lbl = session.env.car_label() if hasattr(session.env, "car_label") else "#1 BLUE"
    boost_line = f"{lbl}  BOOST: {current_boost:5.1f}"
    frames = session.frames
    is_auto = _is_auto(session)
    focus_str = _fmt_focus(session)

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
        sec_i = fr["i"] / 120.0
        sec_tot = len(frames) / 120.0
        body = (
            f"{focus_str}"
            f"{'MODEL 1v1' if is_auto else 'HUMAN PLAYBACK'}  "
            f"i={fr['i']}/{len(frames)-1} ({sec_i:.2f}s/{sec_tot:.2f}s, {len(frames)}f)  "
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
            f"{fr['ball_pos'][2]:.0f}) vel=({fr.get('ball_vel',(0,0,0))[0]:+.0f},{fr.get('ball_vel',(0,0,0))[1]:+.0f},{fr.get('ball_vel',(0,0,0))[2]:+.0f})\n"
            f"dist:  {fr['dist']:.1f}  (h={fr['dist_h']:.1f})  "
            f"closing={closing:+.1f}\n"
        )
        body += _fmt_duel_cars(session, fr)
        body += (
            f"---- reward (Kickoff Potential & Touch) ----\n"
            f"r_dist     = {d.get('r_dist', 0):+8.3f}\n"
            f"r_time     = {d.get('r_time', 0):+8.4f}\n"
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
        sec_i = (fr["i"] + 1) / 120.0
        body = (
            f"{focus_str}"
            f"{'MODEL 1v1' if is_auto else 'HUMAN RECORDING'}  "
            f"step={fr['i']} ({sec_i:.2f}s, {len(frames)}f)  r_cum={fr['r_cum']:+.3f}\n"
            f"keys: [{keys_str}]\n"
            f"dist={fr['dist']:.1f}  spd={car_spd:.1f}  closing={closing:+.1f}\n"
            f"boost={fr['boost']:.1f}  air={fr['air_time']:.2f}  flip={int(fr['is_flipping'])}\n"
        )
        body += _fmt_duel_cars(session, fr)
        body += _fmt_flags(fr)
        if obs_group:
            body += _fmt_obs_group(fr, obs_group)
        if is_auto:
            body += _model_block(session, fr)
        return boost_line, body

    return (
        boost_line,
        f"{focus_str}按任一操作键开始人工录制  (R: 重置当前开球位 | N: 新开球位 | M: 切 MODEL/HUMAN | Enter: 导出CSV)",
    )


def _model_block(session, fr=None):
    probs = fr.get("ppo_probs") if isinstance(fr, dict) and fr.get("ppo_probs") is not None else getattr(session, "last_ppo_probs", None)
    val = fr.get("ppo_value") if isinstance(fr, dict) and fr.get("ppo_value") is not None else getattr(session, "last_ppo_value", 0.0)
    act = fr["input"].get("ppo_channels") if isinstance(fr, dict) and "ppo_channels" in fr.get("input", {}) else getattr(session, "current_ppo_action", [1, 1, 1, 0, 0, 0])
    if probs is None:
        return "\n---- PPO Pure-MLP (waiting) ----\n"
    p_ws = probs[0]
    p_ad = probs[1]
    p_qe = probs[2]
    p_jp = probs[3][1]
    p_bs = probs[4][1]
    p_hb = probs[5][1]
    return (
        f"\n---- PPO Pure-MLP 36K (V={val:+.2f}) ----\n"
        f"act={list(act)}\n"
        f"W/0/S: W={p_ws[2]:.0%} 0={p_ws[1]:.0%} S={p_ws[0]:.0%}\n"
        f"A/0/D: A={p_ad[0]:.0%} 0={p_ad[1]:.0%} D={p_ad[2]:.0%}\n"
        f"Q/0/E: Q={p_qe[0]:.0%} 0={p_qe[1]:.0%} E={p_qe[2]:.0%}\n"
        f"Jump={p_jp:.0%}  Boost={p_bs:.0%}  Slide={p_hb:.0%}\n"
    )
