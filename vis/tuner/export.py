import csv
import os
import time
from core.nn import controls_to_multi_discrete, multi_discrete_to_key_str


def _gv(parts, grp, k):
    v = parts.get(grp)
    return 0.0 if v is None else float(v[k])


def export_frames(frames, out_dir="data", mode="full"):
    if not frames:
        print("[tuner] no frames")
        return None
    os.makedirs(out_dir, exist_ok=True)
    src_tag = frames[0].get("source", "session")
    sp0 = frames[0].get("spawn_state") or {}
    spawn_id = sp0.get("spawn_id", "0000")
    n_f = len(frames)
    mode_suffix = "_compact" if mode == "compact" else ""
    fn = f"{out_dir}/_{spawn_id}_{n_f}f_{src_tag}{mode_suffix}_{int(time.time())}.csv"

    spawn_cols = [
        "source", "spawn_id", "spawn_mode",
        "init_cx", "init_cy", "init_cz",
        "init_cvx", "init_cvy", "init_cvz",
        "init_cax", "init_cay", "init_caz",
        "init_cfx", "init_cfy", "init_cfz",
        "init_crx", "init_cry", "init_crz",
        "init_cux", "init_cuy", "init_cuz",
        "init_cboost",
        "init_bx", "init_by", "init_bz",
        "init_bvx", "init_bvy", "init_bvz",
        "init_bax", "init_bay", "init_baz",
    ]

    if mode == "compact":
        header = [
            "i",
            "thr", "str", "ptc", "yaw", "roll", "jump", "boost", "handbrake",
            "ch0_ws", "ch1_ad", "ch2_qe", "ch3_jump", "ch4_boost", "ch5_slide", "keys_str",
            "on_ground", "is_flipping", "is_supersonic", "boost_wasted", "boost_used", "air_time", "car_boost",
            "dist", "min_dist", "dist_h", "align", "closing", "peak_closing", "closing_accum",
            "r_total", "r_cum", "r_dist", "r_time",
            "r_progress", "r_closing_bonus", "r_boost_waste", "r_boost_used", "r_touch",
            "b_fwd", "b_right", "b_up", "|d|",
            "bv_fwd", "bv_right", "bv_up", "|bv|",
            "v_fwd", "v_right", "v_up", "|v|",
            "car_pitch", "car_yaw", "car_roll",
            "flip_or_jump", "is_jumping", "closing_spd",
            *spawn_cols,
        ]
    else:
        header = [
            "i",
            "thr", "str", "ptc", "yaw", "roll", "jump", "boost", "handbrake",
            "ch0_ws", "ch1_ad", "ch2_qe", "ch3_jump", "ch4_boost", "ch5_slide", "keys_str",
            "px", "py", "pz", "vx", "vy", "vz",
            "bx", "by", "bz", "bvx", "bvy", "bvz",
            "up_z", "on_ground", "is_flipping", "is_supersonic", "boost_wasted", "boost_used", "air_time", "car_boost",
            "dist", "min_dist", "dist_h", "align", "closing", "peak_closing", "closing_accum",
            "r_total", "r_cum",
            "r_dist", "r_time", "r_align",
            "r_progress", "r_closing_bonus", "r_boost_waste", "r_boost_used", "r_touch", "r_timeout",
            "obs_a_nx", "obs_a_px", "obs_a_ny", "obs_a_py",
            "obs_a_floor", "obs_a_ceil",
            "obs_b_pitch", "obs_b_yaw", "obs_b_roll",
            "obs_c_vfwd", "obs_c_vright", "obs_c_vup", "obs_c_vmag",
            "obs_d_bfwd", "obs_d_bright", "obs_d_bup", "obs_d_bdist",
            "obs_e_bvfwd", "obs_e_bvright", "obs_e_bvup", "obs_e_bvmag",
            "obs_f_wx", "obs_f_wy", "obs_f_wz",
            "obs_g_onground", "obs_g_flip_or_jump", "obs_g_is_flipping",
            "obs_g_is_jumping", "obs_g_air_time", "obs_g_boost",
            "obs_h_closing",
            "fl_on_ground", "fl_jumping", "fl_flipping", "fl_auto_flip",
            "fl_has_flipped", "fl_has_djump", "fl_flip_or_jump",
            "fl_flip_reset", "fl_world_cont",
            "fl_air_time", "fl_flip_time", "fl_jump_time", "fl_boost_time",
            *spawn_cols,
        ]

    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for fr in frames:
            inp   = fr["input"]
            d     = fr["detail"]
            parts = fr.get("obs_parts") or {}
            fl    = fr.get("flags") or {}
            sp    = fr.get("spawn_state") or {}

            ch = inp.get("ppo_channels")
            if ch is None:
                ch = controls_to_multi_discrete(inp)
            keys_str = inp.get("keys_str") or multi_discrete_to_key_str(ch)
            hb_int = int(inp.get("handbrake", False))
            ss_int = int(fr.get("is_supersonic", False))
            bw_val = round(float(fr.get("boost_wasted", 0.0)), 4)
            bu_val = round(float(fr.get("boost_used", 0.0)), 4)

            sp_vals = [
                fr.get("source", "session"),
                sp.get("spawn_id", "0000"),
                sp.get("mode", "ground"),
                *(sp.get("car_pos") or (0.0, 0.0, 0.0)),
                *(sp.get("car_vel") or (0.0, 0.0, 0.0)),
                *(sp.get("car_ang") or (0.0, 0.0, 0.0)),
                *(sp.get("car_fwd") or (1.0, 0.0, 0.0)),
                *(sp.get("car_rgt") or (0.0, 1.0, 0.0)),
                *(sp.get("car_up")  or (0.0, 0.0, 1.0)),
                float(sp.get("car_boost", 33.3)),
                *(sp.get("ball_pos") or (0.0, 0.0, 93.15)),
                *(sp.get("ball_vel") or (0.0, 0.0, 0.0)),
                *(sp.get("ball_ang") or (0.0, 0.0, 0.0)),
            ]

            if mode == "compact":
                row = [
                    fr["i"],
                    inp.get("thr",0), inp.get("str",0),
                    inp.get("ptc",0), inp.get("yaw",0), inp.get("roll",0),
                    int(inp.get("jump",False)), int(inp.get("boost",False)), hb_int,
                    int(ch[0]), int(ch[1]), int(ch[2]), int(ch[3]), int(ch[4]), int(ch[5]), keys_str,
                    int(fr["on_ground"]), int(fr["is_flipping"]), ss_int, bw_val, bu_val,
                    fr["air_time"], fr["boost"],
                    fr["dist"], d.get("min_dist", fr["dist"]), fr["dist_h"], fr["align"],
                    fr.get("closing", 0.0),
                    d.get("peak_closing", 0.0),
                    d.get("closing_accum", 0.0),
                    fr["r_total"], fr["r_cum"],
                    d.get("r_dist",0), d.get("r_time",0),
                    d.get("r_progress",0), d.get("r_closing_bonus",0), d.get("r_boost_waste",0), d.get("r_boost_used",0),
                    d.get("r_touch",0),
                    _gv(parts,"D.ball_rel_b",0), _gv(parts,"D.ball_rel_b",1),
                    _gv(parts,"D.ball_rel_b",2), _gv(parts,"D.ball_rel_b",3),
                    _gv(parts,"E.ball_relvel",0), _gv(parts,"E.ball_relvel",1),
                    _gv(parts,"E.ball_relvel",2), _gv(parts,"E.ball_relvel",3),
                    _gv(parts,"C.car_vel_b",0), _gv(parts,"C.car_vel_b",1),
                    _gv(parts,"C.car_vel_b",2), _gv(parts,"C.car_vel_b",3),
                    _gv(parts,"B.car_ang_b",0), _gv(parts,"B.car_ang_b",1),
                    _gv(parts,"B.car_ang_b",2),
                    int(fl.get("flip_or_jump", 0)),
                    int(fl.get("is_jumping", 0)),
                    _gv(parts,"H.closing",0),
                    *sp_vals,
                ]
            else:
                bvel = fr.get("ball_vel") or (0.0, 0.0, 0.0)
                row = [
                    fr["i"],
                    inp.get("thr",0), inp.get("str",0),
                    inp.get("ptc",0), inp.get("yaw",0), inp.get("roll",0),
                    int(inp.get("jump",False)), int(inp.get("boost",False)), hb_int,
                    int(ch[0]), int(ch[1]), int(ch[2]), int(ch[3]), int(ch[4]), int(ch[5]), keys_str,
                    *fr["car_pos"], *fr["car_vel"], *fr["ball_pos"], *bvel,
                    fr["up_z"], int(fr["on_ground"]), int(fr["is_flipping"]), ss_int, bw_val, bu_val,
                    fr["air_time"], fr["boost"],
                    fr["dist"], d.get("min_dist", fr["dist"]), fr["dist_h"], fr["align"],
                    fr.get("closing", 0.0),
                    d.get("peak_closing", 0.0),
                    d.get("closing_accum", 0.0),
                    fr["r_total"], fr["r_cum"],
                    d.get("r_dist",0), d.get("r_time",0), d.get("r_align",0),
                    d.get("r_progress",0), d.get("r_closing_bonus",0), d.get("r_boost_waste",0), d.get("r_boost_used",0),
                    d.get("r_touch",0), d.get("r_timeout",0),
                    _gv(parts,"A.face_dists",0), _gv(parts,"A.face_dists",1),
                    _gv(parts,"A.face_dists",2), _gv(parts,"A.face_dists",3),
                    _gv(parts,"A.face_dists",4), _gv(parts,"A.face_dists",5),
                    _gv(parts,"B.car_ang_b",0), _gv(parts,"B.car_ang_b",1),
                    _gv(parts,"B.car_ang_b",2),
                    _gv(parts,"C.car_vel_b",0), _gv(parts,"C.car_vel_b",1),
                    _gv(parts,"C.car_vel_b",2), _gv(parts,"C.car_vel_b",3),
                    _gv(parts,"D.ball_rel_b",0), _gv(parts,"D.ball_rel_b",1),
                    _gv(parts,"D.ball_rel_b",2), _gv(parts,"D.ball_rel_b",3),
                    _gv(parts,"E.ball_relvel",0), _gv(parts,"E.ball_relvel",1),
                    _gv(parts,"E.ball_relvel",2), _gv(parts,"E.ball_relvel",3),
                    _gv(parts,"F.ball_ang_w",0), _gv(parts,"F.ball_ang_w",1),
                    _gv(parts,"F.ball_ang_w",2),
                    _gv(parts,"G.flags",0), _gv(parts,"G.flags",1),
                    _gv(parts,"G.flags",2), _gv(parts,"G.flags",3),
                    _gv(parts,"G.flags",4), _gv(parts,"G.flags",5),
                    _gv(parts,"H.closing",0),
                    int(fl.get("on_ground", 0)),
                    int(fl.get("is_jumping", 0)),
                    int(fl.get("is_flipping", 0)),
                    int(fl.get("auto_flip", 0)),
                    int(fl.get("has_flipped", 0)),
                    int(fl.get("has_djump", 0)),
                    int(fl.get("flip_or_jump", 0)),
                    int(fl.get("flip_reset", 0)),
                    int(fl.get("world_cont", 0)),
                    float(fl.get("air_time", 0.0)),
                    float(fl.get("flip_time", 0.0)),
                    float(fl.get("jump_time", 0.0)),
                    float(fl.get("boost_time", 0.0)),
                    *sp_vals,
                ]
            w.writerow(row)
    print(f"[tuner] exported {fn} ({len(frames)} frames, mode={mode})")
    return fn


def parse_spawn_state_from_row(row):
    if "init_cx" not in row or not row.get("init_cx"):
        return None
    return {
        "mode": row.get("spawn_mode", "ground"),
        "car_pos": (float(row["init_cx"]), float(row["init_cy"]), float(row["init_cz"])),
        "car_vel": (float(row["init_cvx"]), float(row["init_cvy"]), float(row["init_cvz"])),
        "car_ang": (float(row["init_cax"]), float(row["init_cay"]), float(row["init_caz"])),
        "car_fwd": (float(row["init_cfx"]), float(row["init_cfy"]), float(row["init_cfz"])),
        "car_rgt": (float(row["init_crx"]), float(row["init_cry"]), float(row["init_crz"])),
        "car_up":  (float(row["init_cux"]), float(row["init_cuy"]), float(row["init_cuz"])),
        "car_boost": float(row.get("init_cboost", 33.3)),
        "ball_pos": (float(row["init_bx"]), float(row["init_by"]), float(row["init_bz"])),
        "ball_vel": (float(row["init_bvx"]), float(row["init_bvy"]), float(row["init_bvz"])),
        "ball_ang": (float(row["init_bax"]), float(row["init_bay"]), float(row["init_baz"])),
    }
