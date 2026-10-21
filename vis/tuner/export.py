import csv
import os
import time
from core.policy_net import controls_to_multi_discrete, multi_discrete_to_key_str


def _gv(parts, grp, k):
    v = parts.get(grp)
    if v is None or k >= len(v):
        return 0.0
    return float(v[k])


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
        "init_cboost", "init_has_2jump",
        "init_bx", "init_by", "init_bz",
    ]

    if mode == "compact":
        header = [
            "i",
            "thr", "str", "ptc", "yaw", "roll", "jump", "boost", "handbrake",
            "ch0_ws", "ch1_ad", "ch2_qe", "ch3_jump", "ch4_boost", "ch5_slide", "keys_str",
            "on_ground", "is_flipping", "is_supersonic", "air_time", "car_boost",
            "dist", "dist_h", "align", "closing", "car_speed",
            "r_total", "r_cum", "r_dist", "r_slide", "r_time", "r_boost", "r_jump", "r_min_dist", "r_touch", "r_timeout",
            "dir_fwd", "dir_right", "dir_up", "b_fwd", "b_right", "b_up", "|d|",
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
            "bx", "by", "bz",
            "up_z", "on_ground", "is_flipping", "is_supersonic", "air_time", "car_boost",
            "dist", "dist_h", "align", "closing", "car_speed",
            "r_total", "r_cum",
            "r_dist", "r_slide", "r_time", "r_boost", "r_jump", "r_min_dist", "r_touch", "r_timeout",
            "obs_a_nx", "obs_a_px", "obs_a_ny", "obs_a_py",
            "obs_a_floor", "obs_a_ceil",
            "obs_b_pitch", "obs_b_yaw", "obs_b_roll",
            "obs_c_vfwd", "obs_c_vright", "obs_c_vup", "obs_c_vmag",
            "obs_d_dirfwd", "obs_d_dirright", "obs_d_dirup",
            "obs_d_bfwd", "obs_d_bright", "obs_d_bup", "obs_d_bdist",
            "obs_e_onground", "obs_e_flip_or_jump", "obs_e_is_flipping",
            "obs_e_is_jumping", "obs_e_air_time", "obs_e_boost",
            "obs_f_closing",
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

            sp_vals = [
                fr.get("source", "session"),
                sp.get("spawn_id", "0000"),
                sp.get("mode", "all"),
                *(sp.get("car_pos") or (0.0, 0.0, 0.0)),
                *(sp.get("car_vel") or (0.0, 0.0, 0.0)),
                *(sp.get("car_ang") or (0.0, 0.0, 0.0)),
                *(sp.get("car_fwd") or (1.0, 0.0, 0.0)),
                *(sp.get("car_rgt") or (0.0, 1.0, 0.0)),
                *(sp.get("car_up")  or (0.0, 0.0, 1.0)),
                float(sp.get("car_boost", 100.0)),
                int(sp.get("has_second_jump", True)),
                *(sp.get("ball_pos") or (0.0, 0.0, 93.15)),
            ]

            if mode == "compact":
                row = [
                    fr["i"],
                    inp.get("thr",0), inp.get("str",0),
                    inp.get("ptc",0), inp.get("yaw",0), inp.get("roll",0),
                    int(inp.get("jump",False)), int(inp.get("boost",False)), hb_int,
                    int(ch[0]), int(ch[1]), int(ch[2]), int(ch[3]), int(ch[4]), int(ch[5]), keys_str,
                    int(fr["on_ground"]), int(fr["is_flipping"]), ss_int,
                    fr["air_time"], fr["boost"],
                    fr["dist"], fr["dist_h"], fr["align"],
                    fr.get("closing", 0.0), fr.get("car_speed", 0.0),
                    fr["r_total"], fr["r_cum"],
                    d.get("r_dist",0), d.get("r_slide",0), d.get("r_time",0),
                    d.get("r_boost",0), d.get("r_jump",0), d.get("r_min_dist",0),
                    d.get("r_touch",0), d.get("r_timeout",0),
                    _gv(parts,"D.ball_rel_b",0), _gv(parts,"D.ball_rel_b",1),
                    _gv(parts,"D.ball_rel_b",2), _gv(parts,"D.ball_rel_b",3),
                    _gv(parts,"D.ball_rel_b",4), _gv(parts,"D.ball_rel_b",5),
                    _gv(parts,"D.ball_rel_b",6),
                    _gv(parts,"C.car_vel_b",0), _gv(parts,"C.car_vel_b",1),
                    _gv(parts,"C.car_vel_b",2), _gv(parts,"C.car_vel_b",3),
                    _gv(parts,"B.car_ang_b",0), _gv(parts,"B.car_ang_b",1),
                    _gv(parts,"B.car_ang_b",2),
                    int(fl.get("flip_or_jump", 0)),
                    int(fl.get("is_jumping", 0)),
                    _gv(parts,"F.closing",0),
                    *sp_vals,
                ]
            else:
                row = [
                    fr["i"],
                    inp.get("thr",0), inp.get("str",0),
                    inp.get("ptc",0), inp.get("yaw",0), inp.get("roll",0),
                    int(inp.get("jump",False)), int(inp.get("boost",False)), hb_int,
                    int(ch[0]), int(ch[1]), int(ch[2]), int(ch[3]), int(ch[4]), int(ch[5]), keys_str,
                    *fr["car_pos"], *fr["car_vel"], *fr["ball_pos"],
                    fr["up_z"], int(fr["on_ground"]), int(fr["is_flipping"]), ss_int,
                    fr["air_time"], fr["boost"],
                    fr["dist"], fr["dist_h"], fr["align"],
                    fr.get("closing", 0.0), fr.get("car_speed", 0.0),
                    fr["r_total"], fr["r_cum"],
                    d.get("r_dist",0), d.get("r_slide",0), d.get("r_time",0),
                    d.get("r_boost",0), d.get("r_jump",0), d.get("r_min_dist",0),
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
                    _gv(parts,"D.ball_rel_b",4), _gv(parts,"D.ball_rel_b",5),
                    _gv(parts,"D.ball_rel_b",6),
                    _gv(parts,"E.flags",0), _gv(parts,"E.flags",1),
                    _gv(parts,"E.flags",2), _gv(parts,"E.flags",3),
                    _gv(parts,"E.flags",4), _gv(parts,"E.flags",5),
                    _gv(parts,"F.closing",0),
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
    bz = float(row["init_bz"])
    cz = float(row["init_cz"])
    return {
        "mode": row.get("spawn_mode", "all"),
        "is_hover_ball": bool(bz > 120.0),
        "car_on_ground": bool(cz <= 22.0),
        "has_second_jump": bool(int(float(row.get("init_has_2jump", 1)))),
        "car_pos": (float(row["init_cx"]), float(row["init_cy"]), cz),
        "car_vel": (float(row["init_cvx"]), float(row["init_cvy"]), float(row["init_cvz"])),
        "car_ang": (float(row["init_cax"]), float(row["init_cay"]), float(row["init_caz"])),
        "car_fwd": (float(row["init_cfx"]), float(row["init_cfy"]), float(row["init_cfz"])),
        "car_rgt": (float(row["init_crx"]), float(row["init_cry"]), float(row["init_crz"])),
        "car_up":  (float(row["init_cux"]), float(row["init_cuy"]), float(row["init_cuz"])),
        "car_boost": float(row.get("init_cboost", 100.0)),
        "ball_pos": (float(row["init_bx"]), float(row["init_by"]), bz),
        "ball_vel": (0.0, 0.0, 0.0),
        "ball_ang": (0.0, 0.0, 0.0),
    }
