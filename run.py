import sys, os

HERE = os.path.dirname(os.path.abspath(__file__))

CMDS = {
    "tuner":     ["vis/reward_tuner.py"],
    "cli":       ["vis/cli_tuner.py"],
    "eval":      ["vis/cli_tuner.py"],
    "train":     ["train/train_ppo.py"],
    "train_ppo": ["train/train_ppo.py"],
    "focus":     ["tools/focus_cli.py"],
    "bench":     ["tools/bench.py"],
    "tree":      ["tools/gen_tree.py"],
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print("usage: python run.py", "|".join(CMDS.keys()))
        sys.exit(1)
    target = CMDS[sys.argv[1]][0]
    os.execv(sys.executable, [sys.executable, os.path.join(HERE, target), *sys.argv[2:]])
