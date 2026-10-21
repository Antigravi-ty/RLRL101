# tools/gen_tree.py
"""生成 file.md —— 项目文件树。用法：python tools/gen_tree.py [root]"""
import os
import sys

SKIP_DIRS = {'.git', '__pycache__', '.venv', 'venv', 'env',
             'node_modules', '.idea', '.vscode', '.pytest_cache',
             'checkpoints', 'data'}
SKIP_EXT  = {'.pyc', '.pyo', '.pt', '.csv', '.json'}


def walk(path, prefix, lines):
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        return
    dirs  = [e for e in entries
             if os.path.isdir(os.path.join(path, e)) and e not in SKIP_DIRS]
    files = [e for e in entries
             if os.path.isfile(os.path.join(path, e))
             and os.path.splitext(e)[1] not in SKIP_EXT]
    items = dirs + files
    for i, name in enumerate(items):
        is_last   = (i == len(items) - 1)
        connector = '└── ' if is_last else '├── '
        full      = os.path.join(path, name)
        if os.path.isdir(full):
            lines.append(prefix + connector + name + '/')
            walk(full, prefix + ('    ' if is_last else '│   '), lines)
        else:
            lines.append(prefix + connector + name)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    root = os.path.abspath(root)
    lines = [os.path.basename(root) + '/']
    walk(root, '', lines)
    text = '\n'.join(lines)
    print(text)
    out = os.path.join(root, 'file.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(f"\n[gen_tree] written to {out}")


if __name__ == '__main__':
    main()