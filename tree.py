import os, re

EXCLUDE_DIR = re.compile(r'(\.git|__pycache__|\\results\\|/results/|checkpoint-)')
EXCLUDE_FILE = re.compile(r'\.(safetensors|pt|bin)$')

KEEP_EXT = {'.py', '.md', '.txt', '.csv', '.json', '.toml', '.ini'}
KEEP_SPECIAL = {'requirements.txt', '.env'}

lines = []
for root, dirs, files in os.walk('.', topdown=True):
    # 디렉터리 제외
    dirs[:] = [d for d in dirs if not EXCLUDE_DIR.search(os.path.join(root, d))]
    rel_root = os.path.relpath(root, '.')
    if rel_root == '.': rel_root = ''
    lines.append((rel_root + os.sep) if rel_root else './')

    for f in files:
        path = os.path.join(root, f)
        if EXCLUDE_DIR.search(path) or EXCLUDE_FILE.search(f): 
            continue
        if (os.path.splitext(f)[1] in KEEP_EXT) or (f in KEEP_SPECIAL):
            lines.append(os.path.relpath(path, '.'))

with open('structure_clean.txt', 'w', encoding='utf-8') as w:
    w.write('\n'.join(lines))
