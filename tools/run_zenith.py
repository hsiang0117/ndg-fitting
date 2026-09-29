"""Launch an audited 30k-step Zenith run and record completion/failure."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from splatting.training_output import create_run_directory, write_json

output = create_run_directory(ROOT / 'output')
state = dict(output=str(output), pid=os.getpid(), status='running',
             started_at=datetime.now().isoformat(), iterations=30000)
status_path = output / 'run_status.json'
current_path = ROOT / 'temporary-build' / 'zenith-current-run.json'
command = [sys.executable, '-u', '-m', 'splatting.splatting_train',
           '--dataset_path', r'D:\dataset\CloudDatasetZenith',
           '--output_dir', str(output), '--iterations', '30000',
           '--data_device', 'cpu', '--tensorboard']
write_json(output / 'command.json', command)
(output / 'source_commit.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
(output / 'source_changes.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT))
# Include new source files that have not been committed yet.
untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
for name in untracked:
    source = ROOT / name
    if source.suffix in ('.py', '.ps1', '.txt', '.md'):
        target = output / 'source_untracked' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
write_json(status_path, state)
write_json(current_path, state)
try:
    with (output / 'train.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        state['training_pid'] = process.pid
        write_json(status_path, state)
        write_json(current_path, state)
        code = process.wait()
    if code:
        raise RuntimeError(f'Training exited with code {code}; see train.log')
    if not (output / 'training_complete.json').exists():
        raise RuntimeError('Training did not produce its completion record')
    state.update(status='completed', finished_at=datetime.now().isoformat())
except Exception as exc:
    state.update(status='failed', error=str(exc), finished_at=datetime.now().isoformat())
    raise
finally:
    write_json(status_path, state)
    write_json(current_path, state)
print(json.dumps(state, indent=2))
