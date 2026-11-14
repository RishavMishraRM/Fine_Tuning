import subprocess
import sys
import time
import glob

files = sorted(glob.glob("fine_tuning_techniques/[0-9]*.py"))
print(f"Testing all {len(files)} technique files in dry-run mode...\n")

success = 0
for f in files:
    t0 = time.time()
    try:
        res = subprocess.run([sys.executable, f], capture_output=True, text=True, timeout=10)
        dt = time.time() - t0
        lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
        summary = lines[1] if len(lines) > 1 else (lines[0] if lines else "")
        if res.returncode == 0:
            print(f"✓ [{dt:4.1f}s] {f:<50s} -> {summary}", flush=True)
            success += 1
        else:
            print(f"✗ [FAIL] {f:<50s} (exit code {res.returncode})", flush=True)
            print(res.stderr[:300], flush=True)
    except subprocess.TimeoutExpired:
        print(f"⏱️ [TIMEOUT] {f:<50s}", flush=True)

print(f"\nResult: {success}/{len(files)} files passed standalone execution successfully!")
