"""Download everything collected by the website into ./dataset for training.

  vercel link && vercel env pull .env.local     # once, gives this script access
  python3 export_data.py

Each capture becomes dataset/<date>/<id>.jpg plus .json (what the model predicted)
and, when the person typed their own count, .label.json. Existing files are skipped."""
import os, sys
from vercel.blob import sync as blob

def load_env(path='.env.local'):
    if os.path.exists(path):
        for line in open(path):
            if '=' in line and not line.startswith('#'):
                k, v = line.strip().split('=', 1); os.environ.setdefault(k, v.strip('"'))

load_env()
out, n, skipped = 'dataset', 0, 0
for item in blob.query_blobs(prefix='captures/'):
    dest = os.path.join(out, item.pathname.removeprefix('captures/'))
    if os.path.exists(dest):
        skipped += 1; continue
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    open(dest, 'wb').write(blob.get_blob(item.pathname, access='private').body)
    n += 1
print(f'downloaded {n} files, {skipped} already there -> {out}/')
