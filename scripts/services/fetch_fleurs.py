"""Downloads the first 100 clips of FLEURS id_id dev by streaming the archive.

usage: python3 -I fetch_fleurs.py <empty output directory>
Only file names are taken from the archive, never its paths. dev.tsv is fetched separately from
https://huggingface.co/datasets/google/fleurs/resolve/main/data/id_id/dev.tsv.
"""
import sys, tarfile, urllib.request, os
out = sys.argv[1]
url = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/id_id/audio/dev.tar.gz"
n = 0
with urllib.request.urlopen(url, timeout=30) as r, tarfile.open(fileobj=r, mode="r|gz") as t:
    for m in t:
        if not (m.isfile() and m.name.endswith(".wav")):
            continue
        name = os.path.basename(m.name)  # no path components from the archive
        with open(os.path.join(out, name), "wb") as f:
            f.write(t.extractfile(m).read())
        n += 1
        if n == 100:
            break
print("wavs", n)
