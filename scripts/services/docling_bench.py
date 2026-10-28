"""Times one PDF conversion through docling-serve with the egret-xlarge layout model.

usage: python3 docling_bench.py <file.pdf>
Submits to the asynchronous API on :5003 with OCR off, polls until done, and reports the time and
the length of the Markdown result.
"""
import json, subprocess, sys, time, urllib.request
B = "http://127.0.0.1:5003"
cfg = json.dumps({"kind": "docling_layout_default", "model_spec": {"name": "docling_layout_egret_xlarge", "repo_id": "docling-project/docling-layout-egret-xlarge", "revision": "main"}})
t = time.time()
out = subprocess.check_output(["curl", "-s", "-F", f"files=@{sys.argv[1]};type=application/pdf", "-F", "to_formats=md", "-F", "do_ocr=false", "-F", f"layout_custom_config={cfg}", B + "/v1/convert/file/async"])
task = json.loads(out)["task_id"]
while True:
    s = json.load(urllib.request.urlopen(f"{B}/v1/status/poll/{task}"))
    if s["task_status"] in ("success", "failure"): break
    time.sleep(0.5)
dt = time.time() - t
r = json.load(urllib.request.urlopen(f"{B}/v1/result/{task}"))
md = r["document"]["md_content"] or ""
print(json.dumps({"status": s["task_status"], "seconds": round(dt, 1), "md_chars": len(md)}))
