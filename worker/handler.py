"""RunPod Serverless handler.

Runs on each worker. Reads one audio+text "pack" from the network volume
(mounted at /runpod-volume), splits it using the SAME `splitter.py` engine
that powers the local AudioSplitter EXE (splitter.find_pairs / run_batch),
and writes the resulting clips back to the network volume.

Expected job input (all paths are RELATIVE to the network volume root):
{
    "audio": "jobs/<job_id>/input/audio.mp3",
    "text":  "jobs/<job_id>/input/text.txt",
    "name":  "rus",                 # optional, used for the output subfolder
    "language": "ru",               # optional, same as the EXE's language field
    "model_size": "large-v3",       # optional
    "output": "jobs/<job_id>/output"
}
"""
import os
import traceback

import runpod

import splitter

VOLUME_ROOT = os.environ.get("RUNPOD_VOLUME_ROOT", "/runpod-volume")


def _abs(rel_path: str) -> str:
    return os.path.join(VOLUME_ROOT, rel_path)


def handler(job):
    inp = job.get("input", {}) or {}
    try:
        audio_path = _abs(inp["audio"])
        text_path = _abs(inp["text"])
        out_dir = _abs(inp["output"])
        name = inp.get("name") or os.path.splitext(os.path.basename(audio_path))[0]
        language = inp.get("language") or None
        model_size = inp.get("model_size", "large-v3")
    except KeyError as exc:
        return {"error": f"missing required input field: {exc}"}

    if not os.path.isfile(audio_path):
        return {"error": f"audio not found on volume: {inp.get('audio')}"}
    if not os.path.isfile(text_path):
        return {"error": f"text not found on volume: {inp.get('text')}"}

    os.makedirs(out_dir, exist_ok=True)

    log_lines = []

    def progress(msg):
        log_lines.append(msg)
        print(msg, flush=True)

    def pct(_p):
        pass

    pairs = [(name, audio_path, text_path)]

    try:
        batch = splitter.run_batch(
            pairs,
            out_dir,
            language,
            model_size=model_size,
            device_pref="auto",
            progress=progress,
            pct=pct,
        )
    except Exception:
        return {
            "error": "run_batch failed",
            "traceback": traceback.format_exc(),
            "log": log_lines[-100:],
        }

    pair_result = batch.pairs[0] if batch.pairs else None
    ok = bool(pair_result and pair_result.ok)
    n_outputs = len(pair_result.result.outputs) if (pair_result and pair_result.result) else 0
    n_suspicious = len(pair_result.result.suspicious) if (pair_result and pair_result.result) else 0

    return {
        "status": "ok" if ok else "failed",
        "name": name,
        "output_dir": inp["output"],
        "files": n_outputs,
        "suspicious_boundaries": n_suspicious,
        "log": log_lines[-100:],
    }


runpod.serverless.start({"handler": handler})
