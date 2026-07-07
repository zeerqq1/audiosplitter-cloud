"""
Core engine: splits a single voiceover audio file into one audio file per
paragraph of a source text, using forced alignment (WhisperX) + lossless cutting.

Pipeline:
  1. Parse paragraphs from the .txt (blank-line separated, with fallback).
  2. Transcribe + word-align the audio with WhisperX (GPU accelerated).
  3. Fuzzy-match the *original* paragraph words onto the aligned ASR words
     to locate each paragraph boundary in time.
  4. Place each cut in the middle of the silent gap between paragraphs
     (so no word is ever clipped, and nothing is added to the audio).
  5. Cut losslessly with ffmpeg (-c copy) -> numbered audio files.

Nothing is added to the audio: the segments are exact slices of the source.
"""

from __future__ import annotations

import os
import re
import gc
import difflib
import subprocess
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

# ISO language code used by WhisperX for each UI option.
LANG_CODES = {
    "Авто": None,
    "Русский": "ru",
    "English": "en",
    "한국어 (Korean)": "ko",
    "العربية (Arabic)": "ar",
    "Español": "es",
    "Français": "fr",
    "Português": "pt",
}

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".mp4"}
TEXT_EXTS = {".txt", ".md"}

_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)

ProgressCb = Callable[[str], None]
PctCb = Callable[[int], None]


@dataclass
class SplitResult:
    outputs: List[str] = field(default_factory=list)
    paragraph_count: int = 0
    detected_language: Optional[str] = None
    duration: float = 0.0
    warnings: List[str] = field(default_factory=list)
    coverage: List[float] = field(default_factory=list)   # per-paragraph word match 0..1
    gaps: List[Optional[float]] = field(default_factory=list)  # silence (s) at each boundary
    suspicious: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Text parsing
# --------------------------------------------------------------------------- #
def parse_paragraphs(text_path: str) -> List[str]:
    """Split text into paragraphs. Primary rule: blank-line separated blocks.
    Fallback: if that yields a single block, split on single newlines."""
    with open(text_path, "r", encoding="utf-8-sig") as f:
        raw = f.read()
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    blocks = [b.strip() for b in re.split(r"\n[ \t]*\n", raw)]
    paragraphs = [b for b in blocks if b]

    if len(paragraphs) <= 1:
        paragraphs = [ln.strip() for ln in raw.split("\n") if ln.strip()]

    return paragraphs


def _tokens(text: str) -> List[str]:
    """Lowercased, punctuation-stripped word tokens (works for any script)."""
    return [t for t in _WORD_RE.split(text.lower()) if t]


# --------------------------------------------------------------------------- #
# Forced alignment (WhisperX)
# --------------------------------------------------------------------------- #
def transcribe_and_align(
    audio_path: str,
    language: Optional[str],
    bundle: "ModelBundle",
    batch_size: int,
    progress: ProgressCb,
    pct: PctCb,
):
    """Returns (words, detected_language, duration, energy, hop_s, audio).
    `words` is a list of dicts: {'word', 'start', 'end'} for tokens that aligned
    (ASR pass — used only to derive rough windows for pass 2). `audio` is the
    decoded mono PCM, returned so the caller can run a second force-alignment
    pass on the original text without re-decoding. Models come from `bundle`,
    which loads each one once and reuses it across a whole batch."""
    import whisperx  # heavy import, kept lazy

    progress("Загрузка аудио…")
    pct(6)
    audio = _load_audio(audio_path)
    duration = len(audio) / 16000.0
    energy, hop_s = _energy_envelope(audio)

    progress(f"Распознавание речи (Whisper «{bundle.model_size}», {bundle.device.upper()})…")
    pct(12)
    model = bundle.get_asr(language)
    # The ASR pipeline is reused across files in a batch. whisperx has a bug where,
    # if the language changes on a reused pipeline, it passes a token id as `task`
    # (ValueError: '50360' is not a valid task). Resetting the tokenizer forces the
    # clean rebuild path, so each file gets the right language safely.
    try:
        model.tokenizer = None
    except Exception:
        pass
    pct(20)
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    detected = result.get("language", language) or language

    progress(f"Язык: {detected}. Выравнивание по словам…")
    pct(55)
    align_model, metadata = bundle.get_align(detected)
    pct(65)
    aligned = whisperx.align(
        result["segments"], align_model, metadata, audio, bundle.device,
        return_char_alignments=False,
    )

    words = [
        {"word": w["word"], "start": w["start"], "end": w["end"]}
        for w in aligned.get("word_segments", [])
        if w.get("start") is not None and w.get("end") is not None
    ]
    return words, detected, duration, energy, hop_s, audio


def _empty_cuda_cache(device: str) -> None:
    if device.startswith("cuda"):
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


class ModelBundle:
    """Loads the Whisper ASR model once and caches alignment models per language,
    so a whole batch reuses them instead of reloading for every file."""

    def __init__(self, model_size: str, device: str, compute_type: str):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._asr = None
        self._align: dict = {}

    def get_asr(self, language: Optional[str]):
        import whisperx
        if self._asr is None:
            self._asr = whisperx.load_model(
                self.model_size, self.device,
                compute_type=self.compute_type, language=language,
            )
        return self._asr

    def get_align(self, language: str):
        import whisperx
        if language not in self._align:
            self._align[language] = whisperx.load_align_model(
                language_code=language, device=self.device
            )
        return self._align[language]

    def close(self):
        self._asr = None
        self._align.clear()
        gc.collect()
        _empty_cuda_cache(self.device)


# --------------------------------------------------------------------------- #
# Boundary mapping
# --------------------------------------------------------------------------- #
def _energy_envelope(audio: "np.ndarray", sr: int = 16000, frame_s: float = 0.02):
    """Coarse RMS energy per `frame_s` window — a compact silence map of the audio."""
    hop = max(1, int(sr * frame_s))
    n = len(audio) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32), frame_s
    frames = audio[:n * hop].reshape(n, hop).astype(np.float32)
    energy = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    return energy, frame_s


def _snap_to_silence(energy, hop_s, thresh, t0, t1, fallback,
                     pad: float = 0.25, min_sil: float = 0.05) -> float:
    """Return the time at the centre of the longest real silence between t0 and t1
    (slightly padded). Falls back to `fallback` if there is no clear pause there."""
    a = max(0, int((t0 - pad) / hop_s))
    b = min(len(energy), int((t1 + pad) / hop_s) + 1)
    if b - a < 2:
        return fallback
    silent = (energy[a:b] < thresh).astype(np.int8)
    if not silent.any():
        return fallback
    edges = np.flatnonzero(np.diff(np.concatenate(([0], silent, [0]))))
    starts, ends = edges[0::2], edges[1::2]
    lengths = ends - starts
    k = int(np.argmax(lengths))
    if lengths[k] * hop_s < min_sil:
        return fallback
    center_frame = a + (starts[k] + ends[k] - 1) / 2.0
    return center_frame * hop_s


def _paragraph_token_index(paragraphs: List[str]):
    """Flatten paragraphs into a single token stream. Returns
    (orig_tokens, first_idx, last_idx), where first_idx[i]/last_idx[i] are the
    global token indices that bound paragraph i (None if it has no tokens)."""
    orig_tokens: List[str] = []
    first_idx: List[Optional[int]] = []
    last_idx: List[Optional[int]] = []
    for para in paragraphs:
        toks = _tokens(para)
        if toks:
            first_idx.append(len(orig_tokens))
            orig_tokens.extend(toks)
            last_idx.append(len(orig_tokens) - 1)
        else:
            first_idx.append(None)
            last_idx.append(None)
    return orig_tokens, first_idx, last_idx


def _interp_token_times(ts: List[Optional[float]], te: List[Optional[float]],
                        duration: float) -> None:
    """Fill None entries in the parallel start/end arrays by linear interpolation
    between known anchors (in place). Leading/trailing gaps extrapolate toward the
    audio bounds, so every token ends up with a monotonic time estimate."""
    n = len(ts)
    known = [i for i in range(n) if ts[i] is not None and te[i] is not None]
    if not known:
        for i in range(n):
            ts[i] = duration * i / max(n, 1)
            te[i] = duration * (i + 1) / max(n, 1)
        return
    first = known[0]
    for i in range(first):
        ts[i] = te[i] = ts[first] * ((i + 1) / (first + 1))
    last = known[-1]
    for i in range(last + 1, n):
        ts[i] = te[i] = te[last] + (duration - te[last]) * ((i - last) / (n - last))
    for a, b in zip(known, known[1:]):
        if b - a <= 1:
            continue
        t0, t1 = te[a], ts[b]
        for i in range(a + 1, b):
            ts[i] = te[i] = t0 + (t1 - t0) * ((i - a) / (b - a))


def _token_times(orig_tokens: List[str], ref_tokens: List[str],
                 ref_words: List[dict], duration: float):
    """Assign a (start, end) time to every original token by matching it to a
    reference word stream (ASR words, or force-aligned original words). Matched
    tokens take the reference time directly; the rest are interpolated.
    Returns (starts, ends, matched_mask)."""
    n = len(orig_tokens)
    ts: List[Optional[float]] = [None] * n
    te: List[Optional[float]] = [None] * n
    matched = [False] * n
    if ref_words and ref_tokens:
        matcher = difflib.SequenceMatcher(a=orig_tokens, b=ref_tokens, autojunk=False)
        for blk in matcher.get_matching_blocks():
            for k in range(blk.size):
                oi, ri = blk.a + k, blk.b + k
                s, e = ref_words[ri].get("start"), ref_words[ri].get("end")
                if s is None or e is None:
                    continue
                ts[oi], te[oi] = float(s), float(e)
                matched[oi] = True
    _interp_token_times(ts, te, duration)
    return ts, te, matched


def _bounds_from_words(paragraphs: List[str], words: List[dict], duration: float):
    """Return per-paragraph (start, end, coverage) by aligning the original
    paragraph tokens onto `words` (which carry start/end times). `start` is the
    first token's start, `end` is the last token's end. Coverage is the fraction
    of the paragraph's tokens that matched a real word time."""
    orig_tokens, first_idx, last_idx = _paragraph_token_index(paragraphs)
    ref_tokens = [(_tokens(w["word"])[0] if _tokens(w["word"]) else "") for w in words]
    ts, te, matched = _token_times(orig_tokens, ref_tokens, words, duration)
    bounds = []
    for i in range(len(paragraphs)):
        fi, li = first_idx[i], last_idx[i]
        if fi is None:
            bounds.append((None, None, 0.0))
            continue
        cov = sum(1 for k in range(fi, li + 1) if matched[k]) / (li - fi + 1)
        bounds.append((ts[fi], te[li], cov))
    return bounds


def force_align_paragraphs(
    paragraphs: List[str],
    asr_words: List[dict],
    duration: float,
    align_model,
    metadata,
    audio,
    device: str,
    progress: ProgressCb,
    pct: PctCb,
    pad: float = 4.0,
):
    """Pass 2 — force-align the *original* paragraph text to the audio (CTC).

    Pass 1 (ASR) only gives rough windows; ASR text differs from the script, so
    anchoring boundaries on matched ASR words drifts whenever the boundary words
    aren't recognised exactly. Here we instead align the ground-truth words
    themselves: each paragraph is handed to the wav2vec2 aligner as a segment,
    bounded by a padded window from pass 1. The aligner returns an exact
    timestamp for every original word, so each paragraph's true first and last
    word are pinned precisely — regardless of ASR errors, language, or length.
    Returns per-paragraph (start, end, coverage)."""
    import whisperx  # heavy, lazy

    orig_tokens, first_idx, last_idx = _paragraph_token_index(paragraphs)
    n = len(paragraphs)

    asr_tokens = [(_tokens(w["word"])[0] if _tokens(w["word"]) else "") for w in asr_words]
    a_ts, a_te, _ = _token_times(orig_tokens, asr_tokens, asr_words, duration)

    segments = []
    for i in range(n):
        text = " ".join(paragraphs[i].split()) or " "
        fi, li = first_idx[i], last_idx[i]
        if fi is None:
            segments.append({"text": text, "start": 0.0, "end": 0.0})
            continue
        s = max(0.0, a_ts[fi] - pad)
        e = min(duration, a_te[li] + pad)
        if e <= s:
            e = min(duration, s + 0.1)
        segments.append({"text": text, "start": s, "end": e})

    progress("Точное выравнивание абзацев по исходному тексту (CTC)…")
    aligned = whisperx.align(
        segments, align_model, metadata, audio, device,
        return_char_alignments=False,
    )
    flat = [w for w in aligned.get("word_segments", [])
            if w.get("start") is not None and w.get("end") is not None]
    progress(f"Точно выровнено слов: {len(flat)} из {len(orig_tokens)}")
    return _bounds_from_words(paragraphs, flat, duration)


def compute_cut_points(
    paragraphs: List[str],
    bounds: "List[tuple]",
    duration: float,
    warnings: List[str],
    energy: "Optional[np.ndarray]" = None,
    hop_s: float = 0.02,
) -> "tuple[List[float], List[float], List[Optional[float]]]":
    """Turn exact per-paragraph (start, end, coverage) bounds into cut points.
    Returns (cuts, coverage, gaps):
      cuts     - len(paragraphs)+1 points: [0.0, b1, ..., duration]
      coverage - per-paragraph alignment confidence (0..1)
      gaps     - silence in seconds at each internal boundary (or None)

    Each internal cut is placed in the silence between the END of one paragraph's
    last word and the START of the next paragraph's first word — a narrow,
    well-defined window that cannot drift into an internal sentence pause."""
    n = len(paragraphs)
    coverage = [b[2] for b in bounds] if bounds else []
    if n == 0:
        return [0.0, duration], [], []
    if n == 1:
        return [0.0, duration], (coverage or [1.0]), []

    have_energy = energy is not None and len(energy) > 0
    sil_thresh = 0.0
    if have_energy:
        floor = float(np.percentile(energy, 5))
        loud = float(np.percentile(energy, 75))
        sil_thresh = floor + 0.15 * (loud - floor)

    cuts: List[float] = [0.0]
    last_val = 0.0
    gaps: List[Optional[float]] = []
    for i in range(n - 1):
        left = bounds[i][1]       # end of last word of paragraph i
        right = bounds[i + 1][0]  # start of first word of paragraph i+1

        if left is None and right is None:
            gaps.append(None)
            cuts.append(None)  # type: ignore  (interpolated below)
            warnings.append(f"Абзац {i + 1}: не удалось выровнять — граница оценена.")
            continue
        if left is None:
            left = right
        if right is None:
            right = left

        gaps.append(right - left)
        lo, hi = (left, right) if right >= left else (right, left)
        cut = (lo + hi) / 2.0
        if have_energy:
            cut = _snap_to_silence(energy, hop_s, sil_thresh, lo, hi, cut)
        cut = min(max(cut, last_val + 0.02), duration - 0.02)
        cuts.append(cut)
        last_val = cut

    cuts.append(duration)
    _interpolate_missing(cuts)
    for i in range(1, len(cuts)):
        if cuts[i] <= cuts[i - 1]:
            cuts[i] = min(cuts[i - 1] + 0.02, duration)
    return cuts, coverage, gaps


def _interpolate_missing(cuts: List[Optional[float]]) -> None:
    """Linearly fill any None cut points between known anchors (in place)."""
    i = 0
    while i < len(cuts):
        if cuts[i] is None:
            j = i
            while j < len(cuts) and cuts[j] is None:
                j += 1
            lo = cuts[i - 1]
            hi = cuts[j] if j < len(cuts) else cuts[-1]
            span = j - (i - 1)
            for k in range(i, j):
                frac = (k - (i - 1)) / span
                cuts[k] = lo + (hi - lo) * frac
            i = j
        else:
            i += 1


# --------------------------------------------------------------------------- #
# Cutting
# --------------------------------------------------------------------------- #
def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _load_audio(path: str, sr: int = 16000) -> "np.ndarray":
    """Decode any audio file to mono float32 PCM at `sr` Hz using the bundled
    ffmpeg. Replaces whisperx.load_audio, which relies on a system-wide ffmpeg
    being on PATH (not the case here — ffmpeg is provided by imageio-ffmpeg)."""
    ffmpeg = _ffmpeg_exe()
    cmd = [
        ffmpeg, "-nostdin", "-threads", "0", "-i", path,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg не смог прочитать аудио:\n"
            + proc.stderr.decode("utf-8", "ignore")
        )
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0


def cut_audio(
    audio_path: str,
    cuts: List[float],
    output_dir: str,
    progress: ProgressCb,
    pct: PctCb,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    ext = os.path.splitext(audio_path)[1] or ".wav"
    ffmpeg = _ffmpeg_exe()
    n = len(cuts) - 1
    width = max(3, len(str(n)))
    outputs: List[str] = []

    for i in range(n):
        start = cuts[i]
        dur = cuts[i + 1] - cuts[i]
        out = os.path.join(output_dir, f"{i + 1:0{width}d}{ext}")
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}", "-i", audio_path, "-t", f"{dur:.3f}",
            "-c", "copy", out,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            # Fallback: re-encode this segment if stream-copy failed.
            cmd_re = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{start:.3f}", "-i", audio_path, "-t", f"{dur:.3f}",
                out,
            ]
            proc2 = subprocess.run(cmd_re, capture_output=True, text=True)
            if proc2.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg не смог вырезать сегмент {i + 1}:\n{proc.stderr}\n{proc2.stderr}"
                )
        outputs.append(out)
        progress(f"Сохранён {i + 1}/{n}: {os.path.basename(out)}")
        pct(80 + int(20 * (i + 1) / n))

    return outputs


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def pick_device(requested: str) -> tuple[str, str]:
    """Returns (device, compute_type)."""
    if requested == "cpu":
        return "cpu", "int8"
    if requested == "cuda":
        return "cuda", "float16"
    # auto
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _split_one(
    audio_path: str,
    text_path: str,
    output_dir: str,
    language: Optional[str],
    bundle: "ModelBundle",
    batch_size: int,
    progress: ProgressCb,
    pct: PctCb,
) -> SplitResult:
    """Process a single audio+text pair using an already-built ModelBundle."""
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")
    if not os.path.isfile(text_path):
        raise FileNotFoundError(f"Текстовый файл не найден: {text_path}")

    result = SplitResult()
    progress("Чтение текста и разбиение на абзацы…")
    pct(3)
    paragraphs = parse_paragraphs(text_path)
    result.paragraph_count = len(paragraphs)
    if not paragraphs:
        raise ValueError("В текстовом файле не найдено ни одного абзаца.")
    progress(f"Найдено абзацев: {len(paragraphs)}")

    words, detected, duration, energy, hop_s, audio = transcribe_and_align(
        audio_path, language, bundle, batch_size, progress, pct,
    )
    result.detected_language = detected
    result.duration = duration
    progress(f"Слов выровнено: {len(words)}; длительность: {duration:.1f} c")

    progress("Расчёт точек разреза по реальным паузам…")
    pct(78)
    # Map the ORIGINAL paragraph tokens onto the word-aligned ASR stream. Every
    # token gets a time (matched directly, or interpolated between matches), so
    # each paragraph's first and last word are located even when ASR misheard a
    # few words. Cuts are then placed in the silence between paragraphs.
    bounds = _bounds_from_words(paragraphs, words, duration)
    cuts, result.coverage, result.gaps = compute_cut_points(
        paragraphs, bounds, duration, result.warnings,
        energy=energy, hop_s=hop_s,
    )

    progress("Нарезка аудио (без потерь, по паузам)…")
    outputs = cut_audio(audio_path, cuts, output_dir, progress, pct)
    result.outputs = outputs

    pct(100)
    progress(f"Готово: {len(outputs)} файлов из {len(paragraphs)} абзацев.")
    if len(outputs) != len(paragraphs):
        result.warnings.append(
            f"Внимание: создано {len(outputs)} файлов, абзацев {len(paragraphs)}."
        )

    _build_qc_report(result, progress)
    return result


def run_split(
    audio_path: str,
    text_path: str,
    output_dir: str,
    language: Optional[str],
    model_size: str = "large-v3",
    device_pref: str = "auto",
    batch_size: int = 16,
    progress: Optional[ProgressCb] = None,
    pct: Optional[PctCb] = None,
) -> SplitResult:
    progress = progress or (lambda m: None)
    pct = pct or (lambda p: None)
    if not output_dir:
        raise ValueError("Не выбрана папка для сохранения.")

    device, compute_type = pick_device(device_pref)
    progress(f"Устройство: {device.upper()} ({compute_type})")
    bundle = ModelBundle(model_size, device, compute_type)
    try:
        return _split_one(audio_path, text_path, output_dir, language,
                          bundle, batch_size, progress, pct)
    finally:
        bundle.close()


def _build_qc_report(result: SplitResult, progress: ProgressCb) -> None:
    """Log a short quality-check summary and flag suspicious boundaries."""
    cov = result.coverage
    valid_gaps = [g for g in result.gaps if g is not None]

    suspicious: List[str] = []
    for i, c in enumerate(cov):
        if c < 0.5:
            suspicious.append(f"Абзац {i + 1}: покрытие словами {c * 100:.0f}%")
    for i, g in enumerate(result.gaps):
        if g is not None and g < 0.10:
            suspicious.append(f"Стык {i + 1}→{i + 2}: пауза {g * 1000:.0f} мс (мала)")
    result.suspicious = suspicious

    progress("──────── Проверка качества ────────")
    if cov:
        progress(f"Покрытие словами: среднее {sum(cov) / len(cov) * 100:.0f}%, "
                 f"минимум {min(cov) * 100:.0f}%")
    if valid_gaps:
        progress(f"Паузы между абзацами: средняя {sum(valid_gaps) / len(valid_gaps):.2f}s, "
                 f"минимум {min(valid_gaps):.2f}s")
    if suspicious:
        progress(f"⚠ Подозрительных границ: {len(suspicious)} — проверь эти файлы:")
        for s in suspicious[:20]:
            progress(f"   ⚠ {s}")
        if len(suspicious) > 20:
            progress(f"   …и ещё {len(suspicious) - 20}")
    else:
        progress("✅ Все границы уверенные — проблем не найдено.")


# --------------------------------------------------------------------------- #
# Batch processing (many audio+text pairs, matched by file name)
# --------------------------------------------------------------------------- #
@dataclass
class BatchPairResult:
    name: str
    output_dir: str
    ok: bool = False
    result: Optional[SplitResult] = None
    error: Optional[str] = None


@dataclass
class BatchResult:
    pairs: List["BatchPairResult"] = field(default_factory=list)
    unmatched_audio: List[str] = field(default_factory=list)
    unmatched_text: List[str] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        return sum(1 for p in self.pairs if p.ok)

    @property
    def failed_count(self) -> int:
        return sum(1 for p in self.pairs if not p.ok)


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].strip()


def _safe_name(name: str) -> str:
    """Make a string safe to use as a Windows folder name."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip().rstrip(". ")
    return name or "pair"


def find_pairs(audio_paths, text_paths):
    """Match audio files to text files by base file name (case-insensitive).
    Returns (pairs, unmatched_audio, unmatched_text), where each pair is the
    tuple (name, audio_path, text_path)."""
    text_by_stem = {}
    for t in text_paths:
        text_by_stem.setdefault(_stem(t).lower(), t)

    pairs = []
    used_text = set()
    unmatched_audio = []
    for a in audio_paths:
        key = _stem(a).lower()
        t = text_by_stem.get(key)
        if t is not None:
            pairs.append((_stem(a), a, t))
            used_text.add(key)
        else:
            unmatched_audio.append(a)

    unmatched_text = [t for t in text_paths if _stem(t).lower() not in used_text]
    pairs.sort(key=lambda p: p[0].lower())
    return pairs, unmatched_audio, unmatched_text


def run_batch(
    pairs,
    output_root: str,
    language: Optional[str],
    model_size: str = "large-v3",
    device_pref: str = "auto",
    batch_size: int = 16,
    progress: Optional[ProgressCb] = None,
    pct: Optional[PctCb] = None,
    unmatched_audio=None,
    unmatched_text=None,
) -> BatchResult:
    """Process every (name, audio_path, text_path) pair into its own subfolder
    of `output_root`. The heavy models are loaded once and shared across pairs."""
    progress = progress or (lambda m: None)
    pct = pct or (lambda p: None)

    if not pairs:
        raise ValueError("Не найдено ни одной пары «аудио + текст» с совпадающим именем.")
    if not output_root:
        raise ValueError("Не выбрана папка для сохранения.")
    os.makedirs(output_root, exist_ok=True)

    device, compute_type = pick_device(device_pref)
    progress(f"Устройство: {device.upper()} ({compute_type})")
    progress(f"Пар к обработке: {len(pairs)}")

    batch = BatchResult(
        unmatched_audio=list(unmatched_audio or []),
        unmatched_text=list(unmatched_text or []),
    )
    bundle = ModelBundle(model_size, device, compute_type)
    n = len(pairs)
    try:
        for i, (name, audio_path, text_path) in enumerate(pairs):
            subdir = os.path.join(output_root, _safe_name(name))
            progress("")
            progress(f"━━━━━━━ [{i + 1}/{n}] {name} → {os.path.basename(subdir)}\\ ━━━━━━━")

            def local_pct(p, _i=i):
                pct(int((_i * 100 + max(0, min(100, p))) / n))

            try:
                res = _split_one(audio_path, text_path, subdir, language,
                                 bundle, batch_size, progress, local_pct)
                batch.pairs.append(BatchPairResult(name, subdir, True, res))
            except Exception as exc:
                import traceback
                progress(f"❌ Ошибка в «{name}»: {exc}")
                batch.pairs.append(
                    BatchPairResult(name, subdir, False, None, traceback.format_exc())
                )
    finally:
        bundle.close()

    pct(100)
    _batch_summary(batch, progress)
    return batch


def _batch_summary(batch: BatchResult, progress: ProgressCb) -> None:
    progress("")
    progress("════════ Итог пакетной обработки ════════")
    total_files = sum(len(p.result.outputs) for p in batch.pairs if p.result)
    progress(f"Готово пар: {batch.ok_count}/{len(batch.pairs)} · "
             f"файлов создано: {total_files}")

    for p in batch.pairs:
        if not p.ok:
            progress(f"   ❌ {p.name}: ошибка (см. лог выше)")
        elif p.result and p.result.suspicious:
            progress(f"   ⚠ {p.name}: {len(p.result.outputs)} файлов, "
                     f"{len(p.result.suspicious)} границ на проверку")
        elif p.result:
            progress(f"   ✅ {p.name}: {len(p.result.outputs)} файлов, всё чисто")

    if batch.unmatched_audio:
        progress(f"⚠ Аудио без пары ({len(batch.unmatched_audio)}): "
                 + ", ".join(os.path.basename(a) for a in batch.unmatched_audio[:10]))
    if batch.unmatched_text:
        progress(f"⚠ Текст без пары ({len(batch.unmatched_text)}): "
                 + ", ".join(os.path.basename(t) for t in batch.unmatched_text[:10]))
