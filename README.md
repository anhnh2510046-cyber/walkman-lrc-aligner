# Walkman LRC Aligner

Interactive Python CLI for generating synchronized `.lrc` lyrics beside local audio files, intended for Sony Walkman music libraries, including the NW-S644. Speech recognition and forced alignment run locally; optional Genius lookup and model downloads require Internet access.

## 1. Workflow architecture

```text
Audio file or music folder → scan → existing TXT or approved Genius lyrics
                                    ↓ no approved lyrics + explicit consent
                              local Whisper ASR → user approval / manual entry
                                    ↓
Approved lyrics → optional Demucs vocals → FFmpeg PCM decoding
                → Silero vocal onset → stanza-guided stable-ts alignment → review flags → BOM LRC
                → verify LRC exists → remove TXT sidecar
```

### Track discovery and lyric selection

- Accept a single MP3, FLAC, M4A or WAV file, or recursively scan a music folder. Skip symbolic-link audio files and tracks with an existing LRC sidecar; extension/sidecar matching is case-insensitive.
- Use the full filename stem as the search query, preserving `Artist - Title` and removing a leading numeric track index such as `01.` or `02 -`. Read metadata for generic/numeric names, or explicitly with `--use-metadata`; parsing failures retain the filename fallback.
- Read an existing UTF-8 TXT sidecar (with or without BOM) directly, bypassing Genius and ASR.
- Otherwise, search Genius, preview four nonempty lyric lines, and require approval. Rejecting a candidate allows another title or a direct `genius.com` URL. Section headings become blank stanza separators; Genius header/footer metadata is removed.
- Only explicit `y` consent enables local faster-whisper ASR after lookup ends. Generated lyrics require another approval. Empty recognition offers manual entry (blank line finishes) or Skip.
- Approved TXT/Genius lyrics proceed directly to alignment. The CLI does not transcribe them again or call the RapidFuzz comparison helpers in `comparator.py`.

The ASR fallback processes the original audio with `vad_filter=False`, `no_speech_threshold=0.8`, `condition_on_previous_text=False`, and temperature fallback `(0.0, 0.2, 0.4)`. An empty result is retried at temperature `0.2`. A specific PyAV `metadata_errors` keyword incompatibility triggers FFmpeg decoding to 16 kHz mono float32 audio before retrying. These settings reduce some failure modes but do not guarantee accurate singing recognition.

### Vocal separation and audio preparation

When the `demucs` executable is on PATH, alignment first separates the **entire track once** using:

```sh
# CPU
demucs --two-stems=vocals -n htdemucs -d cpu -o <temporary-directory> <audio-file>
# GPU
demucs --two-stems=vocals -n htdemucs_ft -d cuda --shifts 2 --overlap 0.5 -o <temporary-directory> <audio-file>
```

CPU separation uses `htdemucs` with no explicit shifts/overlap overrides. CUDA separation uses the refinement model `htdemucs_ft`, two shifts and 0.5 overlap to improve vocal isolation at additional compute/memory cost. These settings aim to reduce accompaniment artifacts, including heavy beats, but complete removal of noise/instruments or improved recognition is not guaranteed. This preparation occurs after lyric selection; it does not preprocess the separate ASR fallback.

Demucs is optional and installed separately. It uses the same selected device as ASR and alignment, with a 1,800-second timeout. Its vocal output is read from the selected model's directory (`htdemucs` or `htdemucs_ft`). Failure or missing/empty vocal output emits a warning and falls back to the original audio. FFmpeg then decodes the selected source to **16 kHz mono, signed 16-bit PCM WAV**. Temporary stems and decoded audio are removed after processing; the original audio is not modified.

### Silence handling and stanza-guided alignment

The current stable-ts call uses:

```python
original_split=True
vad=False
suppress_silence=True
fast_mode=False
verbose=False
```

Before stanza alignment, the bundled `silero-vad` CPU model runs `get_speech_timestamps` on the prepared 16 kHz mono PCM audio, with no speech padding. The first detected vocal start becomes the lower bound: the first stanza receives only the audio from that onset onward, and its timestamps are translated back to the full track. This prevents alignment from placing the first lyric in the preceding intro/silence. The remaining stanza timing and review heuristics are unchanged.

If no vocals are detected, the track fails without producing an LRC and existing TXT lyrics are preserved. Model errors also fail the track rather than silently dropping the onset guard. Silero's packaged model is reused across tracks and runs on CPU even with `--device cuda`. A detected onset at audio start can legitimately produce `00:00`; VAD is a speech detector and can misclassify singing or instruments, so this guard depends on its detected onset.

Stable-ts retains `vad=False` and `suppress_silence=True` for its existing per-stanza timing behavior; Silero detection is a separate preprocessing step. ASR retains `vad_filter=False`. There is no VAD CLI toggle or automatic `vad=True` retry.

Alignment follows lyric sections rather than submitting all lyrics in one pass:

1. Split normalized lyrics at blank lines; if none exist, group consecutive lines in batches of nine. Musical sections follow supplied stanza breaks rather than automatic section detection.
2. Start the first audio suffix at the Silero-detected vocal onset, then align each stanza against the remaining suffix, preserving its lyric line boundaries.
3. Convert suffix-relative timestamps back to full-track timestamps.
4. Advance the cursor to the last line's end minus **0.5 seconds**, retaining overlap for the next stanza.

This limits the alignment context and aims to reduce accumulated drift. Every nonempty approved line is retained in order. A segment-count mismatch, nonfinite/invalid/out-of-range timing, or alignment that does not advance aborts the track without generating an LRC. If the cursor reaches the final second of audio, all remaining lines receive `max(previous_start, first_vocal_start, duration - 0.5)` as a fallback timestamp and are marked `[?]`; they are not character-weighted estimates.

### Sequential timing and Walkman timestamp order

Each stanza is aligned once against the remaining audio suffix using `save_audio_suffix`; the next stanza retains 0.5 seconds of overlap. There is no maximum forward-jump constraint, per-line recovery search, local audio window or pace interpolation.

Before accepting a line, its full-track start is bounded by the preceding accepted start (`start >= previous_start`). A backward model timestamp is clamped to that start and marked `[?]` for manual review; its end is raised if necessary so `end >= start`. The same bound applies across stanza boundaries. Equal timestamps are allowed and lyric order is preserved.

When audio is nearly exhausted, the fallback anchor is the maximum of the last accepted start, the first vocal onset and `duration - 0.5`. This prevents a late timestamp such as `[03:20.72]` from being followed by an earlier fallback. Fallback lines remain marked `[?]`; several lines can share the final timestamp and require manual timing. Large forward matches remain the model's responsibility and are not replaced with estimated timings.

### Independent review flags and phonetic-verification limits

A separate review pass evaluates the aligned timings and available segment confidence. It prefixes uncertain lyric text with `[?]`:

```text
[00:12.34] An ordinary lyric line
[00:16.78] [?] A line requiring manual review
```

| Review signal | Current rule |
| --- | --- |
| Nearby/reversed line starts | Previous or next start gap below 1.2 seconds |
| Apparent excessive speed | Next-start gap below the current line's whitespace-separated word count divided by 2.8 |
| Compressed long line | More than 20 characters aligned into less than 1.8 seconds |
| Low model confidence | Finite `avg_logprob` below -0.9, when supplied by the segment |
| Timing cluster | Start gap below 1.5 seconds flags the cluster seed and its immediate neighbors; propagation is one hop |
| Possible outro mismatch | Last lyric starts more than 12 seconds before audio ends |
| Backward start correction | Starts clamped to the previous accepted start are marked for review |
| Exhausted audio | Remaining lines use the final fallback timestamp |

These are review heuristics, not independent phonetic verification against a second recognition result. Missing confidence is not treated as a measured low score. Segment text differences are not compared by this pass: output uses the approved lyric text. `comparator.py` provides ordered text-window comparisons with RapidFuzz, but the current CLI does not invoke them. Dense vocals, long outros, and languages without space-delimited words can require additional manual judgment. Listen to flagged lines and adjust their timestamps/text before copying the LRC to a player.

## 2. Sony Walkman NW-S644 compatibility and file management

### UTF-8 BOM and multilingual text

New TXT and LRC sidecars use **UTF-8 with BOM (`utf-8-sig`)**, LF newlines, and centisecond timestamps (`[mm:ss.xx]`). Text normalization applies Unicode NFKC, regularizes spaces/quotes, and removes selected invisible characters while preserving lyric lines and stanza separators.

The BOM supports encoding detection for Vietnamese, Japanese and English text on classic players. It cannot add glyphs absent from the NW-S644's installed firmware/font, guarantee sharp rendering, or universally prevent `□` placeholders. The automated suite verifies encoding bytes and Unicode preservation; physical NW-S644 display compatibility has not been verified. Check a representative multilingual LRC on your device before processing a full library.

For Vietnamese font compatibility, enable `--strip-vietnamese-accents`. Immediately before LRC writing, the application decomposes text with `unicodedata.normalize('NFD', text)`, removes combining marks (`Mn`), and maps `đ`/`Đ` to `d`/`D`. For example, `Tiếng Việt bão táp` becomes `Tieng Viet bao tap`. Timestamps, `[?]`, punctuation and line breaks are preserved; alignment uses the original accented lyrics. This uses Python's standard library, with no Unikey executable or extra transliteration dependency.

The flag defaults to off. The requested NFD/Mn algorithm removes combining marks in other languages too (including Japanese voiced marks), so use it for Vietnamese material rather than mixed-language lyrics requiring those marks. It improves Vietnamese character compatibility but cannot guarantee every symbol renders on every firmware.

Keep matching audio/LRC basenames, for example:

```text
Music/
  Artist - Song.mp3
  Artist - Song.lrc
```

### Automatic safe TXT cleanup

- LRC output uses exclusive creation, preventing an existing sidecar from being overwritten.
- After the LRC write completes and `lrc_path.is_file()` confirms it exists, the associated TXT is deleted. This includes **pre-existing user-supplied TXT** and case variants such as `.TXT`, as well as newly generated intermediate TXT.
- Alignment failure, LRC write failure, interruption before cleanup, or Skip preserves an existing TXT for the next run. Existing LRC files are skipped.
- Existing TXT contents are not rewritten before cleanup. Keep a separate backup if you want to retain your source lyrics after successful conversion.
- Existence checking is not an atomic transaction or a durability check. An interrupted/failed write may leave a partial LRC; inspect and remove an incomplete output before rerunning, since existing LRC files are skipped.

Avoid concurrent writers and multiple audio files with the same basename in one folder, because they share sidecar paths. Source audio is never overwritten.

## 3. Installation

### Prerequisites and virtual environment

Use **Python 3.10–3.12** and install FFmpeg, including `ffprobe`, on PATH. Debian/Ubuntu:

```sh
sudo apt update
sudo apt install ffmpeg
```

On Windows, install an FFmpeg distribution and add its `bin` directory to PATH. Confirm availability:

```sh
ffmpeg -version
ffprobe -version
```

Create and activate a virtual environment from the repository root:

| Platform | Create | Activate |
| --- | --- | --- |
| Windows PowerShell | `py -3.12 -m venv .venv` | `.\.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `py -3.12 -m venv .venv` | `.venv\Scripts\activate.bat` |
| Linux/macOS | `python3 -m venv .venv` | `. .venv/bin/activate` |

### PyTorch: CPU or CUDA

Install matching PyTorch/torchaudio packages **before** the remaining dependencies. A CPU configuration validated during cloud onboarding is:

```sh
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For an NVIDIA GPU, select a wheel build supported by your driver using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). For example, a matching CUDA 12.8 pair is:

```sh
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

CUDA-enabled PyTorch is required for stable-ts GPU alignment. Faster-whisper additionally uses CTranslate2 and needs compatible NVIDIA CUDA/cuDNN libraries; consult its [GPU requirements](https://github.com/SYSTRAN/faster-whisper#gpu) for your installed version. `torch.cuda.is_available()` alone does not validate CTranslate2. The aligner preserves the model's default dtype rather than forcing `.half()`/`.float()` conversions. Requesting CUDA without available CUDA-enabled PyTorch/GPU raises a clear error.

Optionally install vocal separation:

```sh
python -m pip install demucs
```

Demucs may download its own weights. Installing it on PATH automatically enables separation during alignment; there is no Demucs CLI toggle in this application.

### Model caches and Genius access

Set `HF_HOME` **before starting Python** to place Hugging Face downloads on a secondary drive rather than filling `C:`. Windows PowerShell example:

```powershell
$env:HF_HOME = 'D:\AI_Cache\huggingface'
```

For future PowerShell sessions, persist the non-secret path with:

```powershell
[Environment]::SetEnvironmentVariable('HF_HOME', 'D:\AI_Cache\huggingface', 'User')
```

Linux/macOS example:

```sh
export HF_HOME=/path/to/secondary-drive/AI_Cache/huggingface
```

`HF_HOME` controls the Hugging Face cache used by faster-whisper. It does not relocate every backend's cache: stable-ts loads OpenAI Whisper models, and Demucs has separate caching. Use `--alignment-model` with a local Whisper checkpoint when needed; `--model` accepts a faster-whisper local model directory. The engines use different model formats. Download required weights before offline use.

For Genius search/direct URL lookup, create a Genius API client and set `GENIUS_ACCESS_TOKEN` securely in your shell or environment settings. Never commit tokens. Both API access and lyrics-page access must work. Without the token, existing TXT lyrics remain usable and the CLI offers explicitly consented local Whisper generation.

## 4. CLI reference

### Session hardware selection

If `--device` is omitted in an interactive terminal, the CLI checks `torch.cuda.is_available()`. When CUDA is available, it shows the GPU name and asks once before processing any track:

```text
GPU detected: [GPU Name]. Do you want to process with [1] GPU (fast, high quality) or [2] CPU (default [1]):
```

Press Enter or `1` for CUDA, or `2` for CPU. Invalid answers are requested again before processing starts. The chosen device is reused for every song in that invocation: faster-whisper ASR, stable-ts alignment and Demucs separation all receive it. This choice is not saved between separate CLI runs. GPU refinement may use more VRAM and download additional `htdemucs_ft` weights; speed and output quality depend on the hardware and material.

An explicit `--device cuda` or `--device cpu` overrides automatic selection and never prompts. Without CUDA, or with redirected/captured non-interactive stdin, the default is CPU and no hardware input is requested. If an interactive prompt encounters EOF/unavailable input, it uses its displayed default, GPU. `--scan-only` and scans with no pending tracks skip hardware selection entirely. Existing lyric-approval prompts remain interactive; use explicit device flags for predictable hardware selection in scripts.

The default ASR compute type follows the final selection (`float16` on CUDA, `int8` on CPU); `--compute-type` overrides it. Silero VAD and FFmpeg decoding retain their CPU implementation.

Run commands from the repository root with the virtual environment active:

```sh
python main.py "E:\Music" --scan-only
python main.py "E:\Music" --language vi  # interactive hardware choice
python main.py "E:\Music" --device cuda --language vi
python main.py "E:\Music\song.mp3" --language vi --strip-vietnamese-accents
python main.py "E:\Music\Japanese Album" --device cpu --language ja
python main.py /path/to/Music --language en --use-metadata
python main.py --help
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `location` (required positional) | — | Supported audio **file** or music **folder** to scan recursively |
| `--device` | Interactive choice on CUDA hosts; otherwise `cpu` | Explicit `cpu` or `cuda` for ASR, alignment and Demucs; bypasses hardware prompt |
| `--language` | `en` | Audio language code, for example `en`, `vi`, or `ja` |
| `--model` | `small` | Faster-whisper ASR model name or local model directory; used only on fallback |
| `--alignment-model` | `small` | Stable-ts/OpenAI Whisper model name or local checkpoint |
| `--compute-type` | `int8` on CPU; `float16` on CUDA | Faster-whisper compute type; does not set stable-ts precision |
| `--use-metadata` | Disabled | Prefer embedded artist/title metadata over descriptive filenames |
| `--strip-vietnamese-accents` | Disabled | Strip combining accents and map đ/Đ to d/D in exported LRC text only |
| `--scan-only` | Disabled | List pending tracks without loading models or writing sidecars |
| `-h`, `--help` | — | Show CLI help |

Single-song paths use the same sidecar skipping, metadata, alignment and cleanup logic as folder scans. Unsupported extensions or missing paths produce an argument error. Quote paths containing spaces. Existing LRC sidecars are skipped for both input forms.

Exit status: `0` for completion and normal skip paths, `1` when a track exception is recorded (including declining the final Whisper fallback with no approved lyrics), `130` for interruption/EOF. Argument validation errors use argparse's exit status `2`.

## 5. Quality assurance

**Latest validation: 67/67 tests passed (59 retained tests plus 8 sequential-timing regressions)**, with no failures or skips, on Python 3.12 in the prepared CPU environment. Run the full unit and workflow integration regression suite with:

```sh
python -m unittest discover -s tests -v
```

| Test module | Tests | Coverage |
| --- | ---: | --- |
| `test_compatibility.py` | 5 | Genius constructor API, metadata stream handling/fallback, PyAV decoder compatibility |
| `test_genius_urls.py` | 6 | URL validation, preview/approval, replacement queries, explicit fallback consent |
| `test_gpu_alignment.py` | 3 | Mocked CUDA availability, dtype preservation and alignment options |
| `test_lyric_selection.py` | 3 | Filename queries, Genius bypass of ASR, consented ASR workflow |
| `test_no_speech.py` | 4 | Empty-recognition retry, manual lyrics and Skip |
| `test_timing_fallbacks.py` | 9 | Stanza/cursor logic, Demucs invocation, invalid timing, review flags, exhausted audio and WAV duration |
| `test_walkman_encoding.py` | 3 | Unicode normalization, BOM bytes and exclusive output creation |
| `test_workflow.py` | 6 | Recursive scan, consent, timestamp formatting, LRC output and safe TXT cleanup |
| `test_practical_features.py` | 10 | Single-file scans, invalid paths, accent export, CPU VAD, onset bounds and no-vocal failure |
| `test_sequential_alignment.py` | 8 | Nondecreasing starts within/across stanzas, safe exhausted-audio anchors, equal starts, rounding and suffix-only alignment |
| `test_device_selection.py` | 10 | One-time GPU/CPU prompt, unattended defaults, session-wide ASR/aligner device propagation, Demucs commands/stem paths and failure fallback |
| **Total** | **67** | **Unit and workflow integration regressions** |

Model/API doubles isolate most external operations. The duration test reads an actual generated PCM WAV using Python's `wave` module; it is not an ffprobe integration test. New coverage checks the real bundled Silero model on generated silence on CPU and uses mocked detections to verify vocal-onset cropping and stanza overlap. Existing timing/GPU fixtures mock the onset helper to isolate their original assertions. Passing these tests does not establish live Genius access, real singing/alignment quality, CUDA execution, Demucs separation quality, or physical NW-S644 rendering. Those checks require the relevant dependencies, weights, credentials, hardware and representative audio.

### Implementation map and recent changes

| Module | Responsibility |
| --- | --- |
| `main.py` | CLI, interactive orchestration, optional accent removal, output creation and post-success TXT cleanup |
| `scanner.py` | File/folder track discovery, filename queries and optional metadata |
| `genius_fetcher.py` | Genius approval flow, URL host checks and stanza-preserving text cleanup |
| `comparator.py` | Fallback ASR, decoder compatibility and separately available text-comparison helpers |
| `aligner.py` | Optional vocal separation, CPU Silero detection, sequential stanza alignment, nondecreasing timestamp validation, review flags and BOM writing |

Recent implementation history: `a6c15b9` fixed Genius/PyAV compatibility; `f355429` introduced the stanza engine and review flags; `5c433eb` updated stanza, GPU-option and TXT-cleanup regressions; `40426da` updated ignore rules. This README describes the resulting implementation rather than earlier whole-song alignment/retry behavior.

Commit `7870287` added file-path inputs, the Silero onset guard and optional accent stripping. The experimental anti-jump/recovery logic has been removed; sequential stanza alignment now enforces nondecreasing timestamps, including exhausted-audio fallbacks. Its obsolete recovery tests have been replaced with eight sequential-timing regressions. Hardware selection now propagates a session-wide CPU/CUDA choice to Demucs, ASR and alignment, with a refinement profile for CUDA. The text-comparison heuristics in `comparator.py` remain unchanged.
