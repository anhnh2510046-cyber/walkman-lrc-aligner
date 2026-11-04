# Walkman LRC Aligner

Interactive local audio alignment for Sony Walkman libraries. Genius lookup uses
its free API/website; speech recognition and forced alignment run locally.

## Install

Use Python 3.10–3.12 and install FFmpeg **and ffprobe** on PATH. For example,
`sudo apt install ffmpeg` on Debian/Ubuntu.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows activate with `.venv\Scripts\activate`. Set `GENIUS_ACCESS_TOKEN` in
your environment for Genius search; never commit tokens. Model downloads and
Genius requests need Internet access. Cache/download models before offline use.
CUDA requires compatible PyTorch, CTranslate2 and NVIDIA libraries.

```sh
python main.py "E:\Music" --scan-only
python main.py "E:\Music" --device cuda --language vi
python main.py "E:\Music" --use-metadata
```

## Workflow

- Recursively scan MP3, FLAC, M4A and WAV, skipping existing LRC sidecars.
- Search by the full filename stem, preserving `Artist - Title`. Remove only a
  leading numeric track index such as `01.` or `02 -`. Metadata is read only for
  generic/numeric names or with `--use-metadata`.
- Existing UTF-8 TXT lyrics bypass Genius. Otherwise preview and approve Genius
  lyrics, retry another title, or paste a Genius URL for direct page lookup.
- Approved Genius/TXT lyrics go **directly to alignment**. No ASR or RapidFuzz
  comparison runs on this path.
- Only after Genius lookup is exhausted, explicit Y consent enables Whisper
  ASR. Its output must also be approved. Empty recognition offers manual entry
  (blank line finishes) or skip. Disabling previous-text conditioning and using
  temperature fallback reduces repetition but cannot guarantee accurate lyrics.

The default alignment model is `small`; override with `--alignment-model`.
The aligner first tries fast alignment, retries suspect timings with `vad=True`,
and then uses Demucs vocal isolation if its executable is installed on PATH.
Install Demucs separately with `python -m pip install demucs` if desired; it may
need a model download. Separation runs on CPU to limit GPU memory competition,
uses a temporary directory, and never modifies the input audio.

Timing validation flags missing/invalid times, reversed or overlapping segments,
and long phrases (20 non-space characters) starting less than 0.3 seconds apart.
This is a heuristic: dense legitimate vocals can also trigger it. A short silence
between otherwise normal phrases is not treated as collapse.

Every nonempty canonical lyric line is preserved. Unresolved lines receive
character-weighted timing estimates between the previous valid **end** and next
valid **start**, using audio start/end for boundary gaps. Estimated entries use
`[mm:ss.xx] [?] Lyrics` for manual review. These are estimates, not measured word
timings. Text mismatches are marked rather than shifting later lines.

New TXT/LRC outputs use UTF-8 with BOM and normalized Unicode/spaces. Exclusive
creation prevents overwriting existing audio or sidecars. Existing TXT remains
unchanged; normalization applies to the alignment text and new outputs. BOM
helps encoding detection but cannot add glyphs missing from Walkman firmware.
Avoid concurrent writers or different audio files sharing the same basename.

Exit codes: 0 for completion/user skips, 1 for track errors, 130 for interruption.

## Validation

```sh
python -m unittest discover -s tests -v
python main.py --help
```

Tests use model/API doubles plus an actual ffprobe duration check. Real Genius,
Whisper, VAD, GPU and Demucs quality checks require their dependencies/models and
representative audio; models and credentials are not bundled.
