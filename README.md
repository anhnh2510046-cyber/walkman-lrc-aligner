# Walkman LRC Aligner

Interactive Python CLI for generating synchronized `.lrc` lyrics beside local audio files, intended for Sony Walkman music libraries, including the NW-S644. Speech recognition and forced alignment run locally; optional Genius lookup and model downloads require Internet access.

## 1. Workflow architecture

```text
Music folder → recursive scan → existing TXT or approved Genius lyrics
                                    ↓ no approved lyrics + explicit consent
                              local Whisper ASR → user approval / manual entry
                                    ↓
Approved lyrics → optional Demucs vocals → FFmpeg PCM decoding
                → stanza-guided stable-ts alignment → review flags → BOM LRC
                → verify LRC exists → remove TXT sidecar
```

### Track discovery and lyric selection

- Recursively scan MP3, FLAC, M4A and WAV files. Skip symbolic-link audio files and tracks with an existing LRC sidecar; extension/sidecar matching is case-insensitive.
- Use the full filename stem as the search query, preserving `Artist - Title` and removing a leading numeric track index such as `01.` or `02 -`. Read metadata for generic/numeric names, or explicitly with `--use-metadata`; parsing failures retain the filename fallback.
- Read an existing UTF-8 TXT sidecar (with or without BOM) directly, bypassing Genius and ASR.
- Otherwise, search Genius, preview four nonempty lyric lines, and require approval. Rejecting a candidate allows another title or a direct `genius.com` URL. Section headings become blank stanza separators; Genius header/footer metadata is removed.
- Only explicit `y` consent enables local faster-whisper ASR after lookup ends. Generated lyrics require another approval. Empty recognition offers manual entry (blank line finishes) or Skip.
- Approved TXT/Genius lyrics proceed directly to alignment. The CLI does not transcribe them again or call the RapidFuzz comparison helpers in `comparator.py`.

The ASR fallback processes the original audio with `vad_filter=False`, `no_speech_threshold=0.8`, `condition_on_previous_text=False`, and temperature fallback `(0.0, 0.2, 0.4)`. An empty result is retried at temperature `0.2`. A specific PyAV `metadata_errors` keyword incompatibility triggers FFmpeg decoding to 16 kHz mono float32 audio before retrying. These settings reduce some failure modes but do not guarantee accurate singing recognition.

### Vocal separation and audio preparation

When the `demucs` executable is on PATH, alignment first separates the **entire track once** using:

```sh
demucs --two-stems=vocals -n htdemucs -d cpu -o <temporary-directory> <audio-file>
```

The `htdemucs` vocal stem reduces instrumental accompaniment and may improve alignment, but complete removal of noise/instruments or improved recognition is not guaranteed. This preparation occurs after lyric selection; it does not preprocess the separate ASR fallback.

Demucs is optional and installed separately. It runs on CPU even when alignment uses CUDA, with a 1,800-second timeout. Failure or missing/empty vocal output emits a warning and falls back to the original audio. FFmpeg then decodes the selected source to **16 kHz mono, signed 16-bit PCM WAV**. Temporary stems and decoded audio are removed after processing; the original audio is not modified.

### Silence handling and stanza-guided alignment

The current stable-ts call uses:

```python
original_split=True
vad=False
suppress_silence=True
fast_mode=False
verbose=False
```

Silence suppression helps adjust timing around silence. **Silero VAD is not enabled in the current pipeline**: alignment uses `vad=False`, and ASR uses `vad_filter=False`. There is no automatic retry with `vad=True` or CLI switch to enable it.

Alignment follows lyric sections rather than submitting all lyrics in one pass:

1. Split normalized lyrics at blank lines; if none exist, group consecutive lines in batches of nine. Musical sections follow supplied stanza breaks rather than automatic section detection.
2. Align each stanza against the remaining audio suffix, preserving its lyric line boundaries.
3. Convert suffix-relative timestamps back to full-track timestamps.
4. Advance the cursor to the last line's end minus **0.5 seconds**, retaining overlap for the next stanza.

This limits the alignment context and aims to reduce accumulated drift. Every nonempty approved line is retained in order. A segment-count mismatch, nonfinite/invalid/out-of-range timing, or alignment that does not advance aborts the track without generating an LRC. If the cursor reaches the final second of audio, all remaining lines receive `max(0, duration - 0.5)` as a fallback timestamp and are marked `[?]`; they are not character-weighted estimates.

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
| Exhausted audio | Remaining lines use the final fallback timestamp |

These are review heuristics, not independent phonetic verification against a second recognition result. Missing confidence is not treated as a measured low score. Segment text differences are not compared by this pass: output uses the approved lyric text. `comparator.py` provides ordered text-window comparisons with RapidFuzz, but the current CLI does not invoke them. Dense vocals, long outros, and languages without space-delimited words can require additional manual judgment. Listen to flagged lines and adjust their timestamps/text before copying the LRC to a player.

## 2. Sony Walkman NW-S644 compatibility and file management

### UTF-8 BOM and multilingual text

New TXT and LRC sidecars use **UTF-8 with BOM (`utf-8-sig`)**, LF newlines, and centisecond timestamps (`[mm:ss.xx]`). Text normalization applies Unicode NFKC, regularizes spaces/quotes, and removes selected invisible characters while preserving lyric lines and stanza separators.

The BOM supports encoding detection for Vietnamese, Japanese and English text on classic players. It cannot add glyphs absent from the NW-S644's installed firmware/font, guarantee sharp rendering, or universally prevent `□` placeholders. The automated suite verifies encoding bytes and Unicode preservation; physical NW-S644 display compatibility has not been verified. Check a representative multilingual LRC on your device before processing a full library.

Keep matching audio/LRC basenames, for example:

```text
Music/
  Artist - Song.mp3