# Walkman LRC Aligner

Interactive Python CLI for creating synchronized lyrics next to Walkman audio.
It follows the constitution and specification in `.specify/memory/constitution.md`
and `specs/001-walkman-lrc/`. Audio processing runs locally; Genius uses its free
API and website. No paid API is used.

## Local installation

Use Python 3.10–3.12 and install FFmpeg on your PATH (for example,
`sudo apt install ffmpeg` on Debian/Ubuntu).

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. CPU is the default; CUDA
requires a compatible PyTorch/CTranslate2 installation and NVIDIA libraries.
The first transcription/alignment run downloads model weights. Subsequent runs
can use cached weights; use `--model` and `--alignment-model` to select local
models/checkpoints for offline operation. These two engines use different model
formats. Model downloads and Genius lookup require Internet access.

For Genius lookup, create a free Genius API client and set
`GENIUS_ACCESS_TOKEN` in your shell/environment. Never commit the token. Without
it, existing TXT lyrics still work and the CLI offers explicitly approved local
Whisper generation. Genius API search may still fail if website scraping is blocked.

## Usage

```sh
python main.py /path/to/Walkman/MUSIC --scan-only
python main.py /path/to/Walkman/MUSIC --language en
python main.py /path/to/Walkman/MUSIC --language vi --model small
```

The scanner recursively finds MP3, FLAC, M4A and WAV files. Tracks with an LRC
sidecar are skipped. Existing UTF-8 TXT sidecars bypass Genius search. Metadata
comes from Mutagen, with a cleaned `Artist - Title` filename fallback.

Genius candidates display four lyric lines and require Y/N/R confirmation.
N/R or a failed search prompts for another title; a blank title ends searching.
Whisper-only lyrics always require explicit consent. The console comparison
matches canonical lines against ordered Whisper segment windows with RapidFuzz.
It normalizes punctuation and `STR8` for comparison while keeping the original
lyrics unchanged. This is a review aid, not a phonetic recognizer: check uncertain
matches yourself. Decline alignment and edit the TXT manually when needed.

After review, stable-ts forces the approved lines onto the audio and writes
`[mm:ss.xx] Lyrics` LRC entries. A line-count mismatch fails without exporting an
LRC. New TXT/LRC files use exclusive creation: existing sidecars and original
audio are never overwritten. A failed LRC write may leave the newly approved TXT
for the next run. Interrupted runs can be restarted. Avoid concurrent writers
or duplicate audio basenames sharing sidecars. Exit status is 1 for track errors,
130 for interruption, and 0 otherwise (including user skips).

## Checks

```sh
python -m unittest discover -s tests -v
python main.py --help
```

Unit tests use model/API doubles; a real audio run additionally requires installed
dependencies, weights, an audio fixture and interactive approval. No models or
credentials are bundled.
