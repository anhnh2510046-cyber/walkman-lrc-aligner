"""Stanza-guided alignment with bidirectional cluster review flags."""
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import warnings
import wave
from functools import lru_cache

from genius_fetcher import normalize_lyrics_text


def write_new(path, text):
    """Create a UTF-8 BOM sidecar without overwriting an existing file."""
    normalized = normalize_lyrics_text(text)

    with Path(path).open('x', encoding='utf-8-sig', newline='\n') as handle:
        handle.write(
            normalized + ('\n' if text.endswith(('\n', '\r')) else '')
        )


def timestamp(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid alignment timestamp')

    centiseconds = int(seconds * 100 + 0.5)
    minutes, remainder = divmod(centiseconds, 6000)
    seconds, fraction = divmod(remainder, 100)
    return f'[{minutes:02d}:{seconds:02d}.{fraction:02d}]'


def split_stanzas(lyrics, group_size=9):
    """Respect blank-line stanza breaks; otherwise use nine-line groups."""
    text = normalize_lyrics_text(lyrics).strip()
    if not text:
        return []

    if re.search(r'\n\s*\n', text):
        return [
            [
                line
                for line in block.splitlines()
                if line.strip()
            ]
            for block in re.split(r'\n\s*\n', text)
            if block.strip()
        ]

    lines = [
        line for line in text.splitlines()
        if line.strip()
    ]
    return [
        lines[i:i + group_size]
        for i in range(0, len(lines), group_size)
    ]


def prepare_audio(path, directory):
    """Optionally isolate the full track once, then decode to PCM WAV."""
    original = Path(path).resolve()
    directory = Path(directory)
    source = original

    demucs = shutil.which('demucs')
    if demucs:
        try:
            subprocess.run(
                [
                    demucs,
                    '--two-stems=vocals',
                    '-n', 'htdemucs',
                    '-d', 'cpu',
                    '-o', str(directory),
                    str(original),
                ],
                check=True,
                capture_output=True,
                timeout=1800,
            )

            vocals = (
                directory
                / 'htdemucs'
                / original.stem
                / 'vocals.wav'
            )
            if not vocals.is_file() or vocals.stat().st_size == 0:
                raise ValueError('Demucs produced no nonempty vocals.wav')

            source = vocals

        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            warnings.warn(
                f'Demucs unavailable ({type(exc).__name__}); '
                'using original audio.',
                stacklevel=2,
            )

    decoded = directory / 'alignment-source.wav'
    subprocess.run(
        [
            'ffmpeg', '-nostdin', '-v', 'error',
            '-i', str(source),
            '-ac', '1',
            '-ar', '16000',
            '-c:a', 'pcm_s16le',
            str(decoded),
        ],
        check=True,
        capture_output=True,
    )

    return decoded


def audio_duration(path):
    with wave.open(str(path), 'rb') as handle:
        duration = handle.getnframes() / handle.getframerate()

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Decoded audio has no samples')

    return duration


@lru_cache(maxsize=1)
def vocal_detector():
    """Reuse the packaged Silero CPU model across tracks."""
    from silero_vad import load_silero_vad
    return load_silero_vad().cpu().eval()


def first_vocal_start(path):
    """Return the first detected speech sample in prepared 16 kHz mono PCM."""
    import torch
    from silero_vad import get_speech_timestamps

    with wave.open(str(path), 'rb') as reader:
        audio_format = (
            reader.getframerate(), reader.getnchannels(), reader.getsampwidth()
        )
        if audio_format != (16000, 1, 2):
            raise ValueError('VAD requires 16 kHz mono 16-bit PCM')
        samples = torch.frombuffer(
            bytearray(reader.readframes(reader.getnframes())), dtype=torch.int16
        ).float() / 32768.0

    with torch.inference_mode():
        segments = get_speech_timestamps(
            samples, vocal_detector(), sampling_rate=16000,
            return_seconds=False, speech_pad_ms=0,
        )
    if not segments:
        raise ValueError('Silero VAD detected no vocals; no LRC generated')
    start = float(segments[0]['start']) / 16000
    if not math.isfinite(start) or not 0 <= start < len(samples) / 16000:
        raise ValueError('Silero VAD returned an invalid vocal start')
    return start


def save_audio_suffix(source, destination, time_cursor):
    """Save cursor-to-EOF audio and return its exact offset and duration."""
    with wave.open(str(source), 'rb') as reader:
        rate = reader.getframerate()
        total = reader.getnframes()
        first = max(0, math.ceil(time_cursor * rate))

        if first >= total:
            raise ValueError('Audio exhausted before all stanzas were aligned')

        reader.setpos(first)

        with wave.open(str(destination), 'wb') as writer:
            writer.setparams(reader.getparams())

            while True:
                frames = reader.readframes(65536)
                if not frames:
                    break
                writer.writeframesraw(frames)

    return first / rate, (total - first) / rate


def segment_logprob(segment):
    """Read optional confidence without inventing absent scores."""
    try:
        value = float(getattr(segment, 'avg_logprob', None))
    except (TypeError, ValueError):
        return None

    return value if math.isfinite(value) else None


def acoustic_sanity_check(
    line,
    start,
    end,
    previous_start=None,
    next_start=None,
    avg_logprob=None,
):
    """Bidirectional local checks; outro detection is applied globally."""
    gap_prev = (
        start - previous_start
        if previous_start is not None
        else None
    )
    gap_next = (
        next_start - start
        if next_start is not None
        else None
    )

    clustered = (
        (gap_prev is not None and gap_prev < 1.2)
        or (gap_next is not None and gap_next < 1.2)
    )

    too_fast = (
        gap_next is not None
        and gap_next < len(line.split()) / 2.8
    )

    compressed = (
        len(line) > 20
        and end - start < 1.8
    )

    low_confidence = (
        avg_logprob is not None
        and avg_logprob < -0.9
    )

    # Negative gaps also detect reversed timing.
    return clustered or too_fast or compressed or low_confidence


def build_review_flags(aligned, duration):
    """Compute initial flags, then propagate cluster seeds one hop."""
    count = len(aligned)
    flags = [False] * count
    cluster_seeds = [False] * count

    for i, item in enumerate(aligned):
        line, start, end, forced_review, confidence = item

        prev_start = aligned[i - 1][1] if i > 0 else None
        next_start = aligned[i + 1][1] if i + 1 < count else None

        flags[i] = forced_review or acoustic_sanity_check(
            line=line,
            start=start,
            end=end,
            previous_start=prev_start,
            next_start=next_start,
            avg_logprob=confidence,
        )

        if i == count - 1 and duration - start > 12.0:
            flags[i] = True

        cluster_seeds[i] = (
            (
                prev_start is not None
                and start - prev_start < 1.5
            )
            or (
                next_start is not None
                and next_start - start < 1.5
            )
        )

    # Use original cluster detections, not flags changed during propagation.
    # This avoids recursively flagging the entire song.
    for i, clustered in enumerate(cluster_seeds):
        if clustered:
            for neighbor in range(
                max(0, i - 1),
                min(count, i + 2),
            ):
                flags[neighbor] = True

    return flags


class Aligner:
    def __init__(self, model='small', device='cpu'):
        import stable_whisper
        import torch

        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA requested, but CUDA-enabled PyTorch/GPU is unavailable'
            )

        self.device = device

        # Preserve default dtype to avoid Float/Half conflicts.
        self.model = stable_whisper.load_model(model, device=device)

    def align(self, path, lyrics, language='en'):
        stanzas = split_stanzas(lyrics)
        if not stanzas:
            raise ValueError('No lyrics to align')

        aligned = []
        time_cursor = 0.0

        with tempfile.TemporaryDirectory(
            prefix='walkman-stanzas-'
        ) as directory:
            root = Path(directory)
            source = prepare_audio(path, root)
            duration = audio_duration(source)
            vocal_start = first_vocal_start(source)
            time_cursor = vocal_start

            for number, lines in enumerate(stanzas, start=1):
                if time_cursor >= duration - 1.0:
                    anchor = max(vocal_start, duration - 0.5)

                    warnings.warn(
                        'Audio nearly exhausted; remaining lyrics use '
                        'the final fallback timestamp and are marked [?].',
                        stacklevel=2,
                    )

                    aligned.extend(
                        (line, anchor, anchor, True, None)
                        for stanza in stanzas[number - 1:]
                        for line in stanza
                    )
                    break

                suffix = root / 'remaining-audio.wav'
                offset, remaining = save_audio_suffix(
                    source,
                    suffix,
                    time_cursor,
                )

                result = self.model.align(
                    str(suffix),
                    '\n'.join(lines),
                    language=language,
                    original_split=True,
                    vad=False,
                    suppress_silence=True,
                    fast_mode=False,
                    verbose=False,
                )

                segments = (
                    list(result.segments)
                    if result is not None
                    else []
                )

                if len(segments) != len(lines):
                    raise ValueError(
                        f'Stanza {number}: returned {len(segments)} '
                        f'segments for {len(lines)} lines. '
                        'Review stanza breaks; no LRC generated.'
                    )

                stanza_times = []
                stanza_confidences = []

                for i, segment in enumerate(segments):
                    try:
                        start = float(segment.start)
                        end = float(segment.end)
                    except (AttributeError, TypeError, ValueError) as exc:
                        raise ValueError(
                            f'Stanza {number}, line {i + 1}: invalid timing'
                        ) from exc

                    if not (
                        math.isfinite(start)
                        and math.isfinite(end)
                        and 0 <= start <= end <= remaining + 0.05
                    ):
                        raise ValueError(
                            f'Stanza {number}, line {i + 1}: '
                            'timing outside remaining audio'
                        )

                    start = min(start, remaining)
                    end = min(end, remaining)

                    stanza_times.append(
                        (offset + start, offset + end)
                    )
                    stanza_confidences.append(
                        segment_logprob(segment)
                    )

                if max(end for _, end in stanza_times) <= offset:
                    raise ValueError(
                        f'Stanza {number}: alignment did not advance; '
                        'review lyrics'
                    )

                aligned.extend(
                    (line, start, end, False, confidence)
                    for line, (start, end), confidence in zip(
                        lines,
                        stanza_times,
                        stanza_confidences,
                    )
                )

                # Retain half a second of overlap for the next stanza.
                last_line_end = stanza_times[-1][1]
                time_cursor = max(
                    time_cursor,
                    max(0.0, last_line_end - 0.5),
                )

        flags = build_review_flags(aligned, duration)

        entries = []
        for i, item in enumerate(aligned):
            line, start, end, forced_review, confidence = item
            marker = '[?] ' if flags[i] else ''
            entries.append(f'{timestamp(start)} {marker}{line}')

        return '\n'.join(entries) + '\n'
