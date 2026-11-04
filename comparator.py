"""Local transcription and canonical-line comparison."""
from dataclasses import dataclass
import re
import subprocess
import unicodedata


def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    text = re.sub(r'\bstr8\b', 'straight', text)
    return ' '.join(re.findall(r'\w+', text))


@dataclass(frozen=True)
class Comparison:
    canonical: str
    heard: str
    score: float


class Transcriber:
    def __init__(self, model='small', device='cpu', compute_type='int8'):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            model, device=device, compute_type=compute_type
        )

    def transcribe(self, path, language=None):
        audio = str(path)
        options = dict(
            language=language,
            vad_filter=False,
            no_speech_threshold=0.8,
            condition_on_previous_text=False,
            temperature=(0.0, 0.2, 0.4),
        )

        def run(source, **kwargs):
            segments, _ = self.model.transcribe(source, **kwargs)
            # Consume lazy segments here to catch decoder errors too.
            return [
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            ]

        try:
            transcript = run(audio, **options)
        except TypeError as exc:
            # Compatibility fallback for older PyAV versions.
            if (
                'metadata_errors' not in str(exc)
                or 'unexpected keyword' not in str(exc)
            ):
                raise

            import numpy as np

            decoded = subprocess.run(
                [
                    'ffmpeg', '-nostdin', '-v', 'error',
                    '-i', str(path),
                    '-f', 'f32le', '-ac', '1', '-ar', '16000',
                    'pipe:1',
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            audio = np.frombuffer(decoded.stdout, dtype='<f4').copy()
            if not audio.size:
                raise ValueError('FFmpeg decoded no audio samples') from exc

            transcript = run(audio, **options)

        if not transcript:
            transcript = run(
                audio,
                **{**options, 'temperature': 0.2},
            )

        return transcript


def compare_lines(lyrics, transcript):
    """Find ordered segment windows; never silently rewrite canonical lyrics."""
    from rapidfuzz.fuzz import ratio

    rows = []
    cursor = 0
    for line in lyrics.splitlines():
        if not line.strip():
            continue

        candidates = []
        for start in range(cursor, len(transcript)):
            for end in range(
                start + 1, min(start + 4, len(transcript)) + 1
            ):
                heard = ' '.join(transcript[start:end])
                candidates.append(
                    (
                        ratio(normalize(line), normalize(heard)),
                        start,
                        end,
                        heard,
                    )
                )

        if candidates:
            score, start, end, heard = max(
                candidates,
                key=lambda item: (item[0], -item[1], -item[2]),
            )
            # Reuse a segment when Whisper put several canonical lines in it.
            cursor = start
            rows.append(Comparison(line, heard, score))
        else:
            rows.append(Comparison(line, '', 0.0))

    return rows


def show_diff(rows, output=print):
    output('Match | Canonical lyrics | Whisper transcript')
    for row in rows:
        output(
            f'{row.score:5.1f}% | {row.canonical} | {row.heard}'
        )
    output(
        'Canonical spelling and line breaks are preserved; '
        'low scores need review.'
    )