"""Local transcription and canonical-line comparison."""
from dataclasses import dataclass
import re
import unicodedata
import subprocess


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
        self.model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, path, language=None):
        try:
            segments, _ = self.model.transcribe(str(path), language=language, vad_filter=True)
            return [s.text.strip() for s in segments if s.text.strip()]
        except TypeError as exc:
            if 'metadata_errors' not in str(exc) or 'unexpected keyword' not in str(exc):
                raise
            import numpy as np
            decoded = subprocess.run(
                ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path),
                 '-f', 'f32le', '-ac', '1', '-ar', '16000', 'pipe:1'],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            audio = np.frombuffer(decoded.stdout, dtype='<f4').copy()
            if not audio.size:
                raise ValueError('FFmpeg decoded no audio samples') from exc
            segments, _ = self.model.transcribe(audio, language=language, vad_filter=True)
            return [s.text.strip() for s in segments if s.text.strip()]


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
            for end in range(start + 1, min(start + 4, len(transcript)) + 1):
                heard = ' '.join(transcript[start:end])
                candidates.append((ratio(normalize(line), normalize(heard)), start, end, heard))
        if candidates:
            score, start, end, heard = max(candidates, key=lambda x: (x[0], -x[1], -x[2]))
            # Reuse a segment when Whisper put several canonical lines in it.
            cursor = start
            rows.append(Comparison(line, heard, score))
        else:
            rows.append(Comparison(line, '', 0.0))
    return rows


def show_diff(rows, output=print):
    output('Match | Canonical lyrics | Whisper transcript')
    for row in rows:
        output(f'{row.score:5.1f}% | {row.canonical} | {row.heard}')
    output('Canonical spelling and line breaks are preserved; low scores need review.')
