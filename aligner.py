"""Forced alignment and exclusive, non-destructive sidecar writing."""
import math
from pathlib import Path


def write_new(path, text):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(text)


def timestamp(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid alignment timestamp')
    centiseconds = int(seconds * 100 + 0.5)
    minutes, remainder = divmod(centiseconds, 6000)
    seconds, fraction = divmod(remainder, 100)
    return f'[{minutes:02d}:{seconds:02d}.{fraction:02d}]'


class Aligner:
    def __init__(self, model='small', device='cpu'):
        import stable_whisper
        self.model = stable_whisper.load_model(model, device=device)

    def align(self, path, lyrics, language='en'):
        lines = [line for line in lyrics.splitlines() if line.strip()]
        result = self.model.align(str(path), '\n'.join(lines), language=language,
                                  original_split=True)
        if result is None or len(result.segments) != len(lines):
            raise ValueError('Alignment did not preserve canonical line count; no LRC written')
        entries = []
        previous = -1.0
        for line, segment in zip(lines, result.segments):
            start = float(segment.start)
            if start < previous:
                raise ValueError('Alignment timestamps are not ordered')
            entries.append(f'{timestamp(start)} {line}')
            previous = start
        if not entries:
            raise ValueError('No lyrics aligned')
        return '\n'.join(entries) + '\n'
