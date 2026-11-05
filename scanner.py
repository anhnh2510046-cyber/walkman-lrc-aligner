"""Discover pending tracks without modifying audio files."""
from dataclasses import dataclass
from pathlib import Path
import re

AUDIO_EXTENSIONS = {'.mp3', '.flac', '.m4a', '.wav'}


@dataclass(frozen=True)
class Track:
    path: Path
    artist: str
    title: str

    @property
    def lyrics_path(self):
        return self.path.with_suffix('.txt')

    @property
    def lrc_path(self):
        return self.path.with_suffix('.lrc')


def filename_query(path):
    """Remove a leading numeric track index, preserving Artist - Title."""
    stem = Path(path).stem
    return re.sub(
        r'^\s*\d+(?:[.)]\s*|\s*-\s*|\s+)',
        '',
        stem,
    ).strip()


def read_metadata(path, use_metadata=False):
    path = Path(path)
    title = filename_query(path)

    meaningless = not title or bool(re.fullmatch(
        r'(?:\d+|(?:track|audio|recording|untitled)[ _-]*\d*)',
        title,
        re.I,
    ))

    if not use_metadata and not meaningless:
        return Track(path, '', title)

    from mutagen import File, MutagenError

    artist = ''
    try:
        with path.open('rb') as stream:
            audio = File(stream, easy=True)
            tags = audio.tags if audio is not None else None
            if tags:
                artist = (tags.get('artist') or [''])[0]
                title = (tags.get('title') or [title])[0]
    except (OSError, ValueError, MutagenError):
        pass

    return Track(path, artist.strip(), title.strip())


def scan_tracks(root, use_metadata=False):
    root = Path(root)
    if root.is_file():
        if root.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f'Unsupported audio file: {root}')
        paths = [root]
    elif root.is_dir():
        paths = sorted(root.rglob('*'))
    else:
        raise ValueError(f'Not an audio file or directory: {root}')

    for path in paths:
        if (
            path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() in AUDIO_EXTENSIONS
        ):
            names = {
                p.name.casefold()
                for p in path.parent.iterdir()
            }
            if path.with_suffix('.lrc').name.casefold() not in names:
                if use_metadata:
                    yield read_metadata(path, use_metadata=True)
                else:
                    yield read_metadata(path)
