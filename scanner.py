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


def read_metadata(path):
    from mutagen import File, MutagenError
    path = Path(path)
    stem = re.sub(r'^\d+[ ._-]+', '', path.stem).replace('_', ' ')
    artist, separator, title = stem.partition(' - ')
    if not separator:
        artist, title = '', stem
    try:
        audio = File(path, easy=True)
        tags = audio.tags if audio is not None else None
        if tags:
            artist = (tags.get('artist') or [artist])[0]
            title = (tags.get('title') or [title])[0]
    except (OSError, ValueError, MutagenError):
        pass
    return Track(path, artist.strip(), title.strip())


def scan_tracks(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'Not a directory: {root}')
    for path in sorted(root.rglob('*')):
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in AUDIO_EXTENSIONS:
            # Compare sidecar names case-insensitively for portable Walkman libraries.
            names = {p.name.casefold() for p in path.parent.iterdir()}
            if path.with_suffix('.lrc').name.casefold() not in names:
                yield read_metadata(path)
