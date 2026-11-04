"""Interactive Genius resolution; transcription fallback always needs consent."""
import re


def clean_lyrics(text):
    lines = text.splitlines()
    if lines and re.search(r'(?:Contributors.*Lyrics| Lyrics)$', lines[0]):
        lines.pop(0)
    text = '\n'.join(lines)
    text = re.sub(r'\d*Embed\s*$', '', text)
    return '\n'.join(line.strip() for line in text.splitlines()
                     if line.strip() and not re.fullmatch(r'\s*\[.*\]\s*', line))


def resolve_lyrics(artist, title, token=None, ask=input, output=print):
    client = None
    if token:
        import lyricsgenius
        client = lyricsgenius.Genius(token, verbose=False, remove_section_headers=True,
                                     retries=2, timeout=15)
    else:
        output('GENIUS_ACCESS_TOKEN is unset; Genius search is unavailable.')
    while client is not None:
        try:
            song = client.search_song(title, artist)
        except Exception as exc:
            # Never print request details, which may contain the access token.
            output(f'Genius search failed ({type(exc).__name__}).')
            song = None
        lyrics = clean_lyrics(song.lyrics) if song else ''
        if lyrics:
            output(f'Candidate: {song.artist} — {song.title}')
            output('\n'.join(lyrics.splitlines()[:4]))
            while True:
                choice = ask('Use these lyrics? [Y]/[N]/[R]: ').strip().lower()
                if choice in {'y', 'n', 'r'}:
                    break
            if choice == 'y':
                return lyrics
        replacement = ask('Alternative song title (blank to stop searching): ').strip()
        if not replacement:
            break
        title = replacement
    if ask('Generate lyrics entirely with local Whisper instead? [y/N]: ').strip().lower() == 'y':
        return None
    raise ValueError('Track skipped: no lyrics approved.')
