"""Interactive Genius resolution with preserved stanza boundaries."""
import re
import unicodedata
from urllib.parse import urlsplit


def normalize_lyrics_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = text.translate(str.maketrans({
        '’': "'",
        '‘': "'",
        '“': '"',
        '”': '"',
    }))

    text = re.sub(r'[\u200B\u200C\u200D\uFEFF\u2060]', '', text)
    text = re.sub(
        r'[\u00A0\u2000-\u200B\u202F\u205F\u3000\uFEFF]',
        ' ',
        text,
    )

    return '\n'.join(
        re.sub(r'[ \t]+', ' ', line).strip()
        for line in text.splitlines()
    )


def clean_lyrics(text):
    """Remove Genius metadata while preserving one blank line per stanza."""
    lines = normalize_lyrics_text(text).strip().splitlines()

    if lines and re.search(
        r'(?:Contributors.*Lyrics| Lyrics)$',
        lines[0],
    ):
        lines.pop(0)

    # Section headings become stanza separators instead of disappearing.
    lines = [
        ''
        if re.fullmatch(r'\s*\[.*\]\s*', line)
        else line
        for line in lines
    ]

    text = '\n'.join(lines)
    text = re.sub(r'\d*Embed\s*$', '', text)

    # Reduce consecutive blank lines to a single blank stanza separator.
    text = re.sub(
        r'\n[ \t]*\n(?:[ \t]*\n)*',
        '\n\n',
        text,
    )
    return text.strip()


def resolve_lyrics(artist, title, token=None, ask=input, output=print):
    client = None
    if token:
        import lyricsgenius

        client = lyricsgenius.Genius(token)
        if hasattr(client, 'verbose'):
            client.verbose = False
    else:
        output('GENIUS_ACCESS_TOKEN is unset; Genius search is unavailable.')

    # Preserve the complete filename query supplied by the scanner.
    query = ' '.join(title.split())

    while client is not None:
        lyrics = ''
        label = query

        try:
            if query.lower().startswith(('http://', 'https://')):
                host = (urlsplit(query).hostname or '').lower()

                if host != 'genius.com' and not host.endswith('.genius.com'):
                    output('Please enter a URL hosted on genius.com.')
                else:
                    output(f'[GENIUS] Fetching URL: {query}')
                    lyrics = clean_lyrics(
                        client.lyrics(song_url=query) or ''
                    )
            else:
                output(
                    f"[GENIUS] Searching for keyword: '{query}'..."
                )
                song = client.search_song(query)
                if song:
                    lyrics = clean_lyrics(song.lyrics or '')
                    label = f'{song.artist} — {song.title}'

        except Exception as exc:
            # Avoid printing request details that might contain credentials.
            output(f'Genius lookup failed ({type(exc).__name__}).')

        if lyrics:
            output(f'[GENIUS FOUND]: {label}')

            # Preview four actual lyric lines without changing stanza breaks.
            preview = [
                line for line in lyrics.splitlines()
                if line.strip()
            ][:4]
            output('\n'.join(preview))

            while True:
                choice = ask(
                    'Use these lyrics? [Y]/[N]/[R]: '
                ).strip().lower()
                if choice in {'y', 'n', 'r'}:
                    break

            if choice == 'y':
                return lyrics

        replacement = ask(
            'Alternative song title / Genius URL '
            '(blank to stop searching): '
        ).strip()
        if not replacement:
            break

        query = replacement

    if ask(
        'Generate lyrics entirely with local Whisper instead? [y/N]: '
    ).strip().lower() == 'y':
        return None

    raise ValueError('Track skipped: no lyrics approved.')