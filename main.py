"""Interactive, local Walkman LRC generation CLI."""
import argparse
import os
from pathlib import Path
import sys

from aligner import Aligner, write_new
from comparator import Transcriber, compare_lines, show_diff
from genius_fetcher import resolve_lyrics
from scanner import scan_tracks


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('directory', type=Path, help='Walkman music directory')
    result.add_argument('--model', default='small', help='Whisper model name or local model path')
    result.add_argument('--alignment-model', default='small', help='stable-ts model name or checkpoint')
    result.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    result.add_argument('--compute-type', default='int8', help='faster-whisper compute type')
    result.add_argument('--language', default='en', help='Audio language code (default: en)')
    result.add_argument('--scan-only', action='store_true', help='List pending files without loading models')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.directory.is_dir():
        parser().error(f'Not a directory: {args.directory}')
    transcriber = aligner = None
    failures = 0
    try:
        tracks = list(scan_tracks(args.directory))
        if args.scan_only:
            for track in tracks:
                print(f'{track.path}: {track.artist} — {track.title}')
            return 0
        from tqdm import tqdm
        for track in tqdm(tracks, desc='Tracks'):
            try:
                sidecars = {p.name.casefold(): p for p in track.path.parent.iterdir()}
                if track.lrc_path.name.casefold() in sidecars:
                    continue
                existing = sidecars.get(track.lyrics_path.name.casefold())
                lyrics = existing.read_text(encoding='utf-8-sig') if existing else resolve_lyrics(
                    track.artist, track.title, os.environ.get('GENIUS_ACCESS_TOKEN'))
                if lyrics is not None and not lyrics.strip():
                    raise ValueError('Lyrics file is empty')
                if transcriber is None:
                    transcriber = Transcriber(args.model, args.device, args.compute_type)
                transcript = transcriber.transcribe(track.path, args.language)
                if not transcript:
                    raise ValueError('Whisper returned no speech')
                if lyrics is None:
                    lyrics = '\n'.join(transcript)
                show_diff(compare_lines(lyrics, transcript))
                if input('Proceed with these canonical lines and alignment? [y/N]: ').strip().lower() != 'y':
                    print(f'Skipped: {track.path}')
                    continue
                if aligner is None:
                    aligner = Aligner(args.alignment_model, args.device)
                lrc = aligner.align(track.path, lyrics, args.language)
                if not existing:
                    write_new(track.lyrics_path, lyrics.rstrip() + '\n')
                write_new(track.lrc_path, lrc)
                print(f'Created: {track.lrc_path}')
            except (EOFError, KeyboardInterrupt):
                raise
            except Exception as exc:
                failures += 1
                print(f'{track.path}: {type(exc).__name__}: {exc}', file=sys.stderr)
    except (EOFError, KeyboardInterrupt):
        print('\nStopped; existing audio and sidecars were preserved.', file=sys.stderr)
        return 130
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
