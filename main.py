"""Interactive, local Walkman LRC generation CLI."""
import argparse
import os
from pathlib import Path
import sys
import unicodedata

from aligner import Aligner, write_new
from comparator import Transcriber
from genius_fetcher import resolve_lyrics, normalize_lyrics_text
from scanner import AUDIO_EXTENSIONS, scan_tracks


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        'directory', type=Path, metavar='location',
        help='Audio file or Walkman music directory'
    )
    result.add_argument(
        '--model',
        default='small',
        help='Whisper model name or local model path',
    )
    result.add_argument(
        '--alignment-model',
        default='small',
        help='stable-ts model name or checkpoint',
    )
    result.add_argument(
        '--device', default=None, choices=['cpu', 'cuda'],
        help='Processing device (prompt on interactive CUDA hosts; otherwise CPU)',
    )
    result.add_argument(
        '--compute-type',
        default=None,
        help='faster-whisper compute type '
             '(default: float16 on CUDA, int8 on CPU)',
    )
    result.add_argument(
        '--language',
        default='en',
        help='Audio language code (default: en)',
    )
    result.add_argument(
        '--use-metadata',
        action='store_true',
        help='Explicitly prefer ID3 metadata over the full filename',
    )
    result.add_argument(
        '--scan-only',
        action='store_true',
        help='List pending files without loading models',
    )
    result.add_argument(
        '--strip-vietnamese-accents', action='store_true',
        help='Remove combining accents and map đ/Đ to d/D in LRC output',
    )
    return result


def strip_vietnamese_accents(text):
    """Apply Unikey-style accent removal without changing LRC punctuation."""
    text = unicodedata.normalize('NFD', text)
    return ''.join(
        char for char in text if unicodedata.category(char) != 'Mn'
    ).replace('đ', 'd').replace('Đ', 'D')


def select_device(requested=None):
    """Choose once per invocation; explicit flags and unattended runs never ask."""
    if requested is not None:
        return requested
    if not sys.stdin.isatty():
        return 'cpu'

    try:
        import torch
    except ImportError:
        return 'cpu'
    if not torch.cuda.is_available():
        return 'cpu'

    name = torch.cuda.get_device_name(0)
    while True:
        try:
            choice = input(
                f'GPU detected: {name}. Do you want to process with '
                '[1] GPU (fast, high quality) or [2] CPU (default [1]): '
            ).strip()
        except (EOFError, OSError):
            return 'cuda'
        if choice in ('', '1'):
            return 'cuda'
        if choice == '2':
            return 'cpu'
        print('Please enter 1 for GPU or 2 for CPU.')


def enter_manual_lyrics(ask=input, output=print):
    output('Whisper could not recognize this song.')

    while True:
        choice = ask(
            '[M] Enter lyrics manually / [S] Skip: '
        ).strip().lower()
        if choice == 's':
            return None
        if choice == 'm':
            break

    output('Paste lyrics one line at a time; enter a blank line to finish.')
    lines = []

    while True:
        line = ask('> ')
        if not line.strip():
            if lines:
                return '\n'.join(lines)
            output('No lyrics entered; skipping song.')
            return None
        lines.append(line)


def main(argv=None):
    args = parser().parse_args(argv)
    if not (args.directory.is_dir() or (
        args.directory.is_file()
        and args.directory.suffix.lower() in AUDIO_EXTENSIONS
    )):
        parser().error(f'Not a supported audio file or directory: {args.directory}')

    transcriber = aligner = None
    failures = 0

    try:
        tracks = list(
            scan_tracks(args.directory, use_metadata=True)
            if args.use_metadata
            else scan_tracks(args.directory)
        )

        if args.scan_only:
            for track in tracks:
                print(f'{track.path}: {track.artist} — {track.title}')
            return 0

        if not tracks:
            return 0
        args.device = select_device(args.device)
        compute_type = args.compute_type or (
            'float16' if args.device == 'cuda' else 'int8'
        )

        from tqdm import tqdm

        with tqdm(
            tracks,
            desc='Tracks',
            file=sys.stdout,
            dynamic_ncols=True,
        ) as pbar:
            for track in pbar:
                pbar.set_description(f'Track: {track.title[:20]}')

                with tqdm.external_write_mode(file=sys.stdout):
                    try:
                        sidecars = {
                            p.name.casefold(): p
                            for p in track.path.parent.iterdir()
                        }
                        if track.lrc_path.name.casefold() in sidecars:
                            continue

                        print('\n' + '=' * 70)
                        print(
                            f'[PROCESSING] Song: '
                            f'{track.artist} - {track.title}'
                        )
                        print(f'[ORIGINAL FILE] : {track.path.name}')
                        print('=' * 70, flush=True)

                        existing = sidecars.get(
                            track.lyrics_path.name.casefold()
                        )
                        if existing:
                            lyrics = existing.read_text(
                                encoding='utf-8-sig'
                            )
                        else:
                            print(
                                'Genius lookup: enter an alternative title '
                                'or paste a Genius URL when prompted.'
                            )
                            lyrics = resolve_lyrics(
                                track.artist,
                                track.title,
                                os.environ.get('GENIUS_ACCESS_TOKEN'),
                            )

                        if lyrics is not None:
                            lyrics = normalize_lyrics_text(lyrics)
                        if lyrics is not None and not lyrics.strip():
                            raise ValueError('Lyrics file is empty')

                        if lyrics is None:
                            # None means explicit consent to ASR fallback.
                            aligner = None
                            if args.device == 'cuda':
                                import torch
                                torch.cuda.empty_cache()

                            print(
                                '[ASR FALLBACK] Recognizing lyrics '
                                'with faster-whisper...',
                                flush=True,
                            )
                            if transcriber is None:
                                transcriber = Transcriber(
                                    args.model,
                                    args.device,
                                    compute_type,
                                )

                            transcript = transcriber.transcribe(
                                track.path, args.language
                            )

                            if transcript:
                                lyrics = normalize_lyrics_text(
                                    '\n'.join(transcript)
                                )
                                print(lyrics)
                                if input(
                                    'Align these AI-generated lyrics? [y/N]: '
                                ).strip().lower() != 'y':
                                    print(f'Skipped: {track.path}')
                                    continue
                            else:
                                lyrics = enter_manual_lyrics()
                                if lyrics is None:
                                    print(f'Skipped: {track.path}')
                                    continue
                                lyrics = normalize_lyrics_text(lyrics)

                            transcriber = None
                            if args.device == 'cuda':
                                import torch
                                torch.cuda.empty_cache()

                        print(
                            '[ALIGNMENT] Aligning approved lyrics '
                            'with stable-ts...',
                            flush=True,
                        )
                        if aligner is None:
                            aligner = Aligner(
                                args.alignment_model,
                                args.device,
                            )

                        lrc = aligner.align(
                            track.path, lyrics, args.language
                        )

                        if not existing:
                            write_new(
                                track.lyrics_path,
                                lyrics.rstrip() + '\n',
                            )

                        # Exclusive creation: existing LRC files are preserved.
                        if args.strip_vietnamese_accents:
                            lrc = strip_vietnamese_accents(lrc)
                        write_new(track.lrc_path, lrc)
                        print(f'Created: {track.lrc_path}')

                        # A failed write/alignment or a skipped track never reaches cleanup.
                        # Delete TXT only after the LRC write has completed.
                        # Use the actual sidecar path to handle .TXT as well.
                        if track.lrc_path.is_file():
                            cleanup_path = (
                                existing
                                if existing is not None
                                else track.lyrics_path
                            )
                            if cleanup_path.is_file():
                                cleanup_path.unlink(missing_ok=True)
                                print(
                                    'Temporary file cleaned up: '
                                    f'{cleanup_path.name}'
                                )

                    except (EOFError, KeyboardInterrupt):
                        raise
                    except Exception as exc:
                        failures += 1
                        print(
                            f'{track.path}: {type(exc).__name__}: {exc}',
                            file=sys.stderr,
                        )

    except (EOFError, KeyboardInterrupt):
        print(
            '\nStopped; existing audio and sidecars were preserved.',
            file=sys.stderr,
        )
        return 130

    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
