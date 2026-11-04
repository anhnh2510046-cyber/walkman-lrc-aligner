import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
from contextlib import nullcontext

from aligner import Aligner, timestamp, write_new
from genius_fetcher import resolve_lyrics
from scanner import scan_tracks
import main


class FakeProgress:
    def __init__(self, tracks, **kwargs):
        self.tracks = tracks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.tracks)

    def set_description(self, description):
        pass

    @staticmethod
    def external_write_mode(**kwargs):
        return nullcontext()


class WorkflowTests(unittest.TestCase):
    def test_timestamp_carry_and_invalid(self):
        self.assertEqual(timestamp(59.999), '[01:00.00]')
        for value in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                timestamp(value)

    def test_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'song.lrc'
            write_new(path, 'original')
            with self.assertRaises(FileExistsError):
                write_new(path, 'replacement')
            self.assertEqual(path.read_text(encoding='utf-8-sig'), 'original')

    def test_scanner_recursive_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sub').mkdir()
            (root / 'skip.MP3').touch()
            (root / 'skip.LRC').touch()
            pending = root / 'sub' / 'song.flac'
            pending.touch()
            with patch('scanner.read_metadata', side_effect=lambda path: path):
                self.assertEqual(list(scan_tracks(root)), [pending])

    def test_fallback_requires_consent(self):
        with self.assertRaises(ValueError):
            resolve_lyrics('', 'song', ask=lambda _: 'n', output=lambda _: None)
        self.assertIsNone(resolve_lyrics('', 'song', ask=lambda _: 'y', output=lambda _: None))

    def test_cli_existing_txt_bypasses_genius(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'song.wav'
            audio.write_bytes(b'original audio')
            audio.with_suffix('.txt').write_text('Canonical lyrics')
            from scanner import Track
            with patch.object(main, 'scan_tracks', return_value=[Track(audio, '', 'song')]), \
                 patch.object(main, 'resolve_lyrics', side_effect=AssertionError('Genius called')), \
                 patch.object(main, 'Transcriber') as transcriber, \
                 patch.object(main, 'Aligner') as aligner, \
                 patch('builtins.print') as printed, \
                 patch.dict('sys.modules', {'tqdm': SimpleNamespace(tqdm=FakeProgress)}):
                transcriber.side_effect = AssertionError('ASR must not run for TXT')
                aligner.return_value.align.return_value = '[00:00.00] Canonical lyrics\n'
                self.assertEqual(main.main([directory]), 0)
            self.assertEqual(audio.read_bytes(), b'original audio')
            self.assertEqual(audio.with_suffix('.txt').read_text(), 'Canonical lyrics')
            self.assertTrue(audio.with_suffix('.lrc').exists())


if __name__ == '__main__':
    unittest.main()
