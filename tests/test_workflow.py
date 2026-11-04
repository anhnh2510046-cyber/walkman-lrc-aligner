import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aligner import Aligner, timestamp, write_new
from genius_fetcher import resolve_lyrics
from scanner import scan_tracks
import main


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
            self.assertEqual(path.read_text(), 'original')

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

    def test_alignment_preserves_lines(self):
        aligner = Aligner.__new__(Aligner)
        aligner.model = SimpleNamespace(align=lambda *a, **kw: SimpleNamespace(
            segments=[SimpleNamespace(start=0), SimpleNamespace(start=2.5)]))
        self.assertEqual(aligner.align('song.wav', 'STR8 UP!\nNext'),
                         '[00:00.00] STR8 UP!\n[00:02.50] Next\n')
        with self.assertRaises(ValueError):
            aligner.align('song.wav', 'only one line')

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
                 patch.object(main, 'compare_lines', return_value=[]), \
                 patch('builtins.input', return_value='y'), \
                 patch.dict('sys.modules', {'tqdm': SimpleNamespace(tqdm=lambda tracks, **kw: tracks)}):
                transcriber.return_value.transcribe.return_value = ['heard']
                aligner.return_value.align.return_value = '[00:00.00] Canonical lyrics\n'
                self.assertEqual(main.main([directory]), 0)
            self.assertEqual(audio.read_bytes(), b'original audio')
            self.assertEqual(audio.with_suffix('.txt').read_text(), 'Canonical lyrics')
            self.assertTrue(audio.with_suffix('.lrc').exists())


if __name__ == '__main__':
    unittest.main()
