"""Fallback-only ASR integration and filename-first discovery."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import main
from scanner import Track, read_metadata, filename_query
from test_workflow import FakeProgress


class FallbackWorkflowTests(unittest.TestCase):
    def test_filename_keeps_artist_and_non_track_digits(self):
        self.assertEqual(filename_query('01. 4batz - act ii date @ 8.mp3'),
                         '4batz - act ii date @ 8')
        self.assertEqual(filename_query('02 - Artist - Song.flac'), 'Artist - Song')
        self.assertEqual(filename_query('4batz - act ii.mp3'), '4batz - act ii')
        with patch.dict('sys.modules', {'mutagen': None}):
            self.assertEqual(read_metadata('01 Artist - Song.mp3').title, 'Artist - Song')

    def run_track(self, canonical, answers, asr_expected):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'Artist - Song.wav'
            path.write_bytes(b'original')
            track = Track(path, '', path.stem)
            answers = iter(answers)
            with patch.object(main, 'scan_tracks', return_value=[track]), \
                 patch.object(main, 'resolve_lyrics', return_value=canonical) as resolve, \
                 patch.object(main, 'Transcriber') as transcriber, \
                 patch.object(main, 'Aligner') as aligner, \
                 patch.dict('sys.modules', {'tqdm': SimpleNamespace(tqdm=FakeProgress)}), \
                 patch('builtins.input', side_effect=lambda _: next(answers)):
                transcriber.return_value.transcribe.return_value = ['Generated']
                aligner.return_value.align.return_value = '[00:00.00] Aligned\n'
                self.assertEqual(main.main([directory]), 0)
                resolve.assert_called_once_with('', 'Artist - Song', unittest.mock.ANY)
                if asr_expected:
                    transcriber.assert_called_once()
                else:
                    transcriber.assert_not_called()
                self.assertEqual(aligner.return_value.align.call_args.args[1],
                                 'Generated' if asr_expected else canonical)
                self.assertEqual(path.read_bytes(), b'original')

    def test_genius_bypasses_asr(self):
        self.run_track('Canonical', [], False)

    def test_consented_fallback_uses_asr(self):
        self.run_track(None, ['y'], True)
