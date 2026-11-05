"""Regression coverage for stanza alignment and independent review flags."""
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

from aligner import (
    Aligner, acoustic_sanity_check, audio_duration, build_review_flags,
    prepare_audio, split_stanzas,
)
from genius_fetcher import clean_lyrics


def result(*rows):
    return NS(segments=[NS(text=text, start=start, end=end)
                        for text, start, end in rows])


class TimingTests(unittest.TestCase):
    def setUp(self):
        vad = patch('aligner.first_vocal_start', return_value=0.0)
        vad.start()
        self.addCleanup(vad.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.audio = Path(self.directory.name) / 'audio.wav'
        with wave.open(str(self.audio), 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b'\x00\x00' * 16000 * 10)
        prepare = patch('aligner.prepare_audio', return_value=self.audio)
        self.prepare = prepare.start()
        self.addCleanup(prepare.stop)

    def aligner(self, results):
        aligner = Aligner.__new__(Aligner)
        aligner.device = 'cpu'
        aligner.model = NS(align=Mock(side_effect=results))
        return aligner

    def test_missing_segments_rejected_without_shifting_lines(self):
        aligner = self.aligner([result(('First', 1, 2), ('Last', 8, 9))])
        with self.assertRaisesRegex(ValueError, 'returned 2 segments for 3 lines'):
            aligner.align(self.audio, 'First\nMissing\nLast')
        self.assertEqual(aligner.model.align.call_count, 1)

    def test_stanza_boundaries_and_default_nine_line_groups(self):
        lyrics = clean_lyrics('[Verse]\nOne\nTwo\n[Chorus]\nThree\n4Embed')
        self.assertEqual(split_stanzas(lyrics), [['One', 'Two'], ['Three']])
        lines = [str(i) for i in range(20)]
        self.assertEqual(split_stanzas('\n'.join(lines)),
                         [lines[:9], lines[9:18], lines[18:]])
        self.assertEqual(split_stanzas(' \n '), [])

    def test_independent_acoustic_review_checks(self):
        cases = [dict(previous_start=2), dict(next_start=1.5),
                 dict(avg_logprob=-1.0)]
        for options in cases:
            with self.subTest(options=options):
                self.assertTrue(acoustic_sanity_check('One', 1, 3, **options))
        self.assertTrue(acoustic_sanity_check('A long compressed lyric line', 1, 2))
        self.assertTrue(acoustic_sanity_check('one two three four five six', 1, 4,
                                            next_start=3))
        self.assertFalse(acoustic_sanity_check('One', 1, 3, next_start=5))
        # Cluster propagation reaches one neighbor, not the entire song.
        aligned = [('One', t, t + 2, False, None)
                   for t in (0, 4, 4.5, 8, 12)]
        self.assertEqual(build_review_flags(aligned, 16),
                         [True, True, True, True, False])

    def test_stanzas_advance_cursor_and_preserve_canonical_text(self):
        aligner = self.aligner([result(('different', 1, 4)),
                                result(('also different', 1, 3))])
        output = aligner.align(self.audio, 'One\n\nTwo', language='vi')
        self.assertEqual(output, '[00:01.00] One\n[00:04.50] Two\n')
        calls = aligner.model.align.call_args_list
        self.assertEqual([call.args[1] for call in calls], ['One', 'Two'])
        self.assertEqual(calls[1].kwargs, dict(language='vi', original_split=True,
                         vad=False, suppress_silence=True, fast_mode=False,
                         verbose=False))
        self.assertFalse(Path(calls[0].args[0]).exists())

    def test_exhausted_audio_marks_remaining_lines_for_review(self):
        aligner = self.aligner([result(('One', 7, 10))])
        with self.assertWarnsRegex(UserWarning, 'Audio nearly exhausted'):
            output = aligner.align(self.audio, 'One\n\nTwo\nThree')
        self.assertEqual(output, '[00:07.00] [?] One\n[00:09.50] [?] Two\n'
                                 '[00:09.50] [?] Three\n')
        self.assertEqual(aligner.model.align.call_count, 1)

    def test_demucs_prepares_full_track_before_decode(self):
        root = Path(self.directory.name)
        def run(command, **kwargs):
            if command[0] == '/bin/demucs':
                vocals = root / 'htdemucs' / self.audio.stem / 'vocals.wav'
                vocals.parent.mkdir(parents=True)
                vocals.write_bytes(b'vocals')
        with patch('aligner.shutil.which', return_value='/bin/demucs'), \
             patch('aligner.subprocess.run', side_effect=run) as commands:
            decoded = prepare_audio(self.audio, root)
        self.assertEqual(decoded, root / 'alignment-source.wav')
        self.assertEqual(commands.call_count, 2)
        self.assertIn('--two-stems=vocals', commands.call_args_list[0].args[0])
        self.assertEqual(commands.call_args_list[1].args[0][5],
                         str(root / 'htdemucs' / self.audio.stem / 'vocals.wav'))

    def test_wave_duration_real_fixture(self):
        self.assertAlmostEqual(audio_duration(self.audio), 10)

    def test_invalid_timing_rejected(self):
        for start, end in [(float('nan'), 2), (-1, 2), (3, 2), (1, 11)]:
            with self.subTest(start=start, end=end):
                aligner = self.aligner([result(('One', start, end))])
                with self.assertRaisesRegex(ValueError, 'timing outside remaining audio'):
                    aligner.align(self.audio, 'One')

    def test_low_confidence_and_outro_markers(self):
        aligned = [('One', 1, 3, False, -1.0), ('Two', 5, 7, False, None)]
        self.assertEqual(build_review_flags(aligned, 10), [True, False])
        self.assertEqual(build_review_flags(aligned, 20), [True, True])
