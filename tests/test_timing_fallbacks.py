from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import Mock, patch
import warnings

from aligner import Aligner, line_timings, check_overlap_and_collapse, interpolate_timings, audio_duration


def result(*rows):
    return NS(segments=[NS(text=text, start=start, end=end) for text, start, end in rows])


class TimingTests(unittest.TestCase):
    def test_omitted_line_does_not_shift_later_line(self):
        lines = ['First', 'Missing', 'Last']
        times = line_timings(result(('First', 1, 2), ('Last', 8, 9)), lines)
        self.assertEqual(times, [(1, 2), None, (8, 9)])
        self.assertEqual(interpolate_timings(times, {1}, lines, 10)[1], (2, 8))

    def test_weighted_interpolation_edges_and_all_missing(self):
        self.assertEqual(interpolate_timings([None, None], {0, 1}, ['a', 'abc'], 8),
                         [(0, 2), (2, 8)])
        self.assertEqual(interpolate_timings([None, (5, 6), None], {0, 2}, ['a', 'b', 'c'], 10),
                         [(0, 5), (5, 6), (6, 10)])

    def test_reverse_overlap_and_collapse_flagged(self):
        for times in ([(4, 5), (2, 3)], [(1, 3), (2, 4)], [(1, 1.1), (1.2, 2)]):
            self.assertEqual(check_overlap_and_collapse(times, ['a' * 30, 'b' * 30], 10), {0, 1})
        self.assertEqual(check_overlap_and_collapse([(1, 3), (3.1, 5)], ['a' * 30] * 2, 10), set())

    def aligner(self, results):
        aligner = Aligner.__new__(Aligner)
        aligner.device = 'cpu'
        aligner.model = NS(align=Mock(side_effect=results))
        return aligner

    def test_vad_retry_recovers_without_marking(self):
        aligner = self.aligner([result(), result(('One', 1, 2), ('Two', 3, 4))])
        with patch('aligner.audio_duration', return_value=10), patch('aligner.shutil.which', return_value=None):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                output = aligner.align('song.wav', 'One\nTwo')
        self.assertNotIn('[?]', output)
        self.assertTrue(aligner.model.align.call_args.kwargs['vad'])

    def test_no_demucs_marks_every_missing_line(self):
        aligner = self.aligner([result(), result()])
        with patch('aligner.audio_duration', return_value=8), patch('aligner.shutil.which', return_value=None):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                output = aligner.align('song.wav', 'a\nabc')
        self.assertEqual(output, '[00:00.00] [?] a\n[00:02.00] [?] abc\n')

    def test_demucs_retry_and_temporary_cleanup(self):
        aligner = self.aligner([result(), result(), result(('One', 1, 2))])
        folders = []
        def separate(command, **kwargs):
            folder = Path(command[command.index('-o') + 1])
            folders.append(folder)
            vocals = folder / 'htdemucs' / 'song' / 'vocals.wav'
            vocals.parent.mkdir(parents=True)
            vocals.touch()
        with patch('aligner.audio_duration', return_value=8), \
             patch('aligner.shutil.which', return_value='/bin/demucs'), \
             patch('aligner.subprocess.run', side_effect=separate):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                output = aligner.align('song.wav', 'One')
        self.assertEqual(output, '[00:01.00] One\n')
        self.assertFalse(folders[0].exists())

    def test_ffprobe_real_fixture(self):
        import wave
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audio.wav'
            with wave.open(str(path), 'wb') as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b'\x00\x00' * 16000)
            self.assertAlmostEqual(audio_duration(path), 1)
