"""Sequential stanza alignment must never export decreasing LRC timestamps."""
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from aligner import Aligner, save_audio_suffix


def result(*times):
    return NS(segments=[NS(start=a, end=b) for a, b in times])


def starts(lrc):
    return [int(row[1:3]) * 60 + float(row[4:9]) for row in lrc.splitlines()]


class SequentialAlignmentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.audio = Path(directory.name) / 'song.wav'
        with wave.open(str(self.audio), 'wb') as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(b'\0\0' * 16000 * 201)
        for target, value in [('prepare_audio', self.audio), ('first_vocal_start', 0.0)]:
            patcher = patch('aligner.' + target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.engine = Aligner.__new__(Aligner)
        self.engine.device = 'cpu'
        self.engine.model = NS(align=Mock())

    def test_backward_starts_within_stanza_are_clamped_and_marked(self):
        self.engine.model.align.return_value = result((20, 22), (14, 16), (18, 19), (25, 27))
        output = self.engine.align(self.audio, 'First\nSecond\nThird\nFourth')
        self.assertEqual(starts(output), [20, 20, 20, 25])
        self.assertIn('[?] Second', output)
        self.assertIn('[?] Third', output)
        self.assertEqual(self.engine.model.align.call_count, 1)

    def test_overlap_cannot_move_next_stanza_backwards(self):
        self.engine.model.align.side_effect = [result((20, 20.1)), result((0, 1))]
        output = self.engine.align(self.audio, 'First\n\nSecond')
        self.assertEqual(starts(output), [20, 20])
        self.assertIn('[?] Second', output)

    def test_exhaustion_cannot_move_200_72_back_to_200_22(self):
        with wave.open(str(self.audio), 'wb') as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(b'\0\0' * 3211520)  # Exactly 200.72 seconds.
        self.engine.model.align.return_value = result((200.72, 200.72))
        with self.assertWarnsRegex(UserWarning, 'Audio nearly exhausted'):
            output = self.engine.align(self.audio, 'First\n\nSecond\nThird')
        self.assertEqual(starts(output), [200.72, 200.72, 200.72])
        self.assertIn('[?] Second', output)
        self.assertEqual(self.engine.model.align.call_count, 1)

    def test_equal_starts_are_allowed(self):
        self.engine.model.align.return_value = result((14, 16), (14, 17))
        output = self.engine.align(self.audio, 'First\nSecond')
        self.assertEqual(starts(output), [14, 14])

    def test_large_forward_jump_is_not_retried_or_estimated(self):
        self.engine.model.align.return_value = result((14, 16), (198, 200))
        output = self.engine.align(self.audio, 'First\nRefrain')
        self.assertEqual(starts(output), [14, 198])
        self.assertEqual(self.engine.model.align.call_count, 1)

    def test_long_segment_is_not_replaced_by_a_pace_estimate(self):
        self.engine.model.align.return_value = result((14, 200))
        output = self.engine.align(self.audio, 'Long verse')
        self.assertEqual(starts(output), [14])
        self.assertEqual(self.engine.model.align.call_count, 1)

    def test_normal_stanzas_use_suffix_and_preserve_half_second_overlap(self):
        self.engine.model.align.side_effect = [result((14, 16)), result((1, 3))]
        with patch('aligner.save_audio_suffix', wraps=save_audio_suffix) as suffix:
            output = self.engine.align(self.audio, 'First\n\nSecond')
        self.assertEqual(starts(output), [14, 16.5])
        self.assertEqual([call.args[2] for call in suffix.call_args_list], [0, 15.5])
        self.assertEqual([call.args[1] for call in self.engine.model.align.call_args_list], ['First', 'Second'])

    def test_centisecond_rounding_preserves_nondecreasing_order(self):
        self.engine.model.align.return_value = result((14.004, 15), (14.003, 15), (14.006, 15))
        output = self.engine.align(self.audio, 'First\nSecond\nThird')
        self.assertEqual(starts(output), [14, 14, 14.01])
