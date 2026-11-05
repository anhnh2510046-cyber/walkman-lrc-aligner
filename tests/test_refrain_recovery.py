"""Regressions for remote chorus anchors and bounded local recovery."""
import tempfile
import unittest
import warnings
import wave
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from aligner import Aligner, confirmed_silence, save_audio_window


def result(*times):
    return NS(segments=[NS(start=a, end=b) for a, b in times])


def starts(lrc):
    return [int(row[1:3]) * 60 + float(row[4:9]) for row in lrc.splitlines()]


class RefrainRecoveryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.audio = Path(directory.name) / 'song.wav'
        with wave.open(str(self.audio), 'wb') as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(b'\0\0' * 16000 * 205)
        for target, value in [('prepare_audio', self.audio),
                              ('first_vocal_start', 0.0),
                              ('vocal_intervals', [(14, 202)])]:
            patcher = patch('aligner.' + target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.engine = Aligner.__new__(Aligner)
        self.engine.model = NS(align=Mock())

    def test_repeated_chorus_never_clumps_remaining_lyrics_at_outro(self):
        self.engine.model.align.side_effect = [
            result((14, 16), (17, 19), (20, 22), (23, 25),
                   (198, 199), (199, 200), (200, 201), (201, 202), (202, 203)),
            None, None, None, None, None,
            result((1, 3), (4, 6)),
        ]
        lyrics = '\n'.join('Lyric line ' + str(i) for i in range(9)) + '\n\nLast one\nLast two'
        with warnings.catch_warnings(record=True) as emitted:
            output = self.engine.align(self.audio, lyrics)
        times = starts(output)
        self.assertEqual(len(times), 11)
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])))
        self.assertLess(max(times), 60)
        self.assertTrue(all('[?]' in line for line in output.splitlines()[4:9]))
        self.assertFalse(any('nearly exhausted' in str(w.message) for w in emitted))
        self.assertEqual(self.engine.model.align.call_count, 7)
        # Every canonical line survives recovery in order.
        self.assertEqual([line.split('] ', 1)[1].removeprefix('[?] ')
                          for line in output.splitlines()], lyrics.replace('\n\n', '\n').splitlines())

    def test_false_anchor_is_replaced_by_successful_local_match(self):
        self.engine.model.align.side_effect = [result((14, 16), (198, 200)), result((2, 4))]
        with patch('aligner.save_audio_window', wraps=save_audio_window) as window:
            output = self.engine.align(self.audio, 'First\nRefrain')
        self.assertEqual(starts(output), [14, 17.5])
        self.assertEqual(window.call_args.args[2:], (15.5, 54.0))
        self.assertEqual(self.engine.model.align.call_args.args[1], 'Refrain')

    def test_real_vad_silence_allows_long_instrumental_break(self):
        self.engine.model.align.return_value = result((14, 16), (198, 200))
        with patch('aligner.vocal_intervals', return_value=[(14, 16), (198, 202)]):
            output = self.engine.align(self.audio, 'First\nRefrain')
        self.assertEqual(starts(output), [14, 198])
        self.assertEqual(self.engine.model.align.call_count, 1)

    def test_intervening_vocals_do_not_confirm_silence(self):
        self.assertFalse(confirmed_silence(16, 198, [(14, 16), (50, 52), (198, 202)]))
        self.assertFalse(confirmed_silence(16, 198, []))
        self.assertFalse(confirmed_silence(16, 198, [(14, 16)]))

    def test_stretched_segment_end_cannot_exhaust_audio(self):
        self.engine.model.align.side_effect = [result((14, 203)), None]
        output = self.engine.align(self.audio, 'Beat masked verse')
        self.assertLess(starts(output)[0], 40)
        self.assertIn('[?]', output)

    def test_recovery_does_not_reuse_an_earlier_anchor(self):
        self.engine.model.align.side_effect = [
            result((14, 16), (198, 200), (15, 17)), None, None,
        ]
        output = self.engine.align(self.audio, 'First\nMasked verse\nNext verse')
        times = starts(output)
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])))
        self.assertEqual(self.engine.model.align.call_count, 3)

    def test_estimate_uses_nearby_vad_onset(self):
        self.engine.model.align.side_effect = [result((14, 16), (198, 200)), None]
        with patch('aligner.vocal_intervals', return_value=[(14, 16), (17, 20), (198, 202)]):
            output = self.engine.align(self.audio, 'First\nMasked verse')
        self.assertEqual(starts(output), [14, 17])
        self.assertIn('[?] Masked verse', output)

    def test_out_of_window_retry_is_discarded(self):
        self.engine.model.align.side_effect = [result((14, 16), (198, 200)), result((180, 182))]
        output = self.engine.align(self.audio, 'First\nRefrain')
        self.assertEqual(starts(output), [14, 16])
        self.assertIn('[?] Refrain', output)

    def test_pcm_window_has_exact_local_duration(self):
        target = self.audio.with_name('window.wav')
        self.assertEqual(save_audio_window(self.audio, target, 15.5, 54), (15.5, 38.5))
        with wave.open(str(target), 'rb') as reader:
            self.assertEqual(reader.getnframes(), 38.5 * 16000)
