import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from comparator import Transcriber
from main import enter_manual_lyrics


class NoSpeechTests(unittest.TestCase):
    def test_empty_transcript_retries_without_vad(self):
        transcriber = Transcriber.__new__(Transcriber)
        call = Mock(side_effect=[(iter([]), None),
                                (iter([SimpleNamespace(text=' soft vocals ')]), None)])
        transcriber.model = SimpleNamespace(transcribe=call)
        self.assertEqual(transcriber.transcribe('song.wav', 'vi'), ['soft vocals'])
        self.assertEqual(call.call_count, 2)
        self.assertEqual(call.call_args_list[0].kwargs,
                         dict(language='vi', vad_filter=False, no_speech_threshold=0.8,
                              condition_on_previous_text=False, temperature=(0.0, 0.2, 0.4)))
        self.assertEqual(call.call_args.kwargs['temperature'], 0.2)

    def test_two_empty_attempts_return_empty(self):
        transcriber = Transcriber.__new__(Transcriber)
        transcriber.model = SimpleNamespace(transcribe=Mock(side_effect=[([], None), ([], None)]))
        self.assertEqual(transcriber.transcribe('song.wav'), [])

    def test_manual_entry(self):
        answers = iter(['m', 'First line', 'Second line', ''])
        self.assertEqual(enter_manual_lyrics(ask=lambda _: next(answers), output=lambda _: None),
                         'First line\nSecond line')

    def test_manual_skip(self):
        self.assertIsNone(enter_manual_lyrics(ask=lambda _: 's', output=lambda _: None))
