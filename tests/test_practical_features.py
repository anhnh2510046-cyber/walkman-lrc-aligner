"""File inputs, output-only accent conversion, and CPU vocal onset guards."""
import codecs
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import silero_vad

from aligner import Aligner, first_vocal_start, vocal_detector
from main import main, parser, strip_vietnamese_accents
from scanner import AUDIO_EXTENSIONS, scan_tracks
from test_workflow import FakeProgress


class InputTests(unittest.TestCase):
    def test_single_files_and_existing_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix in AUDIO_EXTENSIONS:
                path = root / ('Artist - Song' + suffix.upper())
                path.touch()
                self.assertEqual([t.path for t in scan_tracks(path)], [path])
            (root / 'Artist - Song.LRC').touch()
            self.assertEqual(list(scan_tracks(path)), [])

    def test_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'lyrics.txt'
            path.touch()
            for invalid in (path, path.with_name('missing.mp3')):
                with self.subTest(path=invalid), self.assertRaises(ValueError):
                    list(scan_tracks(invalid))
                with patch('sys.stderr'), self.assertRaises(SystemExit) as error:
                    main([str(invalid)])
                self.assertEqual(error.exception.code, 2)

    def test_single_file_scan_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'Artist - Song.mp3'
            path.touch()
            with patch('builtins.print') as output:
                self.assertEqual(main([str(path), '--scan-only']), 0)
            self.assertIn(str(path), output.call_args.args[0])
            self.assertEqual(list(Path(directory).iterdir()), [path])


class AccentTests(unittest.TestCase):
    def test_accents_and_lrc_structure(self):
        self.assertEqual(strip_vietnamese_accents('Tiếng Việt bão táp'), 'Tieng Viet bao tap')
        self.assertEqual(strip_vietnamese_accents('đĐ Đường'), 'dD Duong')
        text = '[01:02.34] [?] Tiếng Việt — 日本語!\n'
        self.assertEqual(strip_vietnamese_accents(text), '[01:02.34] [?] Tieng Viet — 日本語!\n')
        self.assertFalse(parser().parse_args(['music']).strip_vietnamese_accents)

    def test_conversion_only_changes_export(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                audio = Path(directory) / 'Artist - Song.wav'
                audio.write_bytes(b'original')
                audio.with_suffix('.txt').write_text('Tiếng Việt bão táp', encoding='utf-8')
                with patch('main.Aligner') as model, patch.dict(
                    'sys.modules', {'tqdm': SimpleNamespace(tqdm=FakeProgress)}
                ), patch('builtins.print'):
                    model.return_value.align.return_value = '[00:03.00] [?] Tiếng Việt bão táp\n'
                    argv = [str(audio), '--device', 'cpu'] + (['--strip-vietnamese-accents'] if enabled else [])
                    self.assertEqual(main(argv), 0)
                self.assertEqual(model.return_value.align.call_args.args[1], 'Tiếng Việt bão táp')
                data = audio.with_suffix('.lrc').read_bytes()
                self.assertTrue(data.startswith(codecs.BOM_UTF8))
                expected = 'Tieng Viet bao tap' if enabled else 'Tiếng Việt bão táp'
                self.assertEqual(data.decode('utf-8-sig'), '[00:03.00] [?] ' + expected + '\n')
                self.assertEqual(audio.read_bytes(), b'original')
                self.assertFalse(audio.with_suffix('.txt').exists())


class VocalOnsetTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.audio = Path(directory.name) / 'prepared.wav'
        with wave.open(str(self.audio), 'wb') as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(b'\x00\x00' * 16000 * 10)

    def test_mock_detector_receives_cpu_audio(self):
        def detect(samples, model, **kwargs):
            self.assertEqual(samples.device.type, 'cpu')
            self.assertEqual(len(samples), 160000)
            self.assertEqual(kwargs, dict(sampling_rate=16000, return_seconds=False, speech_pad_ms=0))
            return [{'start': 48000, 'end': 64000}]
        with patch.object(silero_vad, 'get_speech_timestamps', side_effect=detect), patch('aligner.vocal_detector'):
            self.assertEqual(first_vocal_start(self.audio), 3.0)

    def test_real_cpu_model_rejects_silence(self):
        self.assertEqual(next(vocal_detector().parameters()).device.type, 'cpu')
        with self.assertRaisesRegex(ValueError, 'no vocals'):
            first_vocal_start(self.audio)

    def test_invalid_detection_is_rejected(self):
        for start in (-1, 160001, float('nan')):
            with self.subTest(start=start), patch('aligner.vocal_detector'), patch.object(
                silero_vad, 'get_speech_timestamps', return_value=[{'start': start}]
            ), self.assertRaisesRegex(ValueError, 'invalid vocal start'):
                first_vocal_start(self.audio)

    def test_onset_bounds_first_stanza_and_preserves_overlap(self):
        aligner = Aligner.__new__(Aligner)
        aligner.device = 'cpu'
        aligner.model = SimpleNamespace(align=Mock(side_effect=[
            SimpleNamespace(segments=[SimpleNamespace(start=0, end=2)]),
            SimpleNamespace(segments=[SimpleNamespace(start=1, end=3)]),
        ]))
        from aligner import save_audio_suffix
        with patch('aligner.prepare_audio', return_value=self.audio), patch(
            'aligner.first_vocal_start', return_value=3.0
        ), patch('aligner.save_audio_suffix', wraps=save_audio_suffix) as suffix:
            output = aligner.align(self.audio, 'First\n\nSecond')
        self.assertEqual([call.args[2] for call in suffix.call_args_list], [3.0, 4.5])
        self.assertEqual(output, '[00:03.00] First\n[00:05.50] Second\n')

    def test_no_vocals_does_not_call_alignment(self):
        aligner = Aligner.__new__(Aligner)
        aligner.device = 'cpu'
        aligner.model = Mock()
        with patch('aligner.prepare_audio', return_value=self.audio), patch(
            'aligner.first_vocal_start', side_effect=ValueError('no vocals')
        ), self.assertRaisesRegex(ValueError, 'no vocals'):
            aligner.align(self.audio, 'First')
        aligner.model.align.assert_not_called()
