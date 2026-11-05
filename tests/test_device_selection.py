"""Session-wide hardware choice and device-aware Demucs preparation."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main
from aligner import prepare_audio
from test_workflow import FakeProgress


def gpu():
    return SimpleNamespace(cuda=SimpleNamespace(
        is_available=Mock(return_value=True),
        get_device_name=Mock(return_value='Test NVIDIA GPU'), empty_cache=Mock(),
    ))


class DeviceSelectionTests(unittest.TestCase):
    def test_explicit_device_never_prompts_or_probes(self):
        torch = gpu()
        with patch.dict('sys.modules', {'torch': torch}), patch('builtins.input') as ask:
            for device in ('cpu', 'cuda'):
                self.assertEqual(main.select_device(device), device)
        ask.assert_not_called()
        torch.cuda.is_available.assert_not_called()

    def test_unattended_run_defaults_to_cpu_without_input(self):
        torch = gpu()
        with patch('main.sys.stdin.isatty', return_value=False), patch.dict(
            'sys.modules', {'torch': torch}
        ), patch('builtins.input') as ask:
            self.assertEqual(main.select_device(), 'cpu')
        ask.assert_not_called()

    def test_cpu_only_host_does_not_prompt(self):
        torch = gpu()
        torch.cuda.is_available.return_value = False
        with patch('main.sys.stdin.isatty', return_value=True), patch.dict(
            'sys.modules', {'torch': torch}
        ), patch('builtins.input') as ask:
            self.assertEqual(main.select_device(), 'cpu')
        ask.assert_not_called()

    def test_interactive_choices_and_gpu_name(self):
        for answer, expected in [('', 'cuda'), ('1', 'cuda'), ('2', 'cpu')]:
            with self.subTest(answer=answer), patch('main.sys.stdin.isatty', return_value=True), patch.dict(
                'sys.modules', {'torch': gpu()}
            ), patch('builtins.input', return_value=answer) as ask:
                self.assertEqual(main.select_device(), expected)
                ask.assert_called_once()
                self.assertIn('GPU detected: Test NVIDIA GPU.', ask.call_args.args[0])

    def test_invalid_choice_retries_before_processing(self):
        with patch('main.sys.stdin.isatty', return_value=True), patch.dict(
            'sys.modules', {'torch': gpu()}
        ), patch('builtins.input', side_effect=['bad', '2']), patch('builtins.print'):
            self.assertEqual(main.select_device(), 'cpu')

    def test_unavailable_console_uses_prompt_default(self):
        for error in (EOFError(), OSError('captured stdin')):
            with self.subTest(error=type(error)), patch('main.sys.stdin.isatty', return_value=True), patch.dict(
                'sys.modules', {'torch': gpu()}
            ), patch('builtins.input', side_effect=error):
                self.assertEqual(main.select_device(), 'cuda')

    def test_scan_and_empty_folder_do_not_select_hardware(self):
        with tempfile.TemporaryDirectory() as directory, patch('main.select_device') as choose:
            self.assertEqual(main.main([directory, '--scan-only']), 0)
            self.assertEqual(main.main([directory]), 0)
        choose.assert_not_called()

    def test_session_choice_applies_to_all_tracks_and_asr_compute_type(self):
        for answer, device, compute_type in [('1', 'cuda', 'float16'), ('2', 'cpu', 'int8')]:
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                for name in ('Artist - One.wav', 'Artist - Two.wav'):
                    (Path(directory) / name).write_bytes(b'audio')
                with patch('main.sys.stdin.isatty', return_value=True), patch.dict(
                    'sys.modules', {'torch': gpu(), 'tqdm': SimpleNamespace(tqdm=FakeProgress)}
                ), patch('builtins.input', side_effect=[answer, 'y', 'y']) as ask, patch(
                    'main.resolve_lyrics', return_value=None
                ), patch('main.Transcriber') as transcriber, patch('main.Aligner') as aligner, patch('builtins.print'):
                    transcriber.return_value.transcribe.return_value = ['Lyrics']
                    aligner.return_value.align.return_value = '[00:03.00] Lyrics\n'
                    self.assertEqual(main.main([directory]), 0)
                hardware_questions = [c for c in ask.call_args_list if 'GPU detected:' in c.args[0]]
                self.assertEqual(len(hardware_questions), 1)
                self.assertEqual(transcriber.call_count, 2)
                self.assertTrue(all(c.args == ('small', device, compute_type) for c in transcriber.call_args_list))
                self.assertEqual(aligner.call_count, 2)
                self.assertTrue(all(c.args == ('small', device) for c in aligner.call_args_list))


class DemucsDeviceTests(unittest.TestCase):
    def test_cpu_and_gpu_commands_use_matching_stem_paths(self):
        for device, model in [('cpu', 'htdemucs'), ('cuda', 'htdemucs_ft')]:
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                audio = root / 'song.wav'
                audio.write_bytes(b'audio')
                vocals = root / model / 'song' / 'vocals.wav'
                def run(command, **kwargs):
                    if command[0] == 'demucs':
                        vocals.parent.mkdir(parents=True)
                        vocals.write_bytes(b'vocals')
                with patch('aligner.shutil.which', return_value='demucs'), patch(
                    'aligner.subprocess.run', side_effect=run
                ) as commands:
                    self.assertEqual(prepare_audio(audio, root, device), root / 'alignment-source.wav')
                command = commands.call_args_list[0].args[0]
                self.assertEqual(command[command.index('-d') + 1], device)
                self.assertEqual(command[command.index('-n') + 1], model)
                if device == 'cuda':
                    self.assertEqual(command[command.index('--shifts') + 1], '2')
                    self.assertEqual(command[command.index('--overlap') + 1], '0.5')
                else:
                    self.assertNotIn('--shifts', command)
                    self.assertNotIn('--overlap', command)
                self.assertEqual(commands.call_args_list[1].args[0][5], str(vocals))
                self.assertEqual(commands.call_args_list[0].kwargs['timeout'], 1800)

    def test_gpu_demucs_failure_still_decodes_original(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'song.wav'
            source.write_bytes(b'audio')
            with patch('aligner.shutil.which', return_value='demucs'), patch(
                'aligner.subprocess.run', side_effect=[subprocess.CalledProcessError(1, 'demucs'), None]
            ) as commands, self.assertWarnsRegex(UserWarning, 'using original audio'):
                prepare_audio(source, directory, 'cuda')
            self.assertEqual(commands.call_args_list[1].args[0][5], str(source))
