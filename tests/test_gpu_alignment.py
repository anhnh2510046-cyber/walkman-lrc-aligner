import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from aligner import Aligner


class GPUAlignmentTests(unittest.TestCase):
    def test_cuda_preserves_default_dtype_and_alignment_options(self):
        model = Mock()
        model.align.return_value = SimpleNamespace(segments=[SimpleNamespace(start=1.0, end=2.0, text='Original')])
        load = Mock(return_value=model)
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        with patch('aligner.prepare_audio'), \
             patch('aligner.save_audio_suffix', return_value=(0, 10)), \
             patch('aligner.audio_duration', return_value=10), patch.dict('sys.modules', {'torch': torch,
                                       'stable_whisper': SimpleNamespace(load_model=load)}):
            aligner = Aligner('small', 'cuda')
            self.assertEqual(aligner.align('audio.wav', 'Original'), '[00:01.00] Original\n')
        load.assert_called_once_with('small', device='cuda')
        model.to.assert_not_called()
        model.half.assert_not_called()
        model.float.assert_not_called()
        self.assertEqual(model.align.call_args.kwargs,
                         dict(language='en', original_split=True, vad=False,
                              suppress_silence=True, fast_mode=False, verbose=False))

    def test_cpu_keeps_full_precision(self):
        model = Mock()
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch('aligner.audio_duration', return_value=10), patch.dict('sys.modules', {'torch': torch,
                                       'stable_whisper': SimpleNamespace(load_model=lambda *a, **kw: model)}):
            Aligner(device='cpu')
        model.to.assert_not_called()

    def test_missing_cuda_is_clear_error(self):
        load = Mock()
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch('aligner.audio_duration', return_value=10), patch.dict('sys.modules', {'torch': torch,
                                       'stable_whisper': SimpleNamespace(load_model=load)}):
            with self.assertRaisesRegex(RuntimeError, 'CUDA requested'):
                Aligner(device='cuda')
        load.assert_not_called()
