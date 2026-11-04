"""Regression checks for dependency API differences, without downloading models."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from comparator import Transcriber
from genius_fetcher import resolve_lyrics
from scanner import read_metadata


class CompatibilityTests(unittest.TestCase):
    def test_genius_constructor_accepts_only_token(self):
        client = SimpleNamespace(search_song=Mock(return_value=SimpleNamespace(
            lyrics='First line\nSecond line', artist='Artist', title='Song')))

        def token_only(token):
            self.assertEqual(token, 'test-token')
            return client

        with patch.dict('sys.modules', {'lyricsgenius': SimpleNamespace(Genius=token_only)}):
            self.assertEqual(resolve_lyrics('Artist', 'Song', 'test-token',
                                           ask=lambda _: 'y', output=lambda _: None),
                             'First line\nSecond line')
        client.search_song.assert_called_once_with('Song')

    def test_metadata_uses_closed_binary_stream(self):
        streams = []

        def read(stream, easy):
            self.assertEqual(stream.read(), b'audio')
            self.assertTrue(easy)
            streams.append(stream)
            return SimpleNamespace(tags={'artist': ['Tagged'], 'title': ['Title']})

        module = SimpleNamespace(File=read, MutagenError=type('MutagenError', (Exception,), {}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '01 Artist - Song.mp3'
            path.write_bytes(b'audio')
            with patch.dict('sys.modules', {'mutagen': module}):
                track = read_metadata(path, use_metadata=True)
            self.assertEqual((track.artist, track.title), ('Tagged', 'Title'))
            self.assertTrue(streams[0].closed)
            self.assertEqual(path.read_bytes(), b'audio')

    def test_metadata_parse_failure_falls_back_to_filename(self):
        error = type('MutagenError', (Exception,), {})
        module = SimpleNamespace(File=Mock(side_effect=error()), MutagenError=error)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '01 Artist - Song.mp3'
            path.touch()
            with patch.dict('sys.modules', {'mutagen': module}):
                track = read_metadata(path, use_metadata=True)
            self.assertEqual((track.artist, track.title), ('', 'Artist - Song'))

    def test_pyav_keyword_error_retries_with_decoded_audio(self):
        transcriber = Transcriber.__new__(Transcriber)
        transcriber.model = SimpleNamespace(transcribe=Mock(side_effect=[
            TypeError("open() got an unexpected keyword argument 'metadata_errors'"),
            (iter([SimpleNamespace(text=' recognized ')]), None)]))
        audio = SimpleNamespace(size=4)
        array = SimpleNamespace(copy=lambda: audio)
        numpy = SimpleNamespace(frombuffer=Mock(return_value=array))
        with patch.dict('sys.modules', {'numpy': numpy}), \
             patch('comparator.subprocess.run', return_value=SimpleNamespace(stdout=b'pcm')) as run:
            self.assertEqual(transcriber.transcribe(Path('song.mp3'), 'vi'), ['recognized'])
        self.assertEqual(run.call_args.args[0][0], 'ffmpeg')
        self.assertEqual(transcriber.model.transcribe.call_args.args, (audio,))
        self.assertEqual(transcriber.model.transcribe.call_args.kwargs['language'], 'vi')

    def test_unrelated_type_error_is_not_suppressed(self):
        transcriber = Transcriber.__new__(Transcriber)
        transcriber.model = SimpleNamespace(transcribe=Mock(side_effect=TypeError('other bug')))
        with patch('comparator.subprocess.run') as run:
            with self.assertRaisesRegex(TypeError, 'other bug'):
                transcriber.transcribe('song.mp3')
            run.assert_not_called()
