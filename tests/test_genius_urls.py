import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from genius_fetcher import resolve_lyrics


class GeniusURLTests(unittest.TestCase):
    def resolve(self, client, answers, output=None):
        answers = iter(answers)
        with patch.dict('sys.modules', {'lyricsgenius': SimpleNamespace(Genius=lambda token: client)}):
            return resolve_lyrics('Old Artist', 'Old Title', 'token',
                                  ask=lambda _: next(answers), output=output or (lambda _: None))

    def test_pasted_url_bypasses_search_and_previews_cleaned_lines(self):
        url = 'https://genius.com/Artist-song-lyrics'
        client = SimpleNamespace(search_song=Mock(return_value=None), lyrics=Mock(
            return_value='[Verse]\nOne\nTwo\nThree\nFour\nFive\n123Embed'))
        output = Mock()
        self.assertEqual(self.resolve(client, [url, 'y'], output), 'One\nTwo\nThree\nFour\nFive')
        client.search_song.assert_called_once_with('Old Title')
        client.lyrics.assert_called_once_with(song_url=url)
        output.assert_any_call('One\nTwo\nThree\nFour')
        output.assert_any_call("[GENIUS] Searching for keyword: 'Old Title'...")
        output.assert_any_call(f'[GENIUS FOUND]: {url}')

    def test_reject_url_then_search_independent_artist_title(self):
        song = SimpleNamespace(lyrics='Approved', artist='New', title='Song')
        client = SimpleNamespace(search_song=Mock(side_effect=[None, song]),
                                 lyrics=Mock(return_value='Rejected'))
        self.assertEqual(self.resolve(client, ['http://www.genius.com/song-lyrics', 'r',
                                               'New Artist - New Song', 'y']), 'Approved')
        self.assertEqual(client.search_song.call_args.args, ('New Artist - New Song',))

    def test_single_word_stands_alone(self):
        song = SimpleNamespace(lyrics='Approved', artist='Old Artist', title='New')
        client = SimpleNamespace(search_song=Mock(side_effect=[None, song]))
        self.assertEqual(self.resolve(client, ['New', 'y']), 'Approved')
        self.assertEqual(client.search_song.call_args.args, ('New',))

    def test_multiword_query_does_not_reuse_artist(self):
        song = SimpleNamespace(lyrics='Approved', artist='New', title='Song')
        client = SimpleNamespace(search_song=Mock(side_effect=[None, song]))
        self.assertEqual(self.resolve(client, ['New Song', 'y']), 'Approved')
        self.assertEqual(client.search_song.call_args.args, ('New Song',))

    def test_failed_url_requires_explicit_whisper_consent(self):
        client = SimpleNamespace(search_song=Mock(return_value=None),
                                 lyrics=Mock(side_effect=RuntimeError('failed')))
        with self.assertRaises(ValueError):
            self.resolve(client, ['https://genius.com/song-lyrics', '', 'n'])
        client.search_song.assert_called_once()

    def test_domain_lookalike_is_not_fetched(self):
        client = SimpleNamespace(search_song=Mock(return_value=None), lyrics=Mock())
        with self.assertRaises(ValueError):
            self.resolve(client, ['https://genius.com.example.org/song', '', 'n'])
        client.lyrics.assert_not_called()
