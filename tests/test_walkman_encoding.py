import codecs
from pathlib import Path
import tempfile
import unittest

from aligner import write_new
from genius_fetcher import clean_lyrics


class WalkmanEncodingTests(unittest.TestCase):
    def test_unicode_cleanup_preserves_lyrics_and_lines(self):
        raw = ('[Verse]\nＦｕｌｌ\u00a0\u2002width\t  words\n'
               '‘I’m’ “here”\nze\u200bro\u200cw\u200di\u2060d\ufeffth\n'
               'Tiếng Việt 中文 е\n123Embed')
        self.assertEqual(clean_lyrics(raw),
                         'Full width words\n\'I\'m\' "here"\nzerowidth\nTiếng Việt 中文 е')

    def test_all_requested_spaces_become_ascii(self):
        for character in '\u00a0\u202f\u205f\u3000' + ''.join(map(chr, range(0x2000, 0x200b))):
            with self.subTest(character=repr(character)):
                self.assertEqual(clean_lyrics('a' + character + 'b'), 'a b')

    def test_new_sidecars_have_bom_and_preserve_unicode(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ('.txt', '.lrc'):
                path = Path(directory) / ('song' + suffix)
                text = 'Tiếng Việt 中文\n'
                write_new(path, text)
                self.assertEqual(path.read_bytes(), codecs.BOM_UTF8 + text.encode('utf-8'))
                self.assertEqual(path.read_text(encoding='utf-8-sig'), text)
                with self.assertRaises(FileExistsError):
                    write_new(path, 'replacement')
