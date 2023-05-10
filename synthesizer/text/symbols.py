""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from . import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _pinyin
    + _silences
)

## for Persian Language
persian_phonemes = ['U', 'Q', 'G', 'AA', 'V', 'N', 'CH', 'R', 'KH', 'B', 'Z', 'SH', 'O', 'A', 'E', 'ZH', 'H', 'SIL', 'AH', 'S', 'D', 'J', 'L', 'F', 'K', 'I', 'T', 'P', 'M', 'Y']
persian_phonemes += ['?', '!', '.', ',', ';', ':']
persian_symbols = (
    [_pad]
    + persian_phonemes
    )
