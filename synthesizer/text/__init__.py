""" from https://github.com/keithito/tacotron """
from . import cleaners
from .symbols import persian_symbols, english_symbols
import re

# Mappings from symbol to numeric ID and vice versa:
_persian_symbol_to_id = {s: i for i, s in enumerate(persian_symbols)}
_persian_id_to_symbol = {i: s for i, s in enumerate(persian_symbols)}

_english_symbol_to_id = {s: i for i, s in enumerate(english_symbols)}
_english_id_to_symbol = {i: s for i, s in enumerate(english_symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

def text_to_sequence(text, cleaner_name):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
    """

    if cleaner_name == 'persian_cleaner':
        text = text.replace('{', '').replace('}', '')
        sequence = [_persian_symbol_to_id[phonem] for phonem in text.split()]

    elif cleaner_name == 'english_cleaner':
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence += _symbols_to_sequence(_clean_text(text, cleaner_name))
                break
            sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_name))
            sequence += _arpabet_to_sequence(m.group(2))
            text = m.group(3)
    return sequence



def sequence_to_text(sequence, cleaner_name):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        if "persian_cleaner" == cleaner_name:
            if symbol_id in _persian_id_to_symbol:
                s = _persian_id_to_symbol[symbol_id]
                result += s
        elif "english_cleaner" == cleaner_name:
            if symbol_id in _english_id_to_symbol:
                s = _english_id_to_symbol[symbol_id]
                # Enclose ARPAbet back in curly braces:
                if len(s) > 1 and s[0] == "@":
                    s = "{%s}" % s[1:]
                result += s
    return result.replace("}{", " ")


def _clean_text(text, cleaner_name):
    cleaner = getattr(cleaners, cleaner_name)
    if not cleaner:
        raise Exception("Unknown cleaner: %s" % cleaner_name)
    text = cleaner(text)
    return text


def _symbols_to_sequence(symbols, cleaner_name):
    if cleaner_name == "persian_cleaner":
        return [_persian_symbol_to_id[s] for s in symbols if _should_keep_symbol(s, _persian_symbol_to_id)]
    elif cleaner_name == "english_cleaner":
        return [_english_symbol_to_id[s] for s in symbols if _should_keep_symbol(s, _english_symbol_to_id)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(["@" + s for s in text.split()])


def _should_keep_symbol(s, _symbol_to_id):
    return s in _symbol_to_id and s != "_" and s != "~"
