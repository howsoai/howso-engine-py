import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Protocol defining a standard object for tokenizing and detokenizing tokenizable strings."""

    def tokenize(self, text: str, **kwargs) -> list[str]:
        """Tokenize a string."""
        pass

    def detokenize(self, tokens: list[str], **kwargs) -> str:
        """Detokenize a list of tokens into a string."""
        pass


class HowsoTokenizer:
    """This is a naive implementation of a general text tokenizer, detokenizer for the Howso Engine."""

    def __init__(self):
        self._tokenize_pattern = re.compile(r'\w+|[^\w\s]')
        self._sentence_pattern = re.compile(r'(?<=[.!?])\s+')

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase tokens."""
        # findall is already quite fast; using a list comp with .lower()
        return [t.lower() for t in self._tokenize_pattern.findall(text)]

    def detokenize(self, tokens: list[str]) -> str:
        """Reconstruct text from tokens."""
        if not tokens:
            return ""

        result = []
        prev_was_apostrophe_or_hyphen = False

        for token in tokens:
            # Skip non-string tokens (malformed data)
            if not isinstance(token, str):
                continue
            # Quickly check if first char is alphanumeric, it's a word
            if token and token[0].isalnum():
                if prev_was_apostrophe_or_hyphen:
                    result.append(token)
                else:
                    result.append(' ')
                    result.append(token)
                prev_was_apostrophe_or_hyphen = False
            else:
                result.append(token)
                prev_was_apostrophe_or_hyphen = (token in ["'", "-"])

        text = ''.join(result).strip()

        # Capitalize sentences
        sentences = self._sentence_pattern.split(text)
        return ' '.join(s.capitalize() for s in sentences)
