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
    """Default tokenizer and detokenizer implementations."""

    def __init__(self):
        self._word_pattern = re.compile(r'\w+')
        self._tokenize_pattern = re.compile(r'\w+|[^\w\s]')
        self._sentence_pattern = re.compile(r'(?<=[.!?])\s+')

    def _tokenize_with_punctuation(self, text):
        """
        Tokenize text, keeping punctuation as separate tokens.

        Parameters
        ----------
        text: string

        Returns
        -------
        list of strings
        """
        # This regex splits on word boundaries but keeps punctuation
        # \w+ matches word characters, [^\w\s] matches punctuation
        tokens = self._tokenize_pattern.findall(text)
        return tokens

    def tokenize(self, text) -> list[str]:
        """
        Tokenize a string using CountVectorizer.

        Parameters
        ----------
            text: String to tokenize

        Returns
        -------
            tokens: List of tokens
            counts: Dictionary mapping tokens to their counts
        """
        return [token.lower() for token in self._tokenize_with_punctuation(text)]

    def detokenize(self, tokens: list[str]) -> str:
        """
        Construct text from the list of tokens.

        Parameters
        ----------
            text: String to tokenize

        Returns
        -------
            tokens: List of tokens
            counts: Dictionary mapping tokens to their counts
        """
        result = []
        for token in tokens:
            if self._word_pattern.match(token):
                result.append(' ' + token)
            else:
                result.append(token)
        text = ''.join(result).strip()
        # Split into sentences and capitalize each one
        sentences = self._sentence_pattern.split(text)
        capitalized_sentences = [s.capitalize() for s in sentences]
        return ' '.join(capitalized_sentences)
