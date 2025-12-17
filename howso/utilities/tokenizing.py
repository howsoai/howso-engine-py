from typing import Protocol, runtime_checkable

from sacremoses import MosesDetokenizer, MosesTokenizer


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
    """Howso wrapper for the Moses tokenizer."""

    def __init__(self, lang='en'):
        """Initialize the tokenizer."""
        self.lang = lang
        self._mt = MosesTokenizer(lang=self.lang)
        self._md = MosesDetokenizer(lang=self.lang)

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenizes a string into a list of strings.

        Parameters
        ----------
        text : str
            The string to be tokenized.

        Returns
        -------
        list of str
            The tokenized string.
        """
        return self._mt.tokenize(text)

    def detokenize(self, tokens: list[str]) -> str:
        """
        Detokenizes a list of strings into a string.

        Parameters
        ----------
        tokens: list of str
            The list of strings to be recombobulated.

        Returns
        -------
        str
            The recombobulated string.
        """
        return self._md.detokenize(tokens)
