from pydantic import BaseModel


class SentencePerplexityContainer(BaseModel):
    """Container for the token-level (pseudo-)perplexities of a sentence."""

    token_ids: list[int]
    token_strings: list[str]
    token_perplexities: list[float]
