"""Small exception types shared across the app.

Services translate third-party errors (NAMS SDK, OpenAI SDK) into these, so the API layer
and the graph only ever deal with two kinds of failure and never leak raw SDK messages.
"""


import openai


class ConfigError(Exception):
    """A required setting (usually an API key) is missing."""


class UpstreamError(Exception):
    """An external service (NAMS, the LLM, the embedding API) failed.

    ``status_code`` is the HTTP status this app should answer with; ``message`` is safe to show.
    """

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def from_openai_error(exc: Exception, action: str, env_hint: str) -> UpstreamError:
    """Translate an OpenAI-SDK exception (chat or embeddings) into a safe UpstreamError.

    ``env_hint`` names the settings to check, e.g. "LLM_API_KEY / LLM_BASE_URL / LLM_MODEL".
    """
    if isinstance(exc, openai.AuthenticationError):
        return UpstreamError(f"The provider rejected the API key while trying to {action}. Check {env_hint}.", 502)
    if isinstance(exc, openai.RateLimitError):
        return UpstreamError(f"Provider rate limit or quota reached while trying to {action}.", 429)
    if isinstance(exc, openai.NotFoundError):
        return UpstreamError(f"Provider could not find the model/endpoint while trying to {action}. Check {env_hint}.", 502)
    if isinstance(exc, openai.BadRequestError):
        # The provider's 400 text describes the request (bad model/parameter), never the key.
        detail = str(getattr(exc, "message", ""))[:300]
        return UpstreamError(f"The provider rejected the request while trying to {action}: {detail} (check {env_hint})", 502)
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return UpstreamError(f"Could not reach the provider while trying to {action}.", 503)
    return UpstreamError(f"Provider error ({type(exc).__name__}) while trying to {action}.", 502)
