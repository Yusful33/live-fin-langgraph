"""Startup configuration helpers with no external dependencies."""
import os


def _require_env(var: str) -> str:
    """Return the value of an environment variable, raising EnvironmentError if unset or empty.

    Raises EnvironmentError with the variable name so operators can immediately
    identify which credential to fix.
    """
    value = os.getenv(var, "").strip()
    if not value:
        raise EnvironmentError(
            f"{var} is not set. Configure it in your .env file or environment before starting."
        )
    return value
