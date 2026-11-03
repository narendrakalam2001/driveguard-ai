"""
api_client.py
=============
Small HTTP client for the DriveGuard FastAPI backend, kept free of Streamlit so it can be unit-tested.

Every failure is raised as :class:`ApiError` with a message an operator can act on (what failed and
what the server said), instead of leaking a raw ``requests`` traceback into the UI.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests


class ApiError(RuntimeError):
    """Raised when the API is unreachable or answers with an error status."""


def normalize_base_url(url: str) -> str:
    """Add ``http://`` when no scheme is given (Render's ``hostport`` value has none) and drop a trailing slash.

    Args:
        url: Raw base URL or ``host:port``.

    Returns:
        A URL with scheme and no trailing slash.
    """
    url = (url or "").strip()
    if not url:
        return "http://localhost:8000"
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url.rstrip("/")


class ApiClient:
    """Thin wrapper over ``requests`` bound to one API base URL (and optional ``X-API-Key``)."""

    def __init__(self, base_url: str, api_key: str = "", timeout: float = 15.0) -> None:
        """Create the client.

        Args:
            base_url: API root, e.g. ``http://localhost:8000``.
            api_key: Value for the ``X-API-Key`` header (only needed if the API sets DRIVEGUARD_API_KEY).
            timeout: Per-request timeout in seconds.
        """
        self.base_url = normalize_base_url(base_url)
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        headers = dict(kwargs.pop("headers", {}) or {})
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        timeout = kwargs.pop("timeout", self.timeout)
        try:
            response = requests.request(method, f"{self.base_url}{path}", headers=headers, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            raise ApiError(f"Cannot reach the API at {self.base_url} ({type(exc).__name__}).") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(f"{method} {path} failed ({response.status_code}): {detail}")
        return response

    def json(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """Call an endpoint and return its decoded JSON body.

        Args:
            method: HTTP method.
            path: Path starting with ``/``.
            params: Query parameters; ``None`` and empty values are dropped.
            **kwargs: Passed to ``requests`` (``json=``, ``data=``, ``files=``, ``timeout=``).

        Raises:
            ApiError: On network failure, error status, or a non-JSON body.
        """
        clean = {k: v for k, v in (params or {}).items() if v not in (None, "", [])}
        response = self._request(method, path, params=clean, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            raise ApiError(f"{method} {path} returned a non-JSON response.") from exc

    def bytes(self, path: str, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> bytes:
        """GET an endpoint and return the raw body (images, PDFs, exports)."""
        clean = {k: v for k, v in (params or {}).items() if v not in (None, "", [])}
        return self._request("GET", path, params=clean, **kwargs).content
