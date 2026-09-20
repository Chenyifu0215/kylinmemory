"""Stable pseudonymous identities for encrypted user-profile storage."""

from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Protocol


class KeyProvider(Protocol):
    def get_key(self) -> bytes: ...


def pseudonymous_user_id(
    *,
    platform: str | None,
    platform_user_id: object | None,
    key_provider: KeyProvider,
) -> str:
    """Derive the storage identity shared by runtime and inspection tools."""

    normalized_platform = str(platform or "local").strip().lower()
    identity = (
        f"{normalized_platform}:{platform_user_id}"
        if platform_user_id
        else "local:default"
    )
    digest = hmac.new(
        key_provider.get_key(), identity.encode("utf-8"), hashlib.sha256
    ).digest()
    return "v1_" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
