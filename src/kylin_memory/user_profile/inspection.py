"""Developer-facing, read-only inspection helpers for encrypted profiles."""

from __future__ import annotations

from pathlib import Path
from kylin_memory.config import get_hermes_home

from .identity import pseudonymous_user_id
from .models import UserProfile
from .service import ProfileService
from .storage import EncryptedFileProfileStore, FileKeyProvider


def default_profile_home() -> Path:
    """Return the schema-driven profile directory for the active profile."""

    return get_hermes_home() / "user_profile"


def resolve_profile_user_id(
    *,
    profile_home: str | Path | None = None,
    user_id: str | None = None,
    platform: str = "cli",
    platform_user_id: str | None = None,
) -> str:
    """Resolve the same pseudonymous identity used by ``AIAgent``.

    Supplying ``user_id`` bypasses derivation and is useful for low-level
    debugging.  Without a platform user ID, all local frontends resolve to the
    shared ``local:default`` identity, matching the runtime adapter.
    """

    if user_id:
        return user_id
    home = Path(profile_home) if profile_home is not None else default_profile_home()
    key_provider = FileKeyProvider(home / "profile.key")
    return pseudonymous_user_id(
        platform=platform,
        platform_user_id=platform_user_id,
        key_provider=key_provider,
    )


def load_profile(
    *,
    profile_home: str | Path | None = None,
    user_id: str | None = None,
    platform: str = "cli",
    platform_user_id: str | None = None,
    apply_decay: bool = False,
) -> UserProfile:
    """Decrypt and return a profile without creating or modifying any files."""

    home = Path(profile_home) if profile_home is not None else default_profile_home()
    key_provider = FileKeyProvider(home / "profile.key")
    resolved_user_id = resolve_profile_user_id(
        profile_home=home,
        user_id=user_id,
        platform=platform,
        platform_user_id=platform_user_id,
    )
    service = ProfileService(
        EncryptedFileProfileStore(home / "profiles", key_provider)
    )
    return service.profile(resolved_user_id, apply_decay=apply_decay)


def show_profile_markdown(
    *,
    profile_home: str | Path | None = None,
    user_id: str | None = None,
    platform: str = "cli",
    platform_user_id: str | None = None,
    max_chars: int = 4_000,
    min_confidence: float = 0.5,
) -> str:
    """Decrypt and render the active profile exactly as prompt Markdown."""

    home = Path(profile_home) if profile_home is not None else default_profile_home()
    key_provider = FileKeyProvider(home / "profile.key")
    resolved_user_id = resolve_profile_user_id(
        profile_home=home,
        user_id=user_id,
        platform=platform,
        platform_user_id=platform_user_id,
    )
    service = ProfileService(
        EncryptedFileProfileStore(home / "profiles", key_provider)
    )
    return service.profile_prompt(
        resolved_user_id,
        max_chars=max_chars,
        min_confidence=min_confidence,
    )
