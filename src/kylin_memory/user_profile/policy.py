"""Deterministic validation, privacy boundaries, conflict handling, and decay."""

from __future__ import annotations

import math
from copy import deepcopy
from datetime import datetime
from typing import Iterable

from .models import ApplyResult, Evidence, ProfileEntry, ProfileUpdate, UserProfile, utc_now
from .schema import FIELD_BY_PATH

MIN_INFERRED_CONFIDENCE = 0.70


def _string_list(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    return set()


def _boundary_paths(profile: UserProfile, boundary_path: str) -> set[str]:
    entry = profile.entries.get(boundary_path)
    return _string_list(entry.value) if entry else set()


def _matches_path(path: str, restrictions: Iterable[str]) -> bool:
    return any(path == item or path.startswith(f"{item}.") for item in restrictions)


def apply_updates(
    profile: UserProfile,
    updates: list[ProfileUpdate],
    now: datetime | None = None,
) -> ApplyResult:
    result = ApplyResult()
    timestamp = now or utc_now()

    # Privacy boundary changes take effect before other updates in the same batch.
    ordered = sorted(updates, key=lambda item: not item.path.startswith("boundaries."))
    for update in ordered:
        field = FIELD_BY_PATH[update.path]
        if field.lifecycle == "boundary" and not update.explicit:
            result.rejected.append(update.path)
            continue
        if update.action == "delete":
            if not update.explicit:
                result.rejected.append(update.path)
                continue
            if profile.entries.pop(update.path, None) is not None:
                result.deleted.append(update.path)
            continue
        if update.value is None or not update.evidence_quote.strip():
            result.rejected.append(update.path)
            continue
        if not update.explicit and update.confidence < MIN_INFERRED_CONFIDENCE:
            result.rejected.append(update.path)
            continue
        if not update.path.startswith("boundaries."):
            never_store = _boundary_paths(profile, "boundaries.retention_policies")
            do_not_infer = _boundary_paths(profile, "boundaries.inference_policies")
            if _matches_path(update.path, never_store):
                profile.entries.pop(update.path, None)
                result.rejected.append(update.path)
                continue
            if not update.explicit and _matches_path(update.path, do_not_infer):
                result.rejected.append(update.path)
                continue
        existing = profile.entries.get(update.path)
        if existing and existing.value != update.value and not update.explicit:
            # Explicit/onboarding facts cannot be silently replaced by an inference.
            if existing.source in {"onboarding", "explicit"} or update.confidence <= existing.confidence:
                result.conflicts.append(update.path)
                continue

        source = "explicit" if update.explicit else "inferred"
        created_at = existing.created_at if existing else timestamp
        evidence = list(existing.evidence[-9:]) if existing else []
        evidence.append(Evidence(quote=update.evidence_quote.strip(), observed_at=timestamp))
        profile.entries[update.path] = ProfileEntry(
            value=update.value,
            confidence=1.0 if update.explicit else update.confidence,
            source=source,
            lifecycle=field.lifecycle,
            evidence=evidence,
            created_at=created_at,
            updated_at=timestamp,
        )
        result.applied.append(update.path)

    profile.updated_at = timestamp
    return result


def effective_profile(profile: UserProfile, now: datetime | None = None) -> UserProfile:
    """Return a view with time-decayed dynamic confidence; never mutates storage."""

    timestamp = now or utc_now()
    view = deepcopy(profile)
    for path, entry in view.entries.items():
        field = FIELD_BY_PATH[path]
        if field.lifecycle != "dynamic" or not field.half_life_days or entry.source == "explicit":
            continue
        age_days = max(0.0, (timestamp - entry.updated_at).total_seconds() / 86_400)
        entry.confidence *= math.pow(0.5, age_days / field.half_life_days)
        entry.archived = entry.confidence < 0.25
    return view
