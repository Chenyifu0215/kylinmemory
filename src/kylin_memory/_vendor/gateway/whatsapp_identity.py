from __future__ import annotations
import json
import logging
import re
from typing import Set
logger = logging.getLogger(__name__)
_SAFE_IDENTIFIER_RE = re.compile('^[A-Za-z0-9@.+\\-]+$')
from kylin_memory._vendor.kylin_agent_runtime_constants import get_hermes_home

def normalize_whatsapp_identifier(value: str) -> str:
    """Strip WhatsApp JID/LID syntax down to its stable numeric identifier.

    Accepts any of the identifier shapes the WhatsApp bridge may emit:
    ``"60123456789@s.whatsapp.net"``, ``"60123456789:47@s.whatsapp.net"``,
    ``"60123456789@lid"``, or a bare ``"+601****6789"`` / ``"60123456789"``.
    Returns just the numeric identifier (``"60123456789"``) suitable for
    equality comparisons.

    Useful for plugins that want to match sender IDs against
    user-supplied config (phone numbers in ``config.yaml``) without
    worrying about which variant the bridge happens to deliver.
    """
    return str(value or '').strip().replace('+', '', 1).split(':', 1)[0].split('@', 1)[0]

def expand_whatsapp_aliases(identifier: str) -> Set[str]:
    """Resolve WhatsApp phone/LID aliases via bridge session mapping files.

    Returns the set of all identifiers transitively reachable through the
    bridge's ``$HERMES_HOME/whatsapp/session/lid-mapping-*.json`` files,
    starting from ``identifier``. The result always includes the
    normalized input itself, so callers can safely ``in`` check against
    the return value without a separate fallback branch.

    Returns an empty set if ``identifier`` normalizes to empty.
    """
    normalized = normalize_whatsapp_identifier(identifier)
    if not normalized:
        return set()
    session_dir = get_hermes_home() / 'whatsapp' / 'session'
    resolved: Set[str] = set()
    queue = [normalized]
    while queue:
        current = queue.pop(0)
        if not current or current in resolved:
            continue
        if not _SAFE_IDENTIFIER_RE.match(current):
            continue
        resolved.add(current)
        for suffix in ('', '_reverse'):
            mapping_path = session_dir / f'lid-mapping-{current}{suffix}.json'
            if not mapping_path.exists():
                continue
            try:
                mapped = normalize_whatsapp_identifier(json.loads(mapping_path.read_text(encoding='utf-8')))
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug('whatsapp_identity: failed to read %s: %s', mapping_path, exc)
                continue
            if mapped and mapped not in resolved:
                queue.append(mapped)
    return resolved

def canonical_whatsapp_identifier(identifier: str) -> str:
    """Return a stable WhatsApp sender identity across phone-JID/LID variants.

    WhatsApp may surface the same person under either a phone-format JID
    (``60123456789@s.whatsapp.net``) or a LID (``1234567890@lid``). This
    applies to a DM ``chat_id`` *and* to the ``participant_id`` of a
    member inside a group chat — both represent a user identity, and the
    bridge may flip between the two for the same human.

    This helper reads the bridge's ``whatsapp/session/lid-mapping-*.json``
    files, walks the mapping transitively, and picks the shortest
    (numeric-preferred) alias as the canonical identity.
    :func:`gateway.session.build_session_key` uses this for both WhatsApp
    DM chat_ids and WhatsApp group participant_ids, so callers get the
    same session-key identity Hermes itself uses.

    Plugins that need per-sender behaviour (role-based routing,
    authorisation, per-contact policy) should use this so their
    bookkeeping lines up with Hermes' session bookkeeping even when
    the bridge reshuffles aliases.

    Returns an empty string if ``identifier`` normalizes to empty. If no
    mapping files exist yet (fresh bridge install), returns the
    normalized input unchanged.
    """
    normalized = normalize_whatsapp_identifier(identifier)
    if not normalized:
        return ''
    aliases = expand_whatsapp_aliases(normalized)
    return min(aliases, key=lambda candidate: (len(candidate), candidate))

