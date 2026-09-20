"""L1 Atom and L2 Scenario memory for the local runtime.

This module is deliberately dependency-light.  L1 is an append-only JSONL
audit stream backed by SQLite/FTS5 for current-version retrieval.  L2 is a
scope-isolated set of Markdown scene blocks with a small JSON index.  LLM
extraction/consolidation can be plugged in later; deterministic callers can
already validate, persist, retrieve, and consolidate records safely.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import quote
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kylin_memory.config import get_hermes_home
from kylin_memory.clock import now as _hermes_now

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Embedding generation failed; callers should retain lexical memory."""


def _normalize_embedding(values: Sequence[float], dimensions: int | None = None) -> list[float]:
    """Validate and L2-normalize one finite embedding vector."""
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise EmbeddingError("embedding contains non-numeric values") from exc
    if dimensions is not None and len(vector) != int(dimensions):
        raise EmbeddingError(
            f"embedding dimension mismatch: expected {int(dimensions)}, got {len(vector)}"
        )
    if not vector or any(not math.isfinite(value) for value in vector):
        raise EmbeddingError("embedding contains non-finite values")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise EmbeddingError("embedding norm must be positive")
    return [value / norm for value in vector]


class OpenAICompatibleEmbedding:
    """Small ``/v1/infer`` client for L1 Atom vectors.

    The client deliberately uses the standard library so L1 remains optional
    and does not add another SDK dependency.  The inference service accepts
    one text per request and returns an OpenAI-style embedding list nested
    under ``output``.  Vectors are normalized before persistence.
    """

    provider_name = "infer"

    def __init__(self, *, model: str, dimensions: int | str | None, base_url: str,
                 api_key: str = "", timeout: float = 10.0, batch_size: int = 32):
        self.model = str(model or "").strip()
        raw_dimensions = dimensions
        self._dimensions_auto = raw_dimensions is None or str(raw_dimensions).strip().lower() == "auto"
        if self._dimensions_auto:
            self.dimensions: int | None = None
        else:
            self.dimensions = int(raw_dimensions)
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.api_key = str(api_key or "").strip()
        self.timeout = min(max(float(timeout), 0.1), 120.0)
        self.batch_size = max(1, min(int(batch_size), 128))
        if not self.model or (self.dimensions is not None and self.dimensions <= 0) or not re.match(r"^https?://", self.base_url, re.I):
            raise EmbeddingError("invalid OpenAI-compatible embedding configuration")

    @property
    def model_id(self) -> str:
        """Return a stable model identity once dimensions are known.

        ``dimensions: auto`` cannot participate in the identity until the
        first successful embedding response tells us the vector size.
        """
        dimensions = self.dimensions if self.dimensions is not None else "auto"
        digest = hashlib.sha256(
            f"{self.provider_name}\0{self.base_url}\0{self.model}\0{dimensions}".encode()
        ).hexdigest()[:24]
        return f"atom_remote_{digest}"

    @property
    def endpoint(self) -> str:
        lower_url = self.base_url.lower()
        if lower_url.endswith("/v1/infer"):
            return self.base_url
        if lower_url.endswith("/v1"):
            return f"{self.base_url}/infer"
        return f"{self.base_url}/v1/infer"

    def _request(self, texts: Sequence[str]) -> list[list[float]]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        vectors: list[Any] = []
        for text in texts:
            payload = {"model": self.model, "input": {"text": str(text)}}
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                raise EmbeddingError("embedding inference request failed") from exc
            output = body.get("output") if isinstance(body, dict) else None
            data = output.get("data") if isinstance(output, dict) else None
            if not isinstance(data, list) or len(data) != 1:
                raise EmbeddingError("embedding response length is invalid")
            item = data[0]
            if not isinstance(item, dict) or "embedding" not in item:
                raise EmbeddingError("embedding response item is invalid")
            if "index" in item and item["index"] != 0:
                raise EmbeddingError("embedding response indexes are invalid")
            vectors.append(item["embedding"])
        dimensions = self.dimensions
        if dimensions is None:
            try:
                dimensions = len(vectors[0])
            except (IndexError, TypeError) as exc:
                raise EmbeddingError("embedding response vector is invalid") from exc
            if dimensions <= 0:
                raise EmbeddingError("embedding response vector is empty")
        normalized = [_normalize_embedding(vector, dimensions) for vector in vectors]
        # Only commit the detected dimension after the complete response has
        # validated successfully; a malformed batch must not poison the
        # client for subsequent retries.
        if self.dimensions is None:
            self.dimensions = dimensions
        return normalized

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        values = [str(text) for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(values), self.batch_size):
            vectors.extend(self._request(values[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        result = self.embed_documents([text])
        if not result:
            raise EmbeddingError("empty query embedding")
        return result[0]


def create_atom_embedding(config: Mapping[str, Any] | None) -> OpenAICompatibleEmbedding | None:
    """Create the configured ``/v1/infer`` L1 embedding service.

    ``mode`` may be ``remote``/``openai-compatible`` (``disabled`` remains
    the safe default).  ``dimensions: auto`` is resolved from the first
    response.  Secrets are resolved from the environment only; an explicitly
    empty ``api_key_env`` enables an unauthenticated local endpoint.
    """
    cfg = dict(config or {})
    mode = str(cfg.get("mode", "disabled") or "disabled").strip().lower()
    if mode not in {"remote", "infer", "openai", "openai-compatible", "compact"}:
        return None
    model = str(cfg.get("model") or os.getenv("OPENAI_EMBEDDING_MODEL") or "qwen-embedding")
    base_url = str(
        cfg.get("base_url")
        or os.getenv("OPENAI_EMBEDDING_BASE_URL")
        or "http://127.0.0.1:18080/v1/infer"
    )
    # Existing installations may still have the former defaults persisted in
    # config.yaml.  Make them work before the v24 config migration is run,
    # while leaving every custom model/endpoint combination unchanged.
    if (
        model == "Qwen3-Embedding-0.6B-Q8_0"
        and base_url.rstrip("/") in {
            "http://127.0.0.1:22370",
            "http://127.0.0.1:22370/v1",
            "http://127.0.0.1:22370/v1/embeddings",
        }
    ):
        model = "qwen-embedding"
        base_url = "http://127.0.0.1:18080/v1/infer"
    # An explicitly empty api_key_env enables anonymous local servers.  Keep
    # the historical OPENAI_API_KEY fallback only when neither key field was
    # supplied at all.
    if "api_key_env" in cfg:
        key_env = str(cfg.get("api_key_env") or "").strip()
    elif "key_env" in cfg:
        key_env = str(cfg.get("key_env") or "").strip()
    else:
        key_env = "OPENAI_API_KEY"
    api_key = os.getenv(key_env, "") if key_env else ""
    if not api_key and key_env:
        try:
            # Reuse the runtime's dotenv/profile secret resolver without
            # storing credentials in memory configuration.
            from kylin_memory.config import get_env_value

            api_key = str(get_env_value(key_env) or "")
        except Exception:
            pass
    # A non-empty key environment remains required for hosted services.  An
    # explicitly empty api_key_env is the opt-in anonymous/local mode.
    if not api_key and key_env:
        return None
    try:
        dimensions = cfg["dimensions"] if "dimensions" in cfg else 1536
        return OpenAICompatibleEmbedding(model=model, dimensions=dimensions, base_url=base_url,
                                         api_key=api_key, timeout=float(cfg.get("timeout", 10.0)),
                                         batch_size=int(cfg.get("batch_size", 32)))
    except (TypeError, ValueError, EmbeddingError):
        logger.info("L1 embedding configuration unavailable; using lexical retrieval")
        return None

ATOM_TYPES = frozenset({"persona", "episodic", "instruction", "work_fact", "work_task", "work_method", "work_artifact"})
CHAT_TYPES = frozenset({"persona", "episodic", "instruction"})
CODE_TYPES = frozenset({"work_fact", "work_task", "work_method", "work_artifact"})
TYPE_ALIASES = {"episode": "episodic", "instruct": "instruction", "preference": "persona"}
PRIORITY_MIN = {"persona": 50, "episodic": 60, "instruction": 70, "work_fact": 70, "work_task": 70, "work_method": 70, "work_artifact": 70}
ALLOWED_METADATA = {
    "persona": set(), "episodic": {"activity_start_time", "activity_end_time"},
    "instruction": set(), "work_fact": {"work_object", "status", "activity_start_time", "activity_end_time"},
    "work_task": {"owner", "deadline", "status"}, "work_method": {"scope", "method_type"},
    "work_artifact": {"artifact_type", "artifact_ref"},
}
_NOISE = re.compile(r"^\s*(?:/\S+|NO_REPLY|bootstrap|session reset|system note)\b", re.I)
_XML = re.compile(r"</?\s*(?:memory|persona|scene|user[-_ ]profile)[^>]*>", re.I)
_SAFE_FILENAME = re.compile(r"[^\w\-.\u3400-\u9fff]+", re.UNICODE)


def utc_iso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _scene_iso() -> str:
    """Return an aware L2 metadata timestamp in the user's wall-clock zone."""
    return _hermes_now().isoformat()


def _localize_legacy_scene_timestamp(value: str) -> str:
    """Render a legacy UTC L2 timestamp in the current Hermes timezone."""
    raw = str(value or "").strip()
    if not (raw.endswith("Z") or raw.endswith("+00:00")):
        return raw
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.astimezone(_hermes_now().tzinfo).isoformat()
    except (TypeError, ValueError):
        return raw


def normalize_scope(team_id: str | None = None, agent_id: str | None = None, user_id: str | None = None, *, global_compat: bool = False) -> str:
    if global_compat and not any((team_id, agent_id, user_id)):
        return "global"
    team = str(team_id or user_id or "default")
    agent = str(agent_id or "default")
    return f"team:{team}|agent:{agent}"


def _safe_scope(scope: str) -> str:
    encoded = quote(str(scope), safe="-_.!~*'()")
    if len(encoded) <= 180:
        return encoded
    return encoded[:140] + "-" + hashlib.sha256(str(scope).encode("utf-8")).hexdigest()[:24]


def sanitize_l0_text(value: Any) -> str:
    if isinstance(value, list):
        value = "\n".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in value)
    text = _XML.sub(" ", str(value or ""))
    text = re.sub(r"\x00|data:image/[^;]+;base64,[^\s]+", " ", text, flags=re.I)
    text = re.sub(r"^\s*\[[^\]]*(?:gateway|telegram|discord|timestamp)[^\]]*\]\s*", "", text, flags=re.I)
    return " ".join(text.split()).strip()


def should_extract_l1(content: str) -> bool:
    text = sanitize_l0_text(content)
    if not text or _NOISE.search(text) or text in {"?", "？"}:
        return False
    if len(text) <= 5 and not any(ch.isalnum() for ch in text):
        return False
    return True


def _now_ms() -> int:
    return int(time.time() * 1000)


def new_atom_id() -> str:
    return f"m_{_now_ms()}_{secrets.token_hex(4)}"


@dataclass(slots=True)
class Atom:
    id: str
    content: str
    type: str
    priority: int
    scene_name: str = ""
    source_message_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamps: list[str] = field(default_factory=list)
    createdAt: str = ""
    updatedAt: str = ""
    version: int = 0
    sessionKey: str = ""
    sessionId: str = ""
    taskId: str = ""
    teamId: str = ""
    userId: str = ""
    agentId: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, mode: str = "chat", new_id: bool = True) -> "Atom":
        mode = str(mode or "chat").strip().lower()
        if mode not in {"chat", "code"}:
            raise ValueError("mode must be chat or code")
        raw_type = str(value.get("type", "")).strip().lower()
        atom_type = TYPE_ALIASES.get(raw_type, raw_type)
        allowed = CHAT_TYPES if mode == "chat" else CODE_TYPES
        if atom_type not in allowed:
            raise ValueError(f"unsupported atom type: {raw_type}")
        content = sanitize_l0_text(value.get("content"))
        if not content:
            raise ValueError("atom content is empty")
        try:
            priority = int(value.get("priority", 50))
        except (TypeError, ValueError) as exc:
            raise ValueError("priority must be an integer") from exc
        if not (priority == -1 and atom_type == "instruction" or 0 <= priority <= 100):
            raise ValueError("priority outside 0..100")
        if priority != -1 and priority < PRIORITY_MIN[atom_type]:
            raise ValueError("priority below type threshold")
        metadata = value.get("metadata") or {}
        if not isinstance(metadata, dict) or set(metadata) - ALLOWED_METADATA[atom_type]:
            raise ValueError("unknown atom metadata")
        # A newly stored Atom always has at least one related timestamp.  An
        # empty LLM timestamp array means "not specified", so anchor it at
        # persistence time rather than creating an untraceable empty record.
        timestamps = value.get("timestamps")
        if timestamps in (None, []):
            timestamps = [utc_iso()]
        if not isinstance(timestamps, list) or any(not isinstance(x, str) for x in timestamps):
            raise ValueError("timestamps must be a string array")
        now = utc_iso()
        return cls(
            id=str(value.get("id") or new_atom_id()) if new_id else str(value["id"]),
            content=content, type=atom_type, priority=priority,
            scene_name=sanitize_l0_text(value.get("scene_name")),
            source_message_ids=[int(x) for x in (value.get("source_message_ids") or [])],
            metadata=metadata, timestamps=sorted(set(timestamps)),
            createdAt=str(value.get("createdAt") or now), updatedAt=now,
            version=int(value.get("version", 0)),
            sessionKey=str(value.get("sessionKey") or ""), sessionId=str(value.get("sessionId") or ""),
            taskId=str(value.get("taskId") or ""), teamId=str(value.get("teamId") or ""),
            userId=str(value.get("userId") or ""), agentId=str(value.get("agentId") or ""),
        )

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "content": self.content, "type": self.type, "priority": self.priority, "scene_name": self.scene_name,
                "source_message_ids": list(self.source_message_ids), "metadata": dict(self.metadata), "timestamps": list(self.timestamps),
                "createdAt": self.createdAt, "updatedAt": self.updatedAt, "version": self.version, "sessionKey": self.sessionKey,
                "sessionId": self.sessionId, "taskId": self.taskId, "teamId": self.teamId, "userId": self.userId, "agentId": self.agentId}


def _fts_terms(query: str) -> list[str]:
    terms = re.findall(r"[A-Za-z0-9_./+#-]+|[\u3400-\u9fff]+", sanitize_l0_text(query))
    return [x for x in terms if len(x) >= 2][:12]


def _index_text(text: str) -> str:
    """Keep CJK characters independently searchable without Jieba."""
    return text


class AtomStore:
    """Durable L1 store.  SQLite contains current versions; JSONL keeps history."""

    def __init__(self, root: str | Path | None = None, *, db_name: str = "vectors.db",
                 embedding: Mapping[str, Any] | None = None):
        self.root = Path(root or get_hermes_home())
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / db_name
        self.records_dir = self.root / "records"
        self.records_dir.mkdir(exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS l1_records (
          id TEXT PRIMARY KEY, content TEXT NOT NULL, type TEXT NOT NULL,
          priority INTEGER NOT NULL, scene_name TEXT, session_key TEXT,
          session_id TEXT, task_id TEXT, team_id TEXT, user_id TEXT, agent_id TEXT,
          version INTEGER NOT NULL, timestamp_str TEXT, timestamp_start TEXT,
          timestamp_end TEXT, created_time TEXT NOT NULL, updated_time TEXT NOT NULL,
          metadata_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_l1_scope ON l1_records(team_id,agent_id,user_id,session_id,updated_time);
        CREATE VIRTUAL TABLE IF NOT EXISTS l1_fts USING fts5(content, content_original UNINDEXED, id UNINDEXED, type UNINDEXED, scene_name UNINDEXED, team_id UNINDEXED, user_id UNINDEXED, agent_id UNINDEXED, session_id UNINDEXED);
        CREATE VIRTUAL TABLE IF NOT EXISTS l1_fts_trigram USING fts5(content, content_original UNINDEXED, id UNINDEXED, type UNINDEXED, scene_name UNINDEXED, team_id UNINDEXED, user_id UNINDEXED, agent_id UNINDEXED, session_id UNINDEXED, tokenize='trigram');
        CREATE TABLE IF NOT EXISTS embedding_meta (
          model_id TEXT PRIMARY KEY, provider TEXT NOT NULL, model TEXT NOT NULL,
          dimensions INTEGER NOT NULL, status TEXT NOT NULL, updated_time TEXT NOT NULL
        );
        """)
        # Older development builds kept a second BLOB vector copy.  sqlite-vec
        # is now the sole vector store; remove that obsolete representation on
        # open so stale copies cannot be mistaken for the active ANN index.
        self._conn.execute("DROP TABLE IF EXISTS l1_vec_blob")
        self._conn.commit()
        self._source_cache: dict[str, dict[str, Any]] | None = None
        self.embedding = None
        self.embedding_meta: dict[str, Any] | None = None
        self.needs_reindex = False
        self._sqlite_vec = None
        self._vec_table: str | None = None
        try:
            # sqlite-vec is loaded from the installed Python package. If it is
            # unavailable, vector writes are skipped and FTS remains usable.
            from kylin_memory.l1_embeddings import load_sqlite_vec_extension

            self._sqlite_vec = load_sqlite_vec_extension(self._conn)
        except Exception:
            self._sqlite_vec = None
        try:
            self.embedding = create_atom_embedding(embedding)
            self._sync_embedding_meta()
        except Exception:
            logger.info("L1 embedding service unavailable; using lexical retrieval", exc_info=True)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _sync_embedding_meta(self) -> bool:
        """Create/update vector metadata once the embedding dimension is known."""
        if self.embedding is None or self.embedding.dimensions is None:
            return False
        meta = {
            "model_id": self.embedding.model_id,
            "provider": self.embedding.provider_name,
            "model": self.embedding.model,
            "dimensions": int(self.embedding.dimensions),
        }
        if self.embedding_meta == meta and self._vec_table is not None:
            return True
        self.embedding_meta = meta
        self._register_embedding_meta()
        return self._vec_table is not None

    def _register_embedding_meta(self) -> None:
        meta = self.embedding_meta
        if not meta or meta.get("dimensions") is None or self._sqlite_vec is None:
            # Embeddings without sqlite-vec have nowhere durable to go. Keep
            # the FTS-only mode explicit instead of pretending a vector index
            # is pending forever.
            self.needs_reindex = False
            return
        with self._lock, self._conn:
            if self._sqlite_vec is not None:
                digest = hashlib.sha256(str(meta["model_id"]).encode("utf-8")).hexdigest()[:20]
                self._vec_table = f"l1_vec_{digest}"
                self._conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._vec_table} USING vec0(embedding float[{int(meta['dimensions'])}] distance_metric=cosine)"
                )
            current = self._conn.execute(
                "SELECT model_id,provider,model,dimensions FROM embedding_meta WHERE status='active' LIMIT 1"
            ).fetchone()
            changed = current is not None and (
                str(current["model_id"]) != str(meta["model_id"])
                or str(current["provider"]) != str(meta["provider"])
                or str(current["model"]) != str(meta["model"])
                or int(current["dimensions"]) != int(meta["dimensions"])
            )
            if changed:
                # Vectors are model/dimension-bound. Metadata and FTS remain
                # usable while the new model backfills the vector index.
                if current is not None and self._sqlite_vec is not None:
                    old_digest = hashlib.sha256(
                        str(current["model_id"]).encode("utf-8")
                    ).hexdigest()[:20]
                    old_table = f"l1_vec_{old_digest}"
                    if old_table != self._vec_table:
                        self._conn.execute(f"DROP TABLE IF EXISTS {old_table}")
                if self._vec_table:
                    self._conn.execute(f"DELETE FROM {self._vec_table}")
                self._conn.execute("UPDATE embedding_meta SET status='retired' WHERE status='active'")
                self.needs_reindex = True
            self._conn.execute(
                "INSERT OR REPLACE INTO embedding_meta(model_id,provider,model,dimensions,status,updated_time) VALUES(?,?,?,?, 'active', ?)",
                (meta["model_id"], meta["provider"], meta["model"], int(meta["dimensions"]), utc_iso()),
            )
            total = int(self._conn.execute("SELECT COUNT(*) FROM l1_records").fetchone()[0])
            indexed = (
                int(self._conn.execute(f"SELECT COUNT(*) FROM {self._vec_table}").fetchone()[0])
                if self._vec_table
                else 0
            )
            self.needs_reindex = self.needs_reindex or total > indexed

    def _atom_embedding_text(self, atom: Atom) -> str:
        return f"{atom.type}\n{atom.scene_name}\n{atom.content}\n" + " ".join(
            f"{key}:{value}" for key, value in sorted(atom.metadata.items())
        )

    def _delete_vector(self, record_id: str) -> None:
        """Remove all derived vector representations for one Atom.

        sqlite-vec uses the stable SQLite rowid of ``l1_records`` as its
        primary key.  Using that rowid avoids Python's process-randomized
        ``hash()`` and means a database can be closed and reopened without
        losing the record/vector mapping.
        """
        row = self._conn.execute(
            "SELECT rowid FROM l1_records WHERE id=?", (record_id,)
        ).fetchone()
        if row is not None and self._vec_table:
            self._conn.execute(
                f"DELETE FROM {self._vec_table} WHERE rowid=?", (int(row[0]),)
            )

    def _table_exists(self, name: str) -> bool:
        return self._conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone() is not None

    def _insert_vector(self, atom: Atom, vector: Sequence[float]) -> None:
        if not self.embedding_meta or not self._vec_table or self._sqlite_vec is None:
            return
        row = self._conn.execute(
            "SELECT rowid FROM l1_records WHERE id=?", (atom.id,)
        ).fetchone()
        if row is None:
            raise sqlite3.IntegrityError("L1 record is missing before vector insert")
        normalized = _normalize_embedding(
            vector, int(self.embedding_meta["dimensions"])
        )
        rowid = int(row[0])
        self._conn.execute(f"DELETE FROM {self._vec_table} WHERE rowid=?", (rowid,))
        self._conn.execute(
            f"INSERT INTO {self._vec_table}(rowid,embedding) VALUES(?,?)",
            (rowid, self._sqlite_vec.serialize(normalized)),
        )

    def _embed_atom(self, atom: Atom) -> list[float] | None:
        if self.embedding is None or self._sqlite_vec is None:
            return None
        try:
            vector = self.embedding.embed_query(self._atom_embedding_text(atom))
            self._sync_embedding_meta()
            if not self.embedding_meta or self._vec_table is None:
                return None
            return _normalize_embedding(vector, int(self.embedding_meta["dimensions"]))
        except Exception as exc:
            logger.warning("L1 embedding failed; retaining metadata/FTS (%s)", type(exc).__name__)
            return None

    def rebuild_vectors(self, *, batch_size: int | None = None) -> dict[str, Any]:
        """Regenerate all L1 vectors from the current Atom rows.

        JSONL remains the recovery source; this operation only rebuilds the
        derived ``l1_vec`` table.  A failed embedding leaves metadata/FTS
        intact and returns a structured failure for callers to retry later.
        """
        if self.embedding is None or self._sqlite_vec is None:
            return {"success": False, "embedded": 0, "reason": "embedding_unavailable"}
        rows = self._conn.execute("SELECT * FROM l1_records ORDER BY updated_time,id").fetchall()
        if not rows:
            with self._lock, self._conn:
                if self._vec_table:
                    self._conn.execute(f"DELETE FROM {self._vec_table}")
            self.needs_reindex = False
            return {
                "success": True,
                "embedded": 0,
                "model_id": self.embedding_meta["model_id"] if self.embedding_meta else None,
            }
        atoms = [self._row(row) for row in rows]
        values = [self._atom_embedding_text(atom) for atom in atoms]
        try:
            vectors = self.embedding.embed_documents(values)
            if len(vectors) != len(atoms):
                raise EmbeddingError("embedding batch length mismatch")
            self._sync_embedding_meta()
            if not self.embedding_meta or self._vec_table is None:
                raise EmbeddingError("embedding index is unavailable")
            normalized = [_normalize_embedding(vector, int(self.embedding_meta["dimensions"])) for vector in vectors]
        except Exception as exc:
            logger.warning("L1 vector rebuild failed (%s)", type(exc).__name__)
            return {"success": False, "embedded": 0, "reason": type(exc).__name__}
        with self._lock, self._conn:
            if self._vec_table:
                self._conn.execute(f"DELETE FROM {self._vec_table}")
            for atom, vector in zip(atoms, normalized):
                self._insert_vector(atom, vector)
        self.needs_reindex = False
        return {"success": True, "embedded": len(atoms), "model_id": self.embedding_meta["model_id"]}

    def _write_jsonl(self, atom: Atom) -> None:
        path = self.records_dir / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
        try:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(atom.as_dict(), ensure_ascii=False) + "\n")
            self._source_cache = None
        except OSError:
            logger.warning("L1 JSONL audit append failed", exc_info=True)

    def upsert(self, atom: Atom, *, replace_ids: Sequence[str] = ()) -> Atom:
        # JSONL is the audit/recovery source. Append first; SQLite is a
        # derived current-version index and can always be rebuilt from it.
        self._write_jsonl(atom)
        vector = self._embed_atom(atom)
        with self._lock, self._conn:
            for old_id in replace_ids:
                self._conn.execute("DELETE FROM l1_fts WHERE id=?", (old_id,))
                self._conn.execute("DELETE FROM l1_fts_trigram WHERE id=?", (old_id,))
                self._delete_vector(old_id)
                self._conn.execute("DELETE FROM l1_records WHERE id=?", (old_id,))
            # INSERT OR REPLACE may remove/recreate a row (and therefore
            # change its SQLite rowid), so clear its old ANN entry first.
            self._delete_vector(atom.id)
            self._conn.execute("INSERT OR REPLACE INTO l1_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                atom.id, atom.content, atom.type, atom.priority, atom.scene_name, atom.sessionKey, atom.sessionId,
                atom.taskId, atom.teamId, atom.userId, atom.agentId, atom.version,
                atom.timestamps[0] if atom.timestamps else None, min(atom.timestamps) if atom.timestamps else None,
                max(atom.timestamps) if atom.timestamps else None, atom.createdAt, atom.updatedAt, json.dumps(atom.metadata, ensure_ascii=False)))
            self._conn.execute("DELETE FROM l1_fts WHERE id=?", (atom.id,))
            self._conn.execute("INSERT INTO l1_fts(content,content_original,id,type,scene_name,team_id,user_id,agent_id,session_id) VALUES (?,?,?,?,?,?,?,?,?)", (_index_text(atom.content), atom.content, atom.id, atom.type, atom.scene_name, atom.teamId, atom.userId, atom.agentId, atom.sessionId))
            self._conn.execute("DELETE FROM l1_fts_trigram WHERE id=?", (atom.id,))
            self._conn.execute("INSERT INTO l1_fts_trigram(content,content_original,id,type,scene_name,team_id,user_id,agent_id,session_id) VALUES (?,?,?,?,?,?,?,?,?)", (_index_text(atom.content), atom.content, atom.id, atom.type, atom.scene_name, atom.teamId, atom.userId, atom.agentId, atom.sessionId))
            if vector is not None:
                self._insert_vector(atom, vector)
        return atom

    def upsert_many(self, atoms: Sequence[Atom]) -> list[Atom]:
        """Append and index a batch, embedding all Atom texts in one call."""
        values = list(atoms)
        if not values:
            return []
        vectors_per_atom: list[list[float] | None] = [None] * len(values)
        if self.embedding is not None and self._sqlite_vec:
            try:
                vectors = self.embedding.embed_documents([self._atom_embedding_text(atom) for atom in values])
                if len(vectors) != len(values):
                    raise EmbeddingError("embedding batch length mismatch")
                self._sync_embedding_meta()
                if not self.embedding_meta or self._vec_table is None:
                    raise EmbeddingError("embedding index is unavailable")
                vectors_per_atom = [_normalize_embedding(vector, int(self.embedding_meta["dimensions"])) for vector in vectors]
            except Exception as exc:
                logger.warning("L1 batch embedding failed; retaining metadata/FTS (%s)", type(exc).__name__)
        for atom in values:
            self._write_jsonl(atom)
        with self._lock, self._conn:
            for atom, vector in zip(values, vectors_per_atom):
                self._delete_vector(atom.id)
                self._conn.execute("INSERT OR REPLACE INTO l1_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    atom.id, atom.content, atom.type, atom.priority, atom.scene_name, atom.sessionKey, atom.sessionId,
                    atom.taskId, atom.teamId, atom.userId, atom.agentId, atom.version,
                    atom.timestamps[0] if atom.timestamps else None, min(atom.timestamps) if atom.timestamps else None,
                    max(atom.timestamps) if atom.timestamps else None, atom.createdAt, atom.updatedAt, json.dumps(atom.metadata, ensure_ascii=False)))
                self._conn.execute("DELETE FROM l1_fts WHERE id=?", (atom.id,))
                self._conn.execute("INSERT INTO l1_fts(content,content_original,id,type,scene_name,team_id,user_id,agent_id,session_id) VALUES (?,?,?,?,?,?,?,?,?)", (_index_text(atom.content), atom.content, atom.id, atom.type, atom.scene_name, atom.teamId, atom.userId, atom.agentId, atom.sessionId))
                self._conn.execute("DELETE FROM l1_fts_trigram WHERE id=?", (atom.id,))
                self._conn.execute("INSERT INTO l1_fts_trigram(content,content_original,id,type,scene_name,team_id,user_id,agent_id,session_id) VALUES (?,?,?,?,?,?,?,?,?)", (_index_text(atom.content), atom.content, atom.id, atom.type, atom.scene_name, atom.teamId, atom.userId, atom.agentId, atom.sessionId))
                if vector is not None:
                    self._insert_vector(atom, vector)
        return values

    def get(self, record_id: str) -> Atom | None:
        row = self._conn.execute("SELECT * FROM l1_records WHERE id=?", (record_id,)).fetchone()
        return self._row(row) if row else None

    def get_many(self, record_ids: Sequence[str]) -> list[Atom]:
        """Return current Atom rows for the requested IDs in stable ID order."""
        ids = sorted({str(record_id) for record_id in record_ids if record_id})
        if not ids:
            return []
        result: list[Atom] = []
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            rows = self._conn.execute(
                "SELECT * FROM l1_records WHERE id IN (%s) ORDER BY id"
                % ",".join("?" for _ in batch),
                batch,
            ).fetchall()
            result.extend(self._row(row) for row in rows)
        return result

    def list_after_cursor(
        self,
        *,
        updated_after: str = "",
        id_after: str = "",
        team_id: str = "",
        user_id: str = "",
        agent_id: str = "",
        limit: int = 5000,
    ) -> list[Atom]:
        """List scoped current Atoms after a stable ``(updatedAt, id)`` cursor."""
        clauses = ["1=1"]
        params: list[Any] = []
        if updated_after:
            clauses.append("(updated_time > ? OR (updated_time = ? AND id > ?))")
            params.extend((updated_after, updated_after, id_after))
        for column, value in (
            ("team_id", team_id),
            ("user_id", user_id),
            ("agent_id", agent_id),
        ):
            if value:
                clauses.append(f"{column}=?")
                params.append(value)
        rows = self._conn.execute(
            "SELECT * FROM l1_records WHERE "
            + " AND ".join(clauses)
            + " ORDER BY updated_time,id LIMIT ?",
            (*params, max(1, min(int(limit), 20_000))),
        ).fetchall()
        return [self._row(row) for row in rows]

    def _row(self, row: sqlite3.Row) -> Atom:
        source_ids = []
        if self._source_cache is None:
            self._source_cache = {}
            for path in sorted(self.records_dir.glob("*.jsonl")):
                try:
                    for line in path.open(encoding="utf-8"):
                        item = json.loads(line)
                        if isinstance(item, dict) and item.get("id"):
                            key = str(item["id"])
                            previous = self._source_cache.get(key)
                            # JSONL is append-only; the final occurrence is
                            # the current version for an atomic update.
                            if previous is None or int(item.get("version", 0) or 0) >= int(previous.get("version", 0) or 0):
                                self._source_cache[key] = {
                                    "source_message_ids": [int(x) for x in item.get("source_message_ids", [])],
                                    "timestamps": list(item.get("timestamps") or []),
                                    "version": int(item.get("version", 0) or 0),
                                }
                except (OSError, ValueError, TypeError):
                    continue
        audit = (self._source_cache or {}).get(str(row["id"]), {})
        source_ids = list(audit.get("source_message_ids", []))
        timestamps = list(audit.get("timestamps", [])) or [x for x in (row["timestamp_str"],) if x]
        return Atom(id=row["id"], content=row["content"], type=row["type"], priority=row["priority"], scene_name=row["scene_name"] or "",
                    source_message_ids=source_ids, metadata=json.loads(row["metadata_json"] or "{}"), timestamps=timestamps,
                    createdAt=row["created_time"], updatedAt=row["updated_time"], version=row["version"], sessionKey=row["session_key"] or "",
                    sessionId=row["session_id"] or "", taskId=row["task_id"] or "", teamId=row["team_id"] or "", userId=row["user_id"] or "", agentId=row["agent_id"] or "")

    def _search_fts(self, query: str, *, limit: int = 5, team_id: str = "", user_id: str = "", agent_id: str = "", session_id: str = "", task_id: str = "", types: Iterable[str] | None = None) -> list[Atom]:
        terms = _fts_terms(query)
        if not terms:
            return []
        indexed_terms = list(terms)
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in indexed_terms)
        clauses, params = ["l1_fts MATCH ?"], [match]
        for col, val in (("team_id", team_id), ("user_id", user_id), ("agent_id", agent_id), ("session_id", session_id), ("task_id", task_id)):
            if val:
                clauses.append(f"l1_records.{col}=?"); params.append(val)
        if types:
            vals = list(types); clauses.append("l1_records.type IN (%s)" % ",".join("?" for _ in vals)); params.extend(vals)
        has_cjk = any(any("\u3400" <= c <= "\u9fff" for c in term) for term in terms)
        short_cjk = has_cjk and any(len(term) < 3 for term in terms)
        if short_cjk:
            like_clauses = clauses[1:]
            like_params = params[1:]
            like_clauses.append("(" + " OR ".join("l1_records.content LIKE ?" for _ in terms) + ")")
            like_params.extend(f"%{term}%" for term in terms)
            rows = self._conn.execute(
                "SELECT l1_records.* FROM l1_records WHERE " + " AND ".join(like_clauses) +
                " ORDER BY updated_time DESC LIMIT ?",
                (*like_params, max(1, min(int(limit), 50))),
            ).fetchall()
        else:
            table = "l1_fts_trigram" if has_cjk else "l1_fts"
            clauses[0] = f"{table} MATCH ?"
            rows = self._conn.execute("SELECT l1_records.* FROM " + table + " JOIN l1_records ON l1_records.id=" + table + ".id WHERE " + " AND ".join(clauses) + " ORDER BY bm25(" + table + ") LIMIT ?", (*params, max(1, min(int(limit), 50)))).fetchall()
            if not rows and has_cjk:
                # Unicode fallback for natural-language CJK queries whose
                # characters are separated by words not represented in the
                # trigram index.  This remains a scoped SQLite query.
                like_clauses = clauses[1:]
                like_params = params[1:]
                like_clauses.append("(" + " OR ".join("l1_records.content LIKE ?" for _ in terms) + ")")
                like_params.extend(f"%{term}%" for term in terms)
                rows = self._conn.execute(
                    "SELECT l1_records.* FROM l1_records WHERE " + " AND ".join(like_clauses) +
                    " ORDER BY updated_time DESC LIMIT ?",
                    (*like_params, max(1, min(int(limit), 50))),
                ).fetchall()
        return [self._row(r) for r in rows]

    def _search_vector(self, query: str, *, limit: int = 5, team_id: str = "", user_id: str = "", agent_id: str = "", session_id: str = "", task_id: str = "", types: Iterable[str] | None = None) -> list[tuple[Atom, float]]:
        """Return cosine-ranked Atom candidates from the active L1 vectors."""
        if self.embedding is None or self._sqlite_vec is None:
            return []
        try:
            query_vector = self.embedding.embed_query(query)
            self._sync_embedding_meta()
            if not self.embedding_meta or self._vec_table is None:
                return []
            dimensions = int(self.embedding_meta["dimensions"])
            query_vector = _normalize_embedding(query_vector, dimensions)
        except Exception as exc:
            logger.debug("L1 vector query unavailable (%s)", type(exc).__name__)
            return []
        requested = max(1, min(int(limit), 50))
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("r.team_id", team_id),
            ("r.user_id", user_id),
            ("r.agent_id", agent_id),
            ("r.session_id", session_id),
            ("r.task_id", task_id),
        ):
            if value:
                clauses.append(f"{column}=?")
                params.append(value)
        if types:
            values = list(types)
            clauses.append("r.type IN (%s)" % ",".join("?" for _ in values))
            params.extend(values)

        # sqlite-vec is the primary L1 ANN index.  KNN is intentionally run
        # against a bounded superset before tenant hydration, so records from
        # other scopes cannot hide the nearest rows for this scope.
        if self._vec_table and self._sqlite_vec is not None:
            candidate_limit = min(1000, max(requested * 8, requested))
            sql = (
                "SELECT r.*,v.distance FROM ("
                f"SELECT rowid,distance FROM {self._vec_table} "
                "WHERE embedding MATCH ? AND k=?"
                ") v JOIN l1_records r ON r.rowid=v.rowid WHERE "
                + (" AND ".join(clauses) if clauses else "1=1")
                + " ORDER BY v.distance LIMIT ?"
            )
            try:
                rows = self._conn.execute(
                    sql,
                    (
                        self._sqlite_vec.serialize(query_vector),
                        candidate_limit,
                        *params,
                        requested,
                    ),
                ).fetchall()
                return [
                    (self._row(row), float(1.0 - float(row["distance"])))
                    for row in rows
                ]
            except sqlite3.DatabaseError as exc:
                logger.debug("sqlite-vec L1 query failed; using FTS-only recall (%s)", type(exc).__name__)
        return []

    def delete(self, record_id: str) -> bool:
        """Delete one current Atom and every derived index row."""
        with self._lock, self._conn:
            exists = self._conn.execute(
                "SELECT 1 FROM l1_records WHERE id=?", (record_id,)
            ).fetchone()
            if exists is None:
                return False
            self._conn.execute("DELETE FROM l1_fts WHERE id=?", (record_id,))
            self._conn.execute("DELETE FROM l1_fts_trigram WHERE id=?", (record_id,))
            self._delete_vector(record_id)
            self._conn.execute("DELETE FROM l1_records WHERE id=?", (record_id,))
            self._source_cache = None
            return True

    def search(self, query: str, *, limit: int = 5, team_id: str = "", user_id: str = "", agent_id: str = "", session_id: str = "", task_id: str = "", types: Iterable[str] | None = None) -> list[Atom]:
        """Hybrid L1 retrieval using FTS and cosine vectors with RRF fusion."""
        requested = max(1, min(int(limit), 50))
        # Fetch a larger pool from both rankers. RRF is rank-based and does not
        # apply a second score threshold after fusion.
        pool = max(requested * 3, requested)
        lexical = self._search_fts(query, limit=pool, team_id=team_id, user_id=user_id,
                                   agent_id=agent_id, session_id=session_id, task_id=task_id, types=types)
        vector = self._search_vector(query, limit=pool, team_id=team_id, user_id=user_id,
                                     agent_id=agent_id, session_id=session_id, task_id=task_id, types=types)
        if not vector:
            return lexical[:requested]
        rrf_k = 60.0
        fused: dict[str, tuple[float, Atom]] = {}
        for rank, atom in enumerate(lexical, 1):
            fused[atom.id] = (fused.get(atom.id, (0.0, atom))[0] + 0.85 / (rrf_k + rank), atom)
        for rank, (atom, _score) in enumerate(vector, 1):
            fused[atom.id] = (fused.get(atom.id, (0.0, atom))[0] + 1.0 / (rrf_k + rank), atom)
        return [item[1] for item in sorted(fused.values(), key=lambda item: item[0], reverse=True)[:requested]]

    def atomic_update(self, record_id: str, patch: Mapping[str, Any], *, mode: str = "chat") -> Atom:
        """Update a current record while retaining its id and createdAt."""
        current = self.get(record_id)
        if current is None:
            raise KeyError(record_id)
        value = current.as_dict(); value.update(dict(patch)); value["id"] = record_id
        value["version"] = current.version + 1; value["createdAt"] = current.createdAt
        # A replacement version is evidenced only by the newly supplied
        # messages, while temporal coverage remains the union of old and new
        # timestamps.  The historical JSONL rows retain the old evidence.
        if "timestamps" in patch:
            value["timestamps"] = sorted(set(current.timestamps) | set(patch.get("timestamps") or []))
        value["source_message_ids"] = list(patch.get("source_message_ids", []))
        atom = Atom.from_mapping(value, mode=mode, new_id=False)
        return self.upsert(atom, replace_ids=[record_id])

    def find_equivalent(self, atom: Atom) -> Atom | None:
        """Find the current scoped row used to make background mirroring idempotent."""
        row = self._conn.execute(
            "SELECT * FROM l1_records WHERE content=? AND type=? AND team_id=? AND agent_id=? AND user_id=? LIMIT 1",
            (atom.content, atom.type, atom.teamId, atom.agentId, atom.userId),
        ).fetchone()
        return self._row(row) if row else None

    def list_updated(self, *, updated_after: str = "", session_id: str = "", team_id: str = "", user_id: str = "", agent_id: str = "", limit: int = 100) -> list[Atom]:
        clauses, params = ["1=1"], []
        for col, val in (("updated_time", updated_after), ("session_id", session_id), ("team_id", team_id), ("user_id", user_id), ("agent_id", agent_id)):
            if val:
                clauses.append(f"{col} > ?" if col == "updated_time" else f"{col} = ?"); params.append(val)
        rows = self._conn.execute("SELECT * FROM l1_records WHERE " + " AND ".join(clauses) + " ORDER BY updated_time,id LIMIT ?", (*params, max(1, int(limit)))).fetchall()
        return [self._row(r) for r in rows]

    def validate_source_ids(self, source_ids: Sequence[int], *, messages: Sequence[Mapping[str, Any]], session_id: str = "", team_id: str = "", user_id: str = "", agent_id: str = "", task_id: str = "") -> bool:
        """Validate evidence IDs against the current L0 batch and isolation."""
        def _get(row: Mapping[str, Any], *names: str) -> Any:
            for name in names:
                if name in row and row[name] not in (None, ""):
                    return row[name]
            return None
        by_id = {int(_get(m, "id", "message_id")): m for m in messages if _get(m, "id", "message_id") is not None}
        for mid in source_ids:
            row = by_id.get(int(mid))
            if row is None:
                return False
            fields = (("session_id", ("session_id", "sessionId"), session_id), ("team_id", ("team_id", "teamId"), team_id),
                      ("user_id", ("user_id", "userId"), user_id), ("agent_id", ("agent_id", "agentId"), agent_id),
                      ("task_id", ("task_id", "taskId"), task_id))
            for _key, names, expected in fields:
                if expected and _get(row, *names) not in (None, "", expected):
                    return False
        return bool(source_ids)


def normalize_scene_filename(name: str, existing: Iterable[str] = ()) -> str:
    base = _SAFE_FILENAME.sub("-", str(name or "scene").strip()).strip("-.") or "scene"
    base = re.sub(r"-+", "-", base).lower()
    if not base.endswith(".md"):
        base += ".md"
    used = set(existing)
    if base not in used:
        return base
    stem = base[:-3]
    n = 2
    while f"{stem}-{n}.md" in used:
        n += 1
    return f"{stem}-{n}.md"


@dataclass(slots=True)
class SceneIndexEntry:
    filename: str
    summary: str
    heat: int
    created: str
    updated: str

    def as_dict(self) -> dict[str, Any]:
        return {"filename": self.filename, "summary": self.summary, "heat": self.heat, "created": self.created, "updated": self.updated}


class ScenarioStore:
    """Scope-isolated L2 Markdown scenes and navigation index."""

    def __init__(self, root: str | Path | None = None, *, scope: str = "global", max_scenes: int = 15, prompt_mode: str = "chat"):
        self.root = Path(root or get_hermes_home())
        self.scope = scope
        self.max_scenes = max(1, int(max_scenes))
        self.prompt_mode = "code" if str(prompt_mode).lower() == "code" else "chat"
        base = self.root if scope == "global" else self.root / "profiles" / _safe_scope(scope)
        self.base = base; self.scene_dir = base / "scene_blocks"; self.meta_dir = base / ".metadata"
        self.scene_dir.mkdir(parents=True, exist_ok=True); self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.meta_dir / "scene_index.json"
        self._lock = threading.RLock()
        if self._migrate_legacy_utc_metadata():
            self.rebuild_index()

    def _migrate_legacy_utc_metadata(self) -> bool:
        """Convert pre-fix UTC scene metadata without changing the instant."""
        changed = False
        timestamp_line = re.compile(r"(?m)^(created|updated):[ \t]*(.*?)[ \t]*$")
        for path in sorted(self.scene_dir.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
                meta, marker, body = text.partition("-----META-END-----")
                if not marker:
                    continue

                def replace_timestamp(match: re.Match[str]) -> str:
                    return f"{match.group(1)}: {_localize_legacy_scene_timestamp(match.group(2))}"

                migrated_meta = timestamp_line.sub(replace_timestamp, meta)
                if migrated_meta == meta:
                    continue
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_text(migrated_meta + marker + body, encoding="utf-8")
                tmp.replace(path)
                changed = True
            except (OSError, UnicodeError):
                logger.warning("Could not migrate L2 scene timestamps in %s", path, exc_info=True)
        return changed

    def _parse_meta(self, path: Path) -> SceneIndexEntry | None:
        text = path.read_text(encoding="utf-8")
        match = re.search(r"-----META-START-----\s*created:\s*(.*?)\s*updated:\s*(.*?)\s*summary:\s*(.*?)\s*heat:\s*(-?\d+)\s*-----META-END-----", text, re.S)
        if not match: return None
        return SceneIndexEntry(path.name, match.group(3).strip(), int(match.group(4)), match.group(1).strip(), match.group(2).strip())

    def rebuild_index(self) -> list[SceneIndexEntry]:
        entries = []
        for path in sorted(self.scene_dir.glob("*.md")):
            try:
                item = self._parse_meta(path)
                if item and "[DELETED]" not in path.read_text(encoding="utf-8"):
                    entries.append(item)
            except (OSError, UnicodeError):
                logger.warning("Could not index scene %s", path, exc_info=True)
        entries.sort(key=lambda x: (-x.heat, x.updated, x.filename))
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps([x.as_dict() for x in entries], ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.index_path)
        return entries

    def index(self) -> list[SceneIndexEntry]:
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            return [SceneIndexEntry(str(x["filename"]), str(x.get("summary", "")), int(x.get("heat", 0)), str(x.get("created", "")), str(x.get("updated", ""))) for x in data]
        except Exception:
            return self.rebuild_index()

    def navigation(self, *, absolute: bool = False) -> str:
        rows = self.index(); prefix = str(self.scene_dir) + "/" if absolute else "scene_blocks/"
        return "\n".join(f"- {prefix}{x.filename} | heat={x.heat} | updated={x.updated} | {x.summary}" for x in rows)

    def read(self, filename: str) -> str:
        raw = str(filename or "")
        candidate = Path(raw)
        if candidate.is_absolute() or candidate.name != raw or candidate.suffix.lower() != ".md":
            raise ValueError("invalid scene filename")
        path = self.scene_dir / candidate
        return path.read_text(encoding="utf-8")

    def write(self, filename: str, body: str) -> str:
        with self._lock:
            raw = str(filename or "")
            candidate = Path(raw)
            if candidate.is_absolute() or candidate.name != raw:
                raise ValueError("invalid scene filename")
            filename = candidate.name
            existing = {x.filename for x in self.index()}
            # A MemoryCore-style soft delete must overwrite the original
            # physical file even though rebuild_index intentionally removes
            # deleted entries from the active index.
            soft_delete = str(body).strip() == "[DELETED]" and (self.scene_dir / filename).exists()
            if filename not in existing and not soft_delete:
                filename = normalize_scene_filename(filename, existing)
            path = self.scene_dir / filename
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(str(body), encoding="utf-8"); tmp.replace(path)
            self.rebuild_index()
            return filename

    @staticmethod
    def _eviction_candidate(entries: Sequence[SceneIndexEntry]) -> SceneIndexEntry | None:
        """Return the coldest, least recently active scene.

        ``heat`` is the existing activity/access-frequency proxy and ``updated``
        is the only persisted recency signal in legacy scene metadata.  Keep
        the policy deterministic so capacity pressure cannot cause arbitrary
        scene deletion.
        """
        if not entries:
            return None
        return min(entries, key=lambda entry: (int(entry.heat), entry.updated, entry.filename))

    @staticmethod
    def _scene_document(*, created: str, updated: str, summary: str,
                        heat: int, body: str) -> str:
        clean_body = str(body or "").strip()
        if not clean_body or clean_body == "[DELETED]":
            raise ValueError("scene body is empty")
        if "-----META-START-----" in clean_body:
            if "-----META-END-----" not in clean_body:
                raise ValueError("invalid scene metadata")
            clean_body = clean_body.split("-----META-END-----", 1)[1].strip()
        return (
            "-----META-START-----\n"
            f"created: {created}\n"
            f"updated: {updated}\n"
            f"summary: {sanitize_l0_text(summary)[:240]}\n"
            f"heat: {int(heat)}\n"
            "-----META-END-----\n\n"
            f"{clean_body}"
        )

    def apply_action(
        self,
        atoms: Sequence[Atom],
        *,
        action: str,
        scene_name: str,
        target_files: Sequence[str] = (),
        delete_files: Sequence[str] = (),
        summary: str = "",
        body: str | None = None,
    ) -> dict[str, Any]:
        """Apply the text-host equivalent of MemoryCore's L2 file tools.

        The LLM proposes CREATE/UPDATE/MERGE and the store performs the
        corresponding bounded file operations.  The caller owns snapshot
        restore, so any exception is a failed L2 job with an unchanged cursor.
        """
        if not atoms:
            return {"skipped": True, "changed": [], "latestCursor": ""}
        action = str(action or "update").strip().lower()
        if action not in {"create", "update", "merge"}:
            raise ValueError("invalid scene action")
        with self._lock:
            entries = self.index()
            by_name = {entry.filename: entry for entry in entries}
            targets = list(dict.fromkeys(str(x) for x in target_files if str(x)))
            deletes = list(dict.fromkeys(str(x) for x in delete_files if str(x)))
            if any(name not in by_name for name in [*targets, *deletes]):
                raise ValueError("unknown scene target")
            if action != "merge" and deletes:
                raise ValueError("only merge may delete scene files")
            if action == "merge" and set(deletes) - set(targets):
                raise ValueError("merge may only delete target files")
            now = _scene_iso()
            clean_summary = summary or scene_name

            if action == "create":
                if targets:
                    return {"skipped": True, "reason": "scene_limit", "changed": [],
                            "latestCursor": max(x.updatedAt for x in atoms)}
                evicted: list[str] = []
                # Evict exactly one cold scene only when the current set is
                # already full.  Reaching the limit after creation is valid.
                if len(entries) >= self.max_scenes:
                    victim = self._eviction_candidate(entries)
                    if victim is None:
                        return {"skipped": True, "reason": "scene_limit", "changed": [],
                                "latestCursor": max(x.updatedAt for x in atoms)}
                    self.write(victim.filename, "[DELETED]")
                    evicted.append(victim.filename)
                    entries = self.index()
                    by_name = {entry.filename: entry for entry in entries}
                filename = normalize_scene_filename(scene_name, by_name)
                document = self._scene_document(
                    created=now, updated=now, summary=clean_summary, heat=1,
                    body=str(body or ""),
                )
                changed = [*evicted, self.write(filename, document)]

            elif action == "update":
                if len(targets) != 1:
                    raise ValueError("update requires exactly one target")
                target = targets[0]
                prior = by_name[target]
                # An UPDATE may also change the scene's semantic name when the
                # same ongoing activity has moved to a new object/goal (for
                # example, 深圳求职 -> 成都求职).  Exclude the target itself from
                # collision detection so an unchanged name remains stable.
                filename = normalize_scene_filename(
                    scene_name or Path(target).stem,
                    set(by_name) - {target},
                )
                document = self._scene_document(
                    created=prior.created, updated=now, summary=clean_summary,
                    heat=prior.heat + 1, body=str(body or ""),
                )
                if filename == target:
                    changed = [self.write(target, document)]
                else:
                    written = self.write(filename, document)
                    self.write(target, "[DELETED]")
                    changed = [target, written]

            else:  # merge
                if len(targets) < 2:
                    raise ValueError("merge requires at least two targets")
                filename = normalize_scene_filename(scene_name, by_name)
                heat = sum(by_name[name].heat for name in targets) + 1
                document = self._scene_document(
                    created=now, updated=now, summary=clean_summary, heat=heat,
                    body=str(body or ""),
                )
                changed = [self.write(filename, document)]
                for old in dict.fromkeys([*targets, *deletes]):
                    if old == changed[0] or old not in by_name:
                        continue
                    self.write(old, "[DELETED]")

            # SceneExtractor removes soft-deleted/META-only artifacts before
            # rebuilding scene_index.json.  Do the same after the proposed
            # action has completed successfully.
            for path in list(self.scene_dir.glob("*.md")):
                try:
                    text = path.read_text(encoding="utf-8")
                    if not text.strip() or text.strip() == "[DELETED]":
                        path.unlink()
                except OSError:
                    raise
            final_index = self.rebuild_index()
            if len(final_index) > self.max_scenes:
                return {"skipped": True, "reason": "scene_limit", "changed": [],
                        "latestCursor": max(x.updatedAt for x in atoms)}
            return {
                "skipped": False,
                "changed": changed,
                "latestCursor": max(x.updatedAt for x in atoms),
                "sceneIndex": [x.as_dict() for x in final_index],
            }

    def consolidate(self, atoms: Sequence[Atom], *, scene_name: str, summary: str = "", body: str | None = None) -> dict[str, Any]:
        if not atoms: return {"skipped": True, "changed": [], "latestCursor": ""}
        with self._lock:
            entries = self.index(); by_name = {x.filename: x for x in entries}
            target = next((x.filename for x in entries if x.filename.rsplit(".", 1)[0].casefold() == scene_name.casefold()), None)
            evicted: list[str] = []
            if target is None:
                if len(entries) >= self.max_scenes:
                    victim = self._eviction_candidate(entries)
                    if victim is None:
                        return {"skipped": True, "reason": "scene_limit", "changed": [], "latestCursor": max(x.updatedAt for x in atoms)}
                    self.write(victim.filename, "[DELETED]")
                    evicted.append(victim.filename)
                    entries = self.index(); by_name = {x.filename: x for x in entries}
                target = normalize_scene_filename(scene_name, by_name)
            now = _scene_iso(); old = self.scene_dir / target
            if body is None:
                prior = old.read_text(encoding="utf-8") if old.exists() else ""
                facts = "\n".join(f"- {a.content}" for a in atoms)
                if prior:
                    # Deterministic mode is used only when L2 intentionally has
                    # no model consolidator.  It must still consume every new
                    # Atom; retaining the old body verbatim would advance the
                    # cursor while silently dropping the new evidence.
                    prior_body = prior.split("-----META-END-----", 1)[-1].strip()
                    existing_facts = {
                        line.strip()[2:].strip()
                        for line in prior_body.splitlines()
                        if line.strip().startswith("- ")
                    }
                    additions = [
                        f"- {a.content}" for a in atoms
                        if a.content not in existing_facts
                    ]
                    body = prior_body
                    if additions:
                        heading = "## 演变轨迹" if self.prompt_mode != "code" else "## 关键事实依据"
                        if heading in body:
                            body = body.rstrip() + "\n" + "\n".join(additions)
                        else:
                            body = body.rstrip() + f"\n\n{heading}\n" + "\n".join(additions)
                elif self.prompt_mode == "code":
                    body = f"## 任务场景\n{sanitize_l0_text(scene_name)}\n\n## 适用条件\n{sanitize_l0_text(summary or scene_name)}\n\n## 核心 SOP\n{facts}\n\n## 判断逻辑\n基于已持久化 Atom 逐项核验，必要时回溯 L0 证据。\n\n## 关键事实依据\n{facts}"
                else:
                    narrative = "；".join(sanitize_l0_text(a.content) for a in atoms)
                    body = (
                        f"## 用户基础信息\n{sanitize_l0_text(summary or scene_name)}\n\n"
                        f"## 用户核心特征\n{narrative[:100]}\n\n"
                        f"## 用户偏好\n{narrative[:240]}\n\n"
                        f"## 隐性信号\n基于已持久化 Atom，未作额外推断。\n\n"
                        f"## 核心叙事\n{narrative[:400]}\n\n"
                        f"## 演变轨迹\n{facts}\n\n"
                        f"## 待确认/矛盾点\n无。"
                    )
            created = by_name[target].created if target in by_name else now
            heat = (by_name[target].heat + 1) if target in by_name else 1
            meta = f"-----META-START-----\ncreated: {created}\nupdated: {now}\nsummary: {sanitize_l0_text(summary or scene_name)[:240]}\nheat: {heat}\n-----META-END-----\n\n"
            # MemoryCore does not impose the historical 1500-character local
            # truncation.  Preserve the complete scene document; prompt-time
            # consumers already apply their own context budgets.
            self.write(target, meta + body)
            return {"skipped": False, "changed": [*evicted, target], "latestCursor": max(x.updatedAt for x in atoms), "sceneIndex": [x.as_dict() for x in self.index()]}


__all__ = [
    "Atom", "AtomStore", "ScenarioStore", "SceneIndexEntry",
    "EmbeddingError", "OpenAICompatibleEmbedding", "create_atom_embedding",
    "normalize_scope", "normalize_scene_filename", "sanitize_l0_text",
    "should_extract_l1", "new_atom_id", "CHAT_TYPES", "CODE_TYPES",
]
