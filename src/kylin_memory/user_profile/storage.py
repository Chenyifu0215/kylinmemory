"""Authenticated encrypted persistence for complete profile documents."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import tempfile
from pathlib import Path
from typing import Protocol

from cryptography.fernet import Fernet, InvalidToken

from .models import UserProfile


class ProfileNotFoundError(KeyError):
    pass


class ProfileDecryptionError(ValueError):
    pass


class KeyProvider(Protocol):
    def get_key(self) -> bytes: ...


class EnvironmentKeyProvider:
    def __init__(self, variable: str = "KYLIN_PROFILE_KEY") -> None:
        self.variable = variable

    def get_key(self) -> bytes:
        value = os.environ.get(self.variable)
        if not value:
            raise RuntimeError(f"missing encryption key environment variable: {self.variable}")
        key = value.encode("ascii")
        Fernet(key)  # Validate before using it for any profile.
        return key


class FileKeyProvider:
    """Local-development key provider; production deployments should use a KMS."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def get_key(self) -> bytes:
        key = self.path.read_bytes().strip()
        Fernet(key)
        return key

    def create(self) -> bytes:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"refusing to overwrite existing key: {self.path}")
        key = Fernet.generate_key()
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(key + b"\n")
        return key


class EncryptedFileProfileStore:
    """Stores one Fernet-authenticated ciphertext file per pseudonymous user key."""

    def __init__(self, directory: str | Path, key_provider: KeyProvider) -> None:
        self.directory = Path(directory)
        self.key_provider = key_provider

    def _key(self) -> bytes:
        return self.key_provider.get_key()

    def _profile_path(self, user_id: str) -> Path:
        raw_key = base64.urlsafe_b64decode(self._key())
        digest = hmac.new(raw_key, user_id.encode("utf-8"), hashlib.sha256).hexdigest()
        return self.directory / f"{digest}.profile.enc"

    def exists(self, user_id: str) -> bool:
        return self._profile_path(user_id).exists()

    def load(self, user_id: str) -> UserProfile:
        path = self._profile_path(user_id)
        if not path.exists():
            raise ProfileNotFoundError(user_id)
        try:
            plaintext = Fernet(self._key()).decrypt(path.read_bytes())
            profile = UserProfile.model_validate_json(plaintext)
        except (InvalidToken, ValueError) as exc:
            raise ProfileDecryptionError(f"cannot decrypt or validate profile for {user_id!r}") from exc
        if not hmac.compare_digest(profile.user_id, user_id):
            raise ProfileDecryptionError("encrypted profile belongs to a different user")
        return profile

    def save(self, profile: UserProfile) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        ciphertext = Fernet(self._key()).encrypt(profile.model_dump_json().encode("utf-8"))
        target = self._profile_path(profile.user_id)
        fd, temporary_name = tempfile.mkstemp(prefix=".profile-", dir=self.directory)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(ciphertext)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, target)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def delete(self, user_id: str) -> bool:
        path = self._profile_path(user_id)
        if not path.exists():
            return False
        path.unlink()
        return True
