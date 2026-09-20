"""Application facade consumed by Kylin or any other host application."""

from __future__ import annotations

from typing import Protocol, Sequence

from .extractors import ProfileExtractor
from .models import ApplyResult, InteractionMessage, OnboardingPrompt, UserProfile
from .onboarding import next_prompt, submit_answer
from .policy import apply_updates, effective_profile
from .prompting import DEFAULT_MAX_CHARS, DEFAULT_MIN_CONFIDENCE, render_profile_prompt
from .storage import ProfileNotFoundError


class ProfileStore(Protocol):
    def exists(self, user_id: str) -> bool: ...
    def load(self, user_id: str) -> UserProfile: ...
    def save(self, profile: UserProfile) -> None: ...
    def delete(self, user_id: str) -> bool: ...


class ProfileService:
    def __init__(self, store: ProfileStore, extractor: ProfileExtractor | None = None) -> None:
        self.store = store
        self.extractor = extractor

    def get_or_create(self, user_id: str) -> UserProfile:
        try:
            return self.store.load(user_id)
        except ProfileNotFoundError:
            profile = UserProfile(user_id=user_id)
            self.store.save(profile)
            return profile

    def onboarding_prompt(self, user_id: str) -> OnboardingPrompt:
        return next_prompt(self.get_or_create(user_id))

    def answer_onboarding(self, user_id: str, answer: str) -> OnboardingPrompt:
        profile = self.get_or_create(user_id)
        prompt = submit_answer(profile, answer)
        self.store.save(profile)
        return prompt

    def observe(
        self,
        user_id: str,
        messages: Sequence[InteractionMessage],
    ) -> ApplyResult:
        if self.extractor is None:
            raise RuntimeError("no profile extractor configured")
        profile = self.get_or_create(user_id)
        batch = self.extractor.extract(user_id, messages, effective_profile(profile))
        result = apply_updates(profile, batch.updates)
        if result.applied or result.deleted:
            self.store.save(profile)
        return result

    def profile(self, user_id: str, *, apply_decay: bool = True) -> UserProfile:
        profile = self.store.load(user_id)
        return effective_profile(profile) if apply_decay else profile

    def profile_prompt(
        self,
        user_id: str,
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> str:
        """Return compact prompt context for the host model."""

        return render_profile_prompt(
            self.profile(user_id, apply_decay=True),
            max_chars=max_chars,
            min_confidence=min_confidence,
        )

    def delete_profile(self, user_id: str) -> bool:
        return self.store.delete(user_id)
