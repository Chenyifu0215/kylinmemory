"""Exactly three resumable onboarding questions requested by the product spec."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Evidence, OnboardingPrompt, ProfileEntry, UserProfile, utc_now
from .schema import FIELD_BY_PATH


@dataclass(frozen=True)
class Question:
    path: str
    text: str


QUESTIONS = (
    Question("basic.preferred_name", "怎么称呼您？"),
    Question("basic.age", "您今年多大？"),
    Question("occupation.category", "您从事什么职业？"),
)


def next_prompt(profile: UserProfile) -> OnboardingPrompt:
    index = profile.onboarding.next_question_index
    if profile.onboarding.completed or index >= len(QUESTIONS):
        return OnboardingPrompt(completed=True)
    question = QUESTIONS[index]
    return OnboardingPrompt(completed=False, field=question.path, question=question.text)


def _normalise_answer(path: str, answer: str) -> str | int:
    cleaned = answer.strip()
    if not cleaned:
        raise ValueError("回答不能为空")
    if path == "basic.age":
        match = re.fullmatch(r"(?:我)?\s*(\d{1,3})\s*(?:岁)?", cleaned)
        if not match:
            raise ValueError("年龄需要是 0 到 120 之间的整数")
        age = int(match.group(1))
        if not 0 <= age <= 120:
            raise ValueError("年龄需要是 0 到 120 之间的整数")
        return age
    if len(cleaned) > 200:
        raise ValueError("回答不能超过 200 个字符")
    return cleaned


def submit_answer(profile: UserProfile, answer: str) -> OnboardingPrompt:
    prompt = next_prompt(profile)
    if prompt.completed or prompt.field is None:
        raise ValueError("首次信息收集已经完成")
    now = utc_now()
    value = _normalise_answer(prompt.field, answer)
    field = FIELD_BY_PATH[prompt.field]
    profile.entries[prompt.field] = ProfileEntry(
        value=value,
        confidence=1.0,
        source="onboarding",
        lifecycle=field.lifecycle,
        evidence=[Evidence(quote=answer.strip(), observed_at=now)],
        created_at=now,
        updated_at=now,
    )
    profile.onboarding.next_question_index += 1
    profile.onboarding.completed = profile.onboarding.next_question_index == len(QUESTIONS)
    profile.updated_at = now
    return next_prompt(profile)

