"""Developer CLI for inspecting and exercising encrypted user profiles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from kylin_memory.config import get_hermes_home

from .extractors import OpenAICompatibleProfileExtractor
from .inspection import load_profile, show_profile_markdown
from .models import InteractionMessage
from .service import ProfileService
from .storage import EncryptedFileProfileStore, FileKeyProvider


def _service(home: Path, with_openai: bool = False) -> ProfileService:
    key_provider = FileKeyProvider(home / "profile.key")
    if not key_provider.path.exists():
        key_provider.create()
    store = EncryptedFileProfileStore(home / "profiles", key_provider)
    extractor = OpenAICompatibleProfileExtractor.from_environment() if with_openai else None
    return ProfileService(store, extractor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or exercise the encrypted Kylin user profile"
    )
    parser.add_argument(
        "--home",
        type=Path,
        default=Path(
            os.environ.get(
                "KYLIN_PROFILE_HOME", str(get_hermes_home() / "user_profile")
            )
        ),
        help=(
            "Profile storage directory (default: "
            "$KYLIN_PROFILE_HOME or active HERMES_HOME/user_profile)"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create the local encryption key")
    init.set_defaults(action="init")

    onboard = subparsers.add_parser("onboard", help="run the three onboarding questions")
    onboard.add_argument("user_id")

    observe = subparsers.add_parser("observe", help="extract profile updates from one user message")
    observe.add_argument("user_id")
    observe.add_argument("message")

    show = subparsers.add_parser(
        "show", help="decrypt and show the current profile for debugging"
    )
    show.add_argument(
        "user_id",
        nargs="?",
        help="Internal pseudonymous user ID; defaults to the selected runtime identity",
    )
    show.add_argument("--platform", default="cli")
    show.add_argument("--platform-user-id")
    show.add_argument("--format", choices=("markdown", "json"), default="markdown")
    show.add_argument("--max-chars", type=int, default=4_000)
    show.add_argument("--min-confidence", type=float, default=0.5)

    prompt = subparsers.add_parser("prompt", help="render compact profile context for an LLM")
    prompt.add_argument("user_id", nargs="?")
    prompt.add_argument("--platform", default="cli")
    prompt.add_argument("--platform-user-id")
    prompt.add_argument("--max-chars", type=int, default=4_000)
    prompt.add_argument("--min-confidence", type=float, default=0.5)

    delete = subparsers.add_parser("delete", help="delete one encrypted profile")
    delete.add_argument("user_id")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "init":
        provider = FileKeyProvider(args.home / "profile.key")
        provider.create()
        print(f"Created key at {provider.path}")
        return

    if args.command == "show":
        identity = {
            "profile_home": args.home,
            "user_id": args.user_id,
            "platform": args.platform,
            "platform_user_id": args.platform_user_id,
        }
        if args.format == "json":
            print(load_profile(**identity).model_dump_json(indent=2))
        else:
            print(
                show_profile_markdown(
                    **identity,
                    max_chars=args.max_chars,
                    min_confidence=args.min_confidence,
                )
            )
        return
    if args.command == "prompt":
        print(
            show_profile_markdown(
                profile_home=args.home,
                user_id=args.user_id,
                platform=args.platform,
                platform_user_id=args.platform_user_id,
                max_chars=args.max_chars,
                min_confidence=args.min_confidence,
            )
        )
        return

    service = _service(args.home, with_openai=args.command == "observe")
    if args.command == "onboard":
        prompt = service.onboarding_prompt(args.user_id)
        while not prompt.completed:
            prompt = service.answer_onboarding(args.user_id, input(f"{prompt.question} "))
        print("首次信息收集完成。")
    elif args.command == "observe":
        result = service.observe(
            args.user_id,
            [InteractionMessage(role="user", content=args.message)],
        )
        print(result.model_dump_json(indent=2))
    elif args.command == "delete":
        print("deleted" if service.delete_profile(args.user_id) else "not found")


if __name__ == "__main__":
    main()
