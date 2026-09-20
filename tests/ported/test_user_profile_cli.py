from __future__ import annotations

import json
from types import SimpleNamespace

from kylin_memory.user_profile.cli import main
from kylin_memory.user_profile.models import ProfileEntry
from kylin_memory.user_profile_runtime import initialize_user_profile


def _seed_profile(tmp_path, monkeypatch, *, platform="cli", platform_user_id=None):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runtime = initialize_user_profile(
        SimpleNamespace(
            platform=platform,
            _user_id=platform_user_id,
            client=None,
        ),
        {},
    )
    profile = runtime.service.get_or_create(runtime.user_id)
    profile.entries["basic.preferred_name"] = ProfileEntry(
        value="小王",
        confidence=1.0,
        source="explicit",
        lifecycle="static",
    )
    runtime.service.store.save(profile)
    return tmp_path / "user_profile"


def test_show_without_user_id_prints_local_profile_markdown(
    tmp_path, monkeypatch, capsys
):
    profile_home = _seed_profile(tmp_path, monkeypatch)

    main(["--home", str(profile_home), "show"])

    output = capsys.readouterr().out
    assert "<user_profile_data>" in output
    assert "小王" in output


def test_show_json_prints_complete_decrypted_document(tmp_path, monkeypatch, capsys):
    profile_home = _seed_profile(tmp_path, monkeypatch)

    main(["--home", str(profile_home), "show", "--format", "json"])

    document = json.loads(capsys.readouterr().out)
    assert document["entries"]["basic.preferred_name"]["value"] == "小王"
    assert document["entries"]["basic.preferred_name"]["source"] == "explicit"


def test_show_can_derive_messaging_platform_identity(tmp_path, monkeypatch, capsys):
    profile_home = _seed_profile(
        tmp_path,
        monkeypatch,
        platform="telegram",
        platform_user_id="telegram-user-123",
    )

    main(
        [
            "--home",
            str(profile_home),
            "show",
            "--platform",
            "telegram",
            "--platform-user-id",
            "telegram-user-123",
        ]
    )

    assert "小王" in capsys.readouterr().out


def test_show_is_read_only_when_profile_key_is_missing(tmp_path):
    missing_home = tmp_path / "missing-user-profile"

    try:
        main(["--home", str(missing_home), "show"])
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("show should fail when no profile key exists")

    assert not missing_home.exists()
