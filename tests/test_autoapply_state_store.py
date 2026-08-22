"""Tests for the auto-apply dedup/audit store."""

from hr_breaker.autoapply.state_store import AutoApplyStore


def test_seen_false_for_unknown_vacancy(tmp_path):
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    assert store.seen("v1") is False


def test_upsert_then_seen_and_get(tmp_path):
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    store.upsert("v1", "seen", title="Dev", company="Acme", url="http://x", trigger_keyword="python")

    assert store.seen("v1") is True
    row = store.get("v1")
    assert row["status"] == "seen"
    assert row["title"] == "Dev"
    assert row["company"] == "Acme"
    assert row["trigger_keyword"] == "python"


def test_upsert_updates_status_and_preserves_unspecified_fields(tmp_path):
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    store.upsert("v1", "seen", title="Dev", company="Acme")
    store.upsert("v1", "ready", cover_letter="Dear hiring manager...", pdf_path="/tmp/x.pdf")

    row = store.get("v1")
    assert row["status"] == "ready"
    assert row["title"] == "Dev"  # preserved from first upsert
    assert row["cover_letter"] == "Dear hiring manager..."
    assert row["pdf_path"] == "/tmp/x.pdf"


def test_list_by_status_filters_correctly(tmp_path):
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    store.upsert("v1", "applied")
    store.upsert("v2", "failed", error="boom")
    store.upsert("v3", "applied")

    applied = store.list_by_status("applied")
    assert {r["vacancy_id"] for r in applied} == {"v1", "v3"}

    failed = store.list_by_status("failed")
    assert len(failed) == 1
    assert failed[0]["error"] == "boom"


def test_store_persists_across_instances(tmp_path):
    db_path = tmp_path / "state.sqlite3"
    AutoApplyStore(db_path).upsert("v1", "applied")

    reopened = AutoApplyStore(db_path)
    assert reopened.seen("v1") is True


def test_is_resolved_false_for_bare_seen(tmp_path):
    """A vacancy interrupted before tailoring finished (still "seen") must be retried,
    not permanently skipped."""
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    store.upsert("v1", "seen")

    assert store.seen("v1") is True
    assert store.is_resolved("v1") is False


def test_is_resolved_true_for_terminal_statuses(tmp_path):
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    for status in ("ready", "applied", "skipped"):
        store.upsert(f"v-{status}", status)
        assert store.is_resolved(f"v-{status}") is True


def test_is_resolved_false_for_failed(tmp_path):
    """Most real-world failures (timeouts, transient rate limits) aren't permanent -
    retry on the next run instead of skipping forever."""
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    store.upsert("v1", "failed", error="boom")

    assert store.is_resolved("v1") is False


def test_is_resolved_false_for_unknown_vacancy(tmp_path):
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    assert store.is_resolved("nope") is False
