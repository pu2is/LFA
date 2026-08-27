"""acquire_mutation_lock (#62, generalized to also gate job/Rescan mutations
in #64)."""
from sqlalchemy import text

from app.shared.database import SessionLocal
from app.shared.locking import acquire_mutation_lock


def test_acquire_mutation_lock_is_mutually_exclusive_across_sessions():
    """Two concurrent mutation requests -- whether both path-mutating (the
    root cause of the `/a` + `/a/b` nesting race, #62) or one path- and one
    job-mutating (a path deleted mid-Rescan, ADR-0001b D1) -- must never both
    pass a check before either commits. Uses two independent
    sessions/connections from SessionLocal rather than the shared
    `client`/`db` fixtures -- those route every request through one
    connection wrapped in a single outer transaction (see conftest.py's `db`
    fixture), so they can't model two backends actually contending for the
    same advisory lock. `pg_try_advisory_xact_lock` (non-blocking) makes the
    assertion deterministic instead of racing real thread timing against a
    blocking call."""
    session_a = SessionLocal()
    session_b = SessionLocal()
    try:
        acquire_mutation_lock(session_a)  # holds the lock, uncommitted

        still_held = session_b.execute(
            text("SELECT pg_try_advisory_xact_lock(hashtext('lfa:mutation'))")
        ).scalar()
        assert still_held is False

        session_a.rollback()  # releases it -- pg_advisory_xact_lock is tied to the transaction

        now_available = session_b.execute(
            text("SELECT pg_try_advisory_xact_lock(hashtext('lfa:mutation'))")
        ).scalar()
        assert now_available is True
    finally:
        session_a.close()
        session_b.rollback()
        session_b.close()
