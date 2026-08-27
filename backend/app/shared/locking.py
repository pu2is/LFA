from sqlalchemy import text
from sqlalchemy.orm import Session


def acquire_mutation_lock(db: Session) -> None:
    """Block until this transaction holds the exclusive gate for path and job
    mutations (#62, generalized to cover Rescan in #64). Every endpoint that
    does a check-then-write on paths or jobs -- duplicate/ancestor check then
    path insert/delete, or Rescan precondition check then job insert -- must
    call this first, so two concurrent requests can never both pass a check
    before either commits (e.g. `/a` and `/a/b` both seeing "no conflict" and
    registering as unrelated roots, see docs/workflow/00a-path-register.md
    "已知邊界").

    One shared lock, not one per resource: ADR-0001b D1 requires path and job
    mutations to serialize against each other too, e.g. a path being deleted
    mid-Rescan while a Rescan's precondition check is reading the registered
    path set. Two independent locks would let that interleave.

    Uses `pg_advisory_xact_lock`, not the session-scoped `pg_advisory_lock`,
    so the lock is released automatically on this transaction's commit or
    rollback -- no manual unlock, and no leak on an exception path.
    """
    db.execute(text("SELECT pg_advisory_xact_lock(hashtext('lfa:mutation'))"))
