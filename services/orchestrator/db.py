import json
import os
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv


def load_project_env() -> None:
    """
    Load .env from the project root if available.
    This file is at services/orchestrator/db.py, so parents[2] should be vht-platform/.
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


load_project_env()


def get_connection():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is not set in .env")

    return psycopg2.connect(
        database_url,
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def get_access_key_with_project(key_value: str) -> dict[str, Any] | None:
    sql = """
        SELECT
            ak.id AS access_key_id,
            ak.key_value,
            ak.is_active,
            ak.quota AS access_key_quota,
            ak.usage_count AS access_key_usage_count,
            p.id AS project_id,
            p.name AS project_name,
            p.is_valid AS project_is_valid,
            p.priority AS project_priority,
            p.quota AS project_quota,
            p.usage_count AS project_usage_count
        FROM access_keys ak
        JOIN projects p ON ak.project_id = p.id
        WHERE ak.key_value = %s
        LIMIT 1;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (key_value,))
            row = cur.fetchone()
            return dict(row) if row else None


def create_session_record(
    project_id: str,
    access_key_id: str,
    task_config: dict[str, Any],
    runtime_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sql = """
        INSERT INTO sessions (
            project_id,
            access_key_id,
            task_config,
            runtime_state
        )
        VALUES (%s, %s, %s, %s)
        RETURNING id, project_id, access_key_id, task_config, runtime_state, summary, created_at, updated_at;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    project_id,
                    access_key_id,
                    psycopg2.extras.Json(task_config),
                    psycopg2.extras.Json(runtime_state)
                    if runtime_state is not None
                    else None,
                ),
            )
            row = cur.fetchone()
            return dict(row)


def get_session_with_project_and_key(session_id: str) -> dict[str, Any] | None:
    sql = """
        SELECT
            s.id AS session_id,
            s.project_id,
            s.access_key_id,
            s.task_config,
            s.runtime_state,
            s.summary,
            s.created_at AS session_created_at,
            s.updated_at AS session_updated_at,

            p.name AS project_name,
            p.is_valid AS project_is_valid,
            p.priority AS project_priority,
            p.quota AS project_quota,
            p.usage_count AS project_usage_count,

            ak.key_value,
            ak.is_active AS access_key_is_active,
            ak.quota AS access_key_quota,
            ak.usage_count AS access_key_usage_count
        FROM sessions s
        JOIN projects p ON s.project_id = p.id
        JOIN access_keys ak ON s.access_key_id = ak.id
        WHERE s.id = %s
        LIMIT 1;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def create_turn(
    session_id: str,
    interaction_mode: str = "free",
    status: str = "processing",
    previous_step: dict[str, Any] | None = None,
    current_step: dict[str, Any] | None = None,
    next_step: dict[str, Any] | None = None,
    prompt_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sql = """
        INSERT INTO turns (
            session_id,
            interaction_mode,
            status,
            previous_step,
            current_step,
            next_step,
            prompt_metadata,
            started_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        RETURNING
            id,
            session_id,
            interaction_mode,
            status,
            previous_step,
            current_step,
            next_step,
            structured_output,
            prompt_metadata,
            usage,
            created_at,
            started_at,
            completed_at;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    session_id,
                    interaction_mode,
                    status,
                    psycopg2.extras.Json(previous_step)
                    if previous_step is not None
                    else None,
                    psycopg2.extras.Json(current_step)
                    if current_step is not None
                    else None,
                    psycopg2.extras.Json(next_step)
                    if next_step is not None
                    else None,
                    psycopg2.extras.Json(prompt_metadata)
                    if prompt_metadata is not None
                    else None,
                ),
            )
            row = cur.fetchone()
            return dict(row)


def update_turn(
    turn_id: str,
    status: str,
    structured_output: dict[str, Any] | None = None,
    prompt_metadata: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sql = """
        UPDATE turns
        SET status = %s,
            structured_output = %s,
            prompt_metadata = COALESCE(%s, prompt_metadata),
            usage = %s,
            completed_at = CASE
                WHEN %s IN ('completed', 'cancelled', 'failed') THEN NOW()
                ELSE completed_at
            END
        WHERE id = %s
        RETURNING
            id,
            session_id,
            interaction_mode,
            status,
            previous_step,
            current_step,
            next_step,
            structured_output,
            prompt_metadata,
            usage,
            created_at,
            started_at,
            completed_at;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    status,
                    psycopg2.extras.Json(structured_output)
                    if structured_output is not None
                    else None,
                    psycopg2.extras.Json(prompt_metadata)
                    if prompt_metadata is not None
                    else None,
                    psycopg2.extras.Json(usage) if usage is not None else None,
                    status,
                    turn_id,
                ),
            )
            row = cur.fetchone()
            return dict(row)


def add_message(
    turn_id: str,
    role: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sql = """
        INSERT INTO messages (
            turn_id,
            role,
            text,
            metadata
        )
        VALUES (%s, %s, %s, %s)
        RETURNING id, turn_id, role, text, metadata, created_at;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    turn_id,
                    role,
                    text,
                    psycopg2.extras.Json(metadata) if metadata is not None else None,
                ),
            )
            row = cur.fetchone()
            return dict(row)


def create_audit_event(
    event_type: str,
    event_status: str,
    project_id: str | None = None,
    access_key_id: str | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    actor_type: str | None = None,
    actor_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sql = """
        INSERT INTO audit_events (
            project_id,
            access_key_id,
            session_id,
            turn_id,
            actor_type,
            actor_id,
            event_type,
            event_status,
            metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING
            id,
            project_id,
            access_key_id,
            session_id,
            turn_id,
            actor_type,
            actor_id,
            event_type,
            event_status,
            metadata,
            created_at;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    project_id,
                    access_key_id,
                    session_id,
                    turn_id,
                    actor_type,
                    actor_id,
                    event_type,
                    event_status,
                    psycopg2.extras.Json(metadata) if metadata is not None else None,
                ),
            )
            row = cur.fetchone()
            return dict(row)


def get_session_messages(session_id: str) -> list[dict[str, Any]]:
    sql = """
        SELECT
            m.id,
            m.turn_id,
            m.role,
            m.text,
            m.metadata,
            m.created_at
        FROM messages m
        JOIN turns t ON m.turn_id = t.id
        WHERE t.session_id = %s
        ORDER BY m.created_at ASC, m.id ASC;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id,))
            rows = cur.fetchall()
            return [dict(row) for row in rows]


def get_session_turns(session_id: str) -> list[dict[str, Any]]:
    sql = """
        SELECT
            id,
            session_id,
            interaction_mode,
            status,
            previous_step,
            current_step,
            next_step,
            structured_output,
            prompt_metadata,
            usage,
            created_at,
            started_at,
            completed_at
        FROM turns
        WHERE session_id = %s
        ORDER BY created_at ASC, id ASC;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id,))
            rows = cur.fetchall()
            return [dict(row) for row in rows]


def increment_usage(
    project_id: str,
    access_key_id: str,
    amount: int = 1,
) -> None:
    """
    usage_count is intentionally unit-flexible.
    For now, amount may be request count or token count.
    """
    update_project_sql = """
        UPDATE projects
        SET usage_count = usage_count + %s,
            updated_at = NOW()
        WHERE id = %s;
    """

    update_key_sql = """
        UPDATE access_keys
        SET usage_count = usage_count + %s,
            updated_at = NOW()
        WHERE id = %s;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(update_project_sql, (amount, project_id))
            cur.execute(update_key_sql, (amount, access_key_id))


def update_session_summary(
    session_id: str,
    summary: str,
) -> dict[str, Any]:
    sql = """
        UPDATE sessions
        SET summary = %s,
            updated_at = NOW()
        WHERE id = %s
        RETURNING id, summary, updated_at;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (summary, session_id))
            row = cur.fetchone()
            return dict(row)


def update_session_runtime_state(
    session_id: str,
    runtime_state: dict[str, Any] | None,
) -> dict[str, Any]:
    sql = """
        UPDATE sessions
        SET runtime_state = %s,
            updated_at = NOW()
        WHERE id = %s
        RETURNING id, runtime_state, updated_at;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    psycopg2.extras.Json(runtime_state)
                    if runtime_state is not None
                    else None,
                    session_id,
                ),
            )
            row = cur.fetchone()
            return dict(row)


def serialise_task_config(task_config: Any) -> dict[str, Any]:
    """
    psycopg2 may return JSONB as dict already.
    This keeps the rest of the code safe.
    """
    if task_config is None:
        return {}

    if isinstance(task_config, dict):
        return task_config

    if isinstance(task_config, str):
        return json.loads(task_config)

    return dict(task_config)
