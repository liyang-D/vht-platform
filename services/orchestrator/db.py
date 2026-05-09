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
) -> dict[str, Any]:
    sql = """
        INSERT INTO sessions (
            project_id,
            access_key_id,
            task_config
        )
        VALUES (%s, %s, %s)
        RETURNING id, project_id, access_key_id, task_config, summary, created_at, updated_at;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    project_id,
                    access_key_id,
                    psycopg2.extras.Json(task_config),
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
            s.summary,
            s.created_at AS session_created_at,
            s.updated_at AS session_updated_at,

            p.name AS project_name,
            p.is_valid AS project_is_valid,
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


def add_message(
    session_id: str,
    role: str,
    text: str,
    structured_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sql = """
        INSERT INTO messages (
            session_id,
            role,
            text,
            structured_output
        )
        VALUES (%s, %s, %s, %s)
        RETURNING id, session_id, role, text, structured_output, created_at;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    session_id,
                    role,
                    text,
                    psycopg2.extras.Json(structured_output)
                    if structured_output is not None
                    else None,
                ),
            )
            row = cur.fetchone()
            return dict(row)


def get_session_messages(session_id: str) -> list[dict[str, Any]]:
    sql = """
        SELECT
            id,
            session_id,
            role,
            text,
            structured_output,
            created_at
        FROM messages
        WHERE session_id = %s
        ORDER BY created_at ASC;
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