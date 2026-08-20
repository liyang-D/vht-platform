import os
from typing import Any

import psycopg2
import psycopg2.extras


LEGACY_MODEL_WEIGHTS = "legacy-unknown"


def get_connection():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured for evaluation history.")
    return psycopg2.connect(database_url, cursor_factory=psycopg2.extras.RealDictCursor)


def _session_select(where_clause: str) -> str:
    return f"""
        SELECT
            s.id,
            s.model_weights,
            s.summary,
            s.task_config,
            s.created_at,
            s.updated_at,
            COALESCE(
                JSONB_AGG(
                    JSONB_BUILD_OBJECT(
                        'id', m.id,
                        'role', m.role,
                        'text', m.text,
                        'created_at', m.created_at
                    ) ORDER BY m.created_at, m.id
                ) FILTER (WHERE m.id IS NOT NULL),
                '[]'::JSONB
            ) AS messages
        FROM sessions s
        JOIN access_keys ak ON ak.id = s.access_key_id
        LEFT JOIN turns t ON t.session_id = s.id
        LEFT JOIN messages m ON m.turn_id = t.id
        WHERE {where_clause}
        GROUP BY s.id
    """


def list_sessions(access_key: str, limit: int = 200) -> dict[str, Any]:
    where_clause = """
        ak.key_value = %s
        AND s.model_weights <> %s
        AND BTRIM(s.model_weights) <> ''
    """
    sessions_sql = _session_select(where_clause) + " ORDER BY s.created_at DESC LIMIT %s;"
    weights_sql = """
        SELECT DISTINCT s.model_weights
        FROM sessions s
        JOIN access_keys ak ON ak.id = s.access_key_id
        WHERE ak.key_value = %s
          AND s.model_weights <> %s
          AND BTRIM(s.model_weights) <> ''
        ORDER BY s.model_weights;
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(weights_sql, (access_key, LEGACY_MODEL_WEIGHTS))
            weights = [row["model_weights"] for row in cursor.fetchall()]
            cursor.execute(sessions_sql, (access_key, LEGACY_MODEL_WEIGHTS, limit))
            sessions = [dict(row) for row in cursor.fetchall()]
    return {"weights": weights, "sessions": sessions}


def get_session(access_key: str, session_id: str) -> dict[str, Any] | None:
    where_clause = """
        ak.key_value = %s
        AND s.id = %s
        AND s.model_weights <> %s
        AND BTRIM(s.model_weights) <> ''
    """
    sql = _session_select(where_clause) + ";"
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, (access_key, session_id, LEGACY_MODEL_WEIGHTS))
            row = cursor.fetchone()
    return dict(row) if row else None
