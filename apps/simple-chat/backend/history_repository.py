import os
from typing import Any

import psycopg2
import psycopg2.extras


LEGACY_MODEL_WEIGHTS = "legacy-unknown"
SIMPLE_CHAT_TASK_NAME = "Simple Chat"


def get_connection():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured for session history.")

    return psycopg2.connect(
        database_url,
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def list_session_history(
    access_key: str,
    model_weights: list[str] | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    selected_weights = [value.strip() for value in model_weights or [] if value.strip()]
    filters: list[str] = [
        "ak.key_value = %s",
        "s.model_weights <> %s",
        "BTRIM(s.model_weights) <> ''",
        "s.task_config->>'task_name' = %s",
    ]
    parameters: list[Any] = [access_key, LEGACY_MODEL_WEIGHTS, SIMPLE_CHAT_TASK_NAME]

    if selected_weights:
        filters.append("s.model_weights = ANY(%s)")
        parameters.append(selected_weights)

    parameters.append(limit)
    where_clause = " AND ".join(filters)
    sessions_sql = f"""
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
                    )
                    ORDER BY m.created_at
                ) FILTER (WHERE m.id IS NOT NULL),
                '[]'::JSONB
            ) AS messages
        FROM sessions s
        JOIN access_keys ak ON ak.id = s.access_key_id
        LEFT JOIN turns t ON t.session_id = s.id
        LEFT JOIN messages m ON m.turn_id = t.id
        WHERE {where_clause}
        GROUP BY s.id
        ORDER BY s.created_at DESC
        LIMIT %s;
    """
    weights_sql = """
        SELECT DISTINCT s.model_weights
        FROM sessions s
        JOIN access_keys ak ON ak.id = s.access_key_id
        WHERE ak.key_value = %s
          AND s.model_weights <> %s
          AND BTRIM(s.model_weights) <> ''
          AND s.task_config->>'task_name' = %s
        ORDER BY s.model_weights;
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(weights_sql, (access_key, LEGACY_MODEL_WEIGHTS, SIMPLE_CHAT_TASK_NAME))
            weights = [row["model_weights"] for row in cursor.fetchall()]

            cursor.execute(sessions_sql, tuple(parameters))
            sessions = [dict(row) for row in cursor.fetchall()]

    return {"weights": weights, "sessions": sessions}
