"""Postgres index for ACE structured state; binary artifacts remain on the volume."""

from __future__ import annotations

import json
import os
from typing import Any

import psycopg2
import psycopg2.extras


def _connect():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured for ACE history.")
    return psycopg2.connect(
        database_url, cursor_factory=psycopg2.extras.RealDictCursor
    )


def upsert(
    *,
    session_id: str,
    status: str,
    config: dict[str, Any],
    runtime_config: dict[str, Any],
    checkpoint: dict[str, Any] | None,
    report: dict[str, Any] | None,
    storage_path: str,
) -> None:
    participant = config.get("patient", {}).get("name")
    with _connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO ace_assessments (
                    session_id, status, participant_name, assessor_name, config,
                    runtime_config, latest_checkpoint, report, storage_path
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    participant_name = EXCLUDED.participant_name,
                    assessor_name = EXCLUDED.assessor_name,
                    config = EXCLUDED.config,
                    runtime_config = EXCLUDED.runtime_config,
                    latest_checkpoint = EXCLUDED.latest_checkpoint,
                    report = EXCLUDED.report,
                    storage_path = EXCLUDED.storage_path,
                    updated_at = CASE
                        WHEN ace_assessments.status IS DISTINCT FROM EXCLUDED.status
                          OR ace_assessments.config IS DISTINCT FROM EXCLUDED.config
                          OR ace_assessments.runtime_config IS DISTINCT FROM EXCLUDED.runtime_config
                          OR ace_assessments.latest_checkpoint IS DISTINCT FROM EXCLUDED.latest_checkpoint
                          OR ace_assessments.report IS DISTINCT FROM EXCLUDED.report
                          OR ace_assessments.storage_path IS DISTINCT FROM EXCLUDED.storage_path
                        THEN NOW()
                        ELSE ace_assessments.updated_at
                    END
                """,
                (
                    session_id,
                    status,
                    participant,
                    config.get("assessor"),
                    json.dumps(config),
                    json.dumps(runtime_config),
                    json.dumps(checkpoint) if checkpoint is not None else None,
                    json.dumps(report) if report is not None else None,
                    storage_path,
                ),
            )


def list_assessments() -> list[dict[str, Any]]:
    with _connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT session_id, status, participant_name AS participant,
                       assessor_name AS assessor, latest_checkpoint, report,
                       created_at, updated_at
                FROM ace_assessments
                ORDER BY updated_at DESC
                """
            )
            return [dict(row) for row in cursor.fetchall()]


def get(session_id: str) -> dict[str, Any] | None:
    with _connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM ace_assessments WHERE session_id = %s",
                (session_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None


def save_review_note(session_id: str, key: str, note: str) -> None:
    with _connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE ace_assessments
                SET review_notes = JSONB_SET(
                        COALESCE(review_notes, '{}'::jsonb),
                        ARRAY[%s], TO_JSONB(%s::text), TRUE
                    ),
                    updated_at = NOW()
                WHERE session_id = %s
                """,
                (key, note, session_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(session_id)
