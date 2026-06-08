import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BASE_DIR / "schema.sql"


def read_sql(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def run_sql_file(cursor, path: Path) -> None:
    sql = read_sql(path)
    if sql.strip():
        cursor.execute(sql)


def seed_defaults(cursor) -> None:
    project_name = os.getenv("DEFAULT_PROJECT_NAME", "Mini Project Template")
    access_key = os.getenv("SIMPLE_CHAT_ACCESS_KEY", "dev-template-key")

    cursor.execute(
        """
        INSERT INTO projects (
            name,
            is_valid,
            quota,
            usage_count
        )
        SELECT
            %s,
            TRUE,
            NULL,
            0
        WHERE NOT EXISTS (
            SELECT 1
            FROM projects
            WHERE name = %s
        );
        """,
        (project_name, project_name),
    )

    cursor.execute(
        """
        INSERT INTO access_keys (
            key_value,
            project_id,
            is_active,
            quota,
            usage_count
        )
        VALUES (
            %s,
            (
                SELECT id
                FROM projects
                WHERE name = %s
                ORDER BY created_at ASC
                LIMIT 1
            ),
            TRUE,
            NULL,
            0
        )
        ON CONFLICT (key_value) DO NOTHING;
        """,
        (access_key, project_name),
    )


def main() -> None:
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL is not set. Add it to your .env file.")

    conn = psycopg2.connect(database_url)

    try:
        with conn:
            with conn.cursor() as cursor:
                run_sql_file(cursor, SCHEMA_PATH)
                seed_defaults(cursor)

        print("Database initialised successfully.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
