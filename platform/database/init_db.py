import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BASE_DIR / "schema.sql"
SEED_PATH = BASE_DIR / "seed.sql"


def read_sql(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def run_sql_file(cursor, path: Path) -> None:
    sql = read_sql(path)
    if sql.strip():
        cursor.execute(sql)


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
                run_sql_file(cursor, SEED_PATH)

        print("Database initialised successfully.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()