import os

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv


def get_database_url() -> str:
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set.")

    return database_url


def connect():
    return psycopg2.connect(
        get_database_url(),
        cursor_factory=psycopg2.extras.RealDictCursor,
    )
