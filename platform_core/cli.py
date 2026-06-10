import argparse
import json
import secrets
import sys
from typing import Any


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def connect_to_database():
    from platform_core.database.connection import connect

    return connect()


def print_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        print("No records found.")
        return

    widths = {
        column: max(
            len(column),
            *(len(format_value(row.get(column))) for row in rows),
        )
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    divider = "  ".join("-" * widths[column] for column in columns)

    print(header)
    print(divider)
    for row in rows:
        print(
            "  ".join(
                format_value(row.get(column)).ljust(widths[column])
                for column in columns
            )
        )


def output_rows(rows: list[dict[str, Any]], columns: list[str], as_json: bool) -> None:
    if as_json:
        print(json.dumps(rows, default=str, indent=2))
        return

    print_table(rows, columns)


def get_project(cursor, project: str) -> dict[str, Any] | None:
    cursor.execute(
        """
        SELECT id, name, is_valid, priority, quota, usage_count, created_at, updated_at
        FROM projects
        WHERE id::text = %s OR name = %s
        ORDER BY created_at ASC
        LIMIT 1;
        """,
        (project, project),
    )
    return cursor.fetchone()


def create_project_if_missing(
    cursor,
    project_name: str,
    quota: int | None,
    priority: int,
) -> dict[str, Any]:
    cursor.execute(
        """
        INSERT INTO projects (name, is_valid, priority, quota, usage_count)
        VALUES (%s, TRUE, %s, %s, 0)
        RETURNING id, name, is_valid, priority, quota, usage_count, created_at, updated_at;
        """,
        (project_name, priority, quota),
    )
    return cursor.fetchone()


def list_projects(args: argparse.Namespace) -> None:
    with connect_to_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id::text AS id,
                    name,
                    is_valid,
                    priority,
                    quota,
                    usage_count,
                    created_at,
                    updated_at
                FROM projects
                ORDER BY created_at DESC;
                """
            )
            rows = [dict(row) for row in cursor.fetchall()]

    output_rows(
        rows,
        ["id", "name", "is_valid", "priority", "quota", "usage_count", "created_at"],
        args.json,
    )


def list_keys(args: argparse.Namespace) -> None:
    params: list[Any] = []
    where_clause = ""

    if args.project:
        where_clause = "WHERE p.id::text = %s OR p.name = %s"
        params.extend([args.project, args.project])

    with connect_to_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    ak.id::text AS key_id,
                    ak.key_value,
                    ak.is_active,
                    ak.quota,
                    ak.usage_count,
                    p.id::text AS project_id,
                    p.name AS project_name,
                    p.is_valid AS project_is_valid,
                    p.priority AS project_priority,
                    p.quota AS project_quota,
                    p.usage_count AS project_usage_count,
                    ak.created_at
                FROM access_keys ak
                JOIN projects p ON p.id = ak.project_id
                {where_clause}
                ORDER BY ak.created_at DESC;
                """,
                params,
            )
            rows = [dict(row) for row in cursor.fetchall()]

    output_rows(
        rows,
        [
            "key_id",
            "key_value",
            "is_active",
            "quota",
            "usage_count",
            "project_name",
            "project_priority",
            "project_quota",
            "project_usage_count",
        ],
        args.json,
    )


def create_key(args: argparse.Namespace) -> None:
    if not args.project and not args.project_id:
        raise SystemExit("Provide --project or --project-id.")

    key_value = args.key or secrets.token_urlsafe(32)

    with connect_to_database() as conn:
        with conn.cursor() as cursor:
            project = None
            if args.project_id:
                project = get_project(cursor, args.project_id)
                if project is None:
                    raise SystemExit(f"Project not found: {args.project_id}")

            if project is None and args.project:
                project = get_project(cursor, args.project)

            if project is None:
                if args.project_id:
                    raise SystemExit(f"Project not found: {args.project_id}")
                project = create_project_if_missing(
                    cursor,
                    args.project,
                    args.project_quota,
                    args.project_priority,
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
                VALUES (%s, %s, TRUE, %s, 0)
                RETURNING
                    id::text AS key_id,
                    key_value,
                    is_active,
                    quota,
                    usage_count,
                    created_at;
                """,
                (key_value, project["id"], args.quota),
            )
            key = dict(cursor.fetchone())
            key["project_id"] = str(project["id"])
            key["project_name"] = project["name"]
            key["project_priority"] = project["priority"]

    output_rows(
        [key],
        [
            "key_id",
            "key_value",
            "is_active",
            "quota",
            "usage_count",
            "project_name",
            "project_priority",
        ],
        args.json,
    )


def delete_key(args: argparse.Namespace) -> None:
    from psycopg2 import errors

    with connect_to_database() as conn:
        with conn.cursor() as cursor:
            if args.hard:
                try:
                    cursor.execute(
                        """
                        DELETE FROM access_keys
                        WHERE key_value = %s OR id::text = %s
                        RETURNING id::text AS key_id, key_value;
                        """,
                        (args.key, args.key),
                    )
                except errors.ForeignKeyViolation as exc:
                    raise SystemExit(
                        "This key is referenced by existing sessions. "
                        "Run without --hard to deactivate it instead."
                    ) from exc
            else:
                cursor.execute(
                    """
                    UPDATE access_keys
                    SET is_active = FALSE,
                        updated_at = NOW()
                    WHERE key_value = %s OR id::text = %s
                    RETURNING id::text AS key_id, key_value, is_active;
                    """,
                    (args.key, args.key),
                )

            row = cursor.fetchone()
            if row is None:
                raise SystemExit(f"Key not found: {args.key}")

    action = "Deleted" if args.hard else "Deactivated"
    print(f"{action} key: {row['key_value']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vht-platform")
    subparsers = parser.add_subparsers(dest="command", required=True)

    projects = subparsers.add_parser("projects")
    project_commands = projects.add_subparsers(dest="project_command", required=True)
    projects_list = project_commands.add_parser("list")
    projects_list.add_argument("--json", action="store_true")
    projects_list.set_defaults(func=list_projects)

    keys = subparsers.add_parser("keys")
    key_commands = keys.add_subparsers(dest="key_command", required=True)

    keys_create = key_commands.add_parser("create")
    keys_create.add_argument("--project", help="Project name or id.")
    keys_create.add_argument("--project-id", help="Existing project id.")
    keys_create.add_argument("--key", help="Access key value. Autogenerated if omitted.")
    keys_create.add_argument("--quota", type=int, help="Access key quota.")
    keys_create.add_argument("--project-quota", type=int, help="Quota for newly created project.")
    keys_create.add_argument(
        "--project-priority",
        type=int,
        default=30,
        help="Priority for newly created project. Lower numbers are scheduled first.",
    )
    keys_create.add_argument("--json", action="store_true")
    keys_create.set_defaults(func=create_key)

    keys_delete = key_commands.add_parser("delete")
    keys_delete.add_argument("key", help="Access key value or key id.")
    keys_delete.add_argument(
        "--hard",
        action="store_true",
        help="Physically delete the key. Default is safe deactivation.",
    )
    keys_delete.set_defaults(func=delete_key)

    keys_list = key_commands.add_parser("list")
    keys_list.add_argument("--project", help="Filter by project name or id.")
    keys_list.add_argument("--json", action="store_true")
    keys_list.set_defaults(func=list_keys)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
