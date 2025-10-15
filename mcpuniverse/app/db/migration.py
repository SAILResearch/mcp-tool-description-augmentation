"""Utilities for database migration."""
import argparse
import asyncio
import os
import threading
from typing import Dict, List, Sequence

from sqlalchemy import text

from mcpuniverse.app.db.database import sessionmanager


def _load_migration_sqls(folder: str = "") -> Dict[str, List[str]]:
    """Load migration SQLs defined in the folder."""
    if folder == "":
        folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "migration")
    if not os.path.isdir(folder):
        raise ValueError(f"Path {folder} is not a folder")

    ups, downs = [], []
    for filename in os.listdir(folder):
        if filename.endswith("up.sql"):
            ups.append(filename)
        elif filename.endswith("down.sql"):
            downs.append(filename)
    ups = sorted(ups)
    downs = sorted(downs, reverse=True)

    up_sqls, down_sqls = [], []
    for filename in ups:
        with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
            up_sqls.append(f.read())
    for filename in downs:
        with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
            down_sqls.append(f.read())
    return {"up": up_sqls, "down": down_sqls}


def run_migration(folder: str = ""):
    """
    Run database migration.

    Args:
        folder (str): The folder including migration SQLs.
    """

    async def _migrate():
        sqls = _load_migration_sqls(folder)
        async with sessionmanager.connect() as conn:
            try:
                for sql in sqls["up"]:
                    await conn.execute(text(sql))
            except Exception:
                pass

    thread = threading.Thread(target=asyncio.run, args=(_migrate(),))
    thread.start()
    thread.join()


def _resolve_db_url(cli_value: str | None) -> str | None:
    """Return the database URL from CLI or environment variables."""

    if cli_value:
        return cli_value

    for env_var in ("DB_SOURCE", "DB_URL", "DATABASE_URL"):
        value = os.getenv(env_var)
        if value:
            return value
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line interface for running database migrations."""

    parser = argparse.ArgumentParser(description="Apply SQL migrations for MCP-Universe.")
    parser.add_argument(
        "--db-url",
        default=None,
        help=(
            "Database connection string. Defaults to DB_SOURCE, DB_URL, or DATABASE_URL environment variables."
        ),
    )
    parser.add_argument(
        "--folder",
        default="",
        help="Directory containing migration SQL files (default: built-in migrations).",
    )

    args = parser.parse_args(argv)
    db_url = _resolve_db_url(args.db_url)
    if not db_url:
        parser.error(
            "Database URL not provided. Use --db-url or set DB_SOURCE/DB_URL/DATABASE_URL."
        )

    sessionmanager.init(db_url)
    try:
        run_migration(folder=args.folder)
    finally:
        asyncio.run(sessionmanager.close())

    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    import sys

    sys.exit(main())
