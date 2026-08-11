from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from pathlib import Path

import uvicorn
from alembic.config import Config
from dotenv import load_dotenv
from loguru import logger as log
from sqlalchemy import create_engine

from alembic import command
from remy.models.recipe import RecipeModel

load_dotenv()


def run_server(args: Namespace) -> None:
    """Run the FastAPI app.

    Args:
        args: Parsed CLI arguments.
    """
    uvicorn.run("remy.app:app", host="127.0.0.1", port=8000, reload=True)


def _project_root() -> Path:
    """Return the project root based on this module path.

    Returns:
        The absolute project root path.
    """
    return Path(__file__).resolve().parents[2]


def _normalize_database_url_for_alembic(database_url: str) -> str:
    """Convert async SQLAlchemy URLs to sync variants for Alembic.

    Alembic's default sync environment cannot connect with async drivers,
    so ``postgresql+asyncpg`` URLs are rewritten to ``postgresql+psycopg2``.

    Args:
        database_url: Raw database URL from environment.

    Returns:
        A URL compatible with Alembic's sync migration environment.
    """
    if database_url.startswith("postgresql+asyncpg://"):
        return database_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
    return database_url


def _build_alembic_config(database_url: str) -> Config:
    """Create Alembic config with runtime database URL.

    Args:
        database_url: Database URL to inject into the Alembic config.

    Returns:
        Configured Alembic ``Config`` instance.
    """
    alembic_ini_path = _project_root() / "alembic.ini"
    alembic_config = Config(str(alembic_ini_path))
    alembic_config.set_main_option("sqlalchemy.url", _normalize_database_url_for_alembic(database_url))
    return alembic_config


def _has_migration_scripts() -> bool:
    """Check whether Alembic revision files exist.

    Returns:
        True when at least one migration revision file exists.
    """
    versions_dir = _project_root() / "alembic" / "versions"
    if not versions_dir.exists() or not versions_dir.is_dir():
        return False

    return any(file_path.suffix == ".py" and file_path.name != "__init__.py" for file_path in versions_dir.iterdir())


def _initialize_database_schema(database_url: str) -> None:
    """Create database schema directly from SQLModel metadata.

    This is a bootstrap path used only when no Alembic revisions exist yet.

    Args:
        database_url: Raw database URL from environment.
    """
    sync_database_url = _normalize_database_url_for_alembic(database_url)
    engine = create_engine(sync_database_url, echo=False)
    with engine.begin() as connection:
        # pyrefly: ignore [missing-attribute]
        RecipeModel.metadata.create_all(connection, tables=[RecipeModel.__table__])

    engine.dispose()


def _upgrade_or_bootstrap(alembic_config: Config, args: Namespace, database_url: str) -> None:
    """Run upgrade command or bootstrap schema when no revisions exist.

    Args:
        alembic_config: Prepared Alembic config.
        args: Parsed CLI arguments.
        database_url: Raw database URL from environment.
    """
    if args.target == "head" and not _has_migration_scripts():
        log.info("No Alembic revisions found. Bootstrapping database schema from SQLModel metadata.")
        _initialize_database_schema(database_url=database_url)
        return

    command.upgrade(alembic_config, args.target, sql=args.sql)


def _run_revision_command(alembic_config: Config, args: Namespace) -> None:
    """Create a new Alembic revision.

    Args:
        alembic_config: Prepared Alembic config.
        args: Parsed CLI arguments.

    Raises:
        ValueError: If revision message is missing.
    """
    if not args.message:
        raise ValueError("--message is required when command is 'revision'")

    command.revision(alembic_config, message=args.message, autogenerate=args.autogenerate)


def migrate(args: Namespace) -> None:
    """Run Alembic migration commands.

    Args:
        args: Parsed CLI arguments.

    Raises:
        ValueError: If required migration options are missing.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        message = "DATABASE_URL must be set to run migrations"
        raise ValueError(message)

    alembic_config = _build_alembic_config(database_url=database_url)
    migration_command = args.command

    log.info("Running Alembic command '{}'", migration_command)
    migration_handlers: dict[str, Callable[[], None]] = {
        "upgrade": lambda: _upgrade_or_bootstrap(alembic_config, args, database_url),
        "downgrade": lambda: command.downgrade(alembic_config, args.target, sql=args.sql),
        "current": lambda: command.current(alembic_config, verbose=args.verbose),
        "history": lambda: command.history(alembic_config, rev_range=args.target, verbose=args.verbose),
        "heads": lambda: command.heads(alembic_config, verbose=args.verbose),
        "stamp": lambda: command.stamp(alembic_config, revision=args.target, sql=args.sql),
        "revision": lambda: _run_revision_command(alembic_config, args),
    }

    migration_handler = migration_handlers.get(migration_command)
    if migration_handler is None:
        raise ValueError(f"Unsupported migration command: {migration_command}")

    migration_handler()


def cli() -> Namespace:
    """Build and parse command line arguments.

    Returns:
        Parsed CLI namespace.
    """
    parser = ArgumentParser(prog="remy", description="Remy Agent CLI")
    subparser = parser.add_subparsers(title="subcommands", dest="subcommand", required=True)

    run_subparser = subparser.add_parser(
        "run",
        help="Run the Remy API service",
    )
    run_subparser.set_defaults(func=run_server)

    migrate_subparser = subparser.add_parser(
        "migrate",
        help="Run database migrations.",
    )
    migrate_subparser.add_argument(
        "command",
        nargs="?",
        default="upgrade",
        choices=["upgrade", "downgrade", "current", "history", "heads", "stamp", "revision"],
        help="Alembic command to execute.",
    )
    migrate_subparser.add_argument(
        "target",
        nargs="?",
        default="head",
        help="Revision target (for upgrade/downgrade/stamp/history).",
    )
    migrate_subparser.add_argument(
        "-m",
        "--message",
        help="Revision message (required for 'revision').",
    )
    migrate_subparser.add_argument(
        "--autogenerate",
        action="store_true",
        help="Autogenerate revision contents from metadata changes.",
    )
    migrate_subparser.add_argument(
        "--sql",
        action="store_true",
        help="Emit SQL to stdout instead of executing it.",
    )
    migrate_subparser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output for history/current/heads.",
    )
    migrate_subparser.set_defaults(func=migrate)

    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = cli()
    args.func(args)
