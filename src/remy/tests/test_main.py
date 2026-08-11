"""Tests for the CLI bootstrap helpers."""

from remy import __main__ as main_mod


def test_initialize_database_schema_enables_pgvector_before_create_all(monkeypatch):
    """Bootstrap should enable pgvector before creating the recipe table."""

    class FakeConnection:
        def __init__(self) -> None:
            self.executed_statements: list[str] = []

        def execute(self, statement) -> None:
            self.executed_statements.append(str(statement))

        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class FakeEngine:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
            self.connection = FakeConnection()

        def begin(self) -> FakeConnection:
            return self.connection

        def dispose(self) -> None:
            return None

    def fake_create_engine(*args, **kwargs):  # noqa: ARG002
        return FakeEngine()

    created_tables: list[object] = []

    def fake_create_all(connection, tables) -> None:  # noqa: ARG002
        created_tables.append(tables)
        assert any("CREATE EXTENSION IF NOT EXISTS vector" in statement for statement in connection.executed_statements)

    monkeypatch.setattr(main_mod, "create_engine", fake_create_engine)
    monkeypatch.setattr(main_mod.RecipeModel.metadata, "create_all", fake_create_all)

    main_mod._initialize_database_schema("postgresql+psycopg2://user:pass@localhost/db")

    assert created_tables
