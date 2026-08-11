---
applyTo: "**/test_*.py"
---
You always write unit tests using `pytest`.
You always use the `pytest-mock` plugin to manage and use test mocks.
Add the plugin to a test by adding `mocker` to the function definition: `def test_function(mocker):`.
When mocking objects, prefer to use the `mock.patch.object` decorator, rather than `mock.patch` unless the import path is not available. This helps when refactoring code.
You do not need to type parameters or return types for test functions.
Never use `monkeypatch` to mock objects, functions, methods or attributes.
You always parametrize tests instead of writing multiple similar tests.
You always use the `pytest.mark.parametrize` decorator to parametrize your tests.
You always use the `pytest.raises` context manager to test for exceptions.
You use fixtures to reuse code across tests.
You always use the `pytest` command to run your tests.
Never use mocked log calls to verify behavior.
