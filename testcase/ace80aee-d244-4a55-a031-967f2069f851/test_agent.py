
import os
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def required_env_vars():
    """Fixture returning the list of required environment variables for the application."""
    # This should match the actual required env vars in your app
    return ["APP_SECRET", "DATABASE_URL", "API_KEY"]

@pytest.fixture
def app_startup_function():
    """
    Fixture returning the application startup function.
    Replace this with the actual import from your app.
    The function should raise RuntimeError with a message listing missing env vars if any are missing.
    """
    def startup():
        missing = [var for var in ["APP_SECRET", "DATABASE_URL", "API_KEY"] if not os.environ.get(var)]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
        # Simulate successful startup
        return True
    return startup

class TestAppStartupConfigIntegration:
    """
    Integration tests for application startup configuration validation.
    """

    def test_startup_fails_with_partial_configuration(self, required_env_vars, app_startup_function):
        """
        Verifies that the application fails to start and lists missing environment variables
        when only some required environment variables are set.
        """
        # Set only one required env var
        partial_env = {required_env_vars[0]: "dummy_value"}
        with patch.dict(os.environ, partial_env, clear=True):
            with pytest.raises(RuntimeError) as excinfo:
                app_startup_function()
            # Should mention the other missing env vars
            missing_vars = set(required_env_vars[1:])
            error_msg = str(excinfo.value)
            for var in missing_vars:
                assert var in error_msg, f"Missing variable {var} not listed in error message"
            # Should not start
            assert "Missing required environment variables" in error_msg

    def test_startup_fails_with_all_configuration_missing(self, required_env_vars, app_startup_function):
        """
        Verifies that the application fails to start and lists all required environment variables
        when none are set.
        """
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError) as excinfo:
                app_startup_function()
            error_msg = str(excinfo.value)
            for var in required_env_vars:
                assert var in error_msg, f"Missing variable {var} not listed in error message"
            assert "Missing required environment variables" in error_msg

    def test_startup_succeeds_with_all_configuration_present(self, required_env_vars, app_startup_function):
        """
        Verifies that the application starts successfully when all required environment variables are set.
        """
        env = {var: "dummy_value" for var in required_env_vars}
        with patch.dict(os.environ, env, clear=True):
            result = app_startup_function()
            assert result is True, "Application should start successfully when all env vars are present"
