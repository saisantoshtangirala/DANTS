from src.utils.config import get


def test_load_config_has_project_section(config):
    assert config["project"]["name"] == "Astra-Trade-QML"


def test_load_regimes_has_bull_trend(regimes_config):
    assert "bull_trend" in regimes_config["regimes"]


def test_get_dotted_key(config):
    assert get(config, "trading.risk_management.max_drawdown_pct") == 0.12


def test_get_dotted_key_missing_returns_default(config):
    assert get(config, "trading.does.not.exist", default="fallback") == "fallback"
