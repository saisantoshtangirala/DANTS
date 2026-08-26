from src.trading.risk_manager import RiskManager

TRADING_CONFIG = {
    "capital": {"initial": 1_000_000},
    "position_sizing": {
        "method": "kelly_fraction",
        "max_risk_per_trade_pct": 0.02,
        "kelly_fraction": 0.25,
        "max_position_size_pct": 0.10,
    },
    "risk_management": {
        "daily_loss_limit_pct": 0.03,
        "consecutive_loss_limit": 3,
        "vix_spike_threshold": 25,
        "max_drawdown_pct": 0.12,
        "max_open_positions": 5,
    },
}


def make_manager() -> RiskManager:
    return RiskManager(TRADING_CONFIG)


def test_position_size_respects_max_position_cap():
    rm = make_manager()
    size = rm.position_size(confidence=1.0, win_rate=0.9, avg_win_pct=0.05, avg_loss_pct=0.005)
    assert size <= TRADING_CONFIG["position_sizing"]["max_position_size_pct"]


def test_position_size_zero_when_halted():
    rm = make_manager()
    rm.state.halted = True
    assert rm.position_size(confidence=1.0, win_rate=0.9, avg_win_pct=0.05, avg_loss_pct=0.005) == 0.0


def test_position_size_scales_with_confidence():
    rm = make_manager()
    high_conf = rm.position_size(confidence=1.0, win_rate=0.7, avg_win_pct=0.02, avg_loss_pct=0.01)
    low_conf = rm.position_size(confidence=0.3, win_rate=0.7, avg_win_pct=0.02, avg_loss_pct=0.01)
    assert high_conf > low_conf


def test_can_open_position_respects_max_open_positions():
    rm = make_manager()
    for _ in range(5):
        assert rm.can_open_position()
        rm.open_position()
    assert not rm.can_open_position()


def test_daily_loss_limit_halts_trading():
    rm = make_manager()
    rm.record_trade_result(pnl=-35_000)  # 3.5% of 1,000,000 -> breaches 3% daily loss limit
    assert rm.state.halted
    assert "daily_loss_limit_breached" in rm.state.halt_reason
    assert rm.position_size(confidence=1.0, win_rate=0.9, avg_win_pct=0.05, avg_loss_pct=0.005) == 0.0


def test_max_drawdown_halts_trading():
    rm = make_manager()
    # Small losses spread across separate days (reset_daily between them)
    # accumulate drawdown from the peak without ever breaching the (larger)
    # daily loss limit on any single day.
    for _ in range(5):
        rm.record_trade_result(pnl=-25_000)
        if rm.state.halted:
            break
        rm.reset_daily()

    assert rm.state.halted
    assert "max_drawdown_breached" in rm.state.halt_reason


def test_consecutive_losses_trigger_size_reduction():
    rm = make_manager()
    for _ in range(3):
        rm.record_trade_result(pnl=-1_000)
    assert rm.consecutive_loss_size_multiplier() == 0.5


def test_win_resets_consecutive_loss_counter():
    rm = make_manager()
    rm.record_trade_result(pnl=-1_000)
    rm.record_trade_result(pnl=-1_000)
    rm.record_trade_result(pnl=1_000)
    assert rm.state.consecutive_losses == 0
    assert rm.consecutive_loss_size_multiplier() == 1.0


def test_reset_daily_clears_halt_and_daily_pnl():
    rm = make_manager()
    rm.record_trade_result(pnl=-35_000)
    assert rm.state.halted
    rm.reset_daily()
    assert not rm.state.halted
    assert rm.state.daily_pnl == 0.0


def test_vix_spike_halts_trading():
    rm = make_manager()
    reason = rm.check_circuit_breakers(india_vix=30)
    assert reason is not None
    assert "vix_spike" in reason
    assert rm.state.halted
