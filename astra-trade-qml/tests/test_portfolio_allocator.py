from src.trading.portfolio_allocator import PortfolioAllocator


def make_report(total_trades=30, expectancy=0.001, sharpe_ratio=1.0):
    return {"total_trades": total_trades, "expectancy": expectancy, "sharpe_ratio": sharpe_ratio}


def test_allocate_splits_capital_across_profitable_symbols():
    allocator = PortfolioAllocator(total_capital=50_000, min_trades=20, min_adtv_cr=10.0, max_symbols=5)
    backtest_results = {
        "RELIANCE": make_report(expectancy=0.002, sharpe_ratio=1.5),
        "TCS": make_report(expectancy=0.001, sharpe_ratio=1.0),
    }
    liquidity = {"RELIANCE": 500.0, "TCS": 300.0}

    result = allocator.allocate(backtest_results, liquidity, tradable_symbols=["RELIANCE", "TCS"])

    assert not result.excluded
    assert {a.symbol for a in result.allocations} == {"RELIANCE", "TCS"}
    assert abs(sum(a.allocated_capital for a in result.allocations) - 50_000) < 1e-6
    # RELIANCE has 2x the expectancy of TCS, so should get 2x the capital.
    by_symbol = {a.symbol: a.allocated_capital for a in result.allocations}
    assert by_symbol["RELIANCE"] > by_symbol["TCS"]


def test_allocate_excludes_symbol_with_too_few_trades():
    allocator = PortfolioAllocator(total_capital=50_000, min_trades=20, min_adtv_cr=10.0)
    backtest_results = {"RELIANCE": make_report(total_trades=5)}
    liquidity = {"RELIANCE": 500.0}

    result = allocator.allocate(backtest_results, liquidity, tradable_symbols=["RELIANCE"])

    assert result.allocations == []
    assert "RELIANCE" in result.excluded
    assert "5" in result.excluded["RELIANCE"]


def test_allocate_excludes_illiquid_symbol():
    allocator = PortfolioAllocator(total_capital=50_000, min_trades=20, min_adtv_cr=10.0)
    backtest_results = {"SMALLCAP": make_report()}
    liquidity = {"SMALLCAP": 2.0}

    result = allocator.allocate(backtest_results, liquidity, tradable_symbols=["SMALLCAP"])

    assert result.allocations == []
    assert "SMALLCAP" in result.excluded
    assert "ADTV" in result.excluded["SMALLCAP"]


def test_allocate_excludes_negative_expectancy_symbol():
    allocator = PortfolioAllocator(total_capital=50_000, min_trades=20, min_adtv_cr=10.0)
    backtest_results = {"LOSER": make_report(expectancy=-0.001)}
    liquidity = {"LOSER": 500.0}

    result = allocator.allocate(backtest_results, liquidity, tradable_symbols=["LOSER"])

    assert result.allocations == []
    assert "LOSER" in result.excluded
    assert "expectancy" in result.excluded["LOSER"]


def test_allocate_caps_at_max_symbols_ranked_by_sharpe():
    allocator = PortfolioAllocator(total_capital=50_000, min_trades=20, min_adtv_cr=10.0, max_symbols=1)
    backtest_results = {
        "HIGH_SHARPE": make_report(sharpe_ratio=2.0),
        "LOW_SHARPE": make_report(sharpe_ratio=0.5),
    }
    liquidity = {"HIGH_SHARPE": 500.0, "LOW_SHARPE": 500.0}

    result = allocator.allocate(
        backtest_results, liquidity, tradable_symbols=["HIGH_SHARPE", "LOW_SHARPE"]
    )

    assert len(result.allocations) == 1
    assert result.allocations[0].symbol == "HIGH_SHARPE"
    assert result.allocations[0].allocated_capital == 50_000


def test_allocate_returns_empty_when_no_symbols_qualify():
    allocator = PortfolioAllocator(total_capital=50_000)
    result = allocator.allocate({}, {}, tradable_symbols=["RELIANCE"])

    assert result.allocations == []
    assert "RELIANCE" in result.excluded


def test_as_capital_map_and_as_dict_round_trip():
    allocator = PortfolioAllocator(total_capital=50_000, min_trades=20, min_adtv_cr=10.0)
    backtest_results = {"RELIANCE": make_report()}
    liquidity = {"RELIANCE": 500.0}

    result = allocator.allocate(backtest_results, liquidity, tradable_symbols=["RELIANCE"])

    assert result.as_capital_map() == {"RELIANCE": 50_000.0}
    as_dict = result.as_dict()
    assert as_dict["allocations"][0]["symbol"] == "RELIANCE"
    assert as_dict["excluded"] == {}
