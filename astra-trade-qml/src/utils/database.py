"""
Database module for Astra-Trade QML.
Manages SQLite/PostgreSQL connections for trade journal, model metrics, and audit logs.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import json

import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class TradeRecord(Base):
    """Database model for trade journal entries."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String(50), nullable=False)
    action = Column(String(10), nullable=False)  # BUY, SELL, SHORT, COVER
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    confidence = Column(Float, nullable=False)
    regime = Column(String(50), nullable=False)
    model_version = Column(String(100), nullable=False)
    quantum_depth = Column(Integer, default=0)
    strategy = Column(String(50), nullable=False)
    status = Column(String(20), default="OPEN")  # OPEN, CLOSED, CANCELLED
    is_paper = Column(Boolean, default=True)
    metadata_json = Column(Text, nullable=True)


class ModelMetrics(Base):
    """Database model for model performance tracking."""
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)  # lstm, xgboost, vqc, ensemble
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    training_duration_seconds = Column(Float, nullable=True)
    quantum_circuit_depth = Column(Integer, nullable=True)
    regime = Column(String(50), nullable=True)


class DailySummary(Base):
    """Database model for daily P&L and risk summaries."""
    __tablename__ = "daily_summaries"

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    starting_capital = Column(Float, nullable=False)
    ending_capital = Column(Float, nullable=False)
    total_pnl = Column(Float, nullable=False)
    total_pnl_pct = Column(Float, nullable=False)
    num_trades = Column(Integer, default=0)
    num_wins = Column(Integer, default=0)
    num_losses = Column(Integer, default=0)
    win_rate = Column(Float, nullable=True)
    max_drawdown_pct = Column(Float, nullable=True)
    vix_at_close = Column(Float, nullable=True)
    regime = Column(String(50), nullable=True)
    notes = Column(Text, nullable=True)


class DatabaseManager:
    """Manages database connections and CRUD operations."""

    def __init__(self, database_url: str = "sqlite:///logs/astra_trade.db"):
        """
        Initialize database manager.

        Args:
            database_url: SQLAlchemy connection string
        """
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    @contextmanager
    def session(self):
        """Context manager for database sessions."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def log_trade(self, trade_data: Dict[str, Any]) -> int:
        """
        Log a trade to the database.

        Args:
            trade_data: Dictionary with trade details

        Returns:
            ID of inserted record
        """
        with self.session() as s:
            record = TradeRecord(**trade_data)
            s.add(record)
            s.flush()
            return record.id

    def update_trade(self, trade_id: int, updates: Dict[str, Any]) -> None:
        """Update an existing trade record."""
        with self.session() as s:
            trade = s.query(TradeRecord).filter_by(id=trade_id).first()
            if trade:
                for key, value in updates.items():
                    setattr(trade, key, value)

    def log_model_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log model training metrics."""
        with self.session() as s:
            record = ModelMetrics(**metrics)
            s.add(record)

    def log_daily_summary(self, summary: Dict[str, Any]) -> None:
        """Log daily trading summary."""
        with self.session() as s:
            record = DailySummary(**summary)
            s.add(record)

    def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        status: Optional[str] = None,
        is_paper: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Retrieve trades with optional filters.

        Returns:
            DataFrame of matching trades
        """
        query = "SELECT * FROM trades WHERE 1=1"
        params = {}

        if symbol:
            query += " AND symbol = :symbol"
            params["symbol"] = symbol
        if start_date:
            query += " AND timestamp >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND timestamp <= :end_date"
            params["end_date"] = end_date
        if status:
            query += " AND status = :status"
            params["status"] = status
        if is_paper is not None:
            query += " AND is_paper = :is_paper"
            params["is_paper"] = is_paper

        query += " ORDER BY timestamp DESC"

        return pd.read_sql_query(query, self.engine, params=params)

    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get performance summary for the last N days.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with performance metrics
        """
        query = """
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
            SUM(pnl) as total_pnl,
            AVG(pnl_pct) as avg_return_pct,
            AVG(confidence) as avg_confidence
        FROM trades 
        WHERE timestamp >= datetime('now', '-{} days')
        AND status = 'CLOSED'
        """.format(days)

        df = pd.read_sql_query(query, self.engine)

        if df.empty or df["total_trades"].iloc[0] == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_return_pct": 0.0,
                "avg_confidence": 0.0,
            }

        total = df["total_trades"].iloc[0]
        wins = df["winning_trades"].iloc[0]

        return {
            "total_trades": int(total),
            "win_rate": float(wins / total) if total > 0 else 0.0,
            "total_pnl": float(df["total_pnl"].iloc[0] or 0),
            "avg_return_pct": float(df["avg_return_pct"].iloc[0] or 0),
            "avg_confidence": float(df["avg_confidence"].iloc[0] or 0),
        }