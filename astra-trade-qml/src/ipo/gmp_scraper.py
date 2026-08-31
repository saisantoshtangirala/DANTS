"""
IPO Grey Market Premium (GMP) and subscription data scraper.

Best-effort scraping of public IPO tracking sites. These sites change
markup frequently and have no stable public API, so every method
degrades gracefully to None/empty results on failure rather than raising
— callers should treat missing GMP data as a neutral feature, not an error.
"""

from typing import Dict, List, Optional

import pandas as pd
import requests


class IPODataScraper:
    """Fetches GMP trend and subscription data for upcoming/recent NSE IPOs."""

    def __init__(self, sources: Optional[List[str]] = None, timeout: int = 15):
        self.sources = sources or ["ipo_watch", "chittorgarh", "nse_ipo_calendar"]
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )

    def get_gmp_data(self, company_name: str) -> Dict[str, Optional[float]]:
        """
        Fetch the latest GMP figure for a company. Site markup is not
        stable enough to hardcode a parser here, so this currently
        confirms reachability and returns neutral values; extend the
        parsing once a stable selector for the target site is confirmed.
        """
        try:
            response = self.session.get("https://www.ipowatch.in", timeout=self.timeout)
            response.raise_for_status()
            return {"gmp": None, "gmp_trend": None, "source": "ipo_watch"}
        except Exception as e:
            return {"gmp": None, "gmp_trend": None, "error": str(e)}

    def get_subscription_data(self, company_name: str) -> Dict[str, Optional[float]]:
        """Fetch retail/QIB/HNI subscription ratios for an IPO (see get_gmp_data note)."""
        try:
            response = self.session.get(
                "https://www.chittorgarh.com/ipo/ipo_subscription.asp", timeout=self.timeout
            )
            response.raise_for_status()
            return {
                "subscription_ratio_retail": None,
                "subscription_ratio_qib": None,
                "subscription_ratio_hni": None,
                "source": "chittorgarh",
            }
        except Exception as e:
            return {
                "subscription_ratio_retail": None,
                "subscription_ratio_qib": None,
                "subscription_ratio_hni": None,
                "error": str(e),
            }

    def get_ipo_calendar(self) -> pd.DataFrame:
        """Fetch the upcoming NSE IPO calendar."""
        try:
            response = self.session.get(
                "https://www.nseindia.com/api/all-upcoming-issues?category=ipo", timeout=self.timeout
            )
            response.raise_for_status()
            return pd.DataFrame(response.json())
        except Exception:
            return pd.DataFrame(columns=["symbol", "companyName", "issueStartDate", "issueEndDate"])
