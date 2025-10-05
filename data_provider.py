# -*- coding: utf-8 -*-
import os
import time
import requests
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from structures import OptionChain, OptionQuote, Greek, MinuteData


# ----------------------------
# 공통 베이스
# ----------------------------
class DataProviderBase:
    """데이터 제공자 기본 클래스"""
    name = "Base"

    def fetch_chain(
        self,
        symbol: str,
        date: str,
        expiration: str,
        strike_step: float,
        center_strike: Optional[float] = None,
        strike_range: int = 10
    ) -> OptionChain:
        raise NotImplementedError

    def fetch_minute_data(self, symbol: str, strike: float, side: str, date: str) -> List[MinuteData]:
        raise NotImplementedError


# ----------------------------
# (호환성 유지를 위한) Mock 스텁
# ----------------------------
class MockDataProvider(DataProviderBase):
    """사용하지 않습니다. (호환성용 스텁)"""
    name = "Mock"

    def fetch_chain(self, *args, **kwargs):
        raise RuntimeError("MockDataProvider is disabled. Use PolygonDataProvider instead.")

    def fetch_minute_data(self, *args, **kwargs):
        raise RuntimeError("MockDataProvider is disabled. Use PolygonDataProvider instead.")


# ----------------------------
# Polygon.io 구현 (per-ticker day + prev, 병렬 처리, OHLC 제공)
# ----------------------------
class PolygonDataProvider(DataProviderBase):
    """Polygon.io API를 사용하는 데이터 제공자"""
    name = "Polygon.io"

    BASE = "https://api.polygon.io"

    def __init__(self):
        api_key, source = self._read_api_key_from_files()
        if not api_key or api_key.lower() in {"your_api_key_here", "<your_api_key_here>"}:
            raise RuntimeError(
                "Polygon API key not found or is a placeholder.\n"
                "Create a text file named 'cd_key.txt' that contains ONLY your actual key.\n"
                f"Checked: {source}"
            )
        self.api_key = api_key.strip()
        self._session = requests.Session()

        # 분데이터용 만기일 컨텍스트(HistoryTab에서 설정)
        self._minute_expiration: Optional[str] = None

        masked = ("*" * (len(self.api_key) - 4)) + self.api_key[-4:]
        print(f"[PolygonDataProvider] API key loaded from {source}: {masked}")

    # ---- 만기일 컨텍스트 주입 (MinuteData용) ----
    def set_minute_expiration(self, expiration: Optional[str]):
        """MinuteData 조회 시 사용할 만기일 컨텍스트를 설정"""
        self._minute_expiration = expiration

    # ---- Key loading: 환경변수 무시, 파일만 사용 ----
    def _read_api_key_from_files(self) -> Tuple[str, str]:
        here = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(here, "cd_key.txt")
        if os.path.exists(local_path):
            try:
                with open(local_path, "r", encoding="utf-8") as f:
                    return f.read().strip(), f"FILE({local_path})"
            except Exception:
                pass

        cwd_path = os.path.join(os.getcwd(), "cd_key.txt")
        if os.path.exists(cwd_path):
            try:
                with open(cwd_path, "r", encoding="utf-8") as f:
                    return f.read().strip(), f"FILE({cwd_path})"
            except Exception:
                pass

        return "", "FILE(not found in __file__ dir or CWD)"

    # ---- 공통 요청 유틸 + 백오프 ----
    def _req(self, method: str, path: str, params: Optional[Dict] = None, retries: int = 3) -> Dict:
        if params is None:
            params = {}
        params["apiKey"] = self.api_key
        url = f"{self.BASE}{path}"
        last_err = None
        backoff = 0.35
        for _ in range(retries + 1):
            try:
                resp = self._session.request(method, url, params=params, timeout=15)
                if resp.status_code == 429:
                    last_err = RuntimeError(f"429 Too Many Requests: {resp.text}")
                    time.sleep(backoff)
                    backoff = min(backoff * 1.7, 2.0)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_err = e
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 2.0)
        raise last_err

    # ---- OCC 티커 유틸 ----
    @staticmethod
    def _to_occ_option_ticker(underlying: str, expiration: str, side: str, strike: float) -> str:
        """
        OCC 옵션 티커 생성
        예: O:SPY241004C00450000
        - expiration: 'YYYY-MM-DD'
        - side: 'C' or 'P'
        - strike: 450.0 -> 00450000
        """
        y = expiration[2:4]
        m = expiration[5:7]
        d = expiration[8:10]
        strike_int = int(round(strike * 1000))
        strike_str = f"{strike_int:08d}"
        return f"O:{underlying.upper()}{y}{m}{d}{side.upper()}{strike_str}"

    # ---- 단일 티커: 해당 일자의 OHLC ----
    def _fetch_ohlc_for_day(self, option_ticker: str, day: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        j = self._req(
            "GET",
            f"/v2/aggs/ticker/{option_ticker}/range/1/day/{day}/{day}",
            params={"adjusted": "true", "limit": 1}
        )
        results = j.get("results") or []
        if results:
            r = results[0]
            o = r.get("o"); h = r.get("h"); l = r.get("l"); c = r.get("c")
            return (
                None if o is None else float(o),
                None if h is None else float(h),
                None if l is None else float(l),
                None if c is None else float(c),
            )
        return (None, None, None, None)

    # ---- 단일 티커: '그 날짜의 전일' close ----
    def _fetch_prev_close_for_day(self, option_ticker: str, day: str) -> Optional[float]:
        base = datetime.strptime(day, "%Y-%m-%d").date()
        for i in range(1, 8):
            prev = (base - timedelta(days=i)).strftime("%Y-%m-%d")
            j = self._req(
                "GET",
                f"/v2/aggs/ticker/{option_ticker}/range/1/day/{prev}/{prev}",
                params={"adjusted": "true", "limit": 1}
            )
            results = j.get("results") or []
            if results:
                c = results[0].get("c")
                return None if c is None else float(c)
        return None

    # ---- 병렬 작업자: (OHLC 당일, 전일 close) ----
    def _fetch_ohlc_and_prev(self, option_ticker: str, trade_day: str) -> Tuple[
        Tuple[Optional[float], Optional[float], Optional[float], Optional[float]],
        Optional[float]
    ]:
        try:
            ohlc = self._fetch_ohlc_for_day(option_ticker, trade_day)
        except Exception:
            ohlc = (None, None, None, None)
        try:
            prev = self._fetch_prev_close_for_day(option_ticker, trade_day)
        except Exception:
            prev = None
        return ohlc, prev

    # --------- 공개 API 구현 ---------
    def fetch_chain(
        self,
        symbol: str,
        date: str,
        expiration: str,
        strike_step: float,
        center_strike: Optional[float] = None,
        strike_range: int = 10
    ) -> OptionChain:
        """
        date 기준의 옵션 체인:
        - 각 행사가에 대해 당일 OHLC와 '그 날짜의 전일' 종가(prev close)를 병렬 수집.
        - OptionQuote.last 는 당일 종가(close)로 채움.
        """
        if center_strike is None:
            center_strike = 100.0

        strikes: List[float] = []
        for i in range(-strike_range, strike_range + 1):
            k = round(center_strike + i * strike_step, 2)
            if k > 0:
                step_pos = (k - center_strike) / strike_step
                if abs(step_pos - round(step_pos)) < 1e-9:
                    strikes.append(k)

        # 옵션 티커 목록 (콜/풋)
        tickers: List[Tuple[str, str, float]] = []
        for side in ("C", "P"):
            for K in strikes:
                tick = self._to_occ_option_ticker(symbol, expiration, side, K)
                tickers.append((side, tick, K))

        # 병렬 수집
        quotes: List[OptionQuote] = []
        max_workers = min(16, max(4, len(tickers)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {ex.submit(self._fetch_ohlc_and_prev, tkr, date): (side, tkr, K) for side, tkr, K in tickers}
            for fut in as_completed(fut_map):
                side, tkr, K = fut_map[fut]
                try:
                    (o, h, l, c), prev_close = fut.result()
                except Exception:
                    o, h, l, c, prev_close = None, None, None, None, None

                last_price = 0.0 if c is None else float(c)
                greeks = Greek(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)  # 시그니처 호환

                oq = OptionQuote(symbol.upper(), side, K, last_price, last_price, 0.0, greeks)
                setattr(oq, "open",  0.0 if o is None else float(o))
                setattr(oq, "high",  0.0 if h is None else float(h))
                setattr(oq, "low",   0.0 if l is None else float(l))
                setattr(oq, "close", 0.0 if c is None else float(c))
                setattr(oq, "prev_close", 0.0 if prev_close is None else float(prev_close))
                quotes.append(oq)

        quotes.sort(key=lambda q: (q.side, q.strike))
        return OptionChain(symbol.upper(), date, expiration, quotes)

    # ---- 분 데이터(만기일 컨텍스트 반영) ----
    def fetch_minute_data(self, symbol: str, strike: float, side: str, date: str) -> List[MinuteData]:
        """
        MinuteData 조회 시, HistoryTab에서 set_minute_expiration()으로 설정해둔
        만기일(self._minute_expiration)을 사용해 OCC 티커를 생성합니다.

        (KST 변환)
        - Polygon 타임스탬프(ms)는 UTC 기준이므로, tz=UTC로 해석 후
          KST(UTC+9)로 변환하여 'HH:MM' 문자열을 생성합니다.
        """
        expiration = self._minute_expiration or date  # 컨텍스트가 없으면 비상용으로 date 사용
        ticker = self._to_occ_option_ticker(symbol, expiration, side, strike)

        j = self._req(
            "GET",
            f"/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}",
            params={"adjusted": "true", "limit": 50000}
        )
        results = j.get("results") or []
        data: List[MinuteData] = []

        KST = timezone(timedelta(hours=9))
        for r in results:
            ts_ms = r.get("t")
            if ts_ms is None:
                continue
            dt_utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            dt_kst = dt_utc.astimezone(KST)
            hhmm = dt_kst.strftime("%H:%M")

            o = float(r.get("o", 0.0))
            h = float(r.get("h", 0.0))
            l = float(r.get("l", 0.0))
            c = float(r.get("c", 0.0))
            v = int(r.get("v", 0))
            data.append(MinuteData(timestamp=hhmm, open=o, high=h, low=l, close=c, volume=v))
        return data
