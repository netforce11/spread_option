# -*- coding: utf-8 -*-
from PyQt5 import QtGui
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QAbstractScrollArea
from PyQt5.QtCore import Qt
from typing import Optional, Dict, List
from structures import OptionQuote, MinuteData, OptionChain
from datetime import datetime

# ─────────────────────────────────────────────────────────
# 테이블 컬럼 정의 (Greeks 제거, OHLC 추가)
# ─────────────────────────────────────────────────────────
CALL_COLUMNS_LEFT = [
    ("전일종가", "prev_close"),
    ("시가", "open"),
    ("고가", "high"),
    ("저가", "low"),
    ("종가", "close"),
    ("Last", "last"),  # 강조
]

PUT_COLUMNS_RIGHT = [
    ("Last", "last"),  # 강조
    ("종가", "close"),
    ("저가", "low"),
    ("고가", "high"),
    ("시가", "open"),
    ("전일종가", "prev_close"),
]

OPTION_MULTIPLIER = 100


def _extract_value(q: Optional[OptionQuote], key: str) -> str:
    """OptionQuote에서 특정 키 값을 문자열로 추출 (OHLC/prev_close/last 위주)"""
    if q is None:
        return ""

    if hasattr(q, key):
        val = getattr(q, key)
        if isinstance(val, (int, float)):
            return f"{val:.2f}"
        return "" if val is None else str(val)

    if hasattr(getattr(q, "greeks", object()), key):
        val = getattr(q.greeks, key)
        if isinstance(val, (int, float)):
            return f"{val:.3f}"
        return "" if val is None else str(val)

    return ""


def populate_option_table(table: QTableWidget, chain: Optional[OptionChain]):
    """옵션 체인 데이터를 QTableWidget에 채워넣는 함수 (OHLC 기반)"""
    # 레이아웃 흔들림/깜빡임 방지
    table.setUpdatesEnabled(False)
    table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
    try:
        if chain is None or not getattr(chain, "quotes", None):
            table.setRowCount(0)
            return

        # strike 기준으로 콜/풋 매핑
        by_strike: Dict[float, Dict[str, OptionQuote]] = {}
        for q in chain.quotes:
            by_strike.setdefault(q.strike, {})[q.side] = q

        strikes = sorted(by_strike.keys())
        table.setRowCount(len(strikes))

        sep_col = 1 + len(CALL_COLUMNS_LEFT)

        for r, K in enumerate(strikes):
            # Strike 컬럼
            strike_item = QTableWidgetItem(f"{K:.2f}")
            strike_item.setBackground(QtGui.QColor(200, 220, 255))  # 파란 톤
            strike_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(r, 0, strike_item)

            c = by_strike[K].get('C')
            p = by_strike[K].get('P')

            # ── 콜 옵션 데이터 (왼쪽) ───────────────────────────
            for i, (_, key) in enumerate(CALL_COLUMNS_LEFT, start=1):
                val = _extract_value(c, key)
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                header = table.horizontalHeaderItem(i)
                if header and header.text() == "Last":
                    item.setBackground(QtGui.QColor(230, 255, 230))
                table.setItem(r, i, item)

            # 구분자 컬럼
            sep_item = QTableWidgetItem("|")
            sep_item.setTextAlignment(Qt.AlignCenter)
            sep_item.setFlags(Qt.ItemIsEnabled)  # 편집/선택 방지
            table.setItem(r, sep_col, sep_item)
            base = sep_col + 1

            # ── 풋 옵션 데이터 (오른쪽) ─────────────────────────
            for i, (_, key) in enumerate(PUT_COLUMNS_RIGHT):
                val = _extract_value(p, key)
                col = base + i
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                header = table.horizontalHeaderItem(col)
                if header and header.text() == "Last":
                    item.setBackground(QtGui.QColor(255, 230, 230))
                table.setItem(r, col, item)
    finally:
        table.setUpdatesEnabled(True)


def resample_minute_data(data: List[MinuteData], interval: int) -> List[MinuteData]:
    """1분봉 데이터를 지정된 분봉 간격으로 리샘플링"""
    if interval == 1 or not data:
        return data

    resampled: List[MinuteData] = []
    for i in range(0, len(data), interval):
        chunk = data[i:i + interval]
        if not chunk:
            continue

        timestamp_dt = datetime.strptime(chunk[0].timestamp, "%H:%M")
        new_timestamp = timestamp_dt.strftime("%H:%M")

        resampled.append(
            MinuteData(
                timestamp=new_timestamp,
                open=chunk[0].open,
                high=max(c.high for c in chunk),
                low=min(c.low for c in chunk),
                close=chunk[-1].close,
                volume=sum(c.volume for c in chunk)
            )
        )
    return resampled
