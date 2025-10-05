# -*- coding: utf-8 -*-
"""
PyQt5 기반 주식 옵션 전략 시뮬레이터 (Polygon.io 연동용 - 확장판)
- 과거 탭 개선:
  1) Strike 조회 범위를 사용자가 입력 (중심 Strike 기준 위/아래 개수)
  2) Strike 클릭 시 분 데이터 팝업 (테이블 + 캔들차트)
  3) 분 데이터에서 2개 선택 후 전략별 손익 계산
"""

import os
import sys
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QGroupBox, QRadioButton, QFormLayout, QSpinBox, QDoubleSpinBox, QGridLayout,
    QMessageBox, QHeaderView, QDialog
)

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OPTION_MULTIPLIER = 100


# ------------------------------ Data Structures ------------------------------
@dataclass
class Greek:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


@dataclass
class OptionQuote:
    symbol: str
    side: str
    strike: float
    last: float
    prev_close: float
    iv: float
    greeks: Greek


@dataclass
class OptionChain:
    underlying: str
    as_of: str
    expiration: str
    quotes: List[OptionQuote]


@dataclass
class MinuteData:
    """분 단위 데이터"""
    timestamp: str  # HH:MM
    open: float
    high: float
    low: float
    close: float
    volume: int


# ------------------------------ Data Provider ------------------------------
class DataProviderBase:
    name = "Base"

    def fetch_chain(self, symbol: str, date: str, expiration: str, strike_step: float,
                    center_strike: Optional[float] = None, strike_range: int = 10) -> OptionChain:
        raise NotImplementedError

    def fetch_minute_data(self, symbol: str, strike: float, side: str, date: str) -> List[MinuteData]:
        raise NotImplementedError


class MockDataProvider(DataProviderBase):
    name = "Mock"

    def fetch_chain(self, symbol: str, date: str, expiration: str, strike_step: float,
                    center_strike: Optional[float] = None, strike_range: int = 10) -> OptionChain:
        random.seed(hash((symbol, date, expiration, strike_step)) & 0xffffffff)

        if center_strike is None:
            spot = round(random.uniform(20, 80), 2)
            center_strike = round(spot / strike_step) * strike_step

        # 중심 Strike 기준 위아래 범위
        strikes = []
        for i in range(-strike_range, strike_range + 1):
            K = round(center_strike + i * strike_step, 2)
            if K > 0:
                strikes.append(K)
        strikes.sort()

        spot = center_strike + random.uniform(-2, 2)
        quotes = []
        for K in strikes:
            call_intrinsic = max(0.0, spot - K)
            put_intrinsic = max(0.0, K - spot)
            time_value = max(0.05, 2.0 - abs(spot - K) * 0.1)
            call_last = round(call_intrinsic + time_value + random.uniform(-0.05, 0.05), 2)
            put_last = round(put_intrinsic + time_value + random.uniform(-0.05, 0.05), 2)
            call_prev = round(max(0.01, call_last + random.uniform(-0.1, 0.1)), 2)
            put_prev = round(max(0.01, put_last + random.uniform(-0.1, 0.1)), 2)
            iv = round(max(0.1, random.uniform(0.2, 0.6)), 2)

            greeks_c = Greek(
                delta=round(min(0.99, max(0.01, 0.5 + (spot - K) / 10 + random.uniform(-0.05, 0.05))), 2),
                gamma=round(max(0.0, random.uniform(0.0, 0.1)), 3),
                theta=round(-abs(random.uniform(0.01, 0.2)), 3),
                vega=round(random.uniform(0.1, 1.0), 2),
                rho=round(random.uniform(0.01, 0.2), 3),
            )
            greeks_p = Greek(
                delta=round(-1 + greeks_c.delta, 2),
                gamma=greeks_c.gamma,
                theta=greeks_c.theta,
                vega=greeks_c.vega,
                rho=-greeks_c.rho,
            )
            quotes.append(OptionQuote(symbol, 'C', K, call_last, call_prev, iv, greeks_c))
            quotes.append(OptionQuote(symbol, 'P', K, put_last, put_prev, iv, greeks_p))
        return OptionChain(symbol, date, expiration, quotes)

    def fetch_minute_data(self, symbol: str, strike: float, side: str, date: str) -> List[MinuteData]:
        """모의 분 데이터 생성 (09:30 ~ 16:00, 1분 단위)"""
        random.seed(hash((symbol, strike, side, date)) & 0xffffffff)
        base_price = random.uniform(1.0, 10.0)

        data = []
        start_time = datetime.strptime("09:30", "%H:%M")
        end_time = datetime.strptime("16:00", "%H:%M")
        current = start_time

        while current <= end_time:
            price = base_price + random.uniform(-0.5, 0.5)
            o = round(price + random.uniform(-0.1, 0.1), 2)
            h = round(max(o, price + random.uniform(0, 0.2)), 2)
            l = round(min(o, price - random.uniform(0, 0.2)), 2)
            c = round(price, 2)
            v = random.randint(10, 1000)

            data.append(MinuteData(
                timestamp=current.strftime("%H:%M"),
                open=o, high=h, low=l, close=c, volume=v
            ))
            current += timedelta(minutes=1)
            base_price = c

        return data


class PolygonDataProvider(DataProviderBase):
    name = "Polygon.io"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY", "")

    def fetch_chain(self, symbol: str, date: str, expiration: str, strike_step: float,
                    center_strike: Optional[float] = None, strike_range: int = 10) -> OptionChain:
        return MockDataProvider().fetch_chain(symbol, date, expiration, strike_step, center_strike, strike_range)

    def fetch_minute_data(self, symbol: str, strike: float, side: str, date: str) -> List[MinuteData]:
        return MockDataProvider().fetch_minute_data(symbol, strike, side, date)


# ------------------------------ Utility ------------------------------
CALL_COLUMNS_LEFT = [
    ("전일종가", "prev_close"),
    ("Δ", "delta"), ("Γ", "gamma"), ("Θ", "theta"), ("V", "vega"), ("ρ", "rho"),
    ("IV", "iv"),
    ("Last", "last"),
]
PUT_COLUMNS_RIGHT = [
    ("Last", "last"),
    ("IV", "iv"),
    ("Δ", "delta"), ("Γ", "gamma"), ("Θ", "theta"), ("V", "vega"), ("ρ", "rho"),
    ("전일종가", "prev_close"),
]


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


# ------------------------------ Minute Data Dialog ------------------------------
class MinuteDataDialog(QDialog):
    """Strike 클릭 시 분 데이터를 보여주는 팝업"""

    def __init__(self, provider: DataProviderBase, symbol: str, strike: float,
                 date: str, side: str, parent=None):
        super().__init__(parent)
        self.provider = provider
        self.symbol = symbol
        self.strike = strike
        self.date = date
        self.side = side
        self.minute_data: List[MinuteData] = []
        self.selected_prices: List[Tuple[str, float]] = []  # (timestamp, price)
        self.strategy = "call"  # 'call' or 'put'

        self.setWindowTitle(f"{symbol} - Strike {strike} ({side}) - 분 데이터")
        self.resize(1000, 700)
        self._build_ui()
        self._load_data()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # 상단 정보
        info_label = QLabel(f"심볼: {self.symbol} | Strike: {self.strike} | 사이드: {self.side} | 날짜: {self.date}")
        layout.addWidget(info_label)

        # 전략 선택
        strat_box = QGroupBox("전략 선택")
        strat_layout = QHBoxLayout()
        self.call_radio = QRadioButton("롱 콜 스프레드")
        self.put_radio = QRadioButton("롱 풋 스프레드")
        self.call_radio.setChecked(True)
        self.call_radio.toggled.connect(self.on_strategy_changed)
        strat_layout.addWidget(self.call_radio)
        strat_layout.addWidget(self.put_radio)
        strat_box.setLayout(strat_layout)
        layout.addWidget(strat_box)

        # 수수료 입력
        fee_layout = QHBoxLayout()
        fee_layout.addWidget(QLabel("수수료($):"))
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setDecimals(2)
        self.fee_spin.setRange(0, 100)
        self.fee_spin.setValue(2.5)
        fee_layout.addWidget(self.fee_spin)
        fee_layout.addStretch()
        layout.addLayout(fee_layout)

        # 테이블과 차트 좌우 배치
        content_layout = QHBoxLayout()

        # 테이블
        table_layout = QVBoxLayout()
        table_layout.addWidget(QLabel("분 데이터 (Close 열 2개 선택)"))
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["시간", "Open", "High", "Low", "Close", "Volume"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.cellClicked.connect(self.on_cell_clicked)
        table_layout.addWidget(self.table)
        content_layout.addLayout(table_layout, 1)

        # 차트
        chart_layout = QVBoxLayout()
        chart_layout.addWidget(QLabel("캔들차트"))
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        chart_layout.addWidget(self.canvas)
        content_layout.addLayout(chart_layout, 1)

        layout.addLayout(content_layout)

        # 결과 표시
        self.result_label = QLabel("분 데이터의 Close 열에서 2개를 선택하세요.")
        layout.addWidget(self.result_label)

    def on_strategy_changed(self):
        self.strategy = "call" if self.call_radio.isChecked() else "put"
        self.selected_prices.clear()
        self.result_label.setText("전략 변경됨. Close 열에서 2개를 다시 선택하세요.")

    def _load_data(self):
        try:
            self.minute_data = self.provider.fetch_minute_data(
                self.symbol, self.strike, self.side, self.date
            )
        except Exception as e:
            QMessageBox.critical(self, "에러", f"분 데이터 로드 실패: {e}")
            return

        self._populate_table()
        self._draw_candlestick()

    def _populate_table(self):
        self.table.setRowCount(len(self.minute_data))
        for i, md in enumerate(self.minute_data):
            self.table.setItem(i, 0, QTableWidgetItem(md.timestamp))
            self.table.setItem(i, 1, QTableWidgetItem(f"{md.open:.2f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{md.high:.2f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{md.low:.2f}"))
            close_item = QTableWidgetItem(f"{md.close:.2f}")
            close_item.setBackground(QtGui.QColor(255, 255, 200))  # 강조
            self.table.setItem(i, 4, close_item)
            self.table.setItem(i, 5, QTableWidgetItem(str(md.volume)))

    def _draw_candlestick(self):
        """간단한 캔들차트 그리기"""
        self.canvas.ax.clear()

        if not self.minute_data:
            self.canvas.draw()
            return

        # 샘플링 (모든 데이터 표시 시 너무 많으므로 10분 간격)
        sampled = self.minute_data[::10]

        for i, md in enumerate(sampled):
            color = 'green' if md.close >= md.open else 'red'
            # 캔들 심지(High-Low)
            self.canvas.ax.plot([i, i], [md.low, md.high], color='black', linewidth=0.8)
            # 캔들 몸통
            body_height = abs(md.close - md.open)
            body_bottom = min(md.open, md.close)
            rect = mpatches.Rectangle(
                (i - 0.3, body_bottom), 0.6, body_height if body_height > 0 else 0.01,
                facecolor=color, edgecolor='black', linewidth=0.5
            )
            self.canvas.ax.add_patch(rect)

        self.canvas.ax.set_title(f"캔들차트 (10분 간격 샘플)")
        self.canvas.ax.set_xlabel("시간 인덱스")
        self.canvas.ax.set_ylabel("가격")
        self.canvas.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def on_cell_clicked(self, row: int, col: int):
        if col != 4:  # Close 열이 아니면 무시
            return

        timestamp = self.table.item(row, 0).text()
        close_price = float(self.table.item(row, 4).text())

        self.selected_prices.append((timestamp, close_price))
        if len(self.selected_prices) > 2:
            self.selected_prices = self.selected_prices[-2:]

        if len(self.selected_prices) == 2:
            self._calculate_spread()
        else:
            self.result_label.setText(f"선택: {timestamp} @ ${close_price:.2f}. 하나 더 선택하세요.")

    def _calculate_spread(self):
        (ts1, p1), (ts2, p2) = self.selected_prices

        # 가격 순서 정렬 (낮은 가격이 매수, 높은 가격이 매도)
        if p1 > p2:
            p1, p2 = p2, p1
            ts1, ts2 = ts2, ts1

        low_price, high_price = p1, p2
        net_debit = low_price - high_price  # 실제로는 음수 (매수가 비쌈)

        # Strike 기반 손익 계산은 실제로는 두 개의 다른 Strike를 선택해야 하지만,
        # 여기서는 동일 Strike의 시간대별 가격이므로 간단히 가격 차이만 계산
        fee = self.fee_spin.value()

        # 스프레드 손익 = (매도가 - 매수가) * 100 - 수수료
        spread_pnl = (high_price - low_price) * OPTION_MULTIPLIER - fee

        result_text = (
            f"선택된 시간: {ts1} (${low_price:.2f}) → {ts2} (${high_price:.2f})\n"
            f"전략: {self.strategy.upper()} 스프레드\n"
            f"가격 차이: ${high_price - low_price:.2f}\n"
            f"손익 (100배): ${spread_pnl:.2f}\n"
            f"수수료: ${fee:.2f}\n\n"
            f"※ 주의: 이는 동일 Strike의 시간대별 가격 비교입니다.\n"
            f"실제 스프레드는 서로 다른 Strike 간 거래입니다."
        )

        self.result_label.setText(result_text)


# ------------------------------ Current Tab ------------------------------
class CurrentTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.provider_map: Dict[str, DataProviderBase] = {
            "Polygon.io": PolygonDataProvider(),
            "Mock": MockDataProvider(),
        }
        self.current_chain: Optional[OptionChain] = None
        self.selected_last_prices: List[Tuple[str, float, float]] = []
        self.strategy: str = "call"

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        top = QGridLayout()
        row = 0
        top.addWidget(QLabel("데이터 소스"), row, 0)
        self.source_combo = QComboBox()
        self.source_combo.addItems(list(self.provider_map.keys()))
        top.addWidget(self.source_combo, row, 1)

        top.addWidget(QLabel("심볼"), row, 2)
        self.symbol_edit = QLineEdit("AAPL")
        top.addWidget(self.symbol_edit, row, 3)

        row += 1
        top.addWidget(QLabel("조회일(YYYY-MM-DD)"), row, 0)
        self.date_edit = QLineEdit("2025-10-03")
        top.addWidget(self.date_edit, row, 1)
        top.addWidget(QLabel("만기일(YYYY-MM-DD)"), row, 2)
        self.expiry_edit = QLineEdit("2025-10-10")
        top.addWidget(self.expiry_edit, row, 3)

        row += 1
        top.addWidget(QLabel("행사가 간격"), row, 0)
        self.step_combo = QComboBox()
        self.step_combo.addItems(["0.5", "1.0", "1.5", "2.0"])
        self.step_combo.setCurrentText("0.5")
        top.addWidget(self.step_combo, row, 1)

        top.addWidget(QLabel("수수료($)"), row, 2)
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setDecimals(2)
        self.fee_spin.setRange(0, 100)
        self.fee_spin.setSingleStep(0.5)
        self.fee_spin.setValue(2.5)
        top.addWidget(self.fee_spin, row, 3)

        row += 1
        self.fetch_btn = QPushButton("체인 불러오기")
        self.fetch_btn.clicked.connect(self.on_fetch)
        top.addWidget(self.fetch_btn, row, 0, 1, 4)

        layout.addLayout(top)

        strat_box = QGroupBox("전략")
        strat_layout = QHBoxLayout()
        self.call_radio = QRadioButton("롱 콜 스프레드")
        self.put_radio = QRadioButton("롱 풋 스프레드")
        self.call_radio.setChecked(True)
        self.call_radio.toggled.connect(self.on_strategy_changed)
        strat_layout.addWidget(self.call_radio)
        strat_layout.addWidget(self.put_radio)
        strat_box.setLayout(strat_layout)
        layout.addWidget(strat_box)

        self.table = QTableWidget()
        self._setup_table_headers()
        self.table.cellClicked.connect(self.on_cell_clicked)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

        bottom = QVBoxLayout()
        self.result_label = QLabel("스프레드: 하단에 손익이 표시됩니다. (Last Price 2개 선택)")
        bottom.addWidget(self.result_label)
        self.canvas = MplCanvas(self, width=6, height=3, dpi=100)
        bottom.addWidget(self.canvas)
        layout.addLayout(bottom)

    def _setup_table_headers(self):
        call_cols = [c[0] for c in CALL_COLUMNS_LEFT]
        put_cols = [c[0] for c in PUT_COLUMNS_RIGHT]
        headers = ["Strike"] + call_cols + ["|"] + put_cols
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

    def on_strategy_changed(self):
        self.strategy = "call" if self.call_radio.isChecked() else "put"
        self.selected_last_prices.clear()
        self.result_label.setText("전략 변경: 해당 사이드 Last Price 2개를 선택하세요.")
        self.canvas.ax.clear()
        self.canvas.draw()

    def on_fetch(self):
        source = self.source_combo.currentText()
        provider = self.provider_map[source]
        symbol = self.symbol_edit.text().strip().upper()
        date = self.date_edit.text().strip()
        expiry = self.expiry_edit.text().strip()
        step = float(self.step_combo.currentText())
        try:
            chain = provider.fetch_chain(symbol, date, expiry, step)
        except Exception as e:
            QMessageBox.critical(self, "에러", f"데이터 수신 실패: {e}")
            return
        self.current_chain = chain
        self.populate_table(chain, step)
        self.selected_last_prices.clear()
        self.result_label.setText("데이터 로드 완료. 전략 사이드의 Last Price 2개를 클릭하세요.")
        self.canvas.ax.clear()
        self.canvas.draw()

    def populate_table(self, chain: OptionChain, strike_step: float):
        by_strike: Dict[float, Dict[str, OptionQuote]] = {}
        for q in chain.quotes:
            by_strike.setdefault(q.strike, {})[q.side] = q
        strikes = sorted(by_strike.keys())
        self.table.setRowCount(len(strikes))
        for r, K in enumerate(strikes):
            self.table.setItem(r, 0, QTableWidgetItem(f"{K:.2f}"))
            c = by_strike[K].get('C')
            p = by_strike[K].get('P')

            def set_call(col_idx: int, text: str):
                item = QTableWidgetItem(text)
                if self.table.horizontalHeaderItem(col_idx).text() == "Last":
                    item.setBackground(QtGui.QColor(230, 255, 230))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, col_idx, item)

            def set_put(col_idx: int, text: str):
                item = QTableWidgetItem(text)
                if self.table.horizontalHeaderItem(col_idx).text() == "Last":
                    item.setBackground(QtGui.QColor(230, 230, 255))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, col_idx, item)

            for i, (name, key) in enumerate(CALL_COLUMNS_LEFT, start=1):
                val = self._extract_value(c, key)
                set_call(i, val)
            sep_col = 1 + len(CALL_COLUMNS_LEFT)
            self.table.setItem(r, sep_col, QTableWidgetItem("|"))
            self.table.item(r, sep_col).setTextAlignment(Qt.AlignCenter)
            base = sep_col + 1
            for i, (name, key) in enumerate(PUT_COLUMNS_RIGHT):
                val = self._extract_value(p, key)
                set_put(base + i, val)

    def _extract_value(self, q: Optional[OptionQuote], key: str) -> str:
        if q is None:
            return ""
        if key == "iv":
            return f"{q.iv:.2f}"
        if key == "prev_close":
            return f"{q.prev_close:.2f}"
        if key == "last":
            return f"{q.last:.2f}"
        if key in ("delta", "gamma", "theta", "vega", "rho"):
            return f"{getattr(q.greeks, key):.3f}" if key != "vega" else f"{getattr(q.greeks, key):.2f}"
        return ""

    def on_cell_clicked(self, row: int, col: int):
        if not self.current_chain:
            return
        header = self.table.horizontalHeaderItem(col).text()
        if header != "Last":
            return
        strike = float(self.table.item(row, 0).text())
        sep_col = 1 + len(CALL_COLUMNS_LEFT)
        if col < sep_col:
            side = 'C'
        elif col > sep_col:
            side = 'P'
        else:
            return
        if (self.strategy == 'call' and side != 'C') or (self.strategy == 'put' and side != 'P'):
            self.result_label.setText("선택한 전략과 사이드가 다릅니다. 올바른 사이드의 Last를 선택하세요.")
            return
        last_price = float(self.table.item(row, col).text())
        self.selected_last_prices.append((side, strike, last_price))
        if len(self.selected_last_prices) > 2:
            self.selected_last_prices = self.selected_last_prices[-2:]
        if len(self.selected_last_prices) == 2:
            self.calculate_and_plot()
        else:
            self.result_label.setText(f"선택됨: {side} @ {strike:.2f} / 프리미엄 {last_price:.2f}. 하나 더 선택하세요.")

    def calculate_and_plot(self):
        (side1, K1, L1), (side2, K2, L2) = self.selected_last_prices
        if K1 == K2:
            self.result_label.setText("행사가가 동일합니다. 다른 행사가를 선택하세요.")
            return
        lowK, highK = (K1, K2) if K1 < K2 else (K2, K1)
        lowL, highL = (L1, L2) if K1 < K2 else (L2, L1)
        fee = self.fee_spin.value()
        net_debit = (lowL - highL)

        def pnl_call(S):
            return (max(0.0, S - lowK) - max(0.0, S - highK) - net_debit) * OPTION_MULTIPLIER - fee

        def pnl_put(S):
            return (max(0.0, highK - S) - max(0.0, lowK - S) - net_debit) * OPTION_MULTIPLIER - fee

        fn = pnl_call if self.strategy == 'call' else pnl_put
        S_min = max(0.0, min(lowK, highK) - 5)
        S_max = max(lowK, highK) + 5
        xs = np.linspace(S_min, S_max, 200)
        ys = [fn(x) for x in xs]
        max_profit_intrinsic = (highK - lowK) * OPTION_MULTIPLIER - net_debit * OPTION_MULTIPLIER - fee
        max_loss = net_debit * OPTION_MULTIPLIER + fee
        breakeven = None
        for i in range(1, len(xs)):
            if ys[i - 1] <= 0 <= ys[i] or ys[i - 1] >= 0 >= ys[i]:
                breakeven = xs[i]
                break
        self.result_label.setText(
            f"선택: {self.strategy.upper()} 스프레드 | 저행사 {lowK:.2f} / 고행사 {highK:.2f} | "
            f"순지급(debit) {net_debit:.2f} | 수수료 ${fee:.2f} | \n"
            f"최대이익≈ ${max_profit_intrinsic:.2f}, 최대손실≈ ${max_loss:.2f}, 손익분기점≈ {breakeven:.2f if breakeven else 'N/A'}"
        )
        self.canvas.ax.clear()
        self.canvas.ax.plot(xs, ys)
        self.canvas.ax.axhline(0, linestyle='--', linewidth=1)
        self.canvas.ax.set_title("스프레드 P&L (가상)")
        self.canvas.ax.set_xlabel("기초자산 가격 (만기)")
        self.canvas.ax.set_ylabel("손익($)")
        self.canvas.draw()


# ------------------------------ History Tab ------------------------------
class HistoryTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.provider_map: Dict[str, DataProviderBase] = {
            "Polygon.io": PolygonDataProvider(),
            "Mock": MockDataProvider(),
        }
        self.buy_chain: Optional[OptionChain] = None
        self.sell_chain: Optional[OptionChain] = None
        self.buy_selection: Optional[Tuple[float, float, float, float]] = None
        self.sell_selection: Optional[Tuple[float, float, float, float]] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        controls = QGridLayout()
        r = 0
        controls.addWidget(QLabel("데이터 소스"), r, 0)
        self.source_combo = QComboBox()
        self.source_combo.addItems(self.provider_map.keys())
        controls.addWidget(self.source_combo, r, 1)
        controls.addWidget(QLabel("심볼"), r, 2)
        self.symbol_edit = QLineEdit("AAPL")
        controls.addWidget(self.symbol_edit, r, 3)

        r += 1
        controls.addWidget(QLabel("매수일(YYYY-MM-DD)"), r, 0)
        self.buy_date_edit = QLineEdit("2025-09-29")
        controls.addWidget(self.buy_date_edit, r, 1)
        controls.addWidget(QLabel("매도일(YYYY-MM-DD)"), r, 2)
        self.sell_date_edit = QLineEdit("2025-10-02")
        controls.addWidget(self.sell_date_edit, r, 3)

        r += 1
        controls.addWidget(QLabel("만기일(금요일, YYYY-MM-DD)"), r, 0)
        self.expiry_edit = QLineEdit("2025-10-03")
        controls.addWidget(self.expiry_edit, r, 1)
        controls.addWidget(QLabel("행사가 간격"), r, 2)
        self.step_combo = QComboBox()
        self.step_combo.addItems(["0.5", "1.0", "1.5", "2.0"])
        self.step_combo.setCurrentText("0.5")
        controls.addWidget(self.step_combo, r, 3)

        r += 1
        controls.addWidget(QLabel("중심 Strike"), r, 0)
        self.center_strike_edit = QLineEdit("50.0")
        controls.addWidget(self.center_strike_edit, r, 1)
        controls.addWidget(QLabel("위/아래 개수"), r, 2)
        self.strike_range_spin = QSpinBox()
        self.strike_range_spin.setRange(1, 50)
        self.strike_range_spin.setValue(15)
        controls.addWidget(self.strike_range_spin, r, 3)

        r += 1
        controls.addWidget(QLabel("수수료($)"), r, 0)
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setDecimals(2)
        self.fee_spin.setRange(0, 100)
        self.fee_spin.setValue(2.5)
        controls.addWidget(self.fee_spin, r, 1)
        controls.addWidget(QLabel("최종 만기일 종가 입력"), r, 2)
        self.final_spot_edit = QLineEdit("")
        controls.addWidget(self.final_spot_edit, r, 3)

        r += 1
        self.load_btn = QPushButton("체인 조회(매수/매도일)")
        self.load_btn.clicked.connect(self.on_load_chains)
        controls.addWidget(self.load_btn, r, 0, 1, 4)

        layout.addLayout(controls)

        mode_box = QGroupBox("정산 모드")
        hb = QHBoxLayout()
        self.expiry_mode = QRadioButton("만기결제 값")
        self.mid_mode = QRadioButton("중간계산")
        self.expiry_mode.setChecked(True)
        hb.addWidget(self.expiry_mode)
        hb.addWidget(self.mid_mode)
        mode_box.setLayout(hb)
        layout.addWidget(mode_box)

        tables_layout = QHBoxLayout()
        left_box = QVBoxLayout()
        right_box = QVBoxLayout()
        left_box.addWidget(QLabel("매수일 체결 후보 (Strike 클릭: 분데이터 / Last 2개 선택: 스프레드)"))
        self.buy_table = QTableWidget()
        self._setup_table(self.buy_table)
        self.buy_table.cellClicked.connect(lambda r, c: self.on_cell_click(self.buy_table, r, c, is_buy=True))
        left_box.addWidget(self.buy_table)

        right_box.addWidget(QLabel("매도일/중간청산 체결 후보 (Strike 클릭: 분데이터 / Last 2개: 스프레드)"))
        self.sell_table = QTableWidget()
        self._setup_table(self.sell_table)
        self.sell_table.cellClicked.connect(lambda r, c: self.on_cell_click(self.sell_table, r, c, is_buy=False))
        right_box.addWidget(self.sell_table)

        tables_layout.addLayout(left_box, 1)
        tables_layout.addLayout(right_box, 1)
        layout.addLayout(tables_layout)

        self.result_label = QLabel("선택 후 결과가 여기에 표시됩니다.")
        layout.addWidget(self.result_label)
        self.canvas = MplCanvas(self, width=6, height=3, dpi=100)
        layout.addWidget(self.canvas)

    def _setup_table(self, table: QTableWidget):
        call_cols = [c[0] for c in CALL_COLUMNS_LEFT]
        put_cols = [c[0] for c in PUT_COLUMNS_RIGHT]
        headers = ["Strike"] + call_cols + ["|"] + put_cols
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)

    def on_load_chains(self):
        provider = self.provider_map[self.source_combo.currentText()]
        symbol = self.symbol_edit.text().strip().upper()
        buy_date = self.buy_date_edit.text().strip()
        sell_date = self.sell_date_edit.text().strip()
        expiry = self.expiry_edit.text().strip()
        step = float(self.step_combo.currentText())

        try:
            center_strike = float(self.center_strike_edit.text().strip())
        except ValueError:
            QMessageBox.warning(self, "입력 오류", "중심 Strike는 숫자여야 합니다.")
            return

        strike_range = self.strike_range_spin.value()

        try:
            self.buy_chain = provider.fetch_chain(symbol, buy_date, expiry, step, center_strike, strike_range)
            self.sell_chain = provider.fetch_chain(symbol, sell_date, expiry, step, center_strike, strike_range)
        except Exception as e:
            QMessageBox.critical(self, "에러", f"조회 실패: {e}")
            return

        self.populate_table(self.buy_table, self.buy_chain)
        self.populate_table(self.sell_table, self.sell_chain)
        self.buy_selection = None
        self.sell_selection = None
        self.result_label.setText("매수일에서 Last 2개 선택 또는 Strike 클릭하여 분 데이터 확인")
        self.canvas.ax.clear()
        self.canvas.draw()

    def populate_table(self, table: QTableWidget, chain: OptionChain):
        by_strike: Dict[float, Dict[str, OptionQuote]] = {}
        for q in chain.quotes:
            by_strike.setdefault(q.strike, {})[q.side] = q
        strikes = sorted(by_strike.keys())
        table.setRowCount(len(strikes))
        for r, K in enumerate(strikes):
            strike_item = QTableWidgetItem(f"{K:.2f}")
            strike_item.setBackground(QtGui.QColor(200, 220, 255))  # Strike 클릭 가능 강조
            table.setItem(r, 0, strike_item)

            c = by_strike[K].get('C')
            p = by_strike[K].get('P')
            sep_col = 1 + len(CALL_COLUMNS_LEFT)

            for i, (_, key) in enumerate(CALL_COLUMNS_LEFT, start=1):
                val = self._extract_value(c, key)
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if self._is_last_col(table, i):
                    item.setBackground(QtGui.QColor(230, 255, 230))
                table.setItem(r, i, item)

            table.setItem(r, sep_col, QTableWidgetItem("|"))
            table.item(r, sep_col).setTextAlignment(Qt.AlignCenter)
            base = sep_col + 1

            for i, (_, key) in enumerate(PUT_COLUMNS_RIGHT):
                val = self._extract_value(p, key)
                col = base + i
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if self._is_last_col(table, col):
                    item.setBackground(QtGui.QColor(230, 230, 255))
                table.setItem(r, col, item)

    def _is_last_col(self, table: QTableWidget, col: int) -> bool:
        return table.horizontalHeaderItem(col).text() == "Last"

    def _extract_value(self, q: Optional[OptionQuote], key: str) -> str:
        if q is None:
            return ""
        if key == "iv":
            return f"{q.iv:.2f}"
        if key == "prev_close":
            return f"{q.prev_close:.2f}"
        if key == "last":
            return f"{q.last:.2f}"
        if key in ("delta", "gamma", "theta", "vega", "rho"):
            return f"{getattr(q.greeks, key):.3f}" if key != "vega" else f"{getattr(q.greeks, key):.2f}"
        return ""

    def on_cell_click(self, table: QTableWidget, row: int, col: int, is_buy: bool):
        # Strike 열 클릭 시 분 데이터 팝업
        if col == 0:
            strike = float(table.item(row, 0).text())
            chain = self.buy_chain if is_buy else self.sell_chain
            if not chain:
                return

            date = self.buy_date_edit.text().strip() if is_buy else self.sell_date_edit.text().strip()

            # 콜/풋 선택 다이얼로그 표시 (간단히 콜로 고정, 확장 가능)
            reply = QMessageBox.question(self, '사이드 선택',
                                         f'Strike {strike}의 분 데이터를 확인합니다.\n콜(Yes) / 풋(No)?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            side = 'C' if reply == QMessageBox.Yes else 'P'

            provider = self.provider_map[self.source_combo.currentText()]
            dialog = MinuteDataDialog(provider, chain.underlying, strike, date, side, self)
            dialog.exec_()
            return

        # Last 열 클릭 시 스프레드 계산
        if not self._is_last_col(table, col):
            return

        sep_col = 1 + len(CALL_COLUMNS_LEFT)
        strike = float(table.item(row, 0).text())
        last = float(table.item(row, col).text())
        side = 'C' if col < sep_col else 'P'

        store = getattr(self, 'buy_selection' if is_buy else 'sell_selection')
        if store is None:
            setattr(self, 'buy_selection' if is_buy else 'sell_selection', (strike, strike, last, last))
            self.result_label.setText("첫 번째 행사가 선택됨. 같은 테이블에서 다른 행사가를 하나 더 선택하세요.")
        else:
            K_low, K_high, L_low, L_high = store
            if strike < K_low:
                K_low, L_low = strike, last
            elif strike > K_high:
                K_high, L_high = strike, last
            else:
                if abs(strike - K_low) < abs(strike - K_high):
                    K_low, L_low = strike, last
                else:
                    K_high, L_high = strike, last
            new_tuple = (K_low, K_high, L_low, L_high)
            if is_buy:
                self.buy_selection = new_tuple
            else:
                self.sell_selection = new_tuple

        self.calculate()

    def calculate(self):
        fee = self.fee_spin.value()
        if not self.buy_selection:
            return
        K_low_b, K_high_b, L_low_b, L_high_b = self.buy_selection
        debit = (L_low_b - L_high_b)
        text = [f"매수 스프레드: 저행사 {K_low_b:.2f} / 고행사 {K_high_b:.2f}, 순지급 {debit:.2f}"]

        if self.expiry_mode.isChecked():
            fs = self.final_spot_edit.text().strip()
            if not fs:
                self.result_label.setText("만기 종가를 입력하세요.")
                return
            try:
                S = float(fs)
            except ValueError:
                self.result_label.setText("만기 종가 숫자 입력 오류.")
                return

            pnl_call = (max(0.0, S - K_low_b) - max(0.0, S - K_high_b) - debit) * OPTION_MULTIPLIER - fee
            pnl_put = (max(0.0, K_high_b - S) - max(0.0, K_low_b - S) - debit) * OPTION_MULTIPLIER - fee
            text.append(f"만기 P&L (콜스프레드 가정): ${pnl_call:.2f}")
            text.append(f"만기 P&L (풋스프레드 가정): ${pnl_put:.2f}")

            xs = np.linspace(min(K_low_b, S) - 5, max(K_high_b, S) + 5, 200)
            ys = [(max(0.0, x - K_low_b) - max(0.0, x - K_high_b) - debit) * OPTION_MULTIPLIER - fee for x in xs]
            self.canvas.ax.clear()
            self.canvas.ax.plot(xs, ys)
            self.canvas.ax.axhline(0, linestyle='--', linewidth=1)
            self.canvas.ax.axvline(S, linestyle=':', linewidth=1)
            self.canvas.ax.set_title("만기 P&L (콜 스프레드 기준)")
            self.canvas.ax.set_xlabel("기초자산 가격")
            self.canvas.ax.set_ylabel("손익($)")
            self.canvas.draw()
        else:
            if not self.sell_selection:
                self.result_label.setText("중간계산 모드: 매도일 테이블에서 Last 2개도 선택하세요.")
                return
            K_low_s, K_high_s, L_low_s, L_high_s = self.sell_selection
            credit_mid = (L_low_s - L_high_s)
            pnl = (credit_mid - debit) * OPTION_MULTIPLIER - fee
            text.append(f"중간청산 P&L: (매도 스프레드 {credit_mid:.2f} - 매수 스프레드 {debit:.2f}) *100 - 수수료 = ${pnl:.2f}")
            self.canvas.ax.clear()
            self.canvas.ax.bar(["P&L"], [pnl])
            self.canvas.ax.set_title("중간청산 손익")
            self.canvas.draw()

        self.result_label.setText("\n".join(text))


# ------------------------------ Main Window ------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("옵션 스프레드 시뮬레이터 (Polygon.io - 확장판)")
        self.resize(1400, 900)
        tabs = QTabWidget()
        self.current_tab = CurrentTab()
        self.history_tab = HistoryTab()
        tabs.addTab(self.current_tab, "현재")
        tabs.addTab(self.history_tab, "과거")
        self.setCentralWidget(tabs)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()