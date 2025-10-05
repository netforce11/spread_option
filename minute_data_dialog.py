# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QTableWidget, QHeaderView,
                             QHBoxLayout, QTableWidgetItem, QMessageBox, QComboBox,
                             QSpinBox, QGroupBox)
import matplotlib.patches as mpatches
import numpy as np
from typing import List
from data_provider import DataProviderBase
from structures import MinuteData
from mpl_canvas import MplCanvas
from utils import resample_minute_data


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

        self.setWindowTitle(f"{symbol} - Strike {strike} ({side}) - 분봉 데이터")
        self.resize(1200, 700)
        self._build_ui()
        self._load_data()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        info_label = QLabel(f"<b>심볼: {self.symbol} | Strike: {self.strike} | 사이드: {self.side} | 날짜: {self.date}</b>")
        layout.addWidget(info_label)

        # 컨트롤 패널
        control_panel = QHBoxLayout()
        control_panel.addWidget(QLabel("분봉 간격:"))
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["1분", "3분", "5분"])
        self.interval_combo.currentIndexChanged.connect(self._update_display)
        control_panel.addWidget(self.interval_combo)

        control_panel.addWidget(QLabel("표시할 캔들 수:"))
        self.candle_count_spin = QSpinBox()
        self.candle_count_spin.setRange(10, 5000)
        self.candle_count_spin.setValue(399)   # 기본값 399
        self.candle_count_spin.valueChanged.connect(self._update_display)
        control_panel.addWidget(self.candle_count_spin)
        control_panel.addStretch()
        layout.addLayout(control_panel)

        content_layout = QHBoxLayout()
        # 테이블
        table_group = QGroupBox("분 데이터")
        table_layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["시간", "Open", "High", "Low", "Close", "Volume"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.cellClicked.connect(self.on_cell_clicked)
        table_layout.addWidget(self.table)
        table_group.setLayout(table_layout)
        content_layout.addWidget(table_group, 1)

        # 차트 (고정 높이 + 여백 조절)
        chart_group = QGroupBox("캔들차트")
        chart_layout = QVBoxLayout()
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas.setFixedHeight(300)  # 고정 높이
        chart_layout.addWidget(self.canvas)
        chart_group.setLayout(chart_layout)
        content_layout.addWidget(chart_group, 1)

        layout.addLayout(content_layout)

    def on_cell_clicked(self, row: int, col: int):
        return  # 클릭 시 액션 없음

    def _load_data(self):
        try:
            self.minute_data = self.provider.fetch_minute_data(
                self.symbol, self.strike, self.side, self.date
            )
        except Exception as e:
            QMessageBox.critical(self, "에러", f"분 데이터 로드 실패: {e}")
            return
        self._update_display()

    def _update_display(self):
        interval_text = self.interval_combo.currentText()
        interval = int(interval_text.replace('분', ''))

        display_data = resample_minute_data(self.minute_data, interval)
        self._populate_table(display_data)
        self._draw_candlestick(display_data)

    def _populate_table(self, data: List[MinuteData]):
        self.table.setRowCount(len(data))
        for i, md in enumerate(data):
            self.table.setItem(i, 0, QTableWidgetItem(md.timestamp))
            self.table.setItem(i, 1, QTableWidgetItem(f"{md.open:.2f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{md.high:.2f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{md.low:.2f}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{md.close:.2f}"))
            self.table.setItem(i, 5, QTableWidgetItem(str(md.volume)))
        self.table.scrollToBottom()

    def _draw_candlestick(self, data: List[MinuteData]):
        self.canvas.ax.clear()

        candle_count = self.candle_count_spin.value()
        display_data = data[-candle_count:]

        if not display_data:
            self.canvas.draw()
            return

        for i, md in enumerate(display_data):
            color = 'green' if md.close >= md.open else 'red'
            self.canvas.ax.plot([i, i], [md.low, md.high], color='black', linewidth=0.8)
            body_height = abs(md.close - md.open)
            body_bottom = min(md.open, md.close)
            rect = mpatches.Rectangle(
                (i - 0.3, body_bottom), 0.6, body_height if body_height > 0 else 0.01,
                facecolor=color, edgecolor='black', linewidth=0.5
            )
            self.canvas.ax.add_patch(rect)

        interval_text = self.interval_combo.currentText()
        self.canvas.ax.set_title(f"{self.symbol} {self.strike} ({self.side}) {interval_text} 캔들차트")
        self.canvas.ax.set_xlabel("시간")
        self.canvas.ax.set_ylabel("가격")

        tick_indices = np.linspace(0, len(display_data) - 1, min(12, len(display_data)), dtype=int)
        self.canvas.ax.set_xticks(tick_indices)
        self.canvas.ax.set_xticklabels([display_data[i].timestamp for i in tick_indices], rotation=30, ha='right')

        self.canvas.ax.grid(True, alpha=0.3)
        # ❌ tight_layout() 대신 고정 여백
        self.canvas.figure.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.18)
        self.canvas.draw()
