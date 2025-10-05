# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import (QVBoxLayout, QGridLayout, QLabel, QLineEdit,
                             QPushButton, QMessageBox)
from datetime import datetime
# 'ui.' 접두사 제거
from base_tab import BaseTab
from utils import populate_option_table


class CurrentTab(BaseTab):
    """현재 시점의 옵션 체인을 분석하고 시뮬레이션하는 탭"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # 상단 컨트롤 패널
        controls = QGridLayout()
        controls.addWidget(QLabel("데이터 소스"), 0, 0)
        controls.addWidget(self.source_combo, 0, 1)
        controls.addWidget(QLabel("심볼"), 0, 2)
        controls.addWidget(self.symbol_edit, 0, 3)

        controls.addWidget(QLabel("조회일"), 1, 0)
        self.date_edit = QLineEdit(datetime.now().strftime("%Y-%m-%d"))
        controls.addWidget(self.date_edit, 1, 1)
        controls.addWidget(QLabel("만기일"), 1, 2)
        self.expiry_edit = QLineEdit("2025-10-10")
        controls.addWidget(self.expiry_edit, 1, 3)

        controls.addWidget(QLabel("행사가 간격"), 2, 0)
        controls.addWidget(self.step_combo, 2, 1)
        controls.addWidget(QLabel("중심 행사가"), 2, 2)
        controls.addWidget(self.center_strike_edit, 2, 3)

        controls.addWidget(QLabel("위/아래 개수"), 3, 0)
        controls.addWidget(self.strike_range_spin, 3, 1)
        controls.addWidget(QLabel("자본금($)"), 3, 2)
        controls.addWidget(self.capital_edit, 3, 3)

        controls.addWidget(QLabel("계약당 수수료($)"), 4, 0)
        controls.addWidget(self.fee_spin, 4, 1)

        fetch_btn = QPushButton("옵션 체인 불러오기")
        fetch_btn.clicked.connect(self.on_fetch)
        controls.addWidget(fetch_btn, 5, 0, 1, 4)

        layout.addLayout(controls)

        # 전략 패널
        strategy_layout = QGridLayout()
        strategy_layout.addWidget(QLabel("전략 선택"), 0, 0)
        strategy_layout.addWidget(self.strategy_combo, 0, 1)
        strategy_layout.addWidget(self.manual_price_check, 0, 2)
        strategy_layout.addWidget(self.manual_price_spin, 0, 3)
        strategy_layout.addWidget(self.strategy_desc_label, 1, 0, 1, 4)
        layout.addLayout(strategy_layout)

        # 테이블
        self.table.cellClicked.connect(lambda r, c: self.on_table_cell_clicked(r, c, self.date_edit.text().strip()))
        layout.addWidget(self.table, 1)

        # 결과
        layout.addWidget(self.result_label)
        layout.addWidget(self.canvas)

    def on_fetch(self):
        provider = self.provider_map[self.source_combo.currentText()]
        symbol = self.symbol_edit.text().strip().upper()
        date = self.date_edit.text().strip()
        expiry = self.expiry_edit.text().strip()
        step = float(self.step_combo.currentText())

        try:
            center_strike = float(self.center_strike_edit.text())
            strike_range = self.strike_range_spin.value()
        except ValueError:
            QMessageBox.warning(self, "입력 오류", "중심 행사가와 범위는 숫자여야 합니다.")
            return

        try:
            chain = provider.fetch_chain(symbol, date, expiry, step, center_strike, strike_range)
        except Exception as e:
            QMessageBox.critical(self, "에러", f"데이터 수신 실패: {e}")
            return

        self.current_chain = chain
        populate_option_table(self.table, chain)

        self.selected_options.clear()
        self.on_strategy_changed(self.strategy_combo.currentText())  # Reset labels