# -*- coding: utf-8 -*-
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit,
                             QComboBox, QPushButton, QTableWidget, QHeaderView,
                             QSpinBox, QDoubleSpinBox, QMessageBox, QGroupBox,
                             QHBoxLayout, QCheckBox, QTextEdit)
from typing import Dict, List, Optional, Tuple

from data_provider import DataProviderBase, MockDataProvider, PolygonDataProvider
from structures import OptionChain
from strategies import STRATEGIES, Strategy
from ui.mpl_canvas import MplCanvas
from ui.minute_data_dialog import MinuteDataDialog
from utils import (populate_option_table, CALL_COLUMNS_LEFT, PUT_COLUMNS_RIGHT,
                   OPTION_MULTIPLIER)


class BaseTab(QWidget):
    """탭의 공통 기능을 정의하는 기본 클래스"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.provider_map: Dict[str, DataProviderBase] = {
            "Mock": MockDataProvider(),
            "Polygon.io": PolygonDataProvider(),
        }
        self.current_chain: Optional[OptionChain] = None
        self.selected_options: List[Tuple[str, float, float]] = []  # (Side, Strike, Price)
        self.active_strategy: Optional[Strategy] = None

        self._build_common_ui()

    def _build_common_ui(self):
        """탭에 공통적으로 들어가는 UI 요소들을 생성"""
        # 데이터 소스 및 기본 정보
        self.source_combo = QComboBox()
        self.source_combo.addItems(self.provider_map.keys())
        self.symbol_edit = QLineEdit("SPY")

        # 날짜 관련 (자식 클래스에서 정의)

        # 행사가 관련
        self.step_combo = QComboBox()
        self.step_combo.addItems(["0.5", "1.0", "1.5", "2.0", "2.5", "5.0"])
        self.step_combo.setCurrentText("1.0")
        self.center_strike_edit = QLineEdit("450.0")
        self.strike_range_spin = QSpinBox()
        self.strike_range_spin.setRange(1, 50)
        self.strike_range_spin.setValue(15)

        # 자본금 및 수수료
        self.capital_edit = QLineEdit("10000")
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setDecimals(2)
        self.fee_spin.setRange(0, 100)
        self.fee_spin.setValue(0.65)  # 계약당 수수료

        # 전략 선택
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(STRATEGIES.keys())
        self.strategy_combo.currentTextChanged.connect(self.on_strategy_changed)

        self.strategy_desc_label = QTextEdit()
        self.strategy_desc_label.setReadOnly(True)
        self.strategy_desc_label.setFixedHeight(60)

        # 가격 수동 입력
        self.manual_price_check = QCheckBox("순수비용/수익 수동 입력")
        self.manual_price_spin = QDoubleSpinBox()
        self.manual_price_spin.setDecimals(2)
        self.manual_price_spin.setRange(-1000, 1000)
        self.manual_price_spin.setEnabled(False)
        self.manual_price_check.toggled.connect(self.manual_price_spin.setEnabled)

        # 테이블
        self.table = QTableWidget()
        self._setup_table_headers()

        # 결과 표시
        self.result_label = QLabel("전략을 선택하고 옵션을 클릭하여 시뮬레이션을 시작하세요.")
        self.result_label.setWordWrap(True)
        self.canvas = MplCanvas(self, width=6, height=3, dpi=100)

        self.on_strategy_changed(self.strategy_combo.currentText())

    def _setup_table_headers(self):
        call_cols = [c[0] for c in CALL_COLUMNS_LEFT]
        put_cols = [c[0] for c in PUT_COLUMNS_RIGHT]
        headers = ["Strike"] + call_cols + ["|"] + put_cols
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

    def on_strategy_changed(self, strategy_name: str):
        self.active_strategy = STRATEGIES[strategy_name]
        self.selected_options.clear()

        desc = (f"<b>[전략] {self.active_strategy.name}</b><br>"
                f"<b>[설명]</b> {self.active_strategy.description}<br>"
                f"<b>[목적]</b> {self.active_strategy.purpose}")
        self.strategy_desc_label.setHtml(desc)

        self.result_label.setText(f"<b>[안내]</b> {self.active_strategy.selection_prompt}")
        self.canvas.ax.clear()
        self.canvas.draw()

    def on_table_cell_clicked(self, row, col, date_str):
        # Strike 컬럼 클릭 시 분 데이터 팝업
        if col == 0:
            if not self.current_chain: return
            strike = float(self.table.item(row, 0).text())
            reply = QMessageBox.question(self, '사이드 선택',
                                         f'Strike {strike}의 분 데이터를 확인합니다.\n콜(Yes) / 풋(No)?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            side = 'C' if reply == QMessageBox.Yes else 'P'
            provider = self.provider_map[self.source_combo.currentText()]
            dialog = MinuteDataDialog(provider, self.current_chain.underlying, strike, date_str, side, self)
            dialog.exec_()
            return

        # Last 가격이 아니면 무시
        if self.table.horizontalHeaderItem(col).text() != "Last": return
        if not self.active_strategy: return

        strike = float(self.table.item(row, 0).text())
        price_text = self.table.item(row, col).text()
        if not price_text: return
        price = float(price_text)

        sep_col = 1 + len(CALL_COLUMNS_LEFT)
        side = 'C' if col < sep_col else 'P'

        self.selected_options.append((side, strike, price))

        # 필요한 레그 수만큼 선택되었는지 확인
        if len(self.selected_options) == self.active_strategy.legs:
            self.calculate_and_plot()
        elif len(self.selected_options) > self.active_strategy.legs:
            self.selected_options = self.selected_options[-self.active_strategy.legs:]
            self.calculate_and_plot()
        else:
            remaining = self.active_strategy.legs - len(self.selected_options)
            self.result_label.setText(f"선택됨: {side} {strike} @ {price:.2f}. "
                                      f"{remaining}개 옵션을 더 선택하세요.")

    def calculate_and_plot(self):
        if not self.active_strategy or len(self.selected_options) != self.active_strategy.legs:
            return

        try:
            capital = float(self.capital_edit.text())
            fee_per_contract = self.fee_spin.value()
        except ValueError:
            QMessageBox.warning(self, "입력 오류", "자본금은 숫자여야 합니다.")
            return

        # 전략별 파라미터 준비
        params, leg_info = self._prepare_strategy_params()

        is_credit_spread = "Credit" in self.active_strategy.description

        if self.manual_price_check.isChecked():
            cost_per_unit = self.manual_price_spin.value()
        else:
            cost_per_unit = params[-1]

        total_fee = fee_per_contract * self.active_strategy.legs

        if is_credit_spread:
            # 크레딧 전략: 수수료를 제외한 금액이 담보금으로 사용될 수 있음
            # 단순화를 위해 여기서는 순수익을 자본금에 더하지 않음
            num_contracts = int(capital / (abs(cost_per_unit) * OPTION_MULTIPLIER)) if cost_per_unit != 0 else 1
        else:  # 데빗 전략
            total_cost_per_unit = (cost_per_unit * OPTION_MULTIPLIER) + total_fee
            num_contracts = int(capital / total_cost_per_unit) if total_cost_per_unit > 0 else 0

        if num_contracts == 0:
            self.result_label.setText("자본금이 부족하여 1계약도 진입할 수 없습니다.")
            return

        # 손익 계산
        S_min = min(p[1] for p in self.selected_options) - 10
        S_max = max(p[1] for p in self.selected_options) + 10
        xs = np.linspace(S_min, S_max, 300)
        ys = [(self.active_strategy.pnl_func(x, params) * OPTION_MULTIPLIER) - total_fee for x in xs]

        max_profit = max(ys)
        max_loss = min(ys)

        # 손익분기점 찾기
        breakeven_points = []
        for i in range(1, len(xs)):
            if (ys[i - 1] < 0 and ys[i] >= 0) or (ys[i - 1] > 0 and ys[i] <= 0):
                # 선형 보간법으로 근사치 계산
                x1, y1 = xs[i - 1], ys[i - 1]
                x2, y2 = xs[i], ys[i]
                bep = x1 - y1 * (x2 - x1) / (y2 - y1)
                if not any(abs(bep - p) < 0.5 for p in breakeven_points):  # 중복 방지
                    breakeven_points.append(bep)

        # 결과 텍스트 생성
        result_text = f"<b>{self.active_strategy.name} 시뮬레이션 결과</b><br>"
        result_text += f"선택된 옵션: {leg_info}<br>"
        result_text += f"단위당 순수비용(수익): ${cost_per_unit:.2f}<br><br>"
        result_text += f"<b>총 자본금: ${capital:,.2f}</b><br>"
        result_text += f"<b>진입 가능 계약 수: {num_contracts} 계약</b><br>"
        result_text += f"<b>최대 수익금 (1계약/총): ${max_profit:,.2f} / ${max_profit * num_contracts:,.2f}</b><br>"
        result_text += f"<b>최대 손실금 (1계약/총): ${max_loss:,.2f} / ${max_loss * num_contracts:,.2f}</b><br>"

        initial_investment = (cost_per_unit * OPTION_MULTIPLIER + total_fee) * num_contracts
        if not is_credit_spread and initial_investment > 0:
            roi = (max_profit * num_contracts) / initial_investment * 100
            result_text += f"<b>최대 수익률 (ROI): {roi:.2f}%</b><br>"

        bep_str = ", ".join([f"${p:.2f}" for p in breakeven_points]) if breakeven_points else "N/A"
        result_text += f"<b>손익분기점(BEP) 근사치: {bep_str}</b>"

        self.result_label.setText(result_text)

        # 차트 그리기
        self.canvas.ax.clear()
        self.canvas.ax.plot(xs, [y * num_contracts for y in ys], label=f"Total P&L ({num_contracts} contracts)")
        self.canvas.ax.axhline(0, color='grey', linestyle='--', linewidth=1)
        self.canvas.ax.set_title("Profit & Loss Diagram at Expiration")
        self.canvas.ax.set_xlabel("Underlying Price ($)")
        self.canvas.ax.set_ylabel("Profit / Loss ($)")
        self.canvas.ax.grid(True, alpha=0.3)
        self.canvas.ax.legend()
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def _prepare_strategy_params(self) -> Tuple[List[float], str]:
        """선택된 옵션들로부터 전략별 파라미터와 정보를 생성"""
        # 정렬: (Strike, Side, Price)
        sorted_options = sorted([(opt[1], opt[0], opt[2]) for opt in self.selected_options])

        name = self.active_strategy.name
        params = []
        leg_info = ""

        if name == "콜 매수 (Long Call)":
            K, _, P = sorted_options[0]
            params = [K, P]
            leg_info = f"Call @ {K:.2f}"
        elif name == "풋 매수 (Long Put)":
            K, _, P = sorted_options[0]
            params = [K, P]
            leg_info = f"Put @ {K:.2f}"
        elif "Spread" in name:
            K1, S1, P1 = sorted_options[0]
            K2, S2, P2 = sorted_options[1]

            if "Bull Call" in name or "Bear Call" in name:
                # Long K1, Short K2
                net_cost = P1 - P2
                params = [K1, K2, net_cost]
            elif "Bear Put" in name or "Bull Put" in name:
                # Long K2, Short K1
                net_cost = P2 - P1
                params = [K1, K2, net_cost]
            leg_info = f"{S1}@{K1:.2f}, {S2}@{K2:.2f}"

        # 여기에 다른 복잡한 전략들의 파라미터 준비 로직 추가
        # ...

        # 기본값으로 마지막 파라미터는 net_cost가 되도록 함
        if not self.manual_price_check.isChecked() and self.manual_price_spin.isEnabled():
            self.manual_price_spin.setValue(params[-1])

        return params, leg_info