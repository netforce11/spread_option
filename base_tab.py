# -*- coding: utf-8 -*-
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit,
                             QComboBox, QPushButton, QTableWidget, QHeaderView,
                             QSpinBox, QDoubleSpinBox, QMessageBox, QGroupBox,
                             QHBoxLayout, QCheckBox, QTextEdit, QSizePolicy)
from PyQt5.QtGui import QColor, QBrush
from typing import Dict, List, Optional, Tuple

from data_provider import DataProviderBase, MockDataProvider, PolygonDataProvider
from structures import OptionChain
from strategies import STRATEGIES, Strategy
from mpl_canvas import MplCanvas
from minute_data_dialog import MinuteDataDialog
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

        # 마지막 하이라이트 셀 (row, col)
        self._last_highlight: Optional[Tuple[int, int]] = None

        # 클릭으로 선택 가능한 헤더
        self.selectable_headers_map = {
            "Last": "last",
            "종가": "close",
            "고가": "high",
            "저가": "low",
            "시가": "open",
        }

        self._build_common_ui()

    def _build_common_ui(self):
        """탭에 공통적으로 들어가는 UI 요소들을 생성"""
        # 데이터 소스 및 기본 정보
        self.source_combo = QComboBox()
        self.source_combo.addItems(self.provider_map.keys())
        # 기본 Polygon.io
        idx = self.source_combo.findText("Polygon.io")
        if idx >= 0:
            self.source_combo.setCurrentIndex(idx)

        self.symbol_edit = QLineEdit("SPY")

        # 행사가 관련
        self.step_combo = QComboBox()
        self.step_combo.addItems(["0.5", "1.0", "1.5", "2.0", "2.5", "5.0"])
        self.step_combo.setCurrentText("1.0")
        self.center_strike_edit = QLineEdit("0")
        self.strike_range_spin = QSpinBox()
        self.strike_range_spin.setRange(1, 50)
        self.strike_range_spin.setValue(15)

        # 자본금 및 수수료
        self.capital_edit = QLineEdit("10000")
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setDecimals(2)
        self.fee_spin.setRange(0, 100)
        self.fee_spin.setValue(1.25)  # 계약당 수수료

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

        # 결과 표시 (레이아웃 안정화)
        self.result_label = QLabel("전략을 선택하고 옵션을 클릭하여 시뮬레이션을 시작하세요.")
        self.result_label.setWordWrap(True)
        self.result_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.result_label.setFixedHeight(84)  # 고정 높이

        self.canvas = MplCanvas(self, width=6, height=3, dpi=100)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.canvas.setFixedHeight(240)  # 고정 높이

        self.on_strategy_changed(self.strategy_combo.currentText())

    def _setup_table_headers(self):
        call_cols = [c[0] for c in CALL_COLUMNS_LEFT]
        put_cols = [c[0] for c in PUT_COLUMNS_RIGHT]
        headers = ["Strike"] + call_cols + ["|"] + put_cols
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

    # 클릭 셀 하이라이트
    def _highlight_cell(self, row: int, col: int):
        if self._last_highlight is not None:
            r, c = self._last_highlight
            old_item = self.table.item(r, c)
            if old_item:
                old_item.setBackground(QBrush())  # 기본 배경
        item = self.table.item(row, col)
        if item:
            item.setBackground(QBrush(QColor("#fff59d")))  # 연노랑
            self._last_highlight = (row, col)

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
        """
        - Strike(0) 클릭: 분 데이터 팝업(콜/풋 선택)
        - 그 외: 허용된 헤더면 가격 선택으로 전략에 반영
        """
        self._highlight_cell(row, col)

        if col == 0:
            if not self.current_chain:
                return
            strike_text = self.table.item(row, 0).text() if self.table.item(row, 0) else ""
            if not strike_text:
                return
            try:
                strike = float(strike_text)
            except ValueError:
                return

            reply = QMessageBox.question(
                self, '사이드 선택',
                f'Strike {strike}의 분 데이터를 확인합니다.\n콜(Yes) / 풋(No)?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            side = 'C' if reply == QMessageBox.Yes else 'P'
            provider = self.provider_map[self.source_combo.currentText()]
            dialog = MinuteDataDialog(provider, self.current_chain.underlying, strike, date_str, side, self)
            dialog.exec_()
            return

        header_item = self.table.horizontalHeaderItem(col)
        if not header_item:
            return
        header_text = header_item.text()
        if header_text not in self.selectable_headers_map:
            return
        if not self.active_strategy:
            return

        strike_item = self.table.item(row, 0)
        price_item = self.table.item(row, col)
        if not strike_item or not price_item:
            return

        try:
            strike = float(strike_item.text())
        except (TypeError, ValueError):
            return

        price_text = price_item.text()
        if not price_text:
            return
        try:
            price = float(price_text)
        except ValueError:
            return

        sep_col = 1 + len(CALL_COLUMNS_LEFT)
        side = 'C' if col < sep_col else 'P'

        self.selected_options.append((side, strike, price))
        if len(self.selected_options) >= self.active_strategy.legs:
            self.selected_options = self.selected_options[-self.active_strategy.legs:]
            self.calculate_and_plot()
        else:
            remaining = self.active_strategy.legs - len(self.selected_options)
            self.result_label.setText(
                f"선택됨: {side} {strike:.2f} @ {header_text}={price:.2f}. "
                f"{remaining}개 옵션을 더 선택하세요."
            )

    def calculate_and_plot(self):
        if not self.active_strategy or len(self.selected_options) != self.active_strategy.legs:
            return

        try:
            capital = float(self.capital_edit.text())
            fee_per_contract = float(self.fee_spin.value())
        except Exception:
            QMessageBox.warning(self, "입력 오류", "자본금/수수료 입력을 확인하세요.")
            return

        try:
            params, leg_info, is_credit = self._prepare_strategy_params()
        except Exception as e:
            self.result_label.setText(f"<b>오류:</b> {e}")
            return

        cost_per_unit = float(self.manual_price_spin.value()) if self.manual_price_check.isChecked() else float(params[-1])
        total_fee = fee_per_contract * self.active_strategy.legs

        try:
            if is_credit:
                strike_vals = [p for p in params if isinstance(p, (int, float))]
                strike_diff = abs(strike_vals[0] - strike_vals[1]) if len(strike_vals) >= 2 else 0.0
                margin_per_unit = (strike_diff - abs(cost_per_unit)) * OPTION_MULTIPLIER
                num_contracts = int(capital / margin_per_unit) if margin_per_unit > 0 else 1
            else:
                total_cost_per_unit = (cost_per_unit * OPTION_MULTIPLIER) + total_fee
                num_contracts = int(capital / total_cost_per_unit) if total_cost_per_unit > 0 else 0
        except Exception:
            num_contracts = 0

        if num_contracts <= 0:
            self.result_label.setText("자본금이 부족하여 1계약도 진입할 수 없습니다.")
            return

        try:
            strikes_in_play = [p for p in params if isinstance(p, (int, float))]
            if not strikes_in_play:
                raise ValueError("유효한 행사가가 없습니다.")
            S_min = min(strikes_in_play)
            S_max = max(strikes_in_play)
            if S_min == S_max:
                S_min -= 1.0
                S_max += 1.0
            else:
                S_min -= 10
                S_max += 10

            xs = np.linspace(S_min, S_max, 300)
            ys = [(self.active_strategy.pnl_func(x, params) * OPTION_MULTIPLIER) - total_fee for x in xs]
        except Exception as e:
            self.result_label.setText(f"<b>계산 오류:</b> {e}")
            return

        max_profit = max(ys) if ys else 0.0
        max_loss = min(ys) if ys else 0.0

        breakeven_points = []
        for i in range(1, len(xs)):
            if (ys[i - 1] < 0 <= ys[i]) or (ys[i - 1] > 0 >= ys[i]):
                x1, y1 = xs[i - 1], ys[i - 1]
                x2, y2 = xs[i], ys[i]
                if (y2 - y1) != 0:
                    bep = x1 - y1 * (x2 - x1) / (y2 - y1)
                    if not any(abs(bep - p) < 0.5 for p in breakeven_points):
                        breakeven_points.append(bep)

        result_text = f"<b>{self.active_strategy.name} 시뮬레이션 결과</b><br>"
        result_text += f"선택된 옵션: {leg_info}<br>"
        result_text += f"단위당 순수비용(수익): ${cost_per_unit:.2f}<br><br>"
        result_text += f"<b>총 자본금: ${capital:,.2f}</b><br>"
        result_text += f"<b>진입 가능 계약 수: {num_contracts} 계약</b><br>"
        result_text += f"<b>최대 수익금 (1계약/총): ${max_profit:,.2f} / ${max_profit * num_contracts:,.2f}</b><br>"
        result_text += f"<b>최대 손실금 (1계약/총): ${max_loss:,.2f} / ${max_loss * num_contracts:,.2f}</b><br>"

        initial_investment = (cost_per_unit * OPTION_MULTIPLIER + total_fee) * num_contracts
        if (not is_credit) and initial_investment > 0:
            roi = (max_profit * num_contracts) / initial_investment * 100
            result_text += f"<b>최대 수익률 (ROI): {roi:.2f}%</b><br>"

        bep_str = ", ".join([f"${p:.2f}" for p in breakeven_points]) if breakeven_points else "N/A"
        result_text += f"<b>손익분기점(BEP) 근사치: {bep_str}</b>"
        self.result_label.setText(result_text)

        # 차트 (tight_layout 사용 금지, 여백만 고정)
        self.canvas.ax.clear()
        self.canvas.ax.plot(xs, [y * num_contracts for y in ys], label=f"Total P&L ({num_contracts} contracts)")
        self.canvas.ax.axhline(0, color='grey', linestyle='--', linewidth=1)
        self.canvas.ax.set_title("Profit & Loss Diagram at Expiration")
        self.canvas.ax.set_xlabel("Underlying Price ($)")
        self.canvas.ax.set_ylabel("Profit / Loss ($)")
        self.canvas.ax.grid(True, alpha=0.3)
        self.canvas.ax.legend(loc="best")
        self.canvas.figure.subplots_adjust(left=0.085, right=0.98, top=0.90, bottom=0.12)
        self.canvas.draw()

    def _prepare_strategy_params(self) -> Tuple[List[float], str, bool]:
        """선택된 옵션들로부터 전략별 파라미터, 정보, 크레딧 여부를 생성 (방어 강화)"""
        if not self.active_strategy:
            raise ValueError("전략이 선택되지 않았습니다.")
        if not self.selected_options:
            raise ValueError("선택된 옵션이 없습니다.")

        need = self.active_strategy.legs
        opts_raw = self.selected_options[-need:]

        cleaned = []
        for side, strike, price in opts_raw:
            if side not in ("C", "P"):
                raise ValueError(f"사이드가 잘못되었습니다: {side}")
            K = float(strike)
            P = float(price)
            cleaned.append((K, side, P))

        if len(cleaned) < need:
            raise ValueError(f"이 전략은 {need}개 레그가 필요합니다. 현재 {len(cleaned)}개 선택됨.")

        opts = sorted(cleaned, key=lambda x: (x[0], x[1]))
        name = self.active_strategy.name
        params, leg_info = [], ""
        is_credit = "Credit" in self.active_strategy.description

        if any(s in name for s in ["Long Call", "Long Put"]):
            K, S, P = opts[0]
            params = [K, P]
            leg_info = f"{S} @ {K:.2f}"

        elif any(s in name for s in ["Bull Call", "Bear Put"]):  # Debit Spreads
            if len(opts) < 2:
                raise ValueError("스프레드 전략에 필요한 2개 레그가 선택되지 않았습니다.")
            (K1, S1, P1), (K2, S2, P2) = opts[0], opts[1]
            if S1 == 'C' and S2 == 'C':        # Bull Call
                net_cost = P1 - P2
            elif S1 == 'P' and S2 == 'P':      # Bear Put
                net_cost = P2 - P1
            else:
                raise ValueError("같은 타입(콜/풋) 2개를 선택하세요.")
            params = [K1, K2, float(net_cost)]
            leg_info = f"Long {S1}@{K1}, Short {S2}@{K2}"

        elif any(s in name for s in ["Bear Call", "Bull Put"]):  # Credit Spreads
            if len(opts) < 2:
                raise ValueError("스프레드 전략에 필요한 2개 레그가 선택되지 않았습니다.")
            (K1, S1, P1), (K2, S2, P2) = opts[0], opts[1]
            if S1 == 'C' and S2 == 'C':        # Bear Call
                net_credit = P1 - P2
            elif S1 == 'P' and S2 == 'P':      # Bull Put
                net_credit = P2 - P1
            else:
                raise ValueError("같은 타입(콜/풋) 2개를 선택하세요.")
            params = [K1, K2, float(net_credit)]
            leg_info = f"Short {S1}@{K1}, Long {S2}@{K2}"

        else:
            params = [x[0] for x in opts] + [opts[-1][2]]
            leg_info = ", ".join(f"{x[1]}@{x[0]}" for x in opts)

        if not self.manual_price_check.isChecked():
            self.manual_price_spin.setValue(float(params[-1]))

        return params, leg_info, is_credit
