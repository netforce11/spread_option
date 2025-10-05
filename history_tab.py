# -*- coding: utf-8 -*-
from typing import Optional, List, Dict, Any
import os
import json
from datetime import datetime, timedelta

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QTableWidget,
    QHBoxLayout, QMessageBox, QHeaderView, QComboBox, QDateEdit,
    QListWidget, QListWidgetItem, QDialog, QDialogButtonBox,
    QGroupBox, QSizePolicy, QFrame
)
from PyQt5.QtCore import QDate, Qt
from PyQt5.QtGui import QColor, QBrush

from concurrent.futures import ThreadPoolExecutor, as_completed

from base_tab import BaseTab
from utils import populate_option_table
from structures import OptionChain, OptionQuote, Greek
from utils import CALL_COLUMNS_LEFT, PUT_COLUMNS_RIGHT, OPTION_MULTIPLIER


class _HistorySelectDialog(QDialog):
    """특정 심볼에 대해 저장된 과거 조회값 선택/삭제"""

    def __init__(self, history_items: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("과거 조회값 선택/삭제")
        self.resize(620, 420)
        self._items = history_items
        self.selected: Optional[Dict[str, Any]] = None
        self.deleted_index: Optional[int] = None

        v = QVBoxLayout(self)
        self.listw = QListWidget()
        for it in history_items:
            s = it.get("symbol", "")
            b = it.get("buy_date", "")
            se = it.get("sell_date", "")
            ex = it.get("expiry", "")
            cs = it.get("center_strike", "")
            st = it.get("step", "")
            rr = it.get("strike_range", "")
            cap = it.get("capital", "")
            fee = it.get("fee", "")
            src = it.get("source", "")
            summary = (
                f"[{s}] 소스:{src}  매수:{b}  매도:{se}  만기:{ex}  "
                f"간격:{st}  중심:{cs}  위/아래:{rr}  자본:{cap}  수수료:{fee}"
            )
            QListWidgetItem(summary, self.listw)
        v.addWidget(self.listw)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        del_btn = btns.addButton("삭제", QDialogButtonBox.DestructiveRole)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        del_btn.clicked.connect(self._on_delete_clicked)
        v.addWidget(btns)
        self.listw.itemDoubleClicked.connect(lambda _: self._on_ok())

    def _on_ok(self):
        idx = self.listw.currentRow()
        if 0 <= idx < len(self._items):
            self.selected = self._items[idx]
            self.accept()
        else:
            self.reject()

    def _on_delete_clicked(self):
        idx = self.listw.currentRow()
        if 0 <= idx < len(self._items):
            self.deleted_index = idx
            self.accept()


class HistoryTab(BaseTab):
    """레이아웃 개편: 상단 우측 위젯들을 깔끔하게 2열로 정리"""

    _DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    _WATCH_PATH = os.path.join(_DATA_DIR, "watchlist.json")
    _HISTORY_PATH = os.path.join(_DATA_DIR, "query_history.json")
    _CHAIN_ROOT = os.path.join(_DATA_DIR, "chains")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sell_chain: Optional[OptionChain] = None
        self.watchlist: List[str] = []
        self.query_history: Dict[str, List[Dict[str, Any]]] = {}

        self._build_summary_labels()
        self._build_ui()
        self._ensure_dirs()
        self._load_watchlist_from_file()
        self._load_history_from_file()
        self._render_watchlist()

        self._clear_summary()
        self.manual_price_check.toggled.connect(lambda _: self._recompute_summary())
        self.manual_price_spin.valueChanged.connect(lambda _: self._recompute_summary())
        self.fee_spin.valueChanged.connect(lambda _: self._recompute_summary())
        self.capital_edit.editingFinished.connect(self._recompute_summary)
        self.strategy_combo.currentTextChanged.connect(lambda _: self._clear_summary())

    def _build_summary_labels(self):
        def lab(title: str) -> QLabel:
            lb = QLabel(f"<b>{title}</b><br>-")
            lb.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            lb.setStyleSheet("QLabel{font-size:12px;}")
            return lb

        self.sum_cost = lab("단위당 순수비용")
        self.sum_contract = lab("계약 수")
        self.sum_maxp1 = lab("최대 수익(1계약)")
        self.sum_maxpt = lab("최대 수익(총)")
        self.sum_maxl1 = lab("최대 손실(1계약)")
        self.sum_maxlt = lab("최대 손실(총)")
        self.sum_roi = lab("최대 수익률(ROI)")
        self.sum_bep = lab("BEP")

    def _ensure_dirs(self):
        for p in (self._DATA_DIR, self._CHAIN_ROOT):
            try:
                os.makedirs(p, exist_ok=True)
            except Exception as e:
                print(f"[HistoryTab] 디렉토리 생성 실패({p}): {e}")

    def _watchlist_path(self) -> str:
        return self._WATCH_PATH

    def _history_path(self) -> str:
        return self._HISTORY_PATH

    def _chain_dir(self, symbol: str, expiry: str, step: float, center: float, strike_range: int) -> str:
        key = f"step_{step}_center_{center}_range_{strike_range}"
        d = os.path.join(self._CHAIN_ROOT, symbol.upper(), expiry, key)
        os.makedirs(d, exist_ok=True)
        return d

    def _chain_path(self, symbol: str, expiry: str, step: float, center: float, strike_range: int, date: str) -> str:
        d = self._chain_dir(symbol, expiry, step, center, strike_range)
        return os.path.join(d, f"{date}.json")

    def _load_watchlist_from_file(self):
        try:
            p = self._watchlist_path()
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.watchlist = [str(x).upper().strip() for x in data if isinstance(x, (str,))]
        except Exception as e:
            print(f"[HistoryTab] watchlist 로딩 실패: {e}")

    def _save_watchlist_to_file(self):
        try:
            with open(self._watchlist_path(), "w", encoding="utf-8") as f:
                json.dump(self.watchlist, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[HistoryTab] watchlist 저장 실패: {e}")

    def _load_history_from_file(self):
        try:
            p = self._history_path()
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    normalized: Dict[str, List[Dict[str, Any]]] = {}
                    for k, v in data.items():
                        sym = str(k).upper().strip()
                        if isinstance(v, list):
                            items = [x for x in v if isinstance(x, dict)]
                            if items:
                                normalized[sym] = items
                    self.query_history = normalized
        except Exception as e:
            print(f"[HistoryTab] history 로딩 실패: {e}")

    def _save_history_to_file(self):
        try:
            with open(self._history_path(), "w", encoding="utf-8") as f:
                json.dump(self.query_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[HistoryTab] history 저장 실패: {e}")

    def _extract_chain_date(self, chain: OptionChain, fallback_date: str) -> str:
        for attr in ("date", "trade_date", "as_of", "query_date"):
            v = getattr(chain, attr, None)
            if isinstance(v, str) and v:
                return v
        return fallback_date

    @staticmethod
    def _chain_to_json(chain: OptionChain, date_value: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "underlying": getattr(chain, "underlying", ""),
            "date": date_value,
            "expiration": getattr(chain, "expiration", ""),
            "quotes": []
        }
        for q in (getattr(chain, "quotes", []) or []):
            out["quotes"].append({
                "symbol": getattr(q, "symbol", out["underlying"]),
                "side": getattr(q, "side", ""),
                "strike": float(getattr(q, "strike", 0.0)),
                "last": float(getattr(q, "last", getattr(q, "close", 0.0))),
                "open": float(getattr(q, "open", 0.0)),
                "high": float(getattr(q, "high", 0.0)),
                "low": float(getattr(q, "low", 0.0)),
                "close": float(getattr(q, "close", getattr(q, "last", 0.0))),
                "prev_close": float(getattr(q, "prev_close", 0.0)),
            })
        return out

    @staticmethod
    def _json_to_chain(payload: Dict[str, Any]) -> OptionChain:
        underlying = payload.get("underlying", "")
        date = payload.get("date", "")
        expiration = payload.get("expiration", "")
        quotes_json = payload.get("quotes", [])
        quotes: List[OptionQuote] = []
        for j in quotes_json:
            sym = j.get("symbol", underlying)
            side = j.get("side", "")
            strike = float(j.get("strike", 0.0))
            last = float(j.get("last", j.get("close", 0.0)))
            greeks = Greek(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
            oq = OptionQuote(sym, side, strike, last, last, 0.0, greeks)
            setattr(oq, "open", float(j.get("open", 0.0)))
            setattr(oq, "high", float(j.get("high", 0.0)))
            setattr(oq, "low", float(j.get("low", 0.0)))
            setattr(oq, "close", float(j.get("close", last)))
            setattr(oq, "prev_close", float(j.get("prev_close", 0.0)))
            quotes.append(oq)
        quotes.sort(key=lambda q: (q.side, q.strike))
        try:
            return OptionChain(underlying, date, expiration, quotes)
        except Exception:
            oc = OptionChain.__new__(OptionChain)
            oc.underlying = underlying
            oc.date = date
            oc.expiration = expiration
            oc.quotes = quotes
            return oc

    def _load_chain_from_file(self, symbol: str, expiry: str, step: float, center: float, strike_range: int,
                              date: str) -> Optional[OptionChain]:
        path = self._chain_path(symbol, expiry, step, center, strike_range, date)
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                return self._json_to_chain(payload)
        except Exception as e:
            print(f"[HistoryTab] 체인 로딩 실패({path}): {e}")
        return None

    def _save_chain_to_file(self, chain: OptionChain, symbol: str, expiry: str, step: float, center: float,
                            strike_range: int, date: str):
        chain_date = self._extract_chain_date(chain, date)
        path = self._chain_path(symbol, expiry, step, center, strike_range, chain_date)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._chain_to_json(chain, chain_date), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[HistoryTab] 체인 저장 실패({path}): {e}")

    @staticmethod
    def _date_range_ymd(start_ymd: str, end_ymd: str) -> List[str]:
        s = datetime.strptime(start_ymd, "%Y-%m-%d").date()
        e = datetime.strptime(end_ymd, "%Y-%m-%d").date()
        if e < s: s, e = e, s
        out = []
        cur = s
        while cur <= e:
            out.append(cur.strftime("%Y-%m-%d"))
            cur = cur + timedelta(days=1)
        return out

    @staticmethod
    def _friday_of_week(qdate: QDate) -> QDate:
        dow = qdate.dayOfWeek()
        delta = 5 - dow
        if delta < 0:
            delta += 7
        return qdate.addDays(delta)

    def _build_ui(self):
        if self.layout() is not None:
            while self.layout().count():
                item = self.layout().takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()

        root = QVBoxLayout(self)

        # 관심종목
        wl_box = QHBoxLayout()
        wl_box.addWidget(QLabel("관심종목"))
        self.watch_edit = QLineEdit()
        self.watch_edit.setPlaceholderText("예: SPY / TSLA 등")
        self.watch_edit.setFixedWidth(160)
        wl_box.addWidget(self.watch_edit)
        add_btn = QPushButton("추가")
        del_btn = QPushButton("삭제")
        wl_box.addWidget(add_btn)
        wl_box.addWidget(del_btn)
        self.watch_list = QListWidget()
        wl_v = QVBoxLayout()
        wl_v.addLayout(wl_box)
        wl_v.addWidget(self.watch_list)

        add_btn.clicked.connect(self._on_watch_add)
        del_btn.clicked.connect(self._on_watch_del)
        self.watch_list.itemClicked.connect(self._on_watch_item_clicked)
        self.watch_edit.returnPressed.connect(self._on_watch_add)

        # 우측: 깔끔한 2열 세로 일렬 배치
        controls = QGridLayout()
        row = 0

        # 1열
        controls.addWidget(QLabel("데이터 소스"), row, 0)
        controls.addWidget(self.source_combo, row, 1)

        # 2열
        controls.addWidget(QLabel("심볼"), row, 2)
        self.symbol_edit.setFixedWidth(140)
        controls.addWidget(self.symbol_edit, row, 3)
        row += 1

        # 1열
        controls.addWidget(QLabel("매수일"), row, 0)
        self.buy_date_edit = QDateEdit()
        self.buy_date_edit.setCalendarPopup(True)
        self.buy_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.buy_date_edit.setDate(QDate.currentDate().addDays(-3))
        self.buy_date_edit.dateChanged.connect(self._on_buy_date_changed)
        controls.addWidget(self.buy_date_edit, row, 1)

        # 2열
        controls.addWidget(QLabel("매도일(청산일)"), row, 2)
        self.sell_date_edit = QDateEdit()
        self.sell_date_edit.setCalendarPopup(True)
        self.sell_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.sell_date_edit.setDate(QDate.currentDate().addDays(-1))
        controls.addWidget(self.sell_date_edit, row, 3)
        row += 1

        # 1열
        controls.addWidget(QLabel("만기일"), row, 0)
        self.expiry_edit = QDateEdit()
        self.expiry_edit.setCalendarPopup(True)
        self.expiry_edit.setDisplayFormat("yyyy-MM-dd")
        self.expiry_edit.setDate(QDate.currentDate())
        controls.addWidget(self.expiry_edit, row, 1)

        # 2열
        controls.addWidget(QLabel("행사가 간격"), row, 2)
        controls.addWidget(self.step_combo, row, 3)
        row += 1

        # 1열
        controls.addWidget(QLabel("중심 행사가"), row, 0)
        self.center_strike_edit.setFixedWidth(120)
        controls.addWidget(self.center_strike_edit, row, 1)

        # 2열
        controls.addWidget(QLabel("위/아래 개수"), row, 2)
        self.strike_range_spin.setFixedWidth(100)
        controls.addWidget(self.strike_range_spin, row, 3)
        row += 1

        # 1열
        controls.addWidget(QLabel("자본금($)"), row, 0)
        self.capital_edit.setFixedWidth(120)
        controls.addWidget(self.capital_edit, row, 1)

        # 2열
        controls.addWidget(QLabel("계약당 수수료($)"), row, 2)
        self.fee_spin.setFixedWidth(100)
        controls.addWidget(self.fee_spin, row, 3)
        row += 1

        # 1열
        controls.addWidget(QLabel("옵션 타입"), row, 0)
        self.side_combo = QComboBox()
        self.side_combo.addItems(["양쪽 모두", "콜만", "풋만"])
        self.side_combo.currentIndexChanged.connect(self._refill_tables_with_filter)
        controls.addWidget(self.side_combo, row, 1)

        # 2열
        controls.addWidget(QLabel("만기 종가(기초자산)"), row, 2)
        self.final_spot_edit = QLineEdit("")
        self.final_spot_edit.setPlaceholderText("예: 455.00")
        self.final_spot_edit.setFixedWidth(120)
        controls.addWidget(self.final_spot_edit, row, 3)
        row += 1

        # 버튼들 - 전체 행에 걸쳐 배치
        self.expiry_calc_btn = QPushButton("만기결제 계산")
        self.expiry_calc_btn.clicked.connect(self.on_expiry_settlement_clicked)
        controls.addWidget(self.expiry_calc_btn, row, 0, 1, 2)

        load_btn = QPushButton("과거 체인 조회 (매수~만기 전체)")
        load_btn.clicked.connect(self.on_load_chains)
        controls.addWidget(load_btn, row, 2, 1, 2)
        row += 1

        # 요약 프레임 - 전체 행
        summary_group = QGroupBox("시뮬레이션 요약")
        sum_row = QHBoxLayout(summary_group)
        for w in (self.sum_cost, self.sum_contract, self.sum_maxp1, self.sum_maxpt,
                  self.sum_maxl1, self.sum_maxlt, self.sum_roi, self.sum_bep):
            sum_row.addWidget(w)
        sum_row.addStretch()
        controls.addWidget(summary_group, row, 0, 1, 4)

        top_box = QHBoxLayout()
        top_box.addLayout(wl_v, 1)
        right_v = QVBoxLayout()
        right_v.addLayout(controls)
        top_box.addLayout(right_v, 3)
        root.addLayout(top_box)

        strategy_layout = QGridLayout()
        strategy_layout.addWidget(QLabel("전략 선택"), 0, 0)
        strategy_layout.addWidget(self.strategy_combo, 0, 1)
        strategy_layout.addWidget(self.manual_price_check, 0, 2)
        strategy_layout.addWidget(self.manual_price_spin, 0, 3)
        strategy_layout.addWidget(self.strategy_desc_label, 1, 0, 1, 4)
        root.addLayout(strategy_layout)

        tables_layout = QHBoxLayout()
        buy_box = QVBoxLayout()
        buy_box.addWidget(QLabel("매수일 옵션 체인 (Last = 해당 날짜 종가)"))
        self.buy_table = self.table
        self.buy_table.cellClicked.connect(
            lambda r, c: self._on_buy_table_clicked(r, c, self.buy_date_edit.date().toString("yyyy-MM-dd"))
        )
        buy_box.addWidget(self.buy_table)

        sell_box = QVBoxLayout()
        sell_box.addWidget(QLabel("매도일 옵션 체인 (참고용 / Last = 해당 날짜 종가)"))
        self.sell_table = QTableWidget()
        self._setup_table_headers_for_sell_table(self.sell_table)
        self.sell_table.cellClicked.connect(
            lambda r, c: self._on_sell_table_clicked(r, c, self.sell_date_edit.date().toString("yyyy-MM-dd"))
        )
        sell_box.addWidget(self.sell_table)
        sell_box.addWidget(self.result_label)
        sell_box.addWidget(self.canvas)

        tables_layout.addLayout(buy_box, 1)
        tables_layout.addLayout(sell_box, 1)
        root.addLayout(tables_layout)

        controls.setColumnStretch(0, 0)
        controls.setColumnStretch(1, 1)
        controls.setColumnStretch(2, 0)
        controls.setColumnStretch(3, 1)

    def _setup_table_headers_for_sell_table(self, table: QTableWidget):
        call_cols = [c[0] for c in CALL_COLUMNS_LEFT]
        put_cols = [c[0] for c in PUT_COLUMNS_RIGHT]
        headers = ["Strike"] + call_cols + ["|"] + put_cols
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

    def _render_watchlist(self):
        self.watch_list.clear()
        for sym in self.watchlist:
            QListWidgetItem(sym, self.watch_list)

    def _on_watch_add(self):
        sym = self.watch_edit.text().strip().upper()
        if not sym:
            QMessageBox.information(self, "안내", "추가할 심볼을 입력하세요.")
            return
        if sym in self.watchlist:
            QMessageBox.information(self, "안내", f"{sym} 는 이미 리스트에 있습니다.")
            return
        self.watchlist.append(sym)
        self._render_watchlist()
        self._save_watchlist_to_file()
        self.watch_edit.clear()

    def _on_watch_del(self):
        row = self.watch_list.currentRow()
        if row < 0:
            QMessageBox.information(self, "안내", "삭제할 항목을 선택하세요.")
            return
        sym = self.watch_list.item(row).text()
        self.watch_list.takeItem(row)
        if sym in self.watchlist:
            self.watchlist.remove(sym)
        self._save_watchlist_to_file()

    def _on_watch_item_clicked(self, item: QListWidgetItem):
        sym = item.text().strip().upper()
        self._load_history_from_file()
        hist = self.query_history.get(sym, [])
        if not hist:
            self.symbol_edit.setText(sym)
            return

        dlg = _HistorySelectDialog(hist, self)
        if dlg.exec_() == QDialog.Accepted:
            if dlg.deleted_index is not None:
                idx = dlg.deleted_index
                if 0 <= idx < len(hist):
                    del hist[idx]
                    if hist:
                        self.query_history[sym] = hist
                    else:
                        del self.query_history[sym]
                    self._save_history_to_file()
                return
            if dlg.selected:
                self._apply_snapshot(dlg.selected)

    def _get_chain_with_cache(self, provider, symbol: str, date: str, expiry: str,
                              step: float, center: float, strike_range: int):
        try:
            cached = self._load_chain_from_file(symbol, expiry, step, center, strike_range, date)
            if cached is not None:
                return date, cached, None
        except Exception as e:
            print(f"[HistoryTab] 캐시 읽기 실패({symbol}/{date}): {e}")

        try:
            chain = provider.fetch_chain(symbol, date, expiry, step, center, strike_range)
            try:
                self._save_chain_to_file(chain, symbol, expiry, step, center, strike_range, date)
            except Exception as e:
                print(f"[HistoryTab] 캐시 저장 실패({symbol}/{date}): {e}")
            return date, chain, None
        except Exception as e:
            return date, None, str(e)

    def on_load_chains(self):
        provider = self.provider_map[self.source_combo.currentText()]
        symbol = self.symbol_edit.text().strip().upper()
        buy_date = self.buy_date_edit.date().toString("yyyy-MM-dd")
        sell_date = self.sell_date_edit.date().toString("yyyy-MM-dd")
        expiry = self.expiry_edit.date().toString("yyyy-MM-dd")
        try:
            step = float(self.step_combo.currentText())
        except Exception:
            QMessageBox.warning(self, "입력 오류", "행사가 간격(step)을 올바르게 선택/입력하세요.")
            return

        if hasattr(provider, "set_minute_expiration"):
            provider.set_minute_expiration(expiry)

        try:
            center_strike = float(self.center_strike_edit.text())
            strike_range = self.strike_range_spin.value()
        except ValueError:
            QMessageBox.warning(self, "입력 오류", "중심 행사가와 범위는 숫자여야 합니다.")
            return

        all_dates = self._date_range_ymd(buy_date, expiry)

        results: Dict[str, OptionChain] = {}
        errors: Dict[str, str] = {}
        max_workers = min(8, max(4, len(all_dates)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(self._get_chain_with_cache, provider, symbol, d, expiry, step, center_strike, strike_range)
                for d in all_dates]
            for fut in as_completed(futs):
                d, chain, err = fut.result()
                if err:
                    errors[d] = err
                elif chain:
                    results[d] = chain

        if buy_date in results:
            self.current_chain = results[buy_date]
        else:
            QMessageBox.critical(self, "에러", f"매수일 체인 데이터가 없습니다: {buy_date}\n{errors.get(buy_date, '')}")
            return

        if sell_date in results:
            self.sell_chain = results[sell_date]
        else:
            if sell_date in errors:
                QMessageBox.warning(self, "경고", f"매도일 체인 조회 실패({sell_date}): {errors[sell_date]}")
            self.sell_chain = None

        self._refill_tables_with_filter()
        self.selected_options.clear()
        self.on_strategy_changed(self.strategy_combo.currentText())

        snap = {
            "source": self.source_combo.currentText(),
            "symbol": symbol,
            "buy_date": buy_date,
            "sell_date": sell_date,
            "expiry": expiry,
            "step": step,
            "center_strike": center_strike,
            "strike_range": strike_range,
            "capital": self.capital_edit.text(),
            "fee": self.fee_spin.value(),
            "side_filter": self.side_combo.currentText(),
        }
        self._save_history_snapshot(symbol, snap)

        if errors:
            print(f"[HistoryTab] 일부 날짜 조회 실패: {errors}")

        QMessageBox.information(self, "완료",
                                f"{symbol} {buy_date}~{expiry} ({len(all_dates)}일) 데이터가 준비되었습니다.\n"
                                f"캐시 경로: ./data/chains/{symbol}/{expiry}/...")

        self._clear_summary()

    def _save_history_snapshot(self, symbol: str, snap: Dict[str, Any], keep_last: int = 50):
        lst = self.query_history.setdefault(symbol, [])
        if not lst or lst[-1] != snap:
            lst.append(snap)
        if len(lst) > keep_last:
            del lst[:-keep_last]
        self._save_history_to_file()
        if symbol and symbol not in self.watchlist:
            self.watchlist.append(symbol)
            self._render_watchlist()
            self._save_watchlist_to_file()

    def _refill_tables_with_filter(self):
        if not hasattr(self, "side_combo"):
            return
        sel = self.side_combo.currentText() if self.side_combo.currentText() else "양쪽 모두"

        def filter_chain(chain: Optional[OptionChain]) -> Optional[OptionChain]:
            if chain is None:
                return None
            if sel == "양쪽 모두":
                return chain
            want = "C" if sel == "콜만" else "P"
            filtered = [q for q in getattr(chain, "quotes", []) if getattr(q, "side", None) == want]
            try:
                return OptionChain(getattr(chain, "underlying", ""),
                                   getattr(chain, "date", ""),
                                   getattr(chain, "expiration", ""),
                                   filtered)
            except Exception:
                oc = OptionChain.__new__(OptionChain)
                oc.underlying = getattr(chain, "underlying", "")
                oc.date = getattr(chain, "date", "")
                oc.expiration = getattr(chain, "expiration", "")
                oc.quotes = filtered
                return oc

        populate_option_table(self.buy_table, filter_chain(self.current_chain))
        populate_option_table(self.sell_table, filter_chain(self.sell_chain))

    def _on_buy_table_clicked(self, row: int, col: int, date_str: str):
        item = self.buy_table.item(row, col)
        if item:
            item.setBackground(QBrush(QColor("#fff59d")))
        super().on_table_cell_clicked(row, col, date_str)
        self._recompute_summary()

    def _on_sell_table_clicked(self, row: int, col: int, date_str: str):
        item = self.sell_table.item(row, col)
        if item:
            item.setBackground(QBrush(QColor("#fff59d")))
        if col == 0 and self.sell_chain:
            strike_item = self.sell_table.item(row, 0)
            if not strike_item:
                return
            try:
                strike = float(strike_item.text())
            except ValueError:
                return
            reply = QMessageBox.question(
                self, '사이드 선택',
                f'Strike {strike}의 분 데이터를 확인합니다.\n콜(Yes) / 풋(No)?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            side = 'C' if reply == QMessageBox.Yes else 'P'
            provider = self.provider_map[self.source_combo.currentText()]
            from minute_data_dialog import MinuteDataDialog
            dialog = MinuteDataDialog(provider, self.current_chain.underlying, strike, date_str, side, self)
            dialog.exec_()

    def on_expiry_settlement_clicked(self):
        if not self.active_strategy:
            QMessageBox.information(self, "안내", "먼저 전략을 선택하세요.")
            return
        if len(self.selected_options) != self.active_strategy.legs:
            QMessageBox.information(self, "안내", "매수일 테이블에서 필요한 레그 수만큼 가격을 클릭해 선택하세요.")
            return

        final_spot_text = self.final_spot_edit.text().strip()
        if not final_spot_text:
            QMessageBox.warning(self, "입력 필요", "만기 종가(기초자산)를 입력하세요.")
            return
        try:
            final_spot = float(final_spot_text)
        except ValueError:
            QMessageBox.warning(self, "입력 오류", "만기 종가는 숫자여야 합니다.")
            return

        params, leg_info, is_credit = self._prepare_strategy_params()
        fee_per_contract = self.fee_spin.value()

        pnl_per_unit = (self.active_strategy.pnl_func(final_spot, params) * OPTION_MULTIPLIER) - (
                    fee_per_contract * self.active_strategy.legs * 2)

        capital = float(self.capital_edit.text() or "0")
        if is_credit:
            strike_diff = abs(params[0] - params[1]) if len(params) > 2 else 0
            margin_per_unit = (strike_diff - abs(params[-1])) * OPTION_MULTIPLIER
            num_contracts = int(capital / margin_per_unit) if margin_per_unit > 0 else 1
        else:
            total_cost_per_unit = (params[-1] * OPTION_MULTIPLIER) + (fee_per_contract * self.active_strategy.legs)
            num_contracts = int(capital / total_cost_per_unit) if total_cost_per_unit > 0 else 0

        if num_contracts <= 0:
            QMessageBox.information(self, "안내", "자본금이 부족합니다.")
            return

        total_pnl = pnl_per_unit * num_contracts

        result_text = (f"<b>만기 결제 결과</b><br>"
                       f"포지션: {leg_info}<br>"
                       f"만기 종가(기초자산): ${final_spot:.2f}<br>"
                       f"<b>계약당 손익: ${pnl_per_unit:,.2f}</b><br>"
                       f"<b>진입 가능 계약 수: {num_contracts} 계약</b><br>"
                       f"<b>총 손익: ${total_pnl:,.2f}</b>")
        self.result_label.setText(result_text)

        self.canvas.ax.clear()
        self.canvas.ax.bar(['Total P&L'], [total_pnl], color='skyblue' if total_pnl > 0 else 'salmon')
        self.canvas.ax.set_ylabel("Profit / Loss ($)")
        self.canvas.ax.set_title("Expiration Settlement P&L")
        self.canvas.ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        self.canvas.figure.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.18)
        self.canvas.draw()

    def _clear_summary(self):
        for lb in (self.sum_cost, self.sum_contract, self.sum_maxp1, self.sum_maxpt,
                   self.sum_maxl1, self.sum_maxlt, self.sum_roi, self.sum_bep):
            t = lb.text().split("<br>")[0]
            lb.setText(f"{t}<br>-")

    def _recompute_summary(self):
        if not self.active_strategy or len(self.selected_options) != self.active_strategy.legs:
            self._clear_summary()
            return

        try:
            capital = float(self.capital_edit.text() or "0")
            fee_per_contract = float(self.fee_spin.value())
        except Exception:
            self._clear_summary()
            return

        try:
            params, _, is_credit = self._prepare_strategy_params()
        except Exception:
            self._clear_summary()
            return

        cost_per_unit = float(self.manual_price_spin.value()) if self.manual_price_check.isChecked() else float(
            params[-1])
        total_fee = fee_per_contract * self.active_strategy.legs

        if is_credit:
            strike_vals = [p for p in params if isinstance(p, (int, float))]
            strike_diff = abs(strike_vals[0] - strike_vals[1]) if len(strike_vals) >= 2 else 0.0
            margin_per_unit = (strike_diff - abs(cost_per_unit)) * OPTION_MULTIPLIER
            num_contracts = int(capital / margin_per_unit) if margin_per_unit > 0 else 1
        else:
            total_cost_per_unit = (cost_per_unit * OPTION_MULTIPLIER) + total_fee
            num_contracts = int(capital / total_cost_per_unit) if total_cost_per_unit > 0 else 0

        if num_contracts <= 0:
            self._clear_summary()
            return

        strikes_in_play = [p for p in params if isinstance(p, (int, float))]
        if not strikes_in_play:
            self._clear_summary()
            return
        S_min = min(strikes_in_play) - 10
        S_max = max(strikes_in_play) + 10
        xs = np.linspace(S_min, S_max, 300)
        ys = [(self.active_strategy.pnl_func(x, params) * OPTION_MULTIPLIER) - total_fee for x in xs]

        max_profit = max(ys) if ys else 0.0
        max_loss = min(ys) if ys else 0.0

        initial_investment = (cost_per_unit * OPTION_MULTIPLIER + total_fee) * num_contracts
        roi_txt = "-"
        if (not is_credit) and initial_investment > 0:
            roi = (max_profit * num_contracts) / initial_investment * 100
            roi_txt = f"{roi:.2f}%"

        breakeven_points = []
        for i in range(1, len(xs)):
            if (ys[i - 1] < 0 <= ys[i]) or (ys[i - 1] > 0 >= ys[i]):
                x1, y1 = xs[i - 1], ys[i - 1]
                x2, y2 = xs[i], ys[i]
                if (y2 - y1) != 0:
                    bep = x1 - y1 * (x2 - x1) / (y2 - y1)
                    if not any(abs(bep - p) < 0.5 for p in breakeven_points):
                        breakeven_points.append(bep)
        bep_txt = ", ".join([f"${p:.2f}" for p in breakeven_points]) if breakeven_points else "-"

        self.sum_cost.setText(f"<b>단위당 순수비용</b><br>${cost_per_unit:.2f}")
        self.sum_contract.setText(f"<b>계약 수</b><br>{num_contracts}")
        self.sum_maxp1.setText(f"<b>최대 수익(1계약)</b><br>${max_profit:,.2f}")
        self.sum_maxpt.setText(f"<b>최대 수익(총)</b><br>${max_profit * num_contracts:,.2f}")
        self.sum_maxl1.setText(f"<b>최대 손실(1계약)</b><br>${max_loss:,.2f}")
        self.sum_maxlt.setText(f"<b>최대 손실(총)</b><br>${max_loss * num_contracts:,.2f}")
        self.sum_roi.setText(f"<b>최대 수익률(ROI)</b><br>{roi_txt}")
        self.sum_bep.setText(f"<b>BEP</b><br>{bep_txt}")

    def _apply_snapshot(self, snap: Dict[str, Any]):
        src = snap.get("source", self.source_combo.currentText())
        idx = self.source_combo.findText(src)
        if idx >= 0:
            self.source_combo.setCurrentIndex(idx)
        self.symbol_edit.setText(snap.get("symbol", ""))
        try:
            self.buy_date_edit.setDate(QDate.fromString(snap.get("buy_date", ""), "yyyy-MM-dd"))
            self.sell_date_edit.setDate(QDate.fromString(snap.get("sell_date", ""), "yyyy-MM-dd"))
            self.expiry_edit.setDate(QDate.fromString(snap.get("expiry", ""), "yyyy-MM-dd"))
        except Exception:
            pass
        step = str(snap.get("step", self.step_combo.currentText()))
        i = self.step_combo.findText(step)
        if i >= 0:
            self.step_combo.setCurrentIndex(i)
        self.center_strike_edit.setText(str(snap.get("center_strike", "")))
        try:
            self.strike_range_spin.setValue(int(snap.get("strike_range", self.strike_range_spin.value())))
        except Exception:
            pass
        self.capital_edit.setText(str(snap.get("capital", self.capital_edit.text())))
        try:
            self.fee_spin.setValue(float(snap.get("fee", self.fee_spin.value())))
        except Exception:
            pass
        side = snap.get("side_filter", self.side_combo.currentText())
        j = self.side_combo.findText(side)
        if j >= 0:
            self.side_combo.setCurrentIndex(j)
        self._clear_summary()

    def _on_buy_date_changed(self, qdate: QDate):
        fri = self._friday_of_week(qdate)
        self.sell_date_edit.setDate(fri)
        self.expiry_edit.setDate(fri)
        self._clear_summary()