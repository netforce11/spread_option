# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
import math
import time
import random
from typing import Dict, List, Tuple, Optional

import requests
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
    QDateEdit, QDoubleSpinBox, QSpinBox, QPushButton, QComboBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import QDate, Qt
from PyQt5.QtGui import QColor, QBrush

from data_provider import DataProviderBase, MockDataProvider, PolygonDataProvider
from ws_spread_monitor import SpreadWSWorker


# ----------------------------
# 유틸
# ----------------------------
def _float(val, default=0.0) -> float:
    try:
        return float(val)
    except Exception:
        return float(default)

def _round_to_option_tick(price: float) -> float:
    """옵션 중심행사가로 쓰기 좋은 라운딩(0.5단위)"""
    if price <= 0 or math.isnan(price):
        return 0.0
    return round(price * 2.0) / 2.0


# ----------------------------
# Polygon 주가 조회 (prev close)
# ----------------------------
POLY_BASE = "https://api.polygon.io"

def fetch_polygon_prev_closes(
    symbols: List[str],
    api_key: str,
    chunk_size: int = 6,
    max_retries: int = 3,
    adjusted: bool = True,
) -> Dict[str, Optional[float]]:
    """
    Polygon /v2/aggs/ticker/{SYM}/prev 로 '전일 종가'를 조회.
    - symbols 를 chunk_size 로 나눠 순차 호출
    - 429/5xx 에 대해 지수 백오프 재시도
    - 일부 실패해도 성공분은 유지
    반환: { "SPY": 431.23, "QQQ": 367.10, ... }
    """
    out: Dict[str, Optional[float]] = {s.upper(): None for s in (symbols or [])}
    if not symbols:
        return out

    # 유니크/정리
    uniq = []
    seen = set()
    for s in symbols:
        ss = s.strip().upper()
        if ss and ss not in seen:
            seen.add(ss)
            uniq.append(ss)

    def _fetch_one(sym: str) -> Optional[float]:
        url = f"{POLY_BASE}/v2/aggs/ticker/{sym}/prev"
        params = {"adjusted": "true" if adjusted else "false", "apiKey": api_key}
        backoff = 0.6
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, timeout=10)
                # rate limit/5xx 처리
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{resp.status_code} {resp.reason}")
                resp.raise_for_status()
                j = resp.json()
                results = j.get("results") or []
                if results:
                    r = results[0]
                    c = r.get("c")  # close
                    return float(c) if c is not None else None
                return None
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[SpreadMonitorTab] Polygon fetch error ({sym}): {e}")
                    return None
                time.sleep(backoff + random.uniform(0.0, 0.4))
                backoff = min(backoff * 1.8, 4.0)

    # 순차(청크별로 약간 쉬어줌)
    for i in range(0, len(uniq), chunk_size):
        batch = uniq[i:i + chunk_size]
        for sym in batch:
            out[sym] = _fetch_one(sym)
        if i + chunk_size < len(uniq):
            time.sleep(0.5 + random.uniform(0.0, 0.4))
    return out


# ----------------------------
# 패널(단일 종목)
# ----------------------------
class _SymbolPanel(QWidget):
    """단일 종목 설정 + 결과 테이블"""
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title

        g = QGroupBox(self.title)
        gv = QVBoxLayout(g)

        form = QGridLayout()
        r = 0

        form.addWidget(QLabel("심볼"), r, 0)
        self.symbol_edit = QLineEdit("")
        self.symbol_edit.setPlaceholderText("예: SPY")
        form.addWidget(self.symbol_edit, r, 1)

        form.addWidget(QLabel("매수일"), r, 2)
        self.buy_date = QDateEdit()
        self.buy_date.setCalendarPopup(True)
        self.buy_date.setDisplayFormat("yyyy-MM-dd")
        self.buy_date.setDate(QDate.currentDate().addDays(-1))
        form.addWidget(self.buy_date, r, 3)

        r += 1
        form.addWidget(QLabel("만기일"), r, 0)
        self.expiry = QDateEdit()
        self.expiry.setCalendarPopup(True)
        self.expiry.setDisplayFormat("yyyy-MM-dd")
        self.expiry.setDate(QDate.currentDate())
        form.addWidget(self.expiry, r, 1)

        form.addWidget(QLabel("행사가 간격(step)"), r, 2)
        self.step = QDoubleSpinBox()
        self.step.setDecimals(2)
        self.step.setRange(0.5, 50.0)
        self.step.setSingleStep(0.5)
        self.step.setValue(1.0)
        form.addWidget(self.step, r, 3)

        r += 1
        form.addWidget(QLabel("중심 행사가"), r, 0)
        self.center = QDoubleSpinBox()
        self.center.setDecimals(2)
        self.center.setRange(0, 100000)
        self.center.setValue(0.0)
        form.addWidget(self.center, r, 1)

        form.addWidget(QLabel("위/아래 개수"), r, 2)
        self.range_spin = QSpinBox()
        self.range_spin.setRange(1, 100)
        self.range_spin.setValue(15)
        form.addWidget(self.range_spin, r, 3)

        r += 1
        form.addWidget(QLabel("콜 매수-매도 간격(width)"), r, 0)
        self.width = QDoubleSpinBox()
        self.width.setDecimals(2)
        self.width.setRange(0.5, 100.0)
        self.width.setSingleStep(0.5)
        self.width.setValue(2.0)
        form.addWidget(self.width, r, 1)

        gv.addLayout(form)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "K_long", "K_short", "LongPx", "ShortPx", "NetCost(Long-Short)", "Date", "Expiry"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        gv.addWidget(self.table)

        lay = QVBoxLayout(self)
        lay.addWidget(g)

    def params(self) -> Dict:
        return {
            "symbol": self.symbol_edit.text().strip().upper(),
            "buy_date": self.buy_date.date().toString("yyyy-MM-dd"),
            "expiry": self.expiry.date().toString("yyyy-MM-dd"),
            "step": self.step.value(),
            "center": self.center.value(),
            "range": self.range_spin.value(),
            "width": self.width.value(),
        }

    def clear_results(self):
        self.table.setRowCount(0)

    def populate(self, rows: List[Tuple[float, float, float, float, float, str, str]]):
        """rows: [(K_long, K_short, P_long, P_short, net, date, expiry), ...]"""
        self.table.setRowCount(len(rows))
        for i, (k1, k2, p1, p2, net, d, ex) in enumerate(rows):
            items = [
                QTableWidgetItem(f"{k1:.2f}"),
                QTableWidgetItem(f"{k2:.2f}"),
                QTableWidgetItem(f"{p1:.2f}"),
                QTableWidgetItem(f"{p2:.2f}"),
                QTableWidgetItem(f"{net:.2f}"),
                QTableWidgetItem(d),
                QTableWidgetItem(ex),
            ]
            col = QColor("#fff59d") if net <= 0.05 else QColor("#e8f5e9") if net <= 0.10 else QColor("#ffffff")
            for it in items:
                it.setTextAlignment(Qt.AlignCenter)
                it.setBackground(QBrush(col))
            for c, it in enumerate(items):
                self.table.setItem(i, c, it)


# ----------------------------
# 탭 본체
# ----------------------------
class SpreadMonitorTab(QWidget):
    """
    WebSocket(Polygon) 기반 실시간 감시:
      - 10개 종목 (2행 x 5열)
      - 관심종목 리스트박스 + Polygon 'prev close'로 각 패널의 중심행사가 자동 세팅
      - 콜 수직 스프레드 Long(K) - Short(K+width) 순코스트가 임계값 이하일 때 표에 반영
    """
    # ❌ self를 쓰면 안 됩니다. 클래스 변수로 직접 지정하세요.
    _DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    _WATCH_PATH = os.path.join(_DATA_DIR, "watchlist_spread.json")

    def __init__(self, parent=None):
        super().__init__(parent)

        self.provider_map: Dict[str, DataProviderBase] = {
            "Mock": MockDataProvider(),
            "Polygon.io": PolygonDataProvider(),
        }
        self.ws_worker: Optional[SpreadWSWorker] = None
        self.watchlist: List[str] = []

        self._build_ui()
        self._load_watchlist()

    # ─────────────── 파일/디렉토리 ───────────────
    def _ensure_dir(self):
        try:
            os.makedirs(self._DATA_DIR, exist_ok=True)
        except Exception as e:
            print(f"[SpreadMonitorTab] dir error: {e}")

    def _load_watchlist(self):
        self._ensure_dir()
        if os.path.exists(self._WATCH_PATH):
            try:
                with open(self._WATCH_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.watchlist = [str(x).upper().strip() for x in data if isinstance(x, str)]
            except Exception as e:
                print(f"[SpreadMonitorTab] load watchlist error: {e}")
        self._render_watchlist_into_ui()

    def _save_watchlist(self):
        self._ensure_dir()
        try:
            with open(self._WATCH_PATH, "w", encoding="utf-8") as f:
                json.dump(self.watchlist, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[SpreadMonitorTab] save watchlist error: {e}")

    # ─────────────── UI ───────────────
    def _build_ui(self):
        root = QVBoxLayout(self)

        # 상단 제어부
        ctrl = QGridLayout()
        r = 0

        ctrl.addWidget(QLabel("데이터 소스"), r, 0)
        self.source_combo = QComboBox()
        self.source_combo.addItems(self.provider_map.keys())
        # 기본값 Polygon.io
        idx = self.source_combo.findText("Polygon.io")
        if idx >= 0:
            self.source_combo.setCurrentIndex(idx)
        ctrl.addWidget(self.source_combo, r, 1)

        ctrl.addWidget(QLabel("가격 소스"), r, 2)
        self.price_source = QComboBox()
        self.price_source.addItems(["Last", "종가", "고가", "저가", "시가"])
        self.price_key_map = {"Last": "last", "종가": "close", "고가": "high", "저가": "low", "시가": "open"}
        ctrl.addWidget(self.price_source, r, 3)

        r += 1
        ctrl.addWidget(QLabel("스프레드 임계값(≤)"), r, 0)
        self.threshold = QDoubleSpinBox()
        self.threshold.setDecimals(2)
        self.threshold.setRange(0.01, 50.0)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(0.05)
        ctrl.addWidget(self.threshold, r, 1)

        ctrl.addWidget(QLabel("WebSocket URL"), r, 2)
        self.ws_url_edit = QLineEdit("wss://socket.polygon.io/options")
        ctrl.addWidget(self.ws_url_edit, r, 3)

        r += 1
        ctrl.addWidget(QLabel("API Key 파일"), r, 0)
        self.api_hint = QLabel("프로젝트 폴더의 cd_key.txt 사용")
        ctrl.addWidget(self.api_hint, r, 1)

        self.btn_start = QPushButton("실시간 시작")
        self.btn_stop  = QPushButton("중지")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        ctrl.addWidget(self.btn_start, r, 2)
        ctrl.addWidget(self.btn_stop,  r, 3)

        root.addLayout(ctrl)

        # 상단: 좌측 관심종목 / 우측 Polygon 기준가 버튼
        top = QHBoxLayout()

        # 관심종목 박스
        left_group = QGroupBox("관심종목")
        left_v = QVBoxLayout(left_group)
        hb = QHBoxLayout()
        self.watch_edit = QLineEdit()
        self.watch_edit.setPlaceholderText("예: SPY, QQQ ...")
        self.watch_edit.setFixedWidth(160)
        add_btn = QPushButton("추가")
        del_btn = QPushButton("삭제")
        hb.addWidget(self.watch_edit)
        hb.addWidget(add_btn)
        hb.addWidget(del_btn)
        left_v.addLayout(hb)

        self.watch_list = QListWidget()
        left_v.addWidget(self.watch_list)

        add_btn.clicked.connect(self._on_watch_add)
        del_btn.clicked.connect(self._on_watch_del)
        self.watch_edit.returnPressed.connect(self._on_watch_add)

        # 우측: Polygon 기준가 조회 버튼
        right_v = QVBoxLayout()
        fetch_btn = QPushButton("기준가격 조회 (Polygon prev close)")
        fetch_btn.clicked.connect(self._on_fetch_polygon_clicked)
        fetch_btn.setToolTip("관심종목/패널 심볼들에 대해 Polygon 'prev close'를 가져와 각 패널의 '중심 행사가'에 채웁니다.")
        right_v.addWidget(fetch_btn)
        right_v.addStretch()

        top.addWidget(left_group, 1)
        top.addLayout(right_v, 0)
        root.addLayout(top)

        # 10개 종목 패널(2 x 5)
        grid_group = QGroupBox("스프레드 감시 (10종목)")
        grid = QGridLayout(grid_group)
        self.panels: List[_SymbolPanel] = []
        for i in range(10):
            title = f"종목 {chr(ord('A') + i)}"
            p = _SymbolPanel(title, self)
            self.panels.append(p)
            r = 0 if i < 5 else 1
            c = i if i < 5 else i - 5
            grid.addWidget(p, r, c)
        root.addWidget(grid_group)

        # 상태 라벨
        self.status_label = QLabel("대기 중")
        root.addWidget(self.status_label)

    # ─────────────── 관심종목 핸들러 ───────────────
    def _render_watchlist_into_ui(self):
        self.watch_list.clear()
        for s in self.watchlist:
            QListWidgetItem(s, self.watch_list)

    def _on_watch_add(self):
        s = self.watch_edit.text().strip().upper()
        if not s:
            return
        if s in self.watchlist:
            QMessageBox.information(self, "안내", f"{s} 는 이미 추가되어 있습니다.")
            return
        self.watchlist.append(s)
        self._save_watchlist()
        self._render_watchlist_into_ui()
        self.watch_edit.clear()

    def _on_watch_del(self):
        row = self.watch_list.currentRow()
        if row < 0:
            return
        s = self.watch_list.item(row).text()
        self.watch_list.takeItem(row)
        if s in self.watchlist:
            self.watchlist.remove(s)
            self._save_watchlist()

    # ─────────────── Polygon 기준가 반영 ───────────────
    def _on_fetch_polygon_clicked(self):
        # 패널에 입력된 심볼 + 관심종목 합치기
        syms: List[str] = []
        for p in self.panels:
            t = p.symbol_edit.text().strip().upper()
            if t:
                syms.append(t)
        syms.extend([s for s in self.watchlist if s])
        uniq = sorted(set(syms))
        if not uniq:
            QMessageBox.information(self, "안내", "조회할 심볼이 없습니다. 패널 또는 관심종목에 심볼을 추가하세요.")
            return

        # 데이터 소스는 Polygon.io 여야 함 (API 키 획득)
        provider = self.provider_map[self.source_combo.currentText()]
        if not isinstance(provider, PolygonDataProvider):
            QMessageBox.warning(self, "경고", "Polygon.io 데이터소스를 선택하세요.")
            return
        api_key = provider.api_key

        closes = fetch_polygon_prev_closes(uniq, api_key=api_key)
        if not any(v is not None for v in closes.values()):
            QMessageBox.warning(self, "경고", "Polygon 가격 조회에 실패했습니다.")
            return

        # 패널 채우기: 패널의 심볼이 closes에 있으면 중심행사가에 세팅
        filled_count = 0
        for p in self.panels:
            sym = p.symbol_edit.text().strip().upper()
            if not sym:
                continue
            px = closes.get(sym)
            if px is None:
                continue
            p.center.setValue(_round_to_option_tick(px))
            filled_count += 1

        # 패널에 심볼이 비어있고, 관심종목이 남아있으면 비어있는 패널부터 채워줌
        if filled_count == 0:
            for i, s in enumerate(self.watchlist[:10]):
                self.panels[i].symbol_edit.setText(s)
                px = closes.get(s)
                if px is not None:
                    self.panels[i].center.setValue(_round_to_option_tick(px))
                    filled_count += 1

        QMessageBox.information(self, "완료", f"기준가격(Polygon prev close) 반영: {filled_count}개 패널 업데이트 완료.")

    # ─────────────── WebSocket 제어 ───────────────
    def _collect_params(self) -> List[Dict]:
        params: List[Dict] = []
        for p in self.panels:
            prm = p.params()
            if prm["symbol"]:
                params.append(prm)
        return params

    def _on_start_clicked(self):
        params = self._collect_params()
        if not params:
            QMessageBox.information(self, "안내", "최소 1개 이상의 종목/심볼을 입력하세요.")
            return

        provider = self.provider_map[self.source_combo.currentText()]
        price_key = self.price_key_map[self.price_source.currentText()]
        thr = self.threshold.value()
        ws_url = self.ws_url_edit.text().strip()

        if not isinstance(provider, PolygonDataProvider):
            QMessageBox.warning(self, "경고", "Polygon.io 데이터소스만 WebSocket을 지원합니다.")
            return

        api_key = provider.api_key

        # 기존 워커 종료
        if self.ws_worker:
            try:
                self.ws_worker.stop()
                self.ws_worker.wait(2000)
            except Exception:
                pass
            self.ws_worker = None

        # 테이블 초기화
        for p in self.panels:
            p.clear_results()

        # 워커 시작
        self.ws_worker = SpreadWSWorker(
            provider=provider,
            panels_params=params,
            price_key=price_key,
            threshold=thr,
            ws_url=ws_url,
            api_key=api_key
        )
        self.ws_worker.rows_ready.connect(self._on_rows_ready)
        self.ws_worker.status.connect(self._on_status)
        self.ws_worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_label.setText("실시간 감시 실행 중...")

    def _on_stop_clicked(self):
        if self.ws_worker:
            try:
                self.ws_worker.stop()
                self.ws_worker.wait(2000)
            except Exception:
                pass
            self.ws_worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("중지됨")

    def _on_rows_ready(self, panel_idx: int, rows: list):
        if 0 <= panel_idx < len(self.panels):
            self.panels[panel_idx].populate(rows)

    def _on_status(self, msg: str):
        self.status_label.setText(msg)
