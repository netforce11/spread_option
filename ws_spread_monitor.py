# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import time
import math
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

from PyQt5.QtCore import QThread, pyqtSignal

# pip install websocket-client
from websocket import WebSocketApp

from data_provider import PolygonDataProvider, DataProviderBase
from structures import OptionChain, OptionQuote


def _occ(symbol: str, expiry: str, side: str, strike: float) -> str:
    """OCC 티커 문자열 (Polygon 형식: 'O:UNDERLYINGyymmddSxxxxxxxx')"""
    return PolygonDataProvider._to_occ_option_ticker(symbol, expiry, side, strike)


def _safe_price(q: OptionQuote, key: str) -> float:
    v = getattr(q, key, None)
    return float(v) if isinstance(v, (int, float)) else 0.0


class SpreadWSWorker(QThread):
    """
    Polygon 옵션 틱 WebSocket을 구독해서
    각 패널(심볼)의 콜 수직 스프레드( K vs K+width ) 순코스트를 실시간 계산.
    - rows_ready(panel_idx, rows): 테이블에 채울 행들 방출
      rows = List[ (K_long, K_short, P_long, P_short, net, date, expiry) ]
    - status(msg): 상태 표시용
    """
    rows_ready = pyqtSignal(int, list)
    status = pyqtSignal(str)

    def __init__(
        self,
        provider: DataProviderBase,
        panels_params: List[Dict],
        price_key: str,
        threshold: float,
        ws_url: str,
        api_key: str,
        parent=None
    ):
        super().__init__(parent)
        self.provider = provider
        self.panels_params = panels_params
        self.price_key = price_key      # 'last' | 'close' | 'high' | 'low' | 'open'
        self.threshold = float(threshold)
        self.ws_url = ws_url
        self.api_key = api_key

        self._ws: Optional[WebSocketApp] = None
        self._running = True

        # 구독용 심볼→(OCC 리스트), 계산용 매핑
        self._subs: Set[str] = set()  # e.g. 'T.O:SPY241101C00450000'
        # 각 패널에서 모니터링할 (K, K+width) 페어 목록
        self._pairs_by_panel: Dict[int, List[Tuple[float, float]]] = {}
        # 각 패널: (K -> OCC), (K+width -> OCC)
        self._call_occ_by_panel: Dict[int, Dict[float, str]] = {}
        # 최신 가격 캐시
        self._last_px: Dict[str, float] = {}  # {OCC: last_trade_price}

        # 페어-역참조: 페어 계산 시 빠르게 찾기 위해
        # panel_idx -> { long_occ: [short_occ_candidates], short_occ: [long_occ_candidates] }
        self._pair_links: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    def stop(self):
        self._running = False
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass

    # ---------- 초기 페어 구성 (한 번 REST로 체인 받아 스트라이크/틱커 캐시) ----------
    def _bootstrap_pairs(self):
        self.status.emit("초기 체인 로드 중...")
        for idx, prm in enumerate(self.panels_params):
            sym = prm.get("symbol", "").upper().strip()
            if not sym:
                continue
            date = prm["buy_date"]
            exp  = prm["expiry"]
            step = float(prm["step"])
            center = float(prm["center"])
            rng = int(prm["range"])
            width = float(prm["width"])

            # 비어있으면 skip
            try:
                chain: OptionChain = self.provider.fetch_chain(sym, date, exp, step, center if center > 0 else None, rng)
            except Exception as e:
                self.status.emit(f"[{sym}] 체인 조회 실패: {e}")
                continue

            # 콜만 추림
            calls_by_K: Dict[float, OptionQuote] = {}
            for q in getattr(chain, "quotes", []) or []:
                if getattr(q, "side", "") == "C":
                    calls_by_K[float(q.strike)] = q

            Ks = sorted(calls_by_K.keys())
            if not Ks:
                self.status.emit(f"[{sym}] 콜 체인 없음")
                continue

            # width로 짝을 만들고, 존재하는 쪽만 수집
            tol = 1e-6
            pairs: List[Tuple[float, float]] = []
            occ_map: Dict[float, str] = {}
            for K in Ks:
                target = K + width
                K2 = None
                for s in Ks:
                    if abs(s - target) <= tol or math.isclose(s, target, rel_tol=0, abs_tol=1e-6):
                        K2 = s
                        break
                if K2 is None:
                    continue

                # OCC 미리 생성(구독 리스트 확보)
                occ_long = _occ(sym, exp, 'C', K)
                occ_short = _occ(sym, exp, 'C', K2)
                occ_map[K] = occ_long
                occ_map[K2] = occ_short
                pairs.append((K, K2))

                # 구독 채널 키 조합
                self._subs.add(f"T.{occ_long}")
                self._subs.add(f"T.{occ_short}")

                # 역방향 링크
                self._pair_links[idx][occ_long].append(occ_short)
                self._pair_links[idx][occ_short].append(occ_long)

            self._pairs_by_panel[idx] = pairs
            self._call_occ_by_panel[idx] = occ_map

        total = len(self._subs)
        self.status.emit(f"초기화 완료. 구독 티커 수: {total}")

    # ---------- WebSocket 콜백 ----------
    def _on_open(self, ws: WebSocketApp):
        self.status.emit("WebSocket 연결됨. 인증 중...")
        auth = {"action": "auth", "params": self.api_key}
        ws.send(json.dumps(auth))

        if self._subs:
            params = ",".join(sorted(self._subs))
            sub = {"action": "subscribe", "params": params}
            ws.send(json.dumps(sub))
            self.status.emit(f"구독 완료: {len(self._subs)} 채널")

    def _on_message(self, ws: WebSocketApp, raw: str):
        try:
            data = json.loads(raw)
        except Exception:
            return

        # Polygon은 배열 형태로 여러 이벤트를 보낼 수 있음
        events = data if isinstance(data, list) else [data]

        # 들어온 틱 반영
        updated_occs: Set[str] = set()
        for ev in events:
            # 예) {'ev':'T','sym':'O:SPY241101C00450000','p':0.34, ...}
            sym = ev.get("sym")
            px = ev.get("p")
            if sym and px is not None:
                self._last_px[sym] = float(px)
                updated_occs.add(sym)

        if not updated_occs:
            return

        # 스프레드 계산: 업데이트된 OCC들에 연결된 페어만 재계산
        for panel_idx in self._pair_links:
            impacted_pairs: List[Tuple[float, float, float, float, float, str, str]] = []
            # 어떤 occ가 갱신되었나?
            for changed_occ in updated_occs:
                # 이 changed_occ와 연결된 상대쪽 목록
                others = self._pair_links[panel_idx].get(changed_occ)
                if not others:
                    continue
                # 각 상대와 짝지어서 '둘 다 가격이 있으면' 순코스트 계산
                for other_occ in others:
                    p1 = self._last_px.get(changed_occ)
                    p2 = self._last_px.get(other_occ)
                    if p1 is None or p2 is None:
                        continue

                    # changed_occ가 Long인지 Short인지 구분할 필요 없이
                    # 우리는 (LongK, ShortK2) 형태로 K를 찾아 테이블에 넣을 것.
                    # 패널에 저장된 occ_map을 통해 K를 역조회
                    occ_map = self._call_occ_by_panel.get(panel_idx, {})
                    # 역방향 K검색
                    K_long = None; K_short = None
                    for K, occ in occ_map.items():
                        if occ == changed_occ:
                            K_long = K
                        if occ == other_occ:
                            K_short = K
                    # 둘 다 찾았는지 확인
                    if K_long is None or K_short is None:
                        # 순서 반대일 수도 있으니 한 번 뒤집어서도 검사
                        for K, occ in occ_map.items():
                            if occ == changed_occ and K_short is None:
                                K_short = K
                            if occ == other_occ and K_long is None:
                                K_long = K
                    if K_long is None or K_short is None:
                        continue
                    # width 관계가 올바른지 점검( K_short ~= K_long + width )
                    # (엄격할 필요 없으면 생략 가능)
                    # 순코스트: Long - Short
                    net = p1 - p2 if K_short > K_long else p2 - p1
                    if net <= self.threshold:
                        # 날짜/만기는 패널 파라미터에서 가져오기
                        buy_date = self.panels_params[panel_idx]["buy_date"]
                        expiry = self.panels_params[panel_idx]["expiry"]
                        # 정규화된 Long/Short 순서로 정리
                        if K_short > K_long:
                            long_px, short_px = (p1, p2) if K_short == K_short else (p2, p1)
                            impacted_pairs.append((K_long, K_short, float(long_px), float(short_px), float(net), buy_date, expiry))
                        else:
                            # 만약 순서가 뒤바뀌면 swap
                            impacted_pairs.append((K_short, K_long, float(p2), float(p1), float(net), buy_date, expiry))

            if impacted_pairs:
                # 비용 오름차순
                impacted_pairs.sort(key=lambda t: t[4])
                # 테이블 갱신 신호
                self.rows_ready.emit(panel_idx, impacted_pairs)

    def _on_error(self, ws: WebSocketApp, err):
        self.status.emit(f"WS error: {err}")

    def _on_close(self, ws: WebSocketApp, *a):
        self.status.emit("WebSocket closed")

    def run(self):
        # 초기 페어 구성
        try:
            self._bootstrap_pairs()
        except Exception as e:
            self.status.emit(f"초기화 실패: {e}")
            return

        # WebSocket 루프
        while self._running:
            try:
                self._ws = WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                # 차단 실행
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                self.status.emit(f"WS 예외: {e}")
            if self._running:
                time.sleep(3)  # 재연결 백오프
