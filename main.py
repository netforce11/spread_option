# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from spread_monitor_tab import SpreadMonitorTab
# 기존 탭들
from current_tab import CurrentTab
from history_tab import HistoryTab

# 새로 추가한 스프레드 감시 탭
from spread_monitor_tab import SpreadMonitorTab


class MainWindow(QMainWindow):
    """메인 윈도우"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("옵션 스프레드 시뮬레이터 (Polygon.io - 확장판 v2.2)")
        self.resize(1720, 980)

        tabs = QTabWidget()

        # 탭 인스턴스 생성
        self.current_tab = CurrentTab()
        self.history_tab = HistoryTab()
        self.spread_tab  = SpreadMonitorTab()


        # 탭 연결
        tabs.addTab(self.current_tab, "현재 (Real-time Simulation)")
        tabs.addTab(self.history_tab, "과거 (Backtesting)")

        tabs.addTab(self.spread_tab, "스프레드 감시")
        self.setCentralWidget(tabs)


def main():
    """애플리케이션 실행 함수"""
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
