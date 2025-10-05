# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List

@dataclass
class Greek:
    """옵션 그릭스 데이터 클래스"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

@dataclass
class OptionQuote:
    """개별 옵션 호가 데이터 클래스"""
    symbol: str
    side: str
    strike: float
    last: float
    prev_close: float
    iv: float
    greeks: Greek

@dataclass
class OptionChain:
    """옵션 체인 데이터 클래스"""
    underlying: str
    as_of: str
    expiration: str
    quotes: List[OptionQuote]

@dataclass
class MinuteData:
    """분 단위 시계열 데이터 클래스"""
    timestamp: str  # HH:MM
    open: float
    high: float
    low: float
    close: float
    volume: int