# -*- coding: utf-8 -*-
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple

# 손익 계산 함수 타입 정의
PnlFunction = Callable[[float, List[float]], float]

@dataclass
class Strategy:
    """옵션 전략의 속성을 정의하는 데이터 클래스"""
    name: str
    legs: int  # 필요한 옵션 레그 수
    description: str
    purpose: str
    pnl_func: PnlFunction
    selection_prompt: str

def pnl_long_call(S: float, params: List[float]) -> float:
    K, premium = params
    return max(0, S - K) - premium

def pnl_long_put(S: float, params: List[float]) -> float:
    K, premium = params
    return max(0, K - S) - premium

def pnl_bull_call_spread(S: float, params: List[float]) -> float:
    K_low, K_high, net_debit = params
    return max(0, S - K_low) - max(0, S - K_high) - net_debit

def pnl_bear_put_spread(S: float, params: List[float]) -> float:
    K_low, K_high, net_debit = params
    return max(0, K_high - S) - max(0, K_low - S) - net_debit

def pnl_bear_call_spread(S: float, params: List[float]) -> float:
    K_low, K_high, net_credit = params
    return net_credit - (max(0, S - K_low) - max(0, S - K_high))

def pnl_bull_put_spread(S: float, params: List[float]) -> float:
    K_low, K_high, net_credit = params
    return net_credit - (max(0, K_high - S) - max(0, K_low - S))

def pnl_long_straddle(S: float, params: List[float]) -> float:
    K, call_premium, put_premium = params
    return (max(0, S - K) - call_premium) + (max(0, K - S) - put_premium)

def pnl_long_strangle(S: float, params: List[float]) -> float:
    K_put, K_call, put_premium, call_premium = params
    return (max(0, S - K_call) - call_premium) + (max(0, K_put - S) - put_premium)

def pnl_iron_condor(S: float, params: List[float]) -> float:
    K_p_low, K_p_high, K_c_low, K_c_high, net_credit = params
    bull_put = net_credit - (max(0, K_p_high - S) - max(0, K_p_low - S))
    bear_call = net_credit - (max(0, S - K_c_low) - max(0, S - K_c_high))
    # Note: A proper Iron Condor combines a bull put spread and a bear call spread.
    # The P&L is net_credit - loss_from_put_spread - loss_from_call_spread
    pnl = net_credit
    pnl -= max(0, K_p_high - S) - max(0, K_p_low - S) # Bull Put Spread
    pnl -= max(0, S - K_c_low) - max(0, S - K_c_high) # Bear Call Spread
    return pnl

def pnl_short_iron_butterfly(S: float, params: List[float]) -> float:
    K_low, K_mid, K_high, net_credit = params
    # Short call spread + short put spread
    pnl = net_credit
    pnl -= max(0, S - K_mid) - max(0, S - K_high)  # Bear Call (sell mid, buy high)
    pnl -= max(0, K_mid - S) - max(0, K_low - S)   # Bull Put (sell mid, buy low)
    return pnl


# 사용 가능한 모든 전략 정의
STRATEGIES: Dict[str, Strategy] = {
    "Long Call": Strategy(
        name="콜 매수 (Long Call)", legs=1,
        description="특정 행사가의 콜옵션을 매수합니다.",
        purpose="기초자산 가격이 만기일 전에 크게 상승할 것으로 예상될 때 사용합니다. 잠재적 이익은 무한대, 손실은 지불한 프리미엄으로 제한됩니다.",
        pnl_func=pnl_long_call,
        selection_prompt="매수할 콜옵션의 'Last' 가격 1개를 선택하세요."
    ),
    "Long Put": Strategy(
        name="풋 매수 (Long Put)", legs=1,
        description="특정 행사가의 풋옵션을 매수합니다.",
        purpose="기초자산 가격이 만기일 전에 크게 하락할 것으로 예상될 때 사용합니다. 잠재적 이익은 (행사가-프리미엄)까지, 손실은 지불한 프리미엄으로 제한됩니다.",
        pnl_func=pnl_long_put,
        selection_prompt="매수할 풋옵션의 'Last' 가격 1개를 선택하세요."
    ),
    "Bull Call Spread": Strategy(
        name="강세 콜 스프레드 (Bull Call Spread)", legs=2,
        description="낮은 행사가의 콜옵션을 매수하고, 동시에 높은 행사가의 콜옵션을 매도합니다 (Debit Spread).",
        purpose="기초자산 가격이 완만하게 상승할 것으로 예상될 때 사용합니다. 순수비용(Net Debit)이 발생하며, 최대 이익과 손실이 모두 제한됩니다.",
        pnl_func=pnl_bull_call_spread,
        selection_prompt="매수할 낮은 행사가 콜 1개, 매도할 높은 행사가 콜 1개를 순서대로 선택하세요."
    ),
    "Bear Put Spread": Strategy(
        name="약세 풋 스프레드 (Bear Put Spread)", legs=2,
        description="높은 행사가의 풋옵션을 매수하고, 동시에 낮은 행사가의 풋옵션을 매도합니다 (Debit Spread).",
        purpose="기초자산 가격이 완만하게 하락할 것으로 예상될 때 사용합니다. 순수비용(Net Debit)이 발생하며, 최대 이익과 손실이 모두 제한됩니다.",
        pnl_func=pnl_bear_put_spread,
        selection_prompt="매수할 높은 행사가 풋 1개, 매도할 낮은 행사가 풋 1개를 순서대로 선택하세요."
    ),
     "Bear Call Spread": Strategy(
        name="약세 콜 스프레드 (Bear Call Spread)", legs=2,
        description="낮은 행사가의 콜옵션을 매도하고, 동시에 높은 행사가의 콜옵션을 매수합니다 (Credit Spread).",
        purpose="기초자산 가격이 하락하거나 특정 가격 이상으로 오르지 않을 것으로 예상될 때 사용합니다. 순수수익(Net Credit)으로 시작하며, 최대 이익과 손실이 제한됩니다.",
        pnl_func=pnl_bear_call_spread,
        selection_prompt="매도할 낮은 행사가 콜 1개, 매수할 높은 행사가 콜 1개를 순서대로 선택하세요."
    ),
    "Bull Put Spread": Strategy(
        name="강세 풋 스프레드 (Bull Put Spread)", legs=2,
        description="높은 행사가의 풋옵션을 매도하고, 동시에 낮은 행사가의 풋옵션을 매수합니다 (Credit Spread).",
        purpose="기초자산 가격이 상승하거나 특정 가격 이하로 내리지 않을 것으로 예상될 때 사용합니다. 순수수익(Net Credit)으로 시작하며, 최대 이익과 손실이 제한됩니다.",
        pnl_func=pnl_bull_put_spread,
        selection_prompt="매도할 높은 행사가 풋 1개, 매수할 낮은 행사가 풋 1개를 순서대로 선택하세요."
    ),
    "Long Straddle": Strategy(
        name="롱 스트래들 (Long Straddle)", legs=2,
        description="동일한 행사가와 만기일을 가진 콜옵션과 풋옵션을 동시에 매수합니다.",
        purpose="기초자산 가격이 어느 방향이든 큰 변동성을 보일 것으로 예상하지만 방향을 예측하기 어려울 때 사용합니다. 주가가 행사가에서 멀어질수록 이익이 발생합니다.",
        pnl_func=pnl_long_straddle,
        selection_prompt="동일한 행사가의 콜과 풋 'Last' 가격을 각각 1개씩 선택하세요."
    ),
    "Long Strangle": Strategy(
        name="롱 스트랭글 (Long Strangle)", legs=2,
        description="외가격(OTM) 상태인 콜옵션과 풋옵션을 동시에 매수합니다. (콜 행사가 > 풋 행사가)",
        purpose="스트래들과 유사하지만, 더 적은 비용으로 구성할 수 있습니다. 대신 주가가 더 크게 움직여야 수익이 발생합니다. 극심한 변동성을 예상할 때 사용합니다.",
        pnl_func=pnl_long_strangle,
        selection_prompt="매수할 외가격 풋 1개와 외가격 콜 1개를 선택하세요."
    ),
    "Iron Condor": Strategy(
        name="아이언 콘도르 (Iron Condor)", legs=4,
        description="강세 풋 스프레드(Bull Put Spread)와 약세 콜 스프레드(Bear Call Spread)를 결합한 전략입니다.",
        purpose="기초자산 가격이 특정 범위 내에서 안정적으로 움직일 것으로 예상될 때 사용합니다 (낮은 변동성). 순수수익(Net Credit)으로 시작하며, 주가가 일정 범위 안에 머물면 이익을 얻습니다.",
        pnl_func=pnl_iron_condor,
        selection_prompt="풋(매수-매도), 콜(매도-매수) 순으로 총 4개의 옵션을 선택하세요."
    ),
    "Short Iron Butterfly": Strategy(
        name="숏 아이언 버터플라이 (Short Iron Butterfly)", legs=4,
        description="중심 행사가에서 풋과 콜을 매도하고, 양쪽으로 동일한 간격의 외가격 옵션을 매수하여 위험을 제한합니다.",
        purpose="아이언 콘도르와 유사하지만, 이익 구간이 더 좁고 최대 이익이 더 큽니다. 주가가 중심 행사가에 정확히 머물 때 최대 이익이 발생합니다.",
        pnl_func=pnl_short_iron_butterfly,
        selection_prompt="낮은 풋(매수), 중심 풋(매도), 중심 콜(매도), 높은 콜(매수) 4개를 선택하세요."
    ),
}