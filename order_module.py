import pandas as pd
import numpy as np
import math
import MetaTrader5 as mt5


def Buy(symbol=None,position_size=None):

    price = (mt5.symbol_info_tick(symbol).ask)
    requestBuy = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(position_size),
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "magic": 10,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_TYPE_BUY_LIMIT,
    }

    return (requestBuy)



def Sell(symbol=None,position_size=None):

    price = (mt5.symbol_info_tick(symbol).bid)
    requestBuy = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(position_size),
        "type": mt5.ORDER_TYPE_SELL,
        "price": price,
        "magic": 10,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_TYPE_BUY_LIMIT,
    }

    return (requestBuy)

def close(symbol=None):

    positions = mt5.positions_get(symbol=symbol)

    if positions[0].type == 0:
        price = mt5.symbol_info_tick(symbol).bid
        type = mt5.ORDER_TYPE_SELL
        volume = positions[0].volume

    elif positions[0].type == 1:
        price = mt5.symbol_info_tick(symbol).ask
        type = mt5.ORDER_TYPE_BUY
        volume = (positions[0].volume)

    requestClose = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": type,
        "position": positions[0].ticket,
        "price": price,
        "magic": 10,
        "comment": "Close trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    return (requestClose)
