import ta

import pandas as pd
import tensortrade.env.default as default

from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.instruments import USD, BTC, ETH, LTC
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order



#download data
cdd = CryptoDataDownload()

bitfinex_data = pd.concat([
    cdd.fetch("Bitfinex", "USD", "BTC", "1h").add_prefix("BTC:"),
    cdd.fetch("Bitfinex", "USD", "ETH", "1h").add_prefix("ETH:")
], axis=1)

bitstamp_data = pd.concat([
    cdd.fetch("Bitstamp", "USD", "BTC", "1h").add_prefix("BTC:"),
    cdd.fetch("Bitstamp", "USD", "LTC", "1h").add_prefix("LTC:")
], axis=1)

# define exchange

bitfinex = Exchange("bitfinex", service=execute_order)(
    Stream.source(list(bitfinex_data['BTC:close']), dtype="float").rename("USD-BTC"),
    Stream.source(list(bitfinex_data['ETH:close']), dtype="float").rename("USD-ETH")
)

bitstamp = Exchange("bitstamp", service=execute_order)(
    Stream.source(list(bitstamp_data['BTC:close']), dtype="float").rename("USD-BTC"),
    Stream.source(list(bitstamp_data['LTC:close']), dtype="float").rename("USD-LTC")
)