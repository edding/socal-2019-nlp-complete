import json
from typing import Dict

from fuzzywuzzy import process
from yahoo_fin.stock_info import get_live_price

from chatbot.util import output


class CompanyInfo:
    def __init__(self, name: str, ticker: str):
        self.name = name
        self.ticker = ticker


class Stock:
    def __init__(self, company: CompanyInfo, price: float):
        self.company = company
        self.price = price

    def __str__(self) -> str:
        return "company name: {}, ticker: {}, price: ${:.2f}".format(
            self.company.name, self.company.ticker, self.price
        )

    def report(self) -> str:
        return "The current stock price for {} is {} dollars and {} cents.".format(
            self.company.name, int(self.price), int(self.price * 100 % 100)
        )


def _load_company_info() -> Dict[str, str]:
    """Return a dict mapping from company name to symbol."""
    # Raw data is downloaded from: https://api.iextrading.com/1.0/ref-data/symbols
    with open("chatbot/resources/stock_symbols.json") as fp:
        data = json.load(fp)
        return {c["name"]: c["symbol"] for c in data}


_TICKER_FOR_COMPANY_NAME = _load_company_info()


def company_info_for_company_name(company_name: str) -> CompanyInfo:
    """Use fuzzy match to get company info for matching company name."""
    company_name = process.extractOne(company_name, _TICKER_FOR_COMPANY_NAME.keys())[0]
    return CompanyInfo(company_name, _TICKER_FOR_COMPANY_NAME[company_name])


def pull_stock_data(company_info: CompanyInfo) -> Stock:
    """Pull live stock price for given company (ticker) from Yahoo."""
    output("Pulling stock data for {}".format(company_info.name))
    price = get_live_price(company_info.ticker)
    return Stock(company_info, price)
