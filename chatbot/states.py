from __future__ import annotations

from typing import Optional
from abc import abstractmethod

from chatbot.util import CONFIG, output, get_user_input, extract_entity_from_text
from chatbot.weather import geo_info_for_geo_name, pull_weather_data, GeoInfo
from chatbot.stock import company_info_for_company_name, pull_stock_data, CompanyInfo
from tf_models.models import Intents, intent_model, FlowControls, flow_control_model


class State:
    def __init__(self, text: str = ""):
        self.text = text

    @abstractmethod
    def run(self) -> Optional[State]:
        """A state is expected to handle something and return a new state."""
        raise NotImplementedError


class Initial(State):
    def __init__(self, text: str = "", greeting: bool = True):
        super().__init__(text)
        self.greeting = greeting

    def run(self) -> Optional[State]:
        if self.greeting:
            output("Hi {}, how can I help".format(CONFIG["greeting_name"]))
        else:
            output("Ok what else can I do for you")

        user_input = get_user_input()
        intent = intent_model.predict(
            user_input
        )  # predict which intent (weather / stock)
        if intent == Intents.Weather:
            return WeatherIntent(user_input)
        else:
            return StockIntent(user_input)


class WeatherIntent(State):
    def run(self) -> Optional[State]:
        geo_name = extract_entity_from_text(self.text)
        if geo_name:
            geo_info = geo_info_for_geo_name(geo_name)
            return WeatherPresent(self.text, geo_info)
        else:
            output("Which city or region are you interested in?")
            user_input = get_user_input()
            return WeatherIntent(user_input)


class WeatherPresent(State):
    def __init__(self, text: str, geo_info: GeoInfo):
        super().__init__(text)
        self.geo_info = geo_info

    def run(self) -> Optional[State]:
        weather = pull_weather_data(self.geo_info)
        output(weather.report())
        return FlowControl()


class StockIntent(State):
    def run(self) -> Optional[State]:
        company_name = extract_entity_from_text(self.text, type_="ORG")
        if company_name:
            company_info = company_info_for_company_name(company_name)
            return StockPresent(self.text, company_info)
        else:
            output("Which company are you interested in?")
            user_input = get_user_input()
            # Hack, make it easier to recognize the entity
            return StockIntent("What is the stock price for {}?".format(user_input))


class StockPresent(State):
    def __init__(self, text: str, company_info: CompanyInfo):
        super().__init__(text)
        self.company_info = company_info

    def run(self) -> Optional[State]:
        stock = pull_stock_data(self.company_info)
        output(stock.report())
        return FlowControl()


class FlowControl(State):
    def run(self) -> Optional[State]:
        output("Anything else?")
        user_input = get_user_input()
        flow_control = flow_control_model.predict(
            user_input
        )  # predict whether to continue
        if flow_control == FlowControls.Continue:
            return Initial(greeting=False)
        else:
            return Stop()


class Stop(State):
    def run(self) -> Optional[State]:
        output("Have a great day, {}!".format(CONFIG["greeting_name"]))
        return None
