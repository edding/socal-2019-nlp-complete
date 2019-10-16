import logging
import json

import requests
from geopy import geocoders

from chatbot.util import CONFIG, output

logging.getLogger().setLevel(logging.INFO)


class GeoInfo:
    def __init__(self, geo_name: str, lat: float, long: float):
        self.geo_name = geo_name
        self.lat = lat
        self.long = long


class Weather:
    def __init__(self, geo_info: GeoInfo, temp: float, summary: str):
        self.geo_info = geo_info
        self.temp = temp
        self.summary = summary

    def __str__(self) -> str:
        return "geo_name: {}, temperature: {}, summary: {}".format(
            self.geo_info.geo_name, self.temp, self.summary
        )

    def report(self) -> str:
        return "The current temperature in {} is {} degrees fahrenheit. {}".format(
            self.geo_info.geo_name, int(self.temp), self.summary
        )


def geo_info_for_geo_name(
    geo_name: str, username: str = CONFIG["geonames_username"]
) -> GeoInfo:
    """Get geo information (latitude and longitude) for given region name."""
    logging.info("Decoding latitude and longitude of '{}'...".format(geo_name))
    gn = geocoders.GeoNames(username=username, timeout=10)
    location = gn.geocode(geo_name)
    return GeoInfo(geo_name=geo_name, lat=location.latitude, long=location.longitude)


def pull_weather_data(geo_info: GeoInfo, api_key: str = CONFIG["api_key"]) -> Weather:
    """Pull weather data using Dark Sky api."""
    url = "https://api.darksky.net/forecast/%s/%s,%s" % (
        api_key,
        geo_info.lat,
        geo_info.long,
    )
    output("Pulling weather data for {}...".format(geo_info.geo_name))
    response = requests.get(url)
    data = json.loads(response.text)

    return Weather(
        geo_info=geo_info,
        temp=data["currently"]["temperature"],
        summary=data["hourly"]["summary"],
    )
