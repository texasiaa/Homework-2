import datetime as dt
import json

import requests
from flask import Flask, jsonify, request

# our API_TOKEN
API_TOKEN = "prymaty"
# RSA_KEY from https://weather.visualcrossing.com
RSA_KEY = "2L2YYNKS3DLPKQ86HCW3VJRZN"

app = Flask(__name__)


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv["message"] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/")
def home_page():
    return "<p><h2>KMA HW2.</h2></p>"


@app.route("/weather_by_hour", methods=["POST"])
def weather_endpoint():
    json_data = request.get_json()

    if json_data.get("token") is None:
        raise InvalidUsage("token is required", status_code=400)

    if json_data.get("name") is None:
        raise InvalidUsage("name is required", status_code=400)

    if json_data.get("location") is None:
        raise InvalidUsage("location is required", status_code=400)

    token = json_data.get("token")
    if token != API_TOKEN:
        raise InvalidUsage("wrong API token", status_code=403)

    location = json_data.get("location")
    url_base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    url = f"{url_base_url}/{location}/next24hours?unitGroup=metric&include=hours&key={RSA_KEY}&contentType=json"
    headers = {
        'X-Api-Key': API_TOKEN
    }

    response = requests.get(url, headers=headers)

    response_json = json.loads(response.text)

    curr_hour = dt.datetime.now().hour + 1
    next_hour = 0

    result = {}
    if curr_hour > 11:
        while curr_hour != 24:
            first_res = {
                'temperature': response_json['days'][0]['hours'][curr_hour]['temp'],
                'humidity': response_json['days'][0]['hours'][curr_hour]['humidity'],
                'conditions': response_json['days'][0]['hours'][curr_hour]['conditions'],
                'visibility': response_json['days'][0]['hours'][curr_hour]['visibility'],
                'snow': response_json['days'][0]['hours'][curr_hour]['snow'],
                'solar energy': response_json['days'][0]['hours'][curr_hour]['solarenergy'],
                'wind speed': response_json['days'][0]['hours'][curr_hour]['windspeed'],
                'cloud cover': response_json['days'][0]['hours'][curr_hour]['cloudcover']
            }
            date = f"{location}, {response_json['days'][0]['datetime']}, {response_json['days'][0]['hours'][curr_hour]['datetime']}"
            curr_hour += 1
            result[date] = first_res
        while len(result) != 12:
            second_res = {
                'temperature': response_json['days'][1]['hours'][next_hour]['temp'],
                'humidity': response_json['days'][1]['hours'][next_hour]['humidity'],
                'conditions': response_json['days'][1]['hours'][next_hour]['conditions'],
                'visibility': response_json['days'][1]['hours'][next_hour]['visibility'],
                'snow': response_json['days'][1]['hours'][next_hour]['snow'],
                'solar energy': response_json['days'][1]['hours'][next_hour]['solarenergy'],
                'wind speed': response_json['days'][1]['hours'][next_hour]['windspeed'],
                'cloud cover': response_json['days'][1]['hours'][next_hour]['cloudcover']
            }
            date = f"{location}, {response_json['days'][1]['datetime']}, {response_json['days'][1]['hours'][next_hour]['datetime']}"
            next_hour += 1
            result[date] = second_res
    else:
        for i in range(12):
            first_res = {
                'temperature': response_json['days'][0]['hours'][curr_hour]['temp'],
                'humidity': response_json['days'][0]['hours'][curr_hour]['humidity'],
                'conditions': response_json['days'][0]['hours'][curr_hour]['conditions'],
                'visibility': response_json['days'][0]['hours'][curr_hour]['visibility'],
                'snow': response_json['days'][0]['hours'][curr_hour]['snow'],
                'solar energy': response_json['days'][0]['hours'][curr_hour]['solarenergy'],
                'wind speed': response_json['days'][0]['hours'][curr_hour]['windspeed'],
                'cloud cover': response_json['days'][0]['hours'][curr_hour]['cloudcover']
            }
            date = f"{location}, {response_json['days'][0]['datetime']}, {response_json['days'][0]['hours'][curr_hour]['datetime']}"
            result[date] = first_res
            curr_hour += 1
    new_result = json.dumps(result, indent=4)
    return new_result




