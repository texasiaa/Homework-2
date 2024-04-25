import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import datetime
from scipy.sparse import csr_matrix, vstack
import scipy.sparse as sp
import json
from flask import Flask, request, jsonify
import datetime as dt
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words = set(stopwords.words('english'))

def get_all_words():
    tfidf_matrix = np.load('words.npy', allow_pickle=True)
    all_words = tfidf_matrix.tolist()
    return all_words

def get_model(name):
    with open(name, 'rb') as file:
        model = pickle.load(file)
    return model

# date
def get_date(soup):
    date_span = soup.find('span', class_='submitted')
    date = date_span.find('span').get('content')
    date_object = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S%z')
    return date_object.date()

# title
def get_title(soup):
    title_h1 = soup.find('h1', class_='title')
    return title_h1.text

# main_html
def get_html(soup):
    html_element = soup.find('div', class_='field field-name-body field-type-text-with-summary field-label-hidden')
    return html_element

def get_html_text(soup):
    html_element = get_html(soup)
    html_main = html_element.decode()
    return html_main

# full_url
def get_url(soup):
    full_url_tag = soup.find("link", rel="canonical")
    if full_url_tag:
        full_url = full_url_tag.get("href")
        return full_url

# text
def get_text(soup):
    html_element = get_html(soup)
    html_div = html_element.find('div', class_='field-item even')
    elements = html_div.find_all(['p', 'ul', 'ol', 'div'])
    text = ''
    for element in elements:
        if element.name == 'div' and element.find('hr'):
            break
        text += element.text
    return text

# data for dataframe
def get_data(soup):
    data = []
    data.append(get_date(soup))
    data.append(get_title(soup))
    data.append(get_url(soup))
    data.append(get_html_text(soup))
    data.append(get_text(soup))
    return data

# url for request
def build_url(days_to_substract):
    today = datetime.datetime.today() - datetime.timedelta(days=days_to_substract)
    month_name = calendar.month_name[today.month].lower()
    url = f'https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-{month_name}-{today.day}-{today.year}'
    return url

# dataframe
def get_dataframe():
    df = pd.DataFrame(columns=['date', 'title', 'full_url', 'main_html', 'main_text'])
    days_to_substract = 0
    while True:
        days_to_substract += 1
        url = build_url(days_to_substract)
        answer = requests.get(url)
        if not answer.status_code == 200:
            continue
        html_text = answer.text
        soup = BeautifulSoup(html_text, 'lxml')
        df.loc[0] = get_data(soup)
        return df

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize(report):
    report = report.lower()
    text = re.sub(r'\d+', '', report)
    text = re.sub(r'[^\w\s]', '', text)

    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = nltk.WordNetLemmatizer()
    pos_tags = nltk.pos_tag(filtered_tokens)
    lemmatized_tokens = []
    for token, tag in pos_tags:
        wordnet_tag = get_wordnet_pos(tag)
        if wordnet_tag is None:
            lemmatized_tokens.append(token)
        else:
            lemmatized_tokens.append(lemmatizer.lemmatize(str(token), pos=wordnet_tag))
    return lemmatized_tokens

def get_df_with_words():
    all_words = get_all_words()
    data_frame = get_dataframe()
    data_frame['lemmatized_text'] = data_frame['main_text'].apply(lemmatize)

    lem_text = data_frame['lemmatized_text']

    lem_text[0] = [word for word in lem_text[0] if word in all_words]
    data_frame['lemmatized_text'] = lem_text
    df_copy = data_frame.copy()
    df_copy.loc[len(data_frame)] = [None, None, None, None, None, all_words]
    return df_copy

def get_words_vect():
    df_isw = get_df_with_words()
    lemmatized_text_str = df_isw['lemmatized_text'].apply(lambda x: ' '.join(x))
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(lemmatized_text_str)
    first_row = tfidf_matrix.getrow(0)
    return first_row

def get_weather_response():
    url_base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    RSA_KEY = "FY62VLWDPNHH845BSLWHAFTVQ"
    API_TOKEN = "prymaty"
    location = "Kyiv"
    date = "next24hours"
    url_api_version = "v1"
    url = f"{url_base_url}/{location}/{date}?unitGroup=metric&include=hours&key={RSA_KEY}&contentType=json"
    headers = {
      'X-Api-Key': API_TOKEN
    }
    response = requests.get(url, headers=headers)

    if response.status_code == requests.codes.ok:
        return response
    else:
        print("Error:", response.status_code, response.text)

def get_weather_pred():
    location = "Kyiv"
    response = get_weather_response()
    response_json = json.loads(response.text)

    curr_hour = dt.datetime.now().hour + 1
    next_hour = 0

    result = {}
    if curr_hour > 11:
        while curr_hour != 24:
            first_res = {
                'day_tempmax': response_json['days'][0]['tempmax'],
                'day_tempmin': response_json['days'][0]['tempmin'],
                'day_temp': response_json['days'][0]['temp'],
                'day_dew': response_json['days'][0]['dew'],
                'day_humidity': response_json['days'][0]['humidity'],
                'day_precip': response_json['days'][0]['precip'],
                'day_precipcover': response_json['days'][0]['precipcover'],
                'day_snow': response_json['days'][0]['snow'],
                'day_windgust': response_json['days'][0]['windgust'],
                'day_windspeed': response_json['days'][0]['windspeed'],
                'day_winddir': response_json['days'][0]['winddir'],
                'day_pressure': response_json['days'][0]['pressure'],
                'day_cloudcover': response_json['days'][0]['cloudcover'],
                'day_visibility': response_json['days'][0]['visibility'],
                'day_solarradiation': response_json['days'][0]['solarradiation'],
                'day_solarenergy': response_json['days'][0]['solarenergy'],
                'day_uvindex': response_json['days'][0]['uvindex'],
                'day_moonphase': response_json['days'][0]['moonphase'],
                'hour_temp': response_json['days'][0]['hours'][curr_hour]['temp'],
                'hour_humidity': response_json['days'][0]['hours'][curr_hour]['humidity'],
                'hour_dew': response_json['days'][0]['hours'][curr_hour]['dew'],
                'hour_precip': response_json['days'][0]['hours'][curr_hour]['precip'],
                'hour_precipprob': response_json['days'][0]['hours'][curr_hour]['precipprob'],
                'hour_snow': response_json['days'][0]['hours'][curr_hour]['snow'],
                'hour_snowdepth': response_json['days'][0]['hours'][curr_hour]['snowdepth'],
                'hour_windgust': response_json['days'][0]['hours'][curr_hour]['windgust'],
                'hour_windspeed': response_json['days'][0]['hours'][curr_hour]['windspeed'],
                'hour_winddir': response_json['days'][0]['hours'][curr_hour]['winddir'],
                'hour_pressure': response_json['days'][0]['hours'][curr_hour]['pressure'],
                'hour_visibility': response_json['days'][0]['hours'][curr_hour]['visibility'],
                'hour_cloudcover': response_json['days'][0]['hours'][curr_hour]['cloudcover'],
                'hour_solarradiation': response_json['days'][0]['hours'][curr_hour]['solarradiation'],
                'hour_uvindex': response_json['days'][0]['hours'][curr_hour]['uvindex'],
                'hour_severerisk': response_json['days'][0]['hours'][curr_hour]['severerisk'],
                'day_of_week': dt.datetime.now().day % 7,
                'date1': str(dt.datetime.today().date())
            }
            date = f"{location}, {response_json['days'][0]['datetime']}, {response_json['days'][0]['hours'][curr_hour]['datetime']}"
            curr_hour += 1
            result[date] = first_res
        while len(result) != 12:
            second_res = {
                'day_tempmax': response_json['days'][1]['tempmax'],
                'day_tempmin': response_json['days'][1]['tempmin'],
                'day_temp': response_json['days'][1]['temp'],
                'day_dew': response_json['days'][1]['dew'],
                'day_humidity': response_json['days'][1]['humidity'],
                'day_precip': response_json['days'][1]['precip'],
                'day_precipcover': response_json['days'][1]['precipcover'],
                'day_snow': response_json['days'][1]['snow'],
                'day_windgust': response_json['days'][1]['windgust'],
                'day_windspeed': response_json['days'][1]['windspeed'],
                'day_winddir': response_json['days'][1]['winddir'],
                'day_pressure': response_json['days'][1]['pressure'],
                'day_cloudcover': response_json['days'][1]['cloudcover'],
                'day_visibility': response_json['days'][1]['visibility'],
                'day_solarradiation': response_json['days'][1]['solarradiation'],
                'day_solarenergy': response_json['days'][1]['solarenergy'],
                'day_uvindex': response_json['days'][1]['uvindex'],
                'day_moonphase': response_json['days'][1]['moonphase'],
                'hour_temp': response_json['days'][1]['hours'][next_hour]['temp'],
                'hour_humidity': response_json['days'][1]['hours'][next_hour]['humidity'],
                'hour_dew': response_json['days'][1]['hours'][next_hour]['dew'],
                'hour_precip': response_json['days'][1]['hours'][next_hour]['precip'],
                'hour_precipprob': response_json['days'][1]['hours'][next_hour]['precipprob'],
                'hour_snow': response_json['days'][1]['hours'][next_hour]['snow'],
                'hour_snowdepth': response_json['days'][1]['hours'][next_hour]['snowdepth'],
                'hour_windgust': response_json['days'][1]['hours'][next_hour]['windgust'],
                'hour_windspeed': response_json['days'][1]['hours'][next_hour]['windspeed'],
                'hour_winddir': response_json['days'][1]['hours'][next_hour]['winddir'],
                'hour_pressure': response_json['days'][1]['hours'][next_hour]['pressure'],
                'hour_visibility': response_json['days'][1]['hours'][next_hour]['visibility'],
                'hour_cloudcover': response_json['days'][1]['hours'][next_hour]['cloudcover'],
                'hour_solarradiation': response_json['days'][1]['hours'][next_hour]['solarradiation'],
                'hour_uvindex': response_json['days'][1]['hours'][next_hour]['uvindex'],
                'hour_severerisk': response_json['days'][1]['hours'][next_hour]['severerisk'],
                'day_of_week': (dt.datetime.today() + dt.timedelta(days=1)).day % 7,
                'date1': str(dt.datetime.today().date())
            }
            date = f"{location}, {response_json['days'][1]['datetime']}, {response_json['days'][1]['hours'][next_hour]['datetime']}"
            next_hour += 1
            result[date] = second_res
    else:
        for i in range(12):
            first_res = {
                'day_tempmax': response_json['days'][0]['tempmax'],
                'day_tempmin': response_json['days'][0]['tempmin'],
                'day_temp': response_json['days'][0]['temp'],
                'day_dew': response_json['days'][0]['dew'],
                'day_humidity': response_json['days'][0]['humidity'],
                'day_precip': response_json['days'][0]['precip'],
                'day_precipcover': response_json['days'][0]['precipcover'],
                'day_snow': response_json['days'][0]['snow'],
                'day_windgust': response_json['days'][0]['windgust'],
                'day_windspeed': response_json['days'][0]['windspeed'],
                'day_winddir': response_json['days'][0]['winddir'],
                'day_pressure': response_json['days'][0]['pressure'],
                'day_cloudcover': response_json['days'][0]['cloudcover'],
                'day_visibility': response_json['days'][0]['visibility'],
                'day_solarradiation': response_json['days'][0]['solarradiation'],
                'day_solarenergy': response_json['days'][0]['solarenergy'],
                'day_uvindex': response_json['days'][0]['uvindex'],
                'day_moonphase': response_json['days'][0]['moonphase'],
                'hour_temp': response_json['days'][0]['hours'][curr_hour]['temp'],
                'hour_humidity': response_json['days'][0]['hours'][curr_hour]['humidity'],
                'hour_dew': response_json['days'][0]['hours'][curr_hour]['dew'],
                'hour_precip': response_json['days'][0]['hours'][curr_hour]['precip'],
                'hour_precipprob': response_json['days'][0]['hours'][curr_hour]['precipprob'],
                'hour_snow': response_json['days'][0]['hours'][curr_hour]['snow'],
                'hour_snowdepth': response_json['days'][0]['hours'][curr_hour]['snowdepth'],
                'hour_windgust': response_json['days'][0]['hours'][curr_hour]['windgust'],
                'hour_windspeed': response_json['days'][0]['hours'][curr_hour]['windspeed'],
                'hour_winddir': response_json['days'][0]['hours'][curr_hour]['winddir'],
                'hour_pressure': response_json['days'][0]['hours'][curr_hour]['pressure'],
                'hour_visibility': response_json['days'][0]['hours'][curr_hour]['visibility'],
                'hour_cloudcover': response_json['days'][0]['hours'][curr_hour]['cloudcover'],
                'hour_solarradiation': response_json['days'][0]['hours'][curr_hour]['solarradiation'],
                'hour_uvindex': response_json['days'][0]['hours'][curr_hour]['uvindex'],
                'hour_severerisk': response_json['days'][0]['hours'][curr_hour]['severerisk'],
                'day_of_week': dt.today().day(),
                'date1': str(dt.datetime.today().date())
            }
            date = f"{location}, {response_json['days'][0]['datetime']}, {response_json['days'][0]['hours'][curr_hour]['datetime']}"
            result[date] = first_res
            curr_hour += 1
    return result
    #print(json.dumps(result, indent = 4))
    # for i in range(len(result)):
    #    print(result[i],'\n')

def get_weather_matrix():
    result = get_weather_pred()
    data = json.dumps(result, indent = 4)
    df = pd.read_json(data)
    weather_df = df.T
    weather_df.drop(columns=['date1'], inplace=True)
    weather_df.fillna(0)
    weather_df = weather_df.reset_index(drop=True)
    # because sparce matrix cannot use just float value
    weather_df = weather_df.astype('float64')

    weather_columns = weather_df.columns.tolist()
    weather_matrix = weather_df.to_numpy()

    sparse_weather = csr_matrix(weather_matrix)
    return sparse_weather

def  get_final_matrix():
    words_one_row = get_words_vect()
    weather = get_weather_matrix()
    words = vstack([words_one_row] * 12)
    # merge weather matrix with words matrix
    combine_matrix = sp.hstack((words, weather))
    return combine_matrix

def get_final_pred(model):
    combine_matrix = get_final_matrix()
    y_pred = model.predict(combine_matrix)
    y_pred_probab = model.predict_proba(combine_matrix)

    df = pd.DataFrame(columns=['binary', 'float'])
    hours = 12
    for i in range(hours):
        #print("Example {}: Prediction: {}, Probability: {}".format(i+1, y_pred_5[i], y_pred_proba[i]))
        df.loc[len(df.index)] = [y_pred[i], y_pred_probab[i]]
    return df

app = Flask(__name__)
CORS(app)
@app.route("/prediction", methods=["POST"])

def get_prediction_data():
    json_data = request.get_json()
    location = json_data.get('region')
    location = location.lower()
    model = get_model(f"{location}.pkl")

    prediction = get_final_pred(model)
    prediction['float'] = prediction['float'].apply(lambda x: x.tolist())
    data_dict = prediction.to_dict(orient='records')
    return jsonify(data_dict)
