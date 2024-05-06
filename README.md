# Model that predicts air alarms in Ukraine

<img src="website_models/icon.png" width="200" height="200">

## Table of contents
* [About the project](#about-the-project)
* [General info](#general-info)
* [Technologies](#technologies)
* [Getting started](#getting-started)
* [User Interface](#user-interface)

## About the project
Alert prediction is a platform that will allow users to receive air alert predictions on the territory of Ukraine 12 hours ahead.

## General info
There are files, which we worked on. They divided to groups:
1. "application" - our main app, the code that binds our models and processes the user's request from the site;
2. "website_models" - html & css & js codes for user interface;
3. section "models_training" - files with training of all our models (GradientBoostingClassifier, RandomForestClassifier, DecisionTreeClassifier, Linear Regression, Logistic Regression);
4. "dataset_managing" - merging datasets with weather and alarms by region, data cleaning;
5. "weather", here are files which allow get data from weather api;
6. "isw", here are files which get data from this website;
7. "NLP", there are files with analysis our datasets and vectorization data from isw;

## Technologies
Project is created with:
* Python 3.8
* Matplotlib
* Pandas
* NumPy
* SciPy
* Scikit-learn
* Flask
* Nltk
* Hlxl
* BeautifulSoup4
* Requests 2.31.0
* Cors

## Getting started
Set up the environment:
* Ensure you have Python 3.8 installed on your system.
* Install the required libraries listed in [technologies](#technologies).

Explore the project folders mentioned in [general info](#general-info)
Run the application:
* Start the Flask server by running the main application file.
* Use the website to explore the features the project offers.

## User Interface
* Overview of the Landing page:

<img width="959" alt="image" src="https://github.com/texasiaa/alarm-prediction-model/assets/160049481/317d852f-0470-4b07-9568-d2b47d16eea4">


* Overview of the Main page environment:

<img width="959" alt="image" src="https://github.com/texasiaa/alarm-prediction-model/assets/160049481/0ed97dc4-e86a-47a0-a11b-5782d411ab38">
<img width="959" alt="image" src="https://github.com/texasiaa/alarm-prediction-model/assets/160049481/ac41c103-8e3c-4f8e-af07-6daa8124319f">


