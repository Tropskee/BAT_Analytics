import json
from bs4 import BeautifulSoup
import pandas as pd
from flask import Flask, request
from flask_cors import CORS
import joblib
import xgboost as xgb
from sklearn.pipeline import Pipeline
import requests
import lxml
import cchardet
from scraper_tools import main as scrape

# Instantiate the API
app = Flask(__name__)

# Allow cross-origin-resource-sharing
# ie. allow JSX to access this resource
CORS(app)

VEHICLE_COLUMNS = ['make', 'model', 'model2', 'model_desc', 'year', 'miles', 'color', 'auction_year', 'engine_size', 'cylinders']

def model_pipeline(df):
    '''
    Return prediction from incoming data
    '''
    # Load XGB model
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model('/home/Tropski/mysite/xgb_model.h5')
    preprocessor = joblib.load('/home/Tropski/mysite/preprocessor.joblib')

    model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', xgb_model)
        ])

    return model_pipeline.predict(df)


@app.route('/prediction', methods=['POST'])
def get_prediction():
    values = request.get_json()
    url = values.get("url")

    # get vehicle html
    try:
        html = requests.get(url)
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        print("URL is not correct")

    # Pull html data
    vehicle_data_soup = BeautifulSoup(html.content, 'lxml')

    vehicle_dict = scrape(vehicle_data_soup)

    df = pd.DataFrame()
    for col in VEHICLE_COLUMNS:
        df[col] = pd.Series(vehicle_dict.get(col))

    response = {
        "prediction": str(model_pipeline(df)[0]),
        "image": vehicle_dict.get('image'),
        "model-info": f"{str(vehicle_dict.get('year'))} {vehicle_dict.get('make').upper()} {vehicle_dict.get('model').upper()}",
        "live": vehicle_dict.get('live'),
        "price": vehicle_dict.get('bid_price')
    }

    return response, 201


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port
    app.run(host='0.0.0.0', port=port)