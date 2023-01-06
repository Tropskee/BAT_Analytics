import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import datetime

def get_auction_year_and_price(vehicle_data_soup):
    '''
    Get auction year and price from vehicle html data
    
    :input vehicle_data_soup str: Vehicle html data loaded by BeautifulSoup

    :return auction_year datetime.year: Auction year ie. 2022
    :return sale_price int: Sale price in $
    :return live bool: Whether the auction is currently live or finished
    '''
    try:
        auction_result_string = vehicle_data_soup.find("span", class_ = 'listing-available-countdown')

        # Get auction year & sale price for live auction
        if auction_result_string:
            # Get vehicle sale date
            auction_year = datetime.datetime.now().year
            # Get current vehicle price
            auction_current_price = vehicle_data_soup.find("strong", class_ = 'info-value').text
            sale_price = int(re.findall('[0-9]+,[0-9]+', auction_current_price)[0].replace(",",""))
            live = True

        else: # Auction is over, get historic year and price
            auction_result_string = vehicle_data_soup.find("span", class_ = 'info-value').text
            sale_date = auction_result_string.split()[-1]
            auction_year = (datetime.datetime.strptime(sale_date, '%m/%d/%y').year)
            # Get vehicle sale price
            sale_price = int(re.findall('[0-9]+,[0-9]+', auction_result_string)[0].replace(",",""))
            live = False
    except:
        auction_year = datetime.datetime.now().year
        sale_price = 1
        live="False"

    return auction_year, sale_price, live


def get_make_and_model(vehicle_data_soup):
    '''
    Get make and model from vehicle html data
    
    :input vehicle_data_soup str: Vehicle html data loaded by BeautifulSoup

    :return vehicle_make str: Vehicle make ie. Honda
    :return vehicle_model str: Vehicle model ie. Accord
    '''
    try:
        make_model_lst = [s.text for s in vehicle_data_soup.find_all("button", class_ = 'group-title')][0:3]
        vehicle_make = make_model_lst[0].lower().replace('make', '').strip(' ')
        vehicle_model = make_model_lst[1].lower().replace('model', '').replace(vehicle_make, '').strip(' ')
        vehicle_model2 = vehicle_model
        if "model" in make_model_lst[2].lower():
            vehicle_model2 = make_model_lst[2].lower().replace('model', '').replace(vehicle_make, '').strip(' ')
    except:
        vehicle_make = "BAD"
        vehicle_model = "URL"
    
    return vehicle_make, vehicle_model, vehicle_model2

def get_listing_post_title(vehicle_data_soup, make, model):
    '''
    Get the title of the listing, to extract extra model details
    
    :input vehicle_data_soup str: Vehicle html data loaded by BeautifulSoup

    :return post_title str: Vehicle make ie. "Porsche 911 Carrera 4S Coupe 6-Speed"
    '''
    try:
        post_title_soup = vehicle_data_soup.find("h1", class_ = 'post-title').text
        post_title = ""

        for idx, word in enumerate(post_title_soup.split()):
            if word.isdigit():
                post_title = " ".join(post_title_soup.split()[idx+1:]).lower()
                break
                
        post_title = [post_title.replace(s, '') for s in make.split()][0]
        post_title = [post_title.replace(s, '') for s in model.split()][0]
        
    except:
        post_title = ""

    return post_title.strip(' ')


def get_listing_details(vehicle_data_soup):
    '''
    Get list of auction details from html data
    
    :input vehicle_data_soup str: Vehicle html data loaded by BeautifulSoup

    :return results list: List of vehicle details ie, paint color, miles, etc.
    '''
    try:
        # Get listing details - vin, miles, etc. - second entry
        listing_details_html = vehicle_data_soup.find_all("div", class_="item")

        # Find the listing_details using keyword "Listing Details"
        for detail in listing_details_html:
            if detail.find("strong") and "Listing Details" in detail.find("strong"):
                listing_details = detail
                break

        # Extract details from html "li"
        results = None
        results = [detail.text for detail in listing_details.find_all('li')]
    except:
        results = []

    return results


def get_mileage(listing_details):
    '''
    Extract vehicle mileage from listing details
    
    :input listing_details list: List of auction details from html data

    :return vehicle_mileage str: Mileage of vehicle
    '''
    try:
        # Get vehicle mileage, normally second entry in "results"
        is_mileage_units = True
        mileage_words = ['Miles', 'miles', 'Mile', 'mile'] 
        kilometer_words = ['Kilometers', 'kilometers', 'Kilometer', 'kilometer', 'KM', 'km']
        for result in listing_details:
            result = result.replace("-"," ")
            if any(word in result for word in kilometer_words):
                vehicle_mileage_notes = result
                is_mileage_units = False
                break
            elif any(word in result for word in mileage_words):
                vehicle_mileage_notes = result
                break

        # Extract Mileage figure from vehicle_mileage_notes
        if vehicle_mileage_notes is None:
            vehicle_mileage = None
        else:
            vehicle_mileage = re.findall('[0-9]+,[0-9]+', vehicle_mileage_notes)
            if len(vehicle_mileage) < 1: # If no match is found, mileage must contain 'k' at end i.e., 47k miles
                vehicle_mileage = re.findall('[0-9]+[kK]', vehicle_mileage_notes)
                if len(vehicle_mileage) >= 1: # If match is found, strip k from end
                    vehicle_mileage = [vehicle_mileage[0].strip('k').strip('K') + ',000']
            if len(vehicle_mileage) < 1: # Still no match found, try mileage < 1,000
                vehicle_mileage = re.findall('[0-9]+', vehicle_mileage_notes)
            if vehicle_mileage == []:
                vehicle_mileage = None
        vehicle_mileage = vehicle_mileage if type(vehicle_mileage) is not list else vehicle_mileage[0]

        # Check if units are in miles or km and make adjustments if needed
        if vehicle_mileage is not None:
            vehicle_mileage = int(vehicle_mileage.replace(',',''))
            if is_mileage_units: # units are mileage
                vehicle_kilometers = int(1.60934 * vehicle_mileage)
            else: # units are km
                vehicle_kilometers = vehicle_mileage
                vehicle_mileage = int(0.621371 * vehicle_mileage)
    except:
        vehicle_mileage = 75000
    
    return vehicle_mileage


def get_paint_color(listing_details):
    '''
    Extract paint color from listing details
    
    :input listing_details list: List of auction details from html data

    :return paint_color str: Color of vehicle
    '''
    try:
        colors = ["white", "black", "gray", "silver", "blue", "red", "brown", "green", "orange", "beige", "purple", "gold", "yellow"]
        paint_string = None
        paint_color = None

        for result in listing_details:
            if paint_string:
                break
            results_separated = result.split(",")
            results_separated = [result.strip().lower() for detail in results_separated]
            
            for detail in results_separated:
                if "paint" in detail or any(color in detail for color in colors):
                    paint_string = detail
                    break
            
        for word in paint_string.split():
            if any(color in word for color in colors):
                paint_color = word
    except:
        paint_color = "silver"

    return paint_color


def get_model_year(vehicle_data_soup):
    '''
    Extract model year from vehicle html
    
    :input vehicle_data_soup str: Vehicle html data loaded by BeautifulSoup

    :return model_year str: Model year of vehicle
    '''
    try:
        model_year_text = vehicle_data_soup.find("h1", class_ = 'post-title').text
        model_year = re.findall("(\d{4})", model_year_text)


    except:
        model_year = ["2022"]
    
    # print(model_year)
    return "2022" if not model_year else model_year[0]


def get_engine_size(listing_details):
    '''
    Extract engine size from listing details
    
    :input listing_details list: List of auction details from html data

    :return engine_size float: Engine size of vehicle
    '''
    try:
        ########### Extract engine size string from details ###########
        eng_keywords = ["liter", "v6", "v8", "engine", "inline", "three", "four", "five", "six", "eight", "ci", "cc", "flathead", "cylinder", "dohc", "sohc", "ohc", "turbocharged"]
        eng_size_keywords = ["liter", "ci", "cc"]
        eng_size_re = ["[0-9]+.[0-9]+", "[0-9]+.[0-9]+l"]
        match = False
        engine_size_string = None

        for details in listing_details:
            if match:
                break
            for detail in details.split():
                detail=detail.lower()
                if any(word in detail for word in eng_keywords):
                    match = True
                    if len(re.findall(eng_size_re[0], detail)) > 0 or len(re.findall(eng_size_re[1], detail)) > 0:
                        engine_size_string = detail
                    elif any(w in detail for w in eng_size_keywords):
                        engine_size_string = detail
                        break

        ########### Extract actual engine size from engine size string ###########
        # Best number finding regex ever!
        numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
        rx = re.compile(numeric_const_pattern, re.VERBOSE)
        engine_size = None

        match = rx.findall(engine_size_string)

        # Convert cc or ci to liters
        if "cc" in engine_size_string or "cubic centimeters" in engine_size_string:
            size = engine_size_string.replace("cc", "")
            size = engine_size_string.replace("cubic centimeters", "")
            # print('cc found', size)
            engine_size = (float(size)/1000)

        elif "ci" in engine_size_string or "c.i." in engine_size_string:
            size = engine_size_string.replace("ci", "")
            size = engine_size_string.replace("c.i.", "")
            engine_size = (float(size)*0.0163871)

        else:
            # if liter size > 12, it has errored so default to 2.0
            if float(match[0]) > 12:
                engine_size = 2.0
            else:
                engine_size = (float(match[0]))
    except:
        engine_size = 0.0
            
    return engine_size


def get_num_cylinders(listing_details):
    '''
    Extract number of cylinders from listing details
    
    :input listing_details list: List of auction details from html data

    :return num_cylinders int: Vehicle cylinder count
    '''
    try:
        cylinder_keywords = ["inline", "cylinder", "cyl","two", "three", "four", "five", "six", "eight", "v4", "v6", "v8", "v10", "v12", "v-4", "v-6", "vr6", "v-8", "v-10", "v-12", "w12", "w-12", "flat4", "flat-4", "flat 4", "flat6", "flat 6", "flat-6"]
        # cylinder_re = [""]
        singles = ["1", "single"]
        ones = ["1", "one"]
        twins = ["2", "twin"]
        triples = ["3", "triple"]
        twos = ["2", "two"]
        threes = ["3", "three"]
        fours = ["4", "four"]
        fives = ["5", "five"]
        sixes = ["6", "six"]
        eights = ["8", "eight"]
        tens = ["10", "ten"]
        twelves = ["12", "twelve"]

        num_cyl_string = None
        match = False

        # Extract num of cylinders
        for sentence in listing_details:
            if match:
                break
            words = sentence.lower().split(" ")
            for word in words:
                if any(w in word for w in cylinder_keywords):
                    num_cyl_string = word
                    match = True
                    break

        num_cylinders = None

        # Extract int from cylinder str
        for k in [threes, fours, fives, sixes, eights, tens, twelves, ones, twos, singles, twins, triples]:
            if k[0] in num_cyl_string or k[1] in num_cyl_string:
                num_cylinders = int(k[0])
                match = True
                break
    except:
        num_cylinders = 0
    
    return num_cylinders

def get_auction_result(vehicle_data_soup):
    '''
    Determine if the vehicle actually sold, or if vehicle did not meet reserve

    :input vehicle_data_soup str: Vehicle html data loaded by BeautifulSoup

    :return sold bool: True if vehicle sold, False if reserve not met or not sold for other reasons
    '''
    try:
        auction_result = vehicle_data_soup.find("span", class_ = 'info-value').text.split()[0].lower()
        sold = True
        if auction_result != "sold":
            sold = False
    except:
        sold = False

    return sold

def main(vehicle_data_soup):
    '''
    Combine all vehicle data into exported dict to be imported into XGBoost model
    
    :input vehicle_data_soup str: Vehicle html data loaded by BeautifulSoup

    :return vehicle_dict dict: All extracted vehicle data
    '''
    # Get image
    try:
        image = vehicle_data_soup.find_all('img', class_='post-image')[0]['src']
    except:
        image = None
    # Get vehicle details
    auction_year, sale_price, live = get_auction_year_and_price(vehicle_data_soup)
    vehicle_make, vehicle_model, vehicle_model2 = get_make_and_model(vehicle_data_soup)
    model_desc = get_listing_post_title(vehicle_data_soup, vehicle_make, vehicle_model)
    listing_details = get_listing_details(vehicle_data_soup)
    vehicle_mileage = get_mileage(listing_details)
    paint_color = get_paint_color(listing_details)
    model_year = get_model_year(vehicle_data_soup)
    engine_size = get_engine_size(listing_details)
    num_cylinders = get_num_cylinders(listing_details)
    sold = get_auction_result(vehicle_data_soup)

    vehicle_dict = {
        "make": vehicle_make,
        "model": vehicle_model,
        "model2": vehicle_model2,
        "model_desc": model_desc,
        "year": model_year,
        "miles": vehicle_mileage,
        "color": paint_color,
        "auction_year": auction_year,
        "engine_size": engine_size,
        "cylinders": num_cylinders,
        "bid_price": sale_price,
        "image": image,
        "live": live,
        "sold": sold
    }

    return vehicle_dict