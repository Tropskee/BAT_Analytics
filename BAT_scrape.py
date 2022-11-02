'''
Program to pull auction data from BringaTrailer.com

Created by Tropskee on 10/29/2022
Est. Completion Time: 2 hours
'''

import requests
import re
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


base_url = "https://bringatrailer.com/"
subaru_models = ["subaru/wrx-sti/"]

def get_make_and_model(make_and_model):
    '''
    Accepts URL subdirectory and returns make and model of vehicle
    '''
    split_make_and_model = make_and_model.split('/')
    return split_make_and_model[0], split_make_and_model[1]
    

def get_model_html(model):
    '''
    Get html of overlying vehicle model webpage
    
    :input model str: Subdirectory url of vehicle model
    
    :return model_html str: Html content of vehicle model overview page
    '''
    base_url = "https://bringatrailer.com/"
    driver = webdriver.Safari()
    # driver.implicitly_wait(4)

    
    driver.get(test_url)
    print(test_url)
    
    # Scroll to click 'show more' button to get all previously auctioned vehicle URL's
    try:
        # btn = driver.find_element_by_css_selector("button.load-more-button")
        # btn.click()
        # WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH,
        # '//*[@data-text = "Show More"]'))).click()
        # WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH,
        # '//button[contains(text() = "Show More")]'))).click()
        
        # body > main > div.container > div > div > div.filter-group > div.overlayable > div.auctions-footer.auctions-footer-previous > button
        # btn = driver.find_element(By.CSS_SELECTOR,
        #     'body > main > div.container > div > div > div.filter-group > div.overlayable > div.auctions-footer.auctions-footer-previous > button')
        # btn.click()
        button_selector = 'body > main > div.container > div > div > div.filter-group > div.overlayable > div.auctions-footer.auctions-footer-previous > button'

        while(driver.find_element(By.CSS_SELECTOR, button_selector)):
            WebDriverWait(driver,10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, button_selector))).click()
            time.sleep(1)
            
    except Exception as e:
        print(e)
        print('No more vehicles to show')
    
    html = driver.page_source
    driver.quit()
    model_html = BeautifulSoup(html)
    # print(model_html)
        
    return model_html

def get_vehicle_html(model_url):
    '''
    Get page html of (1) vehicle's auction data
    '''
    try:
        html = requests.get(model_url)
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        print(e)
        print('Model HTML data not acquired')
        pass
        
    # Pull html data
    soup = BeautifulSoup(html.content, 'html.parser')
    
    return soup

def get_model_urls(model_html):
    '''
    Get all urls of previously auction vehicles pertaining to a specific model
    
    :input soup str: Beautiful soup string of html data
    
    :return urls list: List containing urls of previoulsy auction vehicles
    '''
    
    get_models = model_html.find_all("div", class_ = "blocks")
    
    urls = []
    for a in get_models[0].find_all("a", href=True):
        if a["href"] not in urls:
            urls.append(a["href"])
        else:
            pass
    print(f"Found {len(urls)} vehicle auction URL's.")
    return urls


# def get_sale_date(soup):
#     '''
#     Get vehicle auction date
#     '''
#     # date_html = soup.find_all('span', class_ = "data-value")
#     date_html = soup.find_all('span', class_ = "date")
#     # print(date_html[0].text.split()[1])
#     sold_date = date_html[0].text.split()[1]
    
#     return sold_date

def get_auction_result(soup):
    '''
    Get auction result data ex. "Sold for $22,250 on 10/27/22"
    '''
    # for auction_result in soup.find("span", class_ = "data-label"):
    auction_result = soup.find("span", class_ = 'info-value')
    # print(auction_result.text)    
    return auction_result.text
    

# def get_price(soup, sold_bool):
#     '''
#     Get vehicle sale price or max bid if not sold
#     '''
#     if sold_bool:
#         for price in soup.find(class_ = "data-value price"):
#             car_price_str = re.findall('[0-9]+,[0-9]+', price)
#             return int(car_price_str[0].replace(",",""))
#     else:
#         for price in soup.find(class_ = "data-value"):
#             car_price_str = re.findall('[0-9]+,[0-9]+', price)
#             return int(car_price_str[0].replace(",",""))


def get_listing_details(soup):
    '''
    Get vehicle listing details - vin, miles, etc.
    '''
    
    listing_details = soup.find_all("div", class_="item")
    return listing_details

def get_model_year(soup):
    '''
    Get vehicle model year from html
    '''
    model_year_text = soup.find("h1", class_ = 'post-title').text
    model_year = re.findall("(\d{4})", model_year_text)
    return model_year
    

# def get_vin(soup):
#     '''
#     Get vehicle VIN
#     '''
#     # Get VIN
#     for vin in soup.find("ul", class_="listing-essentials-items").find_all("li")[3]:
#         return vin
     
    
# def get_miles(soup):
#     '''
#     Get vehicle miles
#     '''
#     for miles in soup.find("ul",class_="listing-essentials-items").find_all("li")[4]:
#         return miles
            

# def main():
#     soup = get_html(models[0])
#     model_urls = get_model_urls(soup)
#     print(model_urls)
    
# main()


soup = get_model_html(models[0])
model_urls = get_model_urls(soup)

# # Scroll to click 'show more' button to get all previously auctioned vehicle URL's
# btn = dr.find_element_by_css_selector("button.load-more-button")
# btn.click()

import pandas as pd

pd_db = pd.DataFrame()
# Pandas DF column names
COLUMNS = ['Make', 'Model', 'Year', 'Mileage', 'Mileage Notes', 'Sale_Status', 'Final Bid Price', 'Date', 'VIN', 'Details']


# Loop through all urls for a specific model, get listing data, and update db
for model_url in model_urls[-20:]:
    # Initiliaze pandas dataframe variables to None
    vehicle_make, vehicle_model, model_year, vehicle_mileage, vehicle_mileage_notes, sale_status, sale_price, sale_date, vehicle_vin, joined_results = [None] * 10
    # Initialize auction result variables to None
    vehicle_data_soup, auction_result_str, listing_details_html, listing_details = [None] * 4
    # Get vehicle html
    vehicle_data_soup = get_vehicle_html(model_url)

    # Parse html and get auction data
    auction_result_str = get_auction_result(vehicle_data_soup)

    # Get listing details - vin, miles, etc. - second entry
    listing_details_html = get_listing_details(vehicle_data_soup)
    # Find the listing_details using keyword "Listing Details"
    for detail in listing_details_html:
        if detail.find("strong") and "Listing Details" in detail.find("strong"):
            listing_details = detail
        
    # Extract details from html "li"
    results = None
    results = [detail.text for detail in listing_details.find_all('li')]
    # Replace any hyphens in listing details with spaces
    results = [detail.replace("-"," ") for detail in results]
    
    # Get vehicle mileage, normally second entry in "results"
    mileage_words = ['Miles', 'miles', 'Mile', 'mile', 'Kilometers', 'kilometers', 'Kilometer', 'kilometer', 'KM', 'km']
    for result in results:
        if any(word in result for word in mileage_words):
            vehicle_mileage_notes = result
    
    # Extract Mileage figure from vehicle_mileage_notes
    if vehicle_mileage_notes is None:
        mileage_value = None
    else:
        mileage_value = re.findall('[0-9]+,[0-9]+', vehicle_mileage_notes)
        if len(mileage_value) < 1:
            mileage_value = re.findall('[0-9]+[k]', vehicle_mileage_notes)
            if len(mileage_value) >= 1:
                mileage_value = [mileage_value[0].strip('k') + ',000']

    print(mileage_value)
    vehicle_mileage = mileage_value
    
    # Get vehicle make and model
    vehicle_make, vehicle_model = get_make_and_model(subaru_models[0])

    # Get vehicle model year
    model_year = get_model_year(vehicle_data_soup)

    # Check sale status - i.e., sold or not
    sale_status = "Not Sold"
    if "Sold" in auction_result_str or "sold" in auction_result_str:
        sale_status = "Sold"

    # Get vehicle sale date
    sale_date = auction_result_str.split()[-1]

    # Get vehicle sale price
    sale_price = int(re.findall('[0-9]+,[0-9]+', auction_result_str)[0].replace(",",""))

    # Get vehicle vin, first entry in "results"
    vehicle_vin = results[0].split()[-1]

    # Combine results into 1 string
    joined_results = " ,".join(results)

    # Create pd series for ingestion into pd_db
    pd_series = pd.Series([vehicle_make.capitalize(), vehicle_model, model_year[0], vehicle_mileage, vehicle_mileage_notes, sale_status, sale_price, sale_date, vehicle_vin, joined_results])
    
    # Append DataFrame - make, model, year, mileage, sale_status, price, date, vin, other_details
    pd_db = pd_db.append(pd_series, ignore_index=True)

        
pd_db.reset_index()
pd_db.columns = COLUMNS
pd_db.set_index('Date', inplace=True)
pd_db.head()

filepath = './Desktop/sti_auction_prices.xlsx'

pd_db.to_excel(filepath, index=False)