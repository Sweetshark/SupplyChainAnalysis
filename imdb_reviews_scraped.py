# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:00:32 2020
@author: Yating Li
"""
"""
the first 500 reviews of a movie (use Joker as an example) 
- web: https://www.imdb.com/title/tt7286456/reviews?ref_=tt_ov_rt
- Dataset includes 9 variables
- Spoiler is one of the variables, with which we can calculate the spoiler rate in this 500 reviews
- tools: bs4,selenium,pandas,re
  
"""

# %%%% Preliminaries and library loading
import datetime
import os
import pandas as pd
import re
import time

# libraries to crawl websites
from bs4 import BeautifulSoup
from selenium import webdriver

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 5)
pd.set_option('display.width',800)
# %%%
# set up the chrome driver
path = '/Users/yatingli/Desktop/MSBA/bus256'
os.chdir(path)
driver = webdriver.Chrome('./chromedriver')

# Creating the list of links.
links_to_scrape = 'https://www.imdb.com/title/tt7286456/reviews?ref_=tt_ov_rt'

driver.get(links_to_scrape)

# %%%
# the initial number in the first page is 25
# click load more to get more review records untile the number is 500
nums = 25
while nums < 500:
    try:
        driver.find_element_by_xpath("//button[@class='ipl-load-more__button']").click()
        time.sleep(3)
        reviews = driver.find_elements_by_xpath("//div[@class='lister-item mode-detail imdb-user-review  collapsable']")
        spoilers_reviews = driver.find_elements_by_xpath("//div[@class='lister-item mode-detail imdb-user-review  with-spoiler']")
        nums = len(reviews)+len(spoilers_reviews)
        print(nums)
    except:
        break
# %%%
# to click the arrow and expand the reviews that were from spoiler and hiden
# to click the arrow and expand the reviews that were not expanded fully  
buttons = driver.find_elements_by_xpath("//div[@class='expander-icon-wrapper spoiler-warning__control']")
for button in buttons:
    button.click()
    
expand = driver.find_elements_by_xpath("//div[@class='expander-icon-wrapper show-more__control']")
for i in expand:
    try:
        i.click()
    except:
        i = False
# %%% 

# Finding all the reviews in the website and bringing them to python
reviews_one_movie = []

reviews = driver.find_elements_by_xpath("//div[@class='lister-item-content']")
# Since the number of review records is for sure, use for loop instead of while
for r in range(500):
    one_review                   = {}
    one_review['scrapping_date'] = datetime.datetime.now()
    one_review['google_url']     = driver.current_url
    soup                         = BeautifulSoup(reviews[r].get_attribute('innerHTML'))

    # get raw data for this record
    try:
        one_review_raw = soup.text.strip()
    except:
        one_review_raw = ""
    one_review['review_raw'] = one_review_raw
    
    # get the spoiler label: whether the reviewer is a spoiler
    try:
        one_reviewer_spoiler = soup.find('span',attrs={'class':"spoiler-warning"})      
    except:
        one_reviewer_spoiler = False
    one_review['reviewer_spoiler'] = True if one_reviewer_spoiler else False
    
    # get the rating of this review
    try:
        one_review_rating = soup.find('span',attrs={'class':"rating-other-user-rating"}).text.strip()
    except:
        one_review_rating = ""
    one_review['rating'] = one_review_rating
    
    # get the tile of this review
    try:
        one_review_title = soup.find('a',attrs={'class':"title"}).text.strip()
    except:
        one_review_title = ""
    one_review['review_title'] = one_review_title
    
    # get the name of the reviewer
    try:
        one_reviewer = soup.find('span',attrs={'class':"display-name-link"}).text.strip()
    except:
        one_reviewer = ""
    one_review['reviewer'] = one_reviewer
    
    # get the review date
    try:
        one_review_date = soup.find('span',attrs={'class':"review-date"}).text.strip()
    except:
        one_review_date = ""
    one_review['review_date'] = one_review_date
    
    #get the content of this review
    try:
        one_review_text = soup.find('div', attrs={'class':'text show-more__control'}).text.strip()
    except:
        one_review_text = ""
    one_review['review_text'] = one_review_text
    

    reviews_one_movie.append(one_review)
# %%%
# output this dataset as a .csv file    
data = pd.DataFrame(reviews_one_movie)
data['review_raw'] = data['review_raw'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', " ", x))

data.to_csv('./500reviewsForJoker.csv',index = False)

# %%% 



