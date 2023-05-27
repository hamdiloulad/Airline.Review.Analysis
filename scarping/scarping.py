
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Create an empty list to store the reviews
reviewList = []

# Create a list of fields that we want to scrape
reviewSimplelist = ['Aircraft','Type Of Traveller','Seat Type','Route','Date Flown','Recommended']

# Loop through all the pages of reviews
for page in range(1, 36):  # Change the range based on the number of pages of reviews
    url = "https://www.airlinequality.com/airline-reviews/british-airways/page/{}/?sortby=post_date%3ADesc&pagesize=100".format(page)
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')
    print("PAGE " + str(page))
    # Loop through all the reviews on the page
    rating = soup.find_all(itemprop="ratingValue")
    for a in range(1, len(rating)):
        review_text=soup.find_all(itemprop="reviewBody")[a-1].text
        review_rating = soup.find_all(itemprop="ratingValue")[a].text
        table = soup.find_all('table')[a]
        review = {'Aircraft':"none",'Type Of Traveller':"none",'Seat Type':"none",'Route':"none",'Date Flown':"none",'Seat Comfort':"none",'Wifi & Connectivity':"none",'Food & Beverages':"none",'Inflight Entertainment':"none",'Ground Service':"none",'Cabin Staff Service':"none",'Value For Money':"none",'Recommended':"none"}  
        review['review_rating']=review_rating
        review['review_text']=review_text
        trList=table.find_all('tr')
        for tr in trList:
            tdList=tr.find_all('td')
            firstTd=tdList[0].text
            if firstTd in reviewSimplelist:  review[firstTd] = tdList[1].text
            else:  review[firstTd] = len(tdList[1].find_all('span',class_="star fill"))
        reviewList.append(review)   
        
# Convert the list of reviews to a pandas DataFrame and save it as a CSV file
df = pd.DataFrame.from_dict(reviewList)
df.to_csv('D:/DataProject/venv/scarping-nlp/import/main.csv', index=False, encoding='utf-8')

