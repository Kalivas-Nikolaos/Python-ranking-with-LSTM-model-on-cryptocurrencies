from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import pandas as pd 
import numpy as np
import datetime
from datetime import datetime, timedelta

MIN_TRADING_VOLUME = 10000000 # 10m minimum
STARTING_COIN_DATE = datetime.now()
ONE_YEAR_AGO_DATE = STARTING_COIN_DATE - timedelta(days = 365)

#part 1: sort api data with 'sort_name' we give and returns the sorted table.
def coinmarketapi_values(sort_name):
  url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
  parameters = {
    'start':'1',
    'limit':'100',
    'convert':'USD',
    'sort': sort_name
  }
  headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': '5044fdd3-f5fc-4a53-bdaa-d13c1ce5d4a0',
  }

  session = Session()
  session.headers.update(headers)

  try:
    response = session.get(url, params=parameters)
    data = json.loads(response.text)
    tbl = pd.json_normalize(data["data"])

    # now filter tbl to exclude coins with one year old in the Market.
    tbl["date_added"] = pd.to_datetime(tbl["date_added"]).apply(lambda x: x.replace(tzinfo=None))
    tbl = tbl[ tbl["date_added"] < ONE_YEAR_AGO_DATE ]
    # now filter tbl to exclude coins with small volume
    tbl = tbl[ tbl["quote.USD.volume_24h"] >  MIN_TRADING_VOLUME ]

  except (ConnectionError, Timeout, TooManyRedirects) as e:
    print(e)
  return(tbl)

def rank(tbl, rnkColumn):
  ntbl = tbl.sort_values(rnkColumn, ascending=False)
  ntbl["rank"] = range(1, tbl.shape[0]+1, 1) #creates column rank 
  ntbl["ranking_method"] = rnkColumn         #creates column ranking method
  return(ntbl)

def rank_coins():
  coinTble = coinmarketapi_values('market_cap')
  rnk1 = rank(coinTble, "quote.USD.market_cap")
  rnk2 = rank(coinTble, "quote.USD.percent_change_7d")
  rnk3 = rank(coinTble, "quote.USD.volume_24h")

  tbl_rnk = rnk1.append(rnk2).append(rnk3)
  table_pivot = tbl_rnk.pivot_table(values='rank', index='symbol', columns='ranking_method')
  table_pivot["Total"] = table_pivot.loc['Total',:]= table_pivot.sum(axis=1)
  table_pivot["Total"] = table_pivot["Total"] / 3 
  sort_table = table_pivot.sort_values('Total')
  sort_table.columns = ['Market_Cap','Percent_change_7d','Volume_24h','Total']
  # index_table = sort_table.index
  return(sort_table)

total_tbl = rank_coins()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(total_tbl)

print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

#Nomics Api, Description call of the Crypto Coins that we valued before.
def information_of_cryptos(total_tbl):
  k='nothing'
  for i in range (0, 5):
    k = total_tbl.index.values[i]
    new_url = 'https://api.nomics.com/v1/currencies?key=b2c6c863cc61e1c4154c2e2ceec146ea&ids='+k+'&attributes=id,name,description,website_url,whitepaper_url'
    headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': 'b2c6c863cc61e1c4154c2e2ceec146ea',
    }
    session = Session()
    session.headers.update(headers)

    try:
      response = session.get(new_url)
      data = json.loads(response.text)
      print('\nNumber', i, 'Description of our be-loved Coin')
      print(data)
        

    except (ConnectionError, Timeout, TooManyRedirects) as e:
      print(e)

  return()

information_of_cryptos(total_tbl)

print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

def lending_cryptos():
    url = "https://data-api.defipulse.com/api/v1/defipulse/api/GetLendingTokens"
    headers = {'X-MBX-APIKEY' : 'c84855bdf075c2bceae10928ae82e19ffb4a92153a9972bcea85a65ef173'}
    session = Session()
    session.headers.update(headers)
    try:
        response = session.get(url)
        data = json.loads(response.text)
        tbl = pd.DataFrame(data)
        tbl = tbl.rename(columns={0: "lending_token"})
        
    except (ConnectionError, Timeout, TooManyRedirects) as e:
        print(e)
    return(tbl)

df2 = lending_cryptos()
#take the first 30 values of our Ranked Table
df1 = pd.DataFrame(total_tbl.index[0:30])
#compare them with Lending Category Table from De-Fi pulse Api and keep the common elements
lending_table = pd.DataFrame(np.intersect1d(df1,df2))
lending_table = lending_table.rename(columns={0: "Lending Coin"})
print('\nFrom the Coins that we Ranked before, these are the Lending coins ')
print(lending_table)

def DeFi_Pulse_lending_Projects():
    url = "https://data-api.defipulse.com/api/v1/defipulse/api/GetLendingProjects"
    headers = {'X-MBX-APIKEY' : 'c84855bdf075c2bceae10928ae82e19ffb4a92153a9972bcea85a65ef173'}
    session = Session()
    session.headers.update(headers)
    try:
        response = session.get(url)
        data = json.loads(response.text)
        tbl = pd.DataFrame(data)
        lending_project_last_table = pd.DataFrame(tbl['name'], columns=['name'])
        df2 = tbl['token']
        df3 = tbl['chain']
        df4 = tbl['category']
        lending_project_last_table['token'] = df2
        lending_project_last_table['chain'] = df3
        lending_project_last_table['category'] = df4
    except (ConnectionError, Timeout, TooManyRedirects) as e:
        print(e)
    return(lending_project_last_table)

table_of_lending_projects = DeFi_Pulse_lending_Projects()
print('\nFrom this Lending Projects these are the ones who work on Ethereum Block Chain')
print(table_of_lending_projects)
