import pandas as pd
from bs4 import BeautifulSoup, Comment
from urllib.request import urlopen
import requests
import time

def get_player_url(seasons):
    # Gets list of players from specific season #
    player_url = list()
    for season in seasons:
        url = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html".format(season)
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        data_rows = soup.findAll('tr')[2:]
        for row in data_rows:
            for a in row.find_all('a', href=True):
                if 'players' in a['href']:
                    player_url.append(a['href'])
    return list(set(player_url))

def get_player_basic(season):
    # Get basic stats for players #
    url = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html".format(season)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    column_headers = [th.getText() for th in soup.findAll('tr')[0].findAll('th')]
    data_rows = soup.findAll('tr')[1:]
    data = [[td.getText() for td in data_rows[i].findAll('td')] for i in range(len(data_rows))]
    df = pd.DataFrame(data, columns=column_headers[1:])
    df = df.drop_duplicates(subset=['Player'])
    df.to_csv('data/basic_{}.csv'.format(season))
    return df

def get_player_per_poss(season):
    # Get per 100 possession stats for players #
    url = "https://www.basketball-reference.com/leagues/NBA_{}_per_poss.html".format(season)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    column_headers = [th.getText() for th in soup.findAll('tr')[0].findAll('th')]
    data_rows = soup.findAll('tr')[1:]
    data = [[td.getText() for td in data_rows[i].findAll('td')] for i in range(len(data_rows))]
    df = pd.DataFrame(data, columns=column_headers[1:])
    df = df.drop_duplicates(subset=['Player'])
    df.to_csv('data/per_poss_{}.csv'.format(season))
    return df

def get_player_advanced(season):
    # Get advanced possession stats for players #
    url = "https://www.basketball-reference.com/leagues/NBA_{}_advanced.html".format(season)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    column_headers = [th.getText() for th in soup.findAll('tr')[0].findAll('th')]
    data_rows = soup.findAll('tr')[1:]
    data = [[td.getText() for td in data_rows[i].findAll('td')] for i in range(len(data_rows))]
    df = pd.DataFrame(data, columns=column_headers[1:])
    df = df.drop_duplicates(subset=['Player'])
    df.to_csv('data/advanced_{}.csv'.format(season))
    return df

def get_player_salary(season):
    # Get salary information for each player #
    # Needs to be in format ....-.... e.g.2021-2022 #
    url = "https://hoopshype.com/salaries/players/{}/".format(season)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    columns_headers = [th.getText() for th in soup.findAll('tr')[0].findAll('td')]
    columns_headers = [s.strip() for s in columns_headers]
    data_rows = soup.findAll('tr')[1:]
    data = [[td.getText().strip() for td in data_rows[i].findAll('td')][1:] for i in range(len(data_rows))]
    df = pd.DataFrame(data, columns=columns_headers[1:])
    df.to_csv('data/player_salary_{}.csv'.format(season[5:]))
    return df

get_player_basic(2021)
get_player_basic(2022)
get_player_advanced(2021)
get_player_advanced(2022)
get_player_per_poss(2021)
get_player_per_poss(2022)
get_player_salary('2020-2021')
get_player_salary('2021-2022')