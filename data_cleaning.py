import pandas as pd
import numpy as np
import json, unidecode

name_change = json.load(open('namechange.json', encoding='utf8'))
pos_change = json.load(open('poschange.json', encoding='utf-8'))
advanced_2021 = pd.read_csv('data/advanced_2021.csv')
advanced_2022 = pd.read_csv('data/advanced_2022.csv')
basic_2021 = pd.read_csv('data/basic_2021.csv')
basic_2022 = pd.read_csv('data/basic_2022.csv')
per_poss_2021 = pd.read_csv('data/per_poss_2021.csv')
per_poss_2022 = pd.read_csv('data/per_poss_2022.csv')
player_salary_2021 = pd.read_csv('data/player_salary_2021.csv')
player_salary_2022 = pd.read_csv('data/player_salary_2022.csv')

# Model 1: Offensive and Defensive Rating + Offensive and Defensive Box Plus Minus #
basic_col_1 = ["Player", 'Pos', 'Age', 'G', 'GS', 'MP'] 
advanced_col_1 = ['OBPM', 'DBPM']
per_poss_col_1 = ['ORtg', 'DRtg']

basic_2021_mod1 = basic_2021[basic_col_1]
basic_2022_mod1 = basic_2022[basic_col_1]
advanced_2021_mod1 = advanced_2021[advanced_col_1]
advanced_2022_mod1 = advanced_2022[advanced_col_1]
per_poss_2021_mod1 = per_poss_2021[per_poss_col_1]
per_poss_2022_mod1 = per_poss_2022[per_poss_col_1]

mod1_2021 = pd.concat([basic_2021_mod1, advanced_2021_mod1, per_poss_2021_mod1], axis=1)
mod1_2021['Player'] = mod1_2021['Player'].replace(name_change)
mod1_2021 = pd.merge(mod1_2021, player_salary_2021[['Player', '2020/21', '2020/21(*)']], on=['Player'])
mod1_2021.rename(columns={'2020/21': "Salary",'2020/21(*)': "Salary Adjusted"}, inplace=True)

mod1_2022 = pd.concat([basic_2022_mod1, advanced_2022_mod1, per_poss_2022_mod1], axis=1)
mod1_2022['Player'] = mod1_2022['Player'].replace(name_change)
mod1_2022 = pd.merge(mod1_2022, player_salary_2022[['Player', '2021/22','2021/22(*)']], on=['Player'])
mod1_2022.rename(columns={'2021/22': 'Salary','2021/22(*)': 'Salary Adjusted'}, inplace=True)

mod1_df = pd.concat([mod1_2021, mod1_2022])
mod1_df['Pos'] = mod1_df['Pos'].replace(pos_change)
mod1_df = mod1_df[mod1_df['G'] >= 20]
mod1_df['Salary'] = mod1_df['Salary'].str.replace('$', '')
mod1_df['Salary'] = mod1_df['Salary'].str.replace(',', '')
mod1_df['Salary'] = pd.to_numeric(mod1_df['Salary'])
mod1_df['Salary Adjusted'] = mod1_df['Salary Adjusted'].str.replace('$', '')
mod1_df['Salary Adjusted'] = mod1_df['Salary Adjusted'].str.replace(',', '')
mod1_df['Salary Adjusted'] = pd.to_numeric(mod1_df['Salary Adjusted'])
mod1_df = mod1_df.reset_index().drop(columns='index')

mod1_df.to_csv('data/model_1_clean.csv')

# Model 2: Four Factors #
basic_col_2 = ['Player', 'Pos', 'Age', 'G', 'GS', 'MP', 'eFG%', 'FTR']
advanced_col_2 = ['TRB%', 'TOV%']
basic_2021['FTR'] = basic_2021['FT']/basic_2021['FGA']
basic_2021_mod2 = basic_2021[basic_col_2]
basic_2022['FTR'] = basic_2022['FT']/basic_2022['FGA']
basic_2022_mod2 = basic_2022[basic_col_2]
advanced_2021_mod2 = advanced_2021[advanced_col_2]
advanced_2021_mod2['TRB%'] = advanced_2021_mod2['TRB%']/100
advanced_2021_mod2['TOV%'] = advanced_2021_mod2['TOV%']/100
advanced_2022_mod2 = advanced_2021[advanced_col_2]
advanced_2022_mod2['TRB%'] = advanced_2022_mod2['TRB%']/100
advanced_2022_mod2['TOV%'] = advanced_2022_mod2['TOV%']/100

mod2_2021 = pd.concat([basic_2021_mod2, advanced_2021_mod2], axis=1)
mod2_2021['Player'] = mod2_2021['Player'].replace(name_change)
mod2_2021 = pd.merge(mod2_2021, player_salary_2021[['Player', '2020/21','2020/21(*)']], on=['Player'])
mod2_2021.rename(columns={'2020/21': 'Salary','2020/21(*)': 'Salary Adjusted'}, inplace=True)

mod2_2022 = pd.concat([basic_2022_mod2, advanced_2022_mod2], axis=1)
mod2_2022['Player'] = mod2_2022['Player'].replace(name_change)
mod2_2022 = pd.merge(mod2_2022, player_salary_2022[['Player', '2021/22','2021/22(*)']], on=['Player'])
mod2_2022.rename(columns={'2021/22': 'Salary','2021/22(*)': 'Salary Adjusted'}, inplace=True)

mod2_df = pd.concat([mod2_2021, mod2_2022])
mod2_df['Pos'] = mod2_df['Pos'].replace(pos_change)
mod2_df = mod2_df[mod2_df['G'] >= 20]
mod2_df['Salary'] = mod2_df['Salary'].str.replace('$', '')
mod2_df['Salary'] = mod2_df['Salary'].str.replace(',', '')
mod2_df['Salary'] = pd.to_numeric(mod2_df['Salary'])
mod2_df['Salary Adjusted'] = mod2_df['Salary Adjusted'].str.replace('$', '')
mod2_df['Salary Adjusted'] = mod2_df['Salary Adjusted'].str.replace(',', '')
mod2_df['Salary Adjusted'] = pd.to_numeric(mod2_df['Salary Adjusted'])
mod2_df = mod2_df.reset_index().drop(columns='index')

mod2_df.to_csv('data/model_2_clean.csv')