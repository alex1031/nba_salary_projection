# NBA Salary Projection
- Created a tool comparing model projected salary to player actual salary.
- Scraped player advanced and basic stats from basketball-reference.
- Scraped player salary data from hoopshype.com
- Transformed some numerical features to categorical features through binning.
- Optimised Lasso, Random Forest and Support Vector Regressor using GridSearchCV.
- Built a client displaying the tables using Flask and HTML.

## Code and Resources Used
**Python Version**: 3.8.8 

**Packages**: pandas, sklearn, matplotlib, beautifulsoup, flask, json

**Web Framework Requirements**: ``` pip install -r requirements.txt ```

## Web Scraping 
We get data tables from basketball-reference like [this](https://www.basketball-reference.com/leagues/NBA_2022_per_game.html) and from hoopshype like [this](https://hoopshype.com/salaries/players/2021-2022/).

## Data Cleaning
The data retrieved required cleaning in order for it to be usable for the model. The following was done:
- Seperated columns required from different dataframes scraped (basic stats, advanced stats, per 100 possession stats, salary), which was initially seperated by each season.
- Matched names of players (different website have different interpretation of player names).
- Merged the columns into one dataframe.
- Generalised positions into guards, forwards and centers.
- Removed rows for players that played less than 20 games.
- Parsed numeric data out of salary.
- Created new column for free throw rate.

## EDA
