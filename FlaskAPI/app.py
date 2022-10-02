import flask
import pandas as pd
from flask import Flask, render_template


app = Flask(__name__)

disp_table = pd.read_csv('display_df.csv').drop(columns='Unnamed: 0')
disp_table['eFG%'] = disp_table['eFG%'].round(decimals=1)
disp_table['FTR'] = disp_table['FTR'].round(decimals=1)
disp_table['TOV%'] = disp_table['TOV%'].round(decimals=1)
disp_table['Salary'] = disp_table['Salary'].round(decimals=2)
disp_table['Salary Adjusted'] = disp_table['Salary Adjusted'].round(decimals=2)
disp_table['Rating Model Salary'] = disp_table['Rating Model Salary'].round(decimals=2)
disp_table['Four Factors Salary'] = disp_table['Four Factors Salary'].round(decimals=2)
disp_table['Average Predicted Salary'] = disp_table['Average Predicted Salary'].round(decimals=2)
headings = tuple(disp_table.columns)

data = tuple(disp_table.to_records(index=False))

@app.route('/')

def table():
    return render_template('table.html', headings=headings, data=data)

if __name__ == '__main__':
    app.run()
