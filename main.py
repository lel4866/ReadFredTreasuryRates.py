# program to read FRED treasury rates, and a function to return interpolated rate for any period from 1 day to 360 days

import pandas as pd

def read_fred_treasury_rates(name):

    fred_series_names = {'USDONTD156N': 1, 'USD1WKD156N': 7, 'USD1MTD156N': 30, 'USD2MTD156N': 60, 'USD3MTD156N': 90,
                         'USD6MTD156N': 180, 'USD12MD156N': 360}

    overnight_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=USD3MTD156N&cosd=2016-08-27&coed=2021-08-27'
    overnight_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=USD3MTD156N&cosd=2016-08-27&coed=2021-08-27'
    overnight_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=USD3MTD156N&cosd=2016-08-27&coed=2021-08-27'
    overnight_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=USD3MTD156N&cosd=2016-08-27&coed=2021-08-27'
    overnight_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=USD3MTD156N&cosd=2016-08-27&coed=2021-08-27'
    overnight_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=USD3MTD156N&cosd=2016-08-27&coed=2021-08-27'
    overnight_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=USD3MTD156N&cosd=2016-08-27&coed=2021-08-27'
    overnight_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=USD3MTD156N&cosd=2016-08-27&coed=2021-08-27'
    overnight_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=USD3MTD156N&cosd=2016-08-27&coed=2021-08-27'
    series = fred_series_names[]
    fred_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=USD3MTD156N&cosd=2016-08-27&coed=2021-08-27'


if __name__ == '__main__':
    read_fred_treasury_rates()
