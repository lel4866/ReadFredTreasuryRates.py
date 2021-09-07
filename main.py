""" program to read and use FRED interest rates, and a function to return interpolated rate for any period
from 0 days to 360 days.

The intention is that this will be used for things like the risk free rate for Black Scholes computations.

read_FRED_interest_rates() reads the FRED LIBOR rates from https://fred.stlouisfed.org/graph/fredgraph.csv
and creates a dictionary whose keys are the duration in days (1, 7, 30, 60, 90, 180, 360),
and whose values are another dictionary whose keys are a date and whose value is the rate in percent

the FRED LIBOR database goes back to 1986 or so, but has some missing days. Missing days are interpolated so there
will be a key for every day of the year from 1/1/1986 forward

After 12/31/2021, LIBOR rates will be replaced by SOFR and this code will be modified appropriately so the caller
doesn't need to know whether the rate comes from the LIBOR dtatbase or SOFR database
"""

import math
import numpy
import pandas
import datetime
from datetime import timedelta

FRED_interest_rates = {1: {'name': 'USDONTD156N'}, 7: {'name': 'USD1WKD156N'}, 30: {'name': 'USD1MTD156N'},
                       60: {'name': 'USD2MTD156N'}, 90: {'name': 'USD3MTD156N'}, 180: {'name': 'USD6MTD156N'},
                       360: {'name': 'USD12MD156N'}}
global_first_date = numpy.datetime64('1986-01-01')

def read_FRED_interest_rates():
    global FRED_interest_rates
    global global_first_date

    global_first_date = pandas.Timestamp('1986-01-01')
    today = pandas.Timestamp.today()
    today_str = today.strftime('%Y-%m-%d')
    for duration, series in FRED_interest_rates.items():
        series_name = series['name']
        FRED_url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_name}&cosd=1985-01-01&coed={today_str}'
        rate_df = pandas.read_csv(FRED_url, header=0, names=['date', 'rate'], parse_dates=[0], na_values={'rate': '.'},
                                  keep_default_na=False, engine='c')
        rate_df.sort_values(by='date', inplace=True)
        series['df'] = rate_df

        # keep track of overall earliest date of all series read
        first_date = rate_df.iloc[0].date
        if first_date > global_first_date:
            global_first_date = first_date

        # for informational purposes only
        print(series_name)
        row = rate_df.iloc[0]
        print(row.date.date(), row.rate)
        lastrow = len(rate_df.index) - 1
        row = rate_df.iloc[len(rate_df.index) - 1]
        print(row.date.date(), row.rate)
        print()
        break

    # now create numpy array with 1 row for every day between global_first_date and today
    # so, in order to grab a rate we just compute # of days between requested date and global_first_dat, and that's the
    # index into the numpy array
    numrows = ((today - global_first_date) + timedelta(1)).days
    for duration, series in FRED_interest_rates.items():
        rate_array = numpy.full(numrows, numpy.nan, numpy.float)
        rate_df = series['df']
        prev_date = global_first_date - timedelta(1)
        for index, row in rate_df.iterrows():
            # since pandas has a weird way of handling nan's, we have to set numpy array with nan this way:
            i = (row[0] - global_first_date).days
            rate = row[1]
            rate_array[i] = rate if not pandas.isnull(rate) else numpy.nan

        # now use pandas interpolate method to remove NaN's
        pandas_series = pandas.Series(rate_array)
        pandas_series.interpolate(inplace=True)
        rate_array = pandas_series.values
        y = 1

    print("Starting date will be: ", global_first_date)


def FRED_interest_rate(date: datetime, duration: int) -> float:
    return 0


if __name__ == '__main__':
    read_FRED_interest_rates()
