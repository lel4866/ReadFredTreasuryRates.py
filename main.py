"""
program to read FRED LIBOR interest rates, and a function to return the interpolated rate for any duration,
from 0 days to 360 days, for any given date in the FRED series

The intention is that this will be used for things like the risk free rate for Black Scholes computations.

read_FRED_interest_rates() reads the FRED LIBOR rates from https://fred.stlouisfed.org/graph/fredgraph.csv
and creates a dictionary whose keys are the duration in days (1, 7, 30, 60, 90, 180, 360),
and whose values are another dictionary whose keys are a date and whose value is the rate in percent

the FRED LIBOR database goes back to 1986 or so, but has some missing days. Missing days are interpolated so there
will be a value for every day of the year from the start of the FRED series forward

After 12/31/2021, LIBOR rates will be replaced by SOFR and this code will be modified appropriately so the caller
doesn't need to know whether the rate comes from the LIBOR database or SOFR database
"""

import numpy
from scipy import optimize
import pandas
import datetime
from datetime import timedelta
import time

FRED_interest_rates = {1: {'name': 'USDONTD156N'}, 7: {'name': 'USD1WKD156N'}, 30: {'name': 'USD1MTD156N'},
                       60: {'name': 'USD2MTD156N'}, 90: {'name': 'USD3MTD156N'}, 180: {'name': 'USD6MTD156N'},
                       360: {'name': 'USD12MD156N'}}
global_first_date = numpy.datetime64('1986-01-01')


def read_FRED_interest_rates():
    global FRED_interest_rates
    global global_first_date

    start_time = time.time()
    global_first_date = pandas.Timestamp('1986-01-01')
    today = pandas.Timestamp.today()
    today_str = today.strftime('%Y-%m-%d')
    for duration, series in FRED_interest_rates.items():
        series_name = series['name']
        FRED_url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_name}&cosd=1985-01-01&coed={today_str}'
        rate_df = pandas.read_csv(FRED_url, header=0, names=['date', 'rate'], parse_dates=[0], na_values={'rate': '.'},
                                  keep_default_na=False, engine='c')

        # keep track of overall earliest date of all series read
        first_date = rate_df.iloc[0].date
        if first_date > global_first_date:
            global_first_date = first_date

        # save data frame back in FRED_interest_rates dictionary
        series['data'] = rate_df

        # for informational purposes only
        print(series_name)
        row = rate_df.iloc[0]
        print(row.date.date(), row.rate)
        last_row = len(rate_df.index) - 1
        row = rate_df.iloc[last_row - 1]
        print(row.date.date(), row.rate)
        print()  # while debugging

    # now create numpy array with 1 row for EVERY day (including weekends and holidays) between global_first_date and
    # today, and 1 column for each FRED series plus 2 columns for the coefficients of a fitted logarithmic curve that
    # is fitted for each day.
    # once we do this, in order to grab a rate for a specific day and duration, we just compute # of days between
    # requested date and global_first_date, and use that as the index into the numpy rate array, then use the
    # requested duration and curve fit coefficients to compute interpolated_rate = A + B*log(duration)
    numrows = ((today - global_first_date) + timedelta(1)).days
    numcols = 2 + len(FRED_interest_rates)
    rates_array = numpy.empty((numrows, numcols), float)
    icol = 0
    for duration, series in FRED_interest_rates.items():
        rate_df = series['data']
        series_array = numpy.empty((numrows), float)
        for index, row in rate_df.iterrows():
            # since pandas has a weird way of handling nan's, we have to set numpy array with nan this way:
            i = (row[0] - global_first_date).days
            rate = row[1]
            series_array[i] = rate if not pandas.isnull(rate) else numpy.nan

        # now use pandas interpolate method to remove NaN's
        pandas_series = pandas.Series(series_array)
        pandas_series.interpolate(inplace=True)

        # now append rates for a specific series to overall rates array
        rates_array[:,icol] = pandas_series.values
        icol = icol + 1
        y = 1 # debug

    # now fit log function for each row of rates array; put coefficients in last 2 columns
    duration_array = numpy.array([1, 7, 30, 60, 90, 180, 360])
    for row in rates_array:
        row[7:9] = optimize.curve_fit(lambda x, a, b: a+b*numpy.log(x), duration_array, row[0:7])[0]

    end_time = time.time()
    print("Starting date will be: ", global_first_date.date())
    print(f"Elapsed time is {end_time - start_time} seconds")


def FRED_interest_rate(date: datetime, duration: int) -> float:
    # get the two series to interpolate between
    return 0


if __name__ == '__main__':
    read_FRED_interest_rates()
