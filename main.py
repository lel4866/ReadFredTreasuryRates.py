""" program to read and use FRED interest rates,
and a function to return interpolated rate for any period from 1 day to 360 days

The intention is that this will be used for things like the risk free rate for Black Scholes computations.

read_FRED_interest_rates() reads the FRED LIBOR rates from https://fred.stlouisfed.org/graph/fredgraph.csv
and creates a dictionary whose keys are the duration in days (1, 7, 30, 60, 90, 180, 360),
and whose values are another dictionary whose keys are a date and whose value is the rate in percent

the FRED LIBOR database goes back to 1986 or so, but has some missing days. Missing days are interpolated so there
will be a key for every day of the year from 1/1/1986 forward

After 12/31/2021, LIBOR rates will be replaced by SOFR and this code will be modified appropriately so the caller
doesn't need to know whether the rate comes from the LIBOR dtatbase or SOFR database

"""

import pandas as pd

FRED_interest_rates = {1:dict(), 7:dict(), 30:dict(), 60:dict(), 90:dict(), 180:dict(), 360:dict()}

def read_FRED_interest_rates():
    global FRED_interest_rates

    fred_series_names = {'USDONTD156N': 1, 'USD1WKD156N': 7, 'USD1MTD156N': 30, 'USD2MTD156N': 60, 'USD3MTD156N': 90,
                         'USD6MTD156N': 180, 'USD12MD156N': 360}
    fred_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=2016-08-27&coed=2021-08-27'

    series = fred_series_names[]


def FRED_interest_rate(duration: int) -> float:
    return 0;


if __name__ == '__main__':
    read_FRED_interest_rates()
