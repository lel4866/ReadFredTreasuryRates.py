"""
function to read FRED (St Louis Fed database) LIBOR interest rates, and a function to return the interpolated rate for
any duration, from 0 days to 360 days, for any given date in the FRED series back to the beginning of the FRED series,
which appears to be 2001-01-01 (some of the individual rate series go back further, this is the earliest for all the
series (rates_global_first_date)

The intention is that this will be used for things like the risk free rate for Black Scholes computations.

read_FRED_interest_rates() reads the FRED LIBOR rates from https://fred.stlouisfed.org/graph/fredgraph.csv
and creates a dictionary whose keys are the duration in days (1, 7, 30, 60, 90, 180, 360),
and whose values are another dictionary whose keys are a date and whose value is the rate in percent

the FRED LIBOR database goes back to 1986 or so, but has some missing days. Missing days are interpolated so there
will be a value for every day of the year from the start of the FRED series forward

After 12/31/2021, LIBOR rates will be replaced by SOFR and this code will be modified appropriately so the caller
doesn't need to know whether the rate comes from the LIBOR database or SOFR database

There is, in addition a function to read the monthly S&P 500 dividend yield from Nasdaq Data Link (formerly Quandl),
primarily for use in Black Scholes Merton option pricing formula that includes dividends. This is a monthly series,
which I interpolate to daily. Not sure if this interpolation is a good idea...

I make no guarantees of any kind for this program...use at your own risk
Lawrence E. Lewis
"""

import numpy as np
import pandas as pd
import datetime
import time
from multiprocessing import Pool

version = '0.0.3'
version_date = '2021-09-11'

# global variables used for risk free rate functions
rates_valid = False
rates_global_first_date: datetime.date # will hold earliest existing date over all the FRED series
rates_global_last_date: datetime.date  # will hold earliest existing date over all the FRED series
rates_duration_list: np.array  # the durations of the available FRED series
rates_interpolation_vector: np.ndarray  # for each day, has index of series to use to interpolate
rates_array: np.ndarray  # the actual rate vector...1 value per day in percent
today_str: str = datetime.date.today().strftime('%Y-%m-%d')

# the main data structure which is filled in by read_risk_free_rates
fred_interest_rates = {1: {'name': 'USDONTD156N', 'data': None}, 7: {'name': 'USD1WKD156N', 'data': None},
                       30: {'name': 'USD1MTD156N', 'data': None}, 60: {'name': 'USD2MTD156N', 'data': None},
                       90: {'name': 'USD3MTD156N', 'data': None}, 180: {'name': 'USD6MTD156N', 'data': None},
                       360: {'name': 'USD12MD156N', 'data': None}}

# global variables used for sp500 dividend yield functions
dividends_valid = False
dividend_array: np.ndarray  # vector containing sp500 dividend yield in percent
dividends_global_first_date: datetime.date  # will hold earliest existing date in dividend_array
dividends_global_last_date: datetime.date


# read risk free rate series from FRED database for durations specified in rates_duration_list (which must match those
# in FRED_interest_rates below) into global rates_array vector, which has a (interpolated) rate for each day
# assumes missing data is encoded as a '.', which was true on 9/9/2021
# raises Exception if earliest date is earlier than 2000-01-01 or later than today
def read_risk_free_rates(earliest_date: datetime = datetime.date(2000, 1, 1)):
    global rates_global_first_date, rates_global_last_date, rates_interpolation_vector, rates_array,\
        fred_interest_rates, rates_duration_list, rates_valid

    start_time = time.time()

    # make sure earliest date is datetime
    if type(earliest_date) is not datetime.datetime:
        raise ValueError(f'ReadFredTreasuryRates.py:read_risk_free_rates: earliest date is not a datetime. It is a {type(earliest_date)}')

    rates_duration_list = np.array(list(fred_interest_rates.keys()))

    if earliest_date < datetime.datetime(2000, 1, 1):
        raise Exception(f'ReadFredTreasuryRates.py:read_risk_free_rates: earliest date ({earliest_date} is before 2000-01-01')
    if earliest_date > datetime.datetime.today():
        raise Exception(f'ReadFredTreasuryRates.py:read_risk_free_rates: earliest date ({earliest_date} is after today ({datetime.date.today()})')

    # read FRED series in parallel
    pool = Pool(processes=len(rates_duration_list))
    for duration, series in fred_interest_rates.items():
        pool.apply_async(read_fred_series, (earliest_date, duration, series), callback=update_rates)
    pool.close()
    pool.join()

    # get latest first date over all series, earliest last date over all series
    rates_global_first_date = earliest_date
    rates_global_last_date = datetime.datetime(3000, 1, 1)
    for series in fred_interest_rates.values():
        rate_df = series['data']
        first_date = rate_df.iloc[0].date
        if first_date > rates_global_first_date:
            rates_global_first_date = first_date.to_pydatetime()
        last_date = rate_df.iloc[-1].date
        if last_date < rates_global_last_date:
            rates_global_last_date = last_date.to_pydatetime()

    print()
    print("Starting date will be: ", rates_global_first_date)
    print("Ending date will be: ", rates_global_last_date)
    print()

    # now create numpy array with 1 row for EVERY day (including weekends and holidays) between global_first_date and
    # today, and 1 column for each FRED series named rate_array
    # once we do this, in order to grab a rate for a specific day and duration, we just compute # of days between
    # requested date and global_first_date, and use that as the index into the numpy rate_array, then use the
    # requested duration to compute interpolated_rate (see detailed explanation below)
    num_rows = ((rates_global_last_date - rates_global_first_date) + datetime.timedelta(1)).days
    num_cols = len(fred_interest_rates)
    rates_array = np.empty((num_rows, num_cols), float)

    # interpolate to replace NaN's (after annoying setup to reconcile different NaN type, uses pandas.interpolate)
    i_col = 0
    for duration, series in fred_interest_rates.items():
        rate_df = series['data']  # contains date and rate
        pandas_first_date = rates_global_first_date
        rate_df = rate_df[rate_df.date >= pandas_first_date]  # remove unneeded rows

        series_array = np.full(num_rows, np.nan)
        for index, row in rate_df.iterrows():
            # since pd has a weird way of handling nan's, we have to set numpy array with nan this way:
            i = (row[0] - rates_global_first_date).days
            rate = row[1]
            series_array[i] = rate if not pd.isnull(rate) else np.nan
            x = 1

        # now use pandas interpolate method to remove NaN's
        # note that number of rows in series_array might be more than in rates_array
        pandas_series = pd.Series(series_array)
        pandas_series.interpolate(inplace=True)

        # now append first num_rows rates for a specific series to overall rates array (which is only num_rows long)
        rates_array[:, i_col] = pandas_series.values[:num_rows]

        # for informational purposes only
        print("duration = ", rates_duration_list[i_col])
        row = rate_df.iloc[0]
        print(row.date.date(), row.rate)
        row = rate_df.iloc[-1]
        print(row.date.date(), row.rate)
        print()

        i_col = i_col + 1

    # free up global memory
    fred_interest_rates = None

    # do sanity check
    if not rate_sanity_check():
        raise Exception("FRED rate data did not pass sanity check. Some rates might be less than 0 or greater than 30")

    # now create interpolation_vector, which contains a column index into the rates_array. To
    # interpolate an interest rate, first you select the row in the rates_array which corresponds to the date that you
    # want the interest rate for, then you compute the rate by interpolating between the two columns that contain the
    # rates that bracket the given duration. The interpolation array contains a value for every possible duration
    # (currently 361; from 0 to 360). a duration of 0 and 1 are treated the same (uses overnight or 1 day rate)
    # For instance, if you want the 9 day rate (say, for an option expiring in 9 days), for 2020-06-15, you would
    # first get the rate row in the rates_array for 2020-06-15, then look in the interpolation_array in the 9th row
    # (index=8), which would be 1, indicating that you should interpolate between the rate in column 1 of the rate
    # row and column 2 of the rate row
    rates_interpolation_vector = np.empty(361, int)
    duration_start = 0
    i_col = -1
    for duration_end in rates_duration_list:
        for i in range(duration_start, duration_end):
            rates_interpolation_vector[i] = i_col
        i_col = i_col + 1
        duration_start = duration_end
    rates_interpolation_vector[0] = 0  # treat a duration of 0 days the same as 1 day
    rates_interpolation_vector[360] = i_col - 1

    # convert some variables from pandas.Timestamp to Python date
    rates_global_first_date = rates_global_first_date.date()
    rates_global_last_date = rates_global_last_date.date()

    rates_valid = True

    end_time = time.time()
    print(f"read_risk_free_rates: Elapsed time is {end_time - start_time} seconds")


# this function is called in a separate process, so can't reference module's global variables
def read_fred_series(earliest_date: datetime, duration: int, series: dict):
    series_name = series['name']
    print("Reading ", series_name)
    fred_url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_name}&cosd=1985-01-01&coed={today_str}'
    rate_df = pd.read_csv(fred_url, header=0, names=['date', 'rate'], parse_dates=[0], na_values={'rate': '.'},
                          keep_default_na=False, engine='c')

    # remove dates before specified earliest_date (default of 1/1/2000)
    pandas_timestamp = pd.to_datetime(earliest_date)  # convert from Python datetime
    rate_df = rate_df[rate_df['date'] >= pandas_timestamp]
    return duration, rate_df


# callback from Pool.apply_async call to read_fred_series function above
# saves the rates that were read back into the global FRED_interest_rates dictionary
def update_rates(result: (int, pd.DataFrame)):
    duration = result[0]
    rate_df = result[1]
    series = fred_interest_rates[duration]
    series['data'] = rate_df


# make sure rates_df has reasonable values
def rate_sanity_check() -> bool:
    # make sure risk free rates greater than 0 and less than 30
    # make sure the change between any days is less than 0.2
    passed = True
    num_cols = rates_array.shape[1]
    change_array = rates_array[0]
    for index_tuple, rate in np.ndenumerate(rates_array):
        if rate <= 0 or rate >= 30:
            date = rates_global_first_date + pd.DateOffset(index_tuple[0])
            duration = rates_duration_list[index_tuple[1]]
            print(f'ReadFredTreasuryRates.py:risk_free_rate: rate for duration: {duration}, date: {date}, is not reasonable: {rate}')
            passed = False
    return passed


# returns risk free rate for requested date and duration for use in Black Scholes formula
# raises Exception if requested date is before earliest date available or after latest date available
def risk_free_rate(requested_date: datetime, duration: int) -> float:
    if not rates_valid:
        raise Exception('ReadFredTreasuryRates.py:risk_free_rate: rate data not available. Did you call read_risk_free_rates?')
    date_index = (requested_date - rates_global_first_date).days
    if date_index < 0:
        raise ValueError(f'ReadFredTreasuryRates.py:risk_free_rate: requested date ({requested_date}) is before earliest available date in series ({rates_global_first_date})')
    if date_index >= len(rates_array):
        if requested_date > datetime.date.today():
            raise ValueError(f'ReadFredTreasuryRates.py:risk_free_rate: requested date ({requested_date}) is after latest available date in series ({rates_global_last_date})')
        else:
            date_index = len(rates_array) - 1
    if duration < 0:
        raise ValueError(f'ReadFredTreasuryRates.py:risk_free_rate: requested duration ({duration} is less than 0. Must be between 0 and 360')
    if duration > 360:
        raise ValueError(f'ReadFredTreasuryRates.py:risk_free_rate: requested duration ({duration} is greater than 360. Must be between 0 and 360')

    # treat a duration of 0 as 1
    if duration == 0:
        duration = 1

    row = rates_array[date_index]
    column_index = rates_interpolation_vector[duration]
    starting_duration = rates_duration_list[column_index]
    ending_duration = rates_duration_list[column_index + 1]
    starting_val = row[column_index]
    ending_val = row[column_index+1]
    ratio = (duration - starting_duration) / (ending_duration - starting_duration)
    libor_rate = starting_val + ratio*(ending_val - starting_val)

    # convert LIBOR rate to BS convention per Adam Speight in Trading Dominion Mastermind session comment for 2021-07-07
    bs_rate = 360.0/duration * np.log(1.0 + libor_rate*duration/365.0)

    return bs_rate


# read the sp500 dividend yield (in percent) from Nasdaq Data Link (formerly Quandl) into global dividend array
# which has an entry for every day, starting at beginning date
# raises Exception if there are any NaNs, if earliest date is later than today
def read_sp500_dividend_yield(earliest_date: datetime.date = datetime.date(2000, 1, 1)):
    global dividends_global_first_date, dividends_global_last_date, dividend_array, dividends_valid

    # note that series is returned in descending date order (today is the first row)
    url = 'https://data.nasdaq.com/api/v3/datasets/MULTPL/SP500_DIV_YIELD_MONTH.csv?api_key=r1LNaRv-SYEyP9iY8BKj'
    dividend_df = pd.read_csv(url, header=0, names=['date', 'dividend'], parse_dates=[0], engine='c')

    # remove dates before specified earliest_date (default of 1/1/2000)
    pandas_earliest_date = pd.to_datetime(earliest_date)
    dividend_df = dividend_df[dividend_df['date'] >= pandas_earliest_date]

    if dividend_df.dividend.isnull().any():
        raise ValueError('ReadFredTreasuryRates.py:read_sp500_dividend_yield: some values are NaN')

    # create a numpy vector with a slot for every day (including weekends and holidays)
    dates = dividend_df['date']
    last_date = dates.iloc[0]
    dividends_global_first_date = dates.iloc[-1]
    num_rows = ((last_date - dividends_global_first_date) + datetime.timedelta(1)).days
    dividend_array = np.full(num_rows, np.nan)
    for index, row in dividend_df.iterrows():
        date_index = (row.date - dividends_global_first_date).days
        dividend_array[date_index] = row.dividend

    # interpolate to fill NaN's. I know this is a little dicey...but interpolated values are as good as any others
    dividend_series = pd.Series(dividend_array)
    dividend_series.interpolate(inplace=True)
    dividend_array = dividend_series.to_numpy()
    dividends_global_first_date = dividends_global_first_date.to_pydatetime().date()  # convert from pandas Timestamp to Python datetime
    dividends_global_last_date = last_date.to_pydatetime().date()

    dividends_valid = True


def sp500_dividend_yield(requested_date: datetime) -> float:
    if not dividends_valid:
        raise Exception('ReadFredTreasuryRates.py:sp500_dividend_yield: dividend data not available. Did you call read_sp500_dividend_yield?')
    date_index = (requested_date - dividends_global_first_date).days
    if date_index < 0:
        raise ValueError(f'ReadFredTreasuryRates.py:sp500_dividend_yield: requested date ({requested_date}) is before earliest available date in series ({dividends_global_first_date})')
    if date_index >= len(dividend_array):
        if requested_date > datetime.date.today():
            raise ValueError(f'ReadFredTreasuryRates.py:sp500_dividend_yield: requested date ({requested_date}) is after latest available date in series ({dividends_global_last_date})')
        else:
            date_index = len(dividend_array) - 1

    return dividend_array[date_index]


# do not run this test code if this module is accessed via the "import module" mechanism
if __name__ == '__main__':
    read_risk_free_rates(datetime.datetime(2020, 6, 1))  # parameter is earliest date that we want series for

    # tests of risk free rate
    date0 = datetime.date(2020, 6, 15)
    rate1 = risk_free_rate(date0, 1)
    rate9 = risk_free_rate(date0, 9)
    rate47 = risk_free_rate(date0, 47)
    rate200 = risk_free_rate(date0, 200)
    rate360 = risk_free_rate(date0, 360)
    date0 = datetime.datetime(2020, 6, 15).date()
    rate200t = risk_free_rate(datetime.date.today(), 200)

    rate0 = risk_free_rate(date0, 0)
    assert rate0 == rate1, f"0 day rate ({rate0}) does not match 1 day rate ({rate0})"

    exception_caught = False
    try:
        rate6 = risk_free_rate(date0, -1)
    except ValueError:
        exception_caught = True
    assert exception_caught, "Error: no exception caught when requesting rate for duration of -1"

    exception_caught = False
    try:
        rate7 = risk_free_rate(date0, 500)
    except ValueError:
        exception_caught = True
    assert exception_caught, "Error: no exception caught when requesting rate for duration of 500 (greater than 360)"

    date0 = datetime.date(2001, 1, 1)
    exception_caught = False
    try:
        rate9 = risk_free_rate(date0, 9)
    except ValueError:
        exception_caught = True
    assert exception_caught, f"Error: no exception caught when requesting rate for date before first available date ({rates_global_first_date}))"

    read_sp500_dividend_yield(datetime.date(2000, 1, 1))    # parameter is earliest date that we want series for

    # tests of dividend yield
    yield1 = sp500_dividend_yield(datetime.date(2016, 6, 1))
    yield2 = sp500_dividend_yield(datetime.date.today())

    exception_caught = False
    try:
        yield3 = sp500_dividend_yield(datetime.date(1999, 6, 1))
    except ValueError:
        exception_caught = True
    assert exception_caught, f"Error: no exception caught when requesting dividend yield for date before first available date ({dividends_global_first_date})"

    exception_caught = False
    try:
        yield4 = sp500_dividend_yield(datetime.date(2030, 6, 1))
    except ValueError:
        exception_caught = True
    assert exception_caught, f"Error: no exception caught when requesting dividend yield for date after last available date ({dividends_global_last_date})"

    x = 1  # for debug
