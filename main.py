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

Note: if you want to run this program without the parameter checking asserts, use: python -cO main.py

I make no guarantees of any kind for this program...use at your own risk
Lawrence E. Lewis
"""
import numpy
import numpy as np
import pandas as pd
import datetime
import time
from multiprocessing import Pool

version = '0.0.3'
version_date = '2021-09-11'

# global variables used for risk free rate functions
rates_valid = False
rates_global_first_date: datetime.datetime = None  # will hold earliest existing date over all the FRED series
rates_global_last_date: datetime.datetime = None  # will hold earliest existing date over all the FRED series
rates_duration_list: np.array = None  # the durations of the available FRED series
rates_interpolation_vector: np.ndarray = None  # for each day, has index of series to use to interpolate

# the main array that will hold the rates for all durations from 1 to 360 and all dates requested that are available
# if you want, for better speed, you can use this array directly instead of calling the risk_free_rate function
rates_array: np.ndarray = None  # the actual rate vector...1 value per day in percent

# the main data structure which is filled in by read_risk_free_rates
# this will be deleted (set to None) when read_risk_free_rates returns
fred_interest_rates = {1: {'name': 'USDONTD156N', 'data': None}, 7: {'name': 'USD1WKD156N', 'data': None},
                       30: {'name': 'USD1MTD156N', 'data': None}, 60: {'name': 'USD2MTD156N', 'data': None},
                       90: {'name': 'USD3MTD156N', 'data': None}, 180: {'name': 'USD6MTD156N', 'data': None},
                       360: {'name': 'USD12MD156N', 'data': None}}

# global variables used for sp500 dividend yield functions
dividends_valid = False
dividend_array: np.ndarray = None  # vector containing sp500 dividend yield in percent
dividends_global_first_date: datetime.datetime = None  # will hold earliest existing date in dividend_array
dividends_global_last_date: datetime.datetime = None


# read risk free rate series from FRED database for durations specified in rates_duration_list (which must match those
# in FRED_interest_rates below) into global rates_array vector, which has a (interpolated) rate for each day
# assumes missing data is encoded as a '.', which was true on 9/9/2021
# asserts if earliest date is earlier than 2000-01-01 or later than today
# raises ValueError exception if actual data read does not pass basic sanity checks
def read_risk_free_rates(earliest_date: datetime.datetime = datetime.datetime(2000, 1, 1)):
    global rates_global_first_date, rates_global_last_date, rates_interpolation_vector, rates_array,\
        fred_interest_rates, rates_duration_list, rates_valid

    # make sure arguments are valid
    assert type(earliest_date) is datetime.datetime,\
        f'ReadFredTreasuryRates.py:read_risk_free_rates: earliest date is not a datetime. It is a {type(earliest_date)}'
    assert earliest_date >= datetime.datetime(2000, 1, 1),\
        f'ReadFredTreasuryRates.py:read_risk_free_rates: earliest date ({earliest_date} is before 2000-01-01'
    assert earliest_date <= datetime.datetime.today(),\
        f'ReadFredTreasuryRates.py:read_risk_free_rates: earliest date ({earliest_date} is after today ({datetime.date.today()})'

    # read FRED series in parallel
    rates_duration_list = np.array(list(fred_interest_rates.keys()))
    pool = Pool(processes=len(rates_duration_list))
    for duration, series in fred_interest_rates.items():
        pool.apply_async(read_fred_series, (earliest_date, duration, series), callback=update_rates)
    pool.close()
    pool.join()

    # scan all series (all durations), and get latest first date , earliest last date
    # that is, set rates_global_first_date, rates_global_last_date
    get_first_and_last_dates(earliest_date)

    # now create rates_array from fred_interest_rates dataframe, which is a numpy array with 1 row for EVERY day
    # (including weekends and holidays) between global_first_date and today, and 1 column for every valid duration
    # (0 to 360).
    create_rates_array_from_df()

    # free up global memory
    fred_interest_rates = None

    # do a piecewise linear interpolation of nan's along each column that was actually in the fred dataframe
    # that is, columns 1, 7, 30, 60, 90, 180, 360. When we are done, rates_array will have no nan's
    interpolate_rates_array()

    # do sanity check
    if not rate_sanity_check():
        raise ValueError("FRED rate data did not pass sanity check. Some rates might be less than 0 or greater than 30")

    rates_valid = True


# this function is called in a separate process, so can't reference module's global variables
def read_fred_series(earliest_date: datetime, duration: int, series: dict):
    series_name = series['name']
    print("Reading ", series_name)
    today_str: str = datetime.date.today().strftime('%Y-%m-%d')
    fred_url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_name}&cosd=1985-01-01&coed={today_str}'
    rate_df = pd.read_csv(fred_url, header=0, names=['date', 'rate'], parse_dates=['date'], na_values={'rate': '.'},
                          keep_default_na=False, engine='c')

    # remove dates before specified earliest_date (default of 1/1/2000)
    rate_df = rate_df[rate_df['date'] >= earliest_date]
    return duration, rate_df


# callback from Pool.apply_async call to read_fred_series function above
# saves the rates that were read back into the global FRED_interest_rates dictionary
def update_rates(result: (int, pd.DataFrame)):
    duration = result[0]
    rate_df = result[1]
    series = fred_interest_rates[duration]
    series['data'] = rate_df


# get latest first date over all series, earliest last date over all series
def get_first_and_last_dates(earliest_date: datetime):
    global rates_global_first_date, rates_global_last_date

    rates_global_first_date = earliest_date
    rates_global_last_date = datetime.datetime(3000, 1, 1)
    for series in fred_interest_rates.values():
        rate_df = series['data']
        first_date = rate_df.iloc[0].date  # note: accessing 'date' column via attribute
        if first_date > rates_global_first_date:
            rates_global_first_date = first_date.to_pydatetime()
        last_date = rate_df.iloc[-1].date  # note: accessing 'date' column via attribute
        if last_date < rates_global_last_date:
            rates_global_last_date = last_date.to_pydatetime()

    print()
    print("Starting date for risk free rate table will be: ", rates_global_first_date.date())
    print("Ending date for risk free rate table will be: ", rates_global_last_date.date())
    print()


# create rates_array, which will (eventually) contain an interest rate for every day from rates_global_first_date to
# rates_global_last_date (number of rows) and every duration from 0 to 360 (number of columns)
# fills in rates_array from fred_interest_rates dataframe values for durations 1, 7, 30, 60, 90, 180, 360
# note that each of those columns will have nan's (because fred_interest_rates dataframe does not have values
# for weekends or holidays. Also, the remaining columns (NOT 1, 7, 30, etc) will be all nan's
def create_rates_array_from_df():
    global rates_array

    # create numpy rates_array full of nan's for every valid day and duration (0 to 360),
    # unlike fred_interest_rates dataframe, which only has data for weekdays and 7 different durations
    num_rows = ((rates_global_last_date - rates_global_first_date) + datetime.timedelta(1)).days
    rates_array = np.full((num_rows, 361), numpy.nan, float)  # 1 row for each date, 1 column for each possible duration

    pd_global_first_date = pd.to_datetime(rates_global_first_date)  # stupid conversion necessity
    pd_global_last_date = pd.to_datetime(rates_global_last_date)  # stupid conversion necessity
    for duration, data in fred_interest_rates.items():
        # remove unneeded rows from rate_df
        rate_df = data['data']  # contains date and rate columns for a given duration
        rate_df = rate_df[rate_df.date >= pd_global_first_date]
        rate_df = rate_df[rate_df.date <= pd_global_last_date]

        # now place each rate in rate_df dataframe in proper place in ndarray based on date (row) and duration (column)
        for index, row in rate_df.iterrows():
            # since pd has a weird way of handling nan's, we have to set numpy array with nan this way:
            i = (row[0] - pd_global_first_date).days
            rate = row[1]
            rates_array[i, duration] = rate if not pd.isnull(rate) else np.nan


# do a piecewise linear interpolation of nan's along each column that was actually in the fred dataframe
# that is, columns 1, 7, 30, 60, 90, 180, 360. When we are done, those columns will have no nan's. Then
# linearly interpolate each row to fill in for the other columns
def interpolate_rates_array():
    num_rows = rates_array.shape[0]
    num_rows_m1 = num_rows - 1

    for i_col in rates_duration_list:
        col = rates_array[:,i_col]
        # find index of next non-nan
        index_of_prev_non_nan = -1
        nan_found = False
        for index, rate in np.ndenumerate(col):
            idx = index[0]  # because index is actually a tuple
            if np.isnan(rate):
                nan_found = True
                # if last row, just propagate last non-nan value forward
                if idx == num_rows_m1:  # num_rows_m1 = num_rows - 1
                    col[index_of_prev_non_nan+1:] = col[index_of_prev_non_nan]
                break
            if not nan_found:
                index_of_prev_non_nan = idx
                continue
            # we found a non-nan after finding an nan
            nan_found = False
            value = col[idx]
            # interpolate between non-nan just found and previous non-nan
            if index_of_prev_non_nan == -1:
                # this is first non-nan in column...just fill backwards
                col[:idx] = value
            else:
                # interpolate between index_of_prev_non_nan to idx
                idx_range = idx - index_of_prev_non_nan
                interpolated_value = col[index_of_prev_non_nan]  # starting value
                value_range = value - interpolated_value
                value_increment = value_range/idx_range
                for nan_index in range(index_of_prev_non_nan+1, idx):
                    interpolated_value = interpolated_value + value_increment
                    col[nan_index] = interpolated_value

            index_of_prev_non_nan = idx

    # now linearly interpolate across each row to fill in the rest of the colmnns
    for row in rates_array:
        for index, col1 in  np.ndenumerate(rates_duration_list[0:-1]):
            col2 = rates_duration_list[index[0] + 1]
            interpolate(row, col1, col2)  # interpolate between rates in row[col1] and row[col2]

    # now convert LIBOR rate (based on 360 days/year) to rate used in Balck Scholes (based on 365 days/year)
    for duration in range(1,361):
        col = rates_array[:,duration]
        rates_array[:,duration] = black_scholes_rate(duration, col)


# interpolates values between vector[col1] and vector[col2]
# vector[col1] and vector[col2] MUST NOT be nan's.
def interpolate(vector: np.ndarray, col1: int, col2: int):
    idx_range = col2 - col1
    interpolated_value = vector[col1]  # starting value
    value_range = vector[col2] - interpolated_value
    value_increment = value_range / idx_range
    for i in range(col1 + 1, col2):
        interpolated_value = interpolated_value + value_increment
        vector[i] = interpolated_value


def black_scholes_rate(duration: int, libor_rate: float) -> float:
    return 360.0/duration * np.log(1.0 + libor_rate*duration/365.0)


# make sure rates_df has reasonable values
def rate_sanity_check() -> bool:
    # make sure risk free rates greater than 0 and less than 30
    # make sure the change between any days is less than 0.2
    passed = True
    prior_rates = rates_array[0]
    for index_tuple, rate in np.ndenumerate(rates_array):
        duration_index = index_tuple[1]
        # column 0 of rates_array is nan (can't have a duration of 0)
        if duration_index == 0:
            continue
        if rate <= 0 or rate >= 30:
            date = rates_global_first_date + pd.DateOffset(index_tuple[0])
            duration = rates_duration_list[duration_index]
            print(f'ReadFredTreasuryRates.py:rate_sanity_check: rate for duration: {duration}, date: {date}, is not reasonable: {rate}')
            passed = False
            continue

        change_in_rate = abs(rate - prior_rates[duration_index])
        prior_rates[duration_index] = rate
        if change_in_rate > 0.5:
            print(f'ReadFredTreasuryRates.py:rate_sanity_check: change in rate for duration: {duration}, date: {date}, is not reasonable: {change_in_rate}')
            passed = False

    return passed


# returns risk free rate for requested date and duration for use in Black Scholes formula
# asserts if requested date is before earliest date available or after latest date available
def risk_free_rate(requested_date: datetime, duration: int) -> float:
    # check arguments to make sure they are valid
    assert rates_valid,\
        'ReadFredTreasuryRates.py:risk_free_rate: rate data not available. Did you call read_risk_free_rates?'
    assert type(requested_date) is datetime.datetime, f'ReadFredTreasuryRates.py:risk_free_rate: type of requested_date must be datetime, not {type(requested_date)}'
    date_index = (requested_date - rates_global_first_date).days
    assert date_index >= 0,\
        f'ReadFredTreasuryRates.py:risk_free_rate: requested date ({requested_date}) is before earliest available date in series ({rates_global_first_date})'
    if date_index >= len(rates_array):
        assert requested_date <= datetime.datetime.today(),\
            f'ReadFredTreasuryRates.py:risk_free_rate: requested date ({requested_date}) is after latest available date in series ({rates_global_last_date})'
        date_index = len(rates_array) - 1
    assert duration > 0,\
        f'ReadFredTreasuryRates.py:risk_free_rate: requested duration ({duration} is less than or equal to 0. Must be between 1 and 360'
    assert duration <= 360,\
        f'ReadFredTreasuryRates.py:risk_free_rate: requested duration ({duration} is greater than 360. Must be between 0 and 360'
    return rates_array[date_index, duration]


# read the sp500 dividend yield (in percent) from Nasdaq Data Link (formerly Quandl) into global dividend array
# which has an entry for every day, starting at beginning date
# raises Exception if there are any NaNs, if earliest date is later than today
def read_sp500_dividend_yield(earliest_date: datetime.datetime = datetime.date(2000, 1, 1)):
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
    dividends_global_first_date = dividends_global_first_date.to_pydatetime()  # convert from pandas Timestamp to Python datetime
    dividends_global_last_date = last_date.to_pydatetime()

    # do sanity check
    if not dividend_sanity_check():
        raise ValueError("Nasdaq DataLink S&P500 dividend yield data did not pass sanity check.")

    dividends_valid = True


def dividend_sanity_check() -> bool:
    # make sure dividend yield is greater than 0% and less than 5%
    # make sure the change between any days is less than 0.2%
    passed = True
    prior_dividend = dividend_array[0]
    for index, dividend in np.ndenumerate(dividend_array):
        if dividend <= 0 or dividend >= 5:
            date = dividends_global_first_date + datetime.timedelta(days=1)
            print(f'ReadFredTreasuryRates.py:dividend_sanity_check: rate for {date}, is not reasonable: {dividend}')
            passed = False
            continue
        dividend_change = abs(dividend - prior_dividend)
        prior_dividend = dividend
        if dividend_change >= 1:
            date = dividends_global_first_date + datetime.timedelta(days=1)
            print(f'ReadFredTreasuryRates.py:dividend_sanity_check: change in dividend for {date}, is not reasonable: {dividend_change}')
            passed = False

    return passed


def sp500_dividend_yield(requested_date: datetime.datetime) -> float:
    assert dividends_valid,\
        'ReadFredTreasuryRates.py:sp500_dividend_yield: dividend data not available. Did you call read_sp500_dividend_yield?'
    date_index = (requested_date - dividends_global_first_date).days
    assert date_index >= 0,\
        f'ReadFredTreasuryRates.py:sp500_dividend_yield: requested date ({requested_date}) is before earliest available date in series ({dividends_global_first_date})'
    if date_index >= len(dividend_array):
        assert requested_date <= datetime.datetime.today(),\
            f'ReadFredTreasuryRates.py:sp500_dividend_yield: requested date ({requested_date}) is after latest available date in series ({dividends_global_last_date})'
        date_index = len(dividend_array) - 1

    return dividend_array[date_index]


# do not run this test code if this module is accessed via the "import module" mechanism
if __name__ == '__main__':
    start_time = time.time()
    read_risk_free_rates(datetime.datetime(2020, 6, 1))  # parameter is earliest date that we want series for
    end_time = time.time()
    print(f"read_risk_free_rates: Elapsed time was {round(end_time - start_time,3)} seconds")

    #
    # tests of risk free rate
    #
    date0 = datetime.datetime(2020, 6, 15)
    rate1 = risk_free_rate(date0, 1)
    rate9 = risk_free_rate(date0, 9)
    rate47 = risk_free_rate(date0, 47)
    rate200 = risk_free_rate(date0, 200)
    rate360 = risk_free_rate(date0, 360)
    rate200t = risk_free_rate(datetime.datetime.today(), 200)

    exception_caught = False
    try:
        rate0 = risk_free_rate(date0, 0)
    except AssertionError:
        exception_caught = True
    assert exception_caught, "Error: no exception caught when requesting rate for duration of 0"

    exception_caught = False
    try:
        rate7 = risk_free_rate(date0, 500)
    except AssertionError:
        exception_caught = True
    assert exception_caught, "Error: no exception caught when requesting rate for duration of 500 (greater than 360)"

    date0 = datetime.datetime(2001, 1, 1)
    exception_caught = False
    try:
        rate9 = risk_free_rate(date0, 9)
    except AssertionError:
        exception_caught = True
    assert exception_caught, f"Error: no exception caught when requesting rate for date before first available date ({rates_global_first_date}))"

    #
    # tests of dividend yield
    #
    start_time = time.time()
    read_sp500_dividend_yield(datetime.datetime(2000, 1, 1))    # parameter is earliest date that we want series for
    end_time = time.time()
    print(f"read_sp500_dividend_yield: Elapsed time was {round(end_time - start_time,3)} seconds")

    yield1 = sp500_dividend_yield(datetime.datetime(2016, 6, 1))
    yield2 = sp500_dividend_yield(datetime.datetime.today())

    exception_caught = False
    try:
        yield3 = sp500_dividend_yield(datetime.datetime(1999, 6, 1))
    except AssertionError:
        exception_caught = True
    assert exception_caught, f"Error: no exception caught when requesting dividend yield for date before first available date ({dividends_global_first_date})"

    exception_caught = False
    try:
        yield4 = sp500_dividend_yield(datetime.datetime(2030, 6, 1))
    except AssertionError:
        exception_caught = True
    assert exception_caught, f"Error: no exception caught when requesting dividend yield for date after last available date ({dividends_global_last_date})"

    x = 1  # for debug
