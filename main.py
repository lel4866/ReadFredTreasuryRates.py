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

import numpy
import pandas
import datetime
import time

version = '0.0.2'
version_date = '2021-09-09'

# global variables used for risk free rate functions
rates_global_first_date = numpy.datetime64('1980-01-01')  # will hold earliest existing date over all the FRED series
rates_global_last_date = None  # will hold earliest existing date over all the FRED series
rates_duration_list = numpy.array([1, 7, 30, 60, 90, 180, 360])  # the durations of the available FRED series
rates_interpolation_vector: numpy.ndarray  # for each day, has index of series to use to interpolate
rates_array: numpy.ndarray  # the actual rate vector...1 value per day in percent

# global variables used for sp500 dividend yield functions
dividend_array: numpy.ndarray  # vector containing sp500 dividen yield in percent
dividends_global_first_date= numpy.datetime64('1980-01-01')  # will hold earliest existing date in dividend_array
dividends_global_last_date: numpy.datetime64

# read risk free rate series from FRED database for durations specified in rates_duration_list (which must match those
# in FRED_interest_rates below) into global rates_array vector, which has a (interpolated) rate for each day
# assumes missing data is encoded as a '.', which was true on 9/9/2021
# raises Exception if earliest date is earlier than 2000-01-01 or later than today
def read_risk_free_rates(earliest_date: datetime=datetime.date(2000, 1, 1)):
    global rates_global_first_date, rates_global_last_date, rates_interpolation_vector, rates_array

    FRED_interest_rates = {1: {'name': 'USDONTD156N'}, 7: {'name': 'USD1WKD156N'}, 30: {'name': 'USD1MTD156N'},
                           60: {'name': 'USD2MTD156N'}, 90: {'name': 'USD3MTD156N'}, 180: {'name': 'USD6MTD156N'},
                           360: {'name': 'USD12MD156N'}}

    start_time = time.time()

    if (earliest_date < datetime.date(2000, 1, 1)):
        raise Exception(f'ReadFredTresuryRates.py:read_risk_free_rates: earliest date ({earliest_date} is before 2000-01-01')
    if (earliest_date > datetime.date.today()):
        raise Exception(f'ReadFredTresuryRates.py:read_risk_free_rates: earliest date ({earliest_date} is after today (datetime.date.today())')

    #rates_global_first_date = pandas.Timestamp('1986-01-01')
    rates_global_first_date = datetime.datetime(1986, 1, 1)
    #today = pandas.Timestamp.today()
    today = datetime.date.today()
    today_str = today.strftime('%Y-%m-%d')
    for duration, series in FRED_interest_rates.items():
        series_name = series['name']
        print("Reading ", series_name)
        FRED_url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_name}&cosd=1985-01-01&coed={today_str}'
        rate_df = pandas.read_csv(FRED_url, header=0, names=['date', 'rate'], parse_dates=[0], na_values={'rate': '.'},
                                  keep_default_na=False, engine='c')

        # remove dates before specified earliest_date (default of 1/1/2000)
        pandas_timestamp = pandas.to_datetime(earliest_date)  # convert from Python datetime
        rate_df = rate_df[rate_df['date'] >= pandas_timestamp]
        series['data'] = rate_df

        # keep track of overall earliest date of all series read
        first_date = rate_df.iloc[0].date
        if first_date > rates_global_first_date:
            rates_global_first_date = first_date

        # keep track of overall latest date of all series read...so this means the earliest last date
        last_date = rate_df.iloc[-1].date
        if rates_global_last_date == None:
            rates_global_last_date = last_date
        elif last_date < rates_global_last_date:
            rates_global_last_date = last_date

    print()
    print("Starting date will be: ", rates_global_first_date)
    print("Ending date will be: ", rates_global_last_date)
    print()

    # now create numpy array with 1 row for EVERY day (including weekends and holidays) between global_first_date and
    # today, and 1 column for each FRED series named rate_array
    # once we do this, in order to grab a rate for a specific day and duration, we just compute # of days between
    # requested date and global_first_date, and use that as the index into the numpy rate_array, then use the
    # requested duration to compute interpolated_rate (see detailed explanation below)
    numrows = ((rates_global_last_date - rates_global_first_date) + datetime.timedelta(1)).days
    numcols = len(FRED_interest_rates)
    rates_array = numpy.empty((numrows, numcols), float)

    # interpolate to replace NaN's (after annoying setup to reconcile different NaN type, uses pandas.interpolate)
    icol = 0
    for duration, series in FRED_interest_rates.items():
        rate_df = series['data']  # contains date and rate
        rate_df = rate_df[rate_df.date >= rates_global_first_date] # remove unneeded rows

        series_array = numpy.empty(numrows, float)
        for index, row in rate_df.iterrows():
            # since pandas has a weird way of handling nan's, we have to set numpy array with nan this way:
            i = (row[0] - rates_global_first_date).days
            rate = row[1]
            series_array[i] = rate if not pandas.isnull(rate) else numpy.nan

        # now use pandas interpolate method to remove NaN's
        # note that number of rows in series_array might be more than in rates_array
        pandas_series = pandas.Series(series_array)
        pandas_series.interpolate(inplace=True)

        # now append first numrows rates for a specific series to overall rates array (which is only numrows long)
        rates_array[:, icol] = pandas_series.values[:numrows]

        # for informational purposes only
        print("duration = ", rates_duration_list[icol])
        row = rate_df.iloc[0]
        print(row.date.date(), row.rate)
        last_row = len(rate_df.index) - 1
        row = rate_df.iloc[last_row - 1]
        print(row.date.date(), row.rate)
        print()

        icol = icol + 1


    # now create interpolation_vector, which contains a column index into the rates_array. To
    # interpolate an interest rate, first you select the row in the rates_array which corresponds to the date that you
    # want the interest rate for, then you compute the rate by interpolating between the two columns that contain the
    # rates that bracket the given duration. The interpolation array contains a value for every possible duration
    # (currently 361; from 0 to 360). a duration of 0 and 1 are treated the same (uses overnight or 1 day rate)
    # For instance, if you want the 9 day rate (say, for an option expiring in 9 days), for 2020-06-15, you would
    # first get the rate row in the rates_array for 2020-06-15, then look in the interpolation_array in the 9th row
    # (index=8), which would be 1, indicating that you should interpolate between the rate in column 1 of the rate
    # row and column 2 of the rate row
    rates_interpolation_vector = numpy.empty(361, int)
    duration_start = 0
    icol = -1
    for duration_end in rates_duration_list:
        for i in range(duration_start, duration_end):
            rates_interpolation_vector[i] = icol
        icol = icol + 1
        duration_start = duration_end
    rates_interpolation_vector[0] = 0  # treat a duration of 0 days the same as 1 day
    rates_interpolation_vector[360] = icol - 1

    # convert some variables from pandas.Timestamp to Python datetime
    rates_global_first_date = rates_global_first_date.date()
    rates_global_last_date = rates_global_last_date.date()

    end_time = time.time()
    print(f"read_risk_free_rates: Elapsed time is {end_time - start_time} seconds")


# returns risk free rate for requested date and duration for use in Black Scholes formula
# raises Exception if requested date is before earliest date available or after latest date available
def risk_free_rate(requested_date: datetime.date, duration: int) -> float:
    #xx = pandas.Timestamp(requested_date) - rates_global_first_date
    date_index = (requested_date - rates_global_first_date).days
    if (date_index < 0):
        raise ValueError(f'ReadFredTresuryRates.py:risk_free_rate: requested date ({requested_date}) is before earliest available date in series ({rates_global_first_date})')
    if (date_index >= len(rates_array)):
        if requested_date > datetime.date.today():
            raise ValueError(f'ReadFredTresuryRates.py:risk_free_rate: requested date ({requested_date}) is after latest available date in series ({rates_global_last_date})')
        else:
            date_index = len(rates_array) - 1
    if (duration < 0):
        raise ValueError(f'ReadFredTresuryRates.py:risk_free_rate: requested duration ({duration} is less than 0. Must be between 0 and 360')
    if (duration > 360):
        raise ValueError(f'ReadFredTresuryRates.py:risk_free_rate: requested duration ({duration} is greater than 360. Must be between 0 and 360')

    # treate a duration of 0 as 1
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
    bs_rate = 360.0/duration * numpy.log(1.0 + libor_rate*duration/365.0)

    return bs_rate


# read the sp500 dividend yield (in percent) from Nasdaq Data Link (formerly Quandl) into global dividend array
# which has an entry for every day, starting at beginning date
# raises Exception if there are any NaNs, if earliest date is later than today
def read_sp500_dividend_yield(earliest_date: datetime=datetime.date(2000, 1, 1)):
    global dividends_global_first_date, dividends_global_last_date, dividend_array

    # note that series is returned in descending date order (today is the first row)
    url = 'https://data.nasdaq.com/api/v3/datasets/MULTPL/SP500_DIV_YIELD_MONTH.csv?api_key=r1LNaRv-SYEyP9iY8BKj'
    dividend_df = pandas.read_csv(url, header=0, names=['date', 'dividend'], parse_dates=[0], engine='c')

    # remove dates before specified earliest_date (default of 1/1/2000)
    pandas_earliest_date = pandas.to_datetime(earliest_date)
    dividend_df = dividend_df[dividend_df['date'] >= pandas_earliest_date]

    if dividend_df.dividend.isnull().any():
        raise Exception('ReadFredTresuryRates.py:read_sp500_dividend_yield: some values are NaN')

    # create a numpy vector with a slot for every day (including weekends and holidays)
    dates = dividend_df['date']
    last_date = dates.iloc[0]
    dividends_global_first_date = dates.iloc[-1]
    num_rows = ((last_date - dividends_global_first_date) + datetime.timedelta(1)).days
    dividend_array = numpy.full(num_rows, numpy.nan)
    for index, row in dividend_df.iterrows():
        date_index = (row.date - dividends_global_first_date).days
        dividend_array[date_index] = row.dividend

    # interpolate to fill NaN's. I know this is a little dicey...but interpolated values are as good as any others
    dividend_series = pandas.Series(dividend_array)
    dividend_series.interpolate(inplace=True)
    dividend_array = dividend_series.to_numpy()
    dividends_global_first_date = dividends_global_first_date.date()  # convert from pandas Timestamp to Python datetime
    dividends_global_last_date = last_date.date()
    x = 1


def sp500_dividend_yield(requested_date: datetime) -> float:
    date_index = (requested_date - dividends_global_first_date).days
    if (date_index < 0):
        raise ValueError(f'ReadFredTresuryRates.py:sp500_dividend_yield: requested date ({requested_date.date}) is before earliest available date in series ({dividends_global_first_date.date})')
    if (date_index >= len(dividend_array)):
        if requested_date > datetime.date.today():
            raise ValueError(f'ReadFredTresuryRates.py:sp500_dividend_yield: requested date ({requested_date.date}) is after latest available date in series ({dividends_global_last_date.date})')
        else:
            date_index = len(dividend_array) - 1

    return dividend_array[date_index]


if __name__ == '__main__':
    read_risk_free_rates(datetime.date(2000, 1, 1))  # parameter is earliest date that we want series for

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
    except:
        exception_caught = True
    assert exception_caught, "Error: no exception caught when requesting rate for duration of -1"

    exception_caught = False
    try:
        rate7 = risk_free_rate(date0, 500)
    except:
        exception_caught = True
    assert exception_caught, "Error: no exception caught when requesting rate for duration of 500 (greater than 360)"

    date0 = datetime.date(2001, 1, 1)
    exception_caught = False
    try:
        rate9 = risk_free_rate(date0, 9)
    except:
        exception_caught = True
    assert exception_caught, f"Error: no exception caught when requesting rate for date before first available date ({rates_global_first_date}))"

    read_sp500_dividend_yield(datetime.date(2000, 1, 1))    # parameter is earliest date that we want series for

    # tests of dividend yield
    yield1 = sp500_dividend_yield(datetime.date(2016, 6, 1))
    yield2 = sp500_dividend_yield(datetime.date.today())

    exception_caught = False
    try:
        yield3 = sp500_dividend_yield(datetime.date(1999, 6, 1))
    except:
        exception_caught = True
    assert exception_caught, f"Error: no exception caught when requesting dividend yield for date before first available date ({dividends_global_first_date})"

    exception_caught = False
    try:
        yield4 = sp500_dividend_yield(datetime.date(2030, 6, 1))
    except:
        exception_caught = True
    assert exception_caught, f"Error: no exception caught when requesting dividend yield for date after last available date ({dividends_global_last_date})"

    x = 1  # for debug
