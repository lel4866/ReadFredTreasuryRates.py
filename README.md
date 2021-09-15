# ReadFredTreasuryRates.py
Python program to read treasury rates from FRED database, and supply a function
to interpolate rate for any period from 1 day to 360 days

Supplied functions:

**read_risk_free_rates(earliest_date: datetime=datetime.date(2000, 1, 1))**

Reads LIBOR rates of durations: 1, 7, 30, 60, 90, 180, amd 360 from FRED database
Removes NaN's and interpolates linearly so that rates are available for all durations from 0
through 360

**risk_free_rate(requested_date: datetime.date, duration: int) -> float**

Returns a rate (in percent) for a requested date and duration

Asserts (throws AssertException) if date is prior to earliest date of series (saved in global
variable rates_global_first_date) or later than today

Asserts if duration is less than 0 or greater 360

**read_sp500_dividend_yield(earliest_date: datetime=datetime.date(2000, 1, 1))**

Reads S&P 500 dividend yield from Nasdaq Data Link (formerly Quandl)
This series typically only has one value per month, I interpolated between those values. I'm
not sure how good this is, but the values do change over the month so it's probabnly at least
as good as assuming the value stays fixed over the month

**sp500_dividend_yield(date: datetime) -> float**

Returns an annualized dividend yield for the S&P500 for the requested date

# Programming comments:
I use asserts to check for invalid arguments to functions rather than raising exeptions because
I consider invalid arguments a programming error. I raise exceptions (mostly ValueExceptions)
for things like data problems

If you want to run without taking the time to check for valid parameter values, that is, turn
off asserts, use: python -cO main.py

This is written using Python 3.9 and Pycharm 2021.2.1


