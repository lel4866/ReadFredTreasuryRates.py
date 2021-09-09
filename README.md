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

Throws an Exception if date is prior to earliest date of series (saved in global variable
rates_global_first_date) or later than today

Throws an Exception if duration is less than 0 or greater 360

# Programming comments:
This is written using Python 3.9 and Pycharm 2021.2.1


