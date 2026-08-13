#!/usr/bin/env python
# coding: utf-8

# In[39]:


import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


# In[40]:


# Column orders matches the project spec exactly
TRIP_COLUMNS = [
    "trip_id",
    "driver_id",
    "pickup_datetime",
    "dropoff_datetime",
    "passenger_count",
    "pickup_loc_id",
    "dropoff_loc_id",
    "trip_distance",
    "fare_amount",
]

DRIVER_COLUMNS = ["driver_id", "last_name", "given_name"]

LOCATION_COLUMNS = ["location_id", "loc_name"]

DATETIME_FORMAT = "%H:%M:%S,%d-%m-%Y"

# Which trips.csv columns search_trips is allowed to filter on, and what
# kind of value each one expects
SEARCHABLE_COLUMNS = {
            "driver_id": "int",
            "pickup_datetime": "datetime",
            "dropoff_datetime": "datetime",
            "passenger_count": "int",
            "trip_distance": "float",
            "fare_amount": "float",
        }


# In[41]:

class SakayDBError(ValueError):
    """Exception raised for invalid operations or data in SakayDB"""
    pass

# In[44]:


class SakayDB:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    # Creating the helper methods to make implementation consistnet
    def _path(self, filename):
        """Build the full path to a data file inside data_dir."""
        # Centralized here so every method builds paths the same way
        # instead of repeating os.path.join(self.data_dir, ...) everywhere
        return os.path.join(self.data_dir, filename)

    def _load_csv(self, filename, columns):
        """Load a CSV file, or return an empty DataFrame if missing."""
        # A file might not exist yet on a fresh data_dir 
        # (e.g. first ever add_trip call)
        path = self._path(filename)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame(columns=columns)

    # Create some methods that can call each of the 3 CSV
    # Without need for specifying the columns everytime
    # Prevents typos for later
    def _load_trips(self):
        return self._load_csv("trips.csv", TRIP_COLUMNS)

    def _load_drivers(self):
        return self._load_csv("drivers.csv", DRIVER_COLUMNS)

    def _load_locations(self):
        return self._load_csv("locations.csv", LOCATION_COLUMNS)

    # Helper methods for adding and calling drivers and trips
    # Moved from add_trip to another function to prevent clutter
    def _get_or_add_driver(self, df_drivers, driver):
        """Find a driver's id by name, adding a new driver if needed.

        Parameters
        ----------
        df_drivers : pd.DataFrame
            Current drivers table.
        driver : str
            Driver name as "Last name, Given name".

        Returns
        -------
        driver_id : int
        df_drivers : pd.DataFrame
            Possibly updated with a new driver row.
        """
        # split(",", 1) protects against a given name that itself
        # contains a comma (e.g. suffixes); we only ever want 2 parts.
        last_name, given_name = [part.strip() for part in driver.split(",", 1)]

        # Matching is case-insensitive and whitespace-trimmed
        # match: tuple(Bool, Bool, Bool,...)
        match = (
            (df_drivers["last_name"].str.strip().str.lower()
             == last_name.lower())
            & (df_drivers["given_name"].str.strip().str.lower()
               == given_name.lower())
        )
        if match.any() == True:
            # Driver already exists, reuse their id
            # .iloc[0] guards against possible duplicate rows.
            driver_id = int(df_drivers.loc[match, "driver_id"].iloc[0])
            return driver_id, df_drivers

        # If new driver, go to this:
        # New driver: id is "last id + 1", or 1 if the table is empty.
        # incrementing integers rather than needing a separate counter.
        if len(df_drivers):
            driver_id = int(df_drivers["driver_id"].max()) + 1
        else:
            driver_id = 1

        # Defining the new row for new drivers
        new_row = pd.DataFrame([{
            "driver_id": driver_id,
            "last_name": last_name,
            "given_name": given_name,
        }])
        # add it to the existing drivers dataframe

        if df_drivers.empty:
            df_drivers = new_row
        else:
            df_drivers = pd.concat([df_drivers, new_row], ignore_index=True)

        return driver_id, df_drivers

    # Same lookup-or-create pattern as _get_or_add_driver, just for
    # a single-field name instead of a last/given pair
    def _get_or_add_location(self, df_locations, loc_name):
        """Find a location's id by name, adding a new location if needed.

        Parameters
        ----------
        df_locations : pd.DataFrame
            Current locations table.
        loc_name : str
            Zone/location name.

        Returns
        -------
        location_id : int
        df_locations : pd.DataFrame
            Possibly updated with a new location row.
        """
        loc_name = loc_name.strip()
        match = (
            df_locations["loc_name"].str.strip().str.lower()
            == loc_name.lower()
        )
        # If: location already exists
        if match.any() == True:
            location_id = int(df_locations.loc[match, "location_id"].iloc[0])
            return location_id, df_locations
        # Else: location needs to be created with unique ID
        if len(df_locations):
            location_id = int(df_locations["location_id"].max()) + 1
        else: 
            location_id = 1

        new_row = pd.DataFrame([{
            "location_id": location_id,
            "loc_name": loc_name,
        }])

        if df_locations.empty:
            df_trips = new_row    
        else:
            df_locations = pd.concat([df_locations, new_row], 
                                     ignore_index=True)
        
        return location_id, df_locations

# 1. Adding Trips
    def add_trip(self, driver, pickup_datetime, dropoff_datetime,
                passenger_count, pickup_loc_name, dropoff_loc_name,
                trip_distance, fare_amount):
        """
        Add a single trip to the database.

        Parameters
        ----------
        driver : str
            Driver name as "Last name, Given name".
        pickup_datetime, dropoff_datetime : str
            Datetime as "hh:mm:ss,DD-MM-YYYY".
        passenger_count : int
        pickup_loc_name, dropoff_loc_name : str
        trip_distance : float
            Distance in meters.
        fare_amount : float

        Returns
        -------
        int
            The trip_id assigned to the new trip.

        Raises
        ------
        SakayDBError
            If an identical trip already exists in trips.csv.
        """ 
        # First, load all three 3 CSVs as separate df
        # (trips.csv, drivers.csv, locations.csv)
        df_trips = self._load_trips()
        df_drivers = self._load_drivers()
        df_locations = self._load_locations()

        # Fetch driver ID from driver name
        # Use _get_or_add_driver() method created earlier
        # Modifies drivers df if not yet existing
        driver_id, df_drivers = self._get_or_add_driver(df_drivers, driver)

        # Do same for pickup and dropoff
        pickup_loc_id, df_locations = self._get_or_add_location(
            df_locations, pickup_loc_name)

        dropoff_loc_id, df_locations = self._get_or_add_location(
            df_locations, dropoff_loc_name)

        # Now that we have the driver_id, loc_ids 
        # Create logic for checking duplicates
        # use ids instead of resolving duplicates
        # Create pd series(bool, bool, bool...)
        is_duplicate_series = (
            (df_trips["driver_id"] == driver_id)
            & (df_trips["pickup_datetime"] == pickup_datetime)
            & (df_trips["dropoff_datetime"] == dropoff_datetime)
            & (df_trips["passenger_count"] == passenger_count)
            & (df_trips["pickup_loc_id"] == pickup_loc_id)
            & (df_trips["dropoff_loc_id"] == dropoff_loc_id)
            & (df_trips["trip_distance"] == trip_distance)
            & (df_trips["fare_amount"] == fare_amount)
        ).any()

        if is_duplicate_series.any() == True:
            raise SakayDBError("Cannot add a duplicate trip into database")

        # Once we confirm that this is a new trip,
        # add to trips database (recylce same logic as:
        # _get_or_add_drivers()
        if len(df_trips):
            trip_id = int(df_trips["trip_id"].max()) + 1
        else:
            trip_id = 1
        new_row = pd.DataFrame([{
            "trip_id": trip_id,
            "driver_id": driver_id,
            "pickup_datetime": pickup_datetime,
            "dropoff_datetime": dropoff_datetime,
            "passenger_count": passenger_count,
            "pickup_loc_id": pickup_loc_id,
            "dropoff_loc_id": dropoff_loc_id,
            "trip_distance": trip_distance,
            "fare_amount": fare_amount,
        }])

        if df_trips.empty:
            df_trips = new_row
        else:
            df_trips = pd.concat([df_trips, new_row], ignore_index=True)

        # Overwrite all 3 CSVs
        df_trips.to_csv(self._path("trips.csv"), index=False)
        df_drivers.to_csv(self._path("drivers.csv"), index=False)
        df_locations.to_csv(self._path("locations.csv"), index=False)

        return trip_id

    def add_trips(self, trips):
        """Add multiple trips to the database.

        Each trip that is a duplicate or has invalid/incomplete data is
        skipped (with a warning printed), rather than stopping the
        whole batch.

        Parameters
        ----------
        trips : list of dict
            Each dict has the same keys as add_trip's parameters:
            "driver", "pickup_datetime", "dropoff_datetime",
            "passenger_count", "pickup_loc_name", "dropoff_loc_name",
            "trip_distance", "fare_amount".

        Returns
        -------
        list of int
            trip_ids of the trips that were successfully added, in the
            order they were added.
        """
        trip_ids = []

        REQUIRED_TRIP_KEYS = {
        "driver",
        "pickup_datetime",
        "dropoff_datetime",
        "passenger_count",
        "pickup_loc_name",
        "dropoff_loc_name",
        "trip_distance",
        "fare_amount",
        }

        for i, trip in enumerate(trips):
            # Check for any missing kwargs/incompete information
            if REQUIRED_TRIP_KEYS - set(trip.keys()):
                # True means incomplete input
                print(f"Warning: trip index {i} has invalid or incomplete "
                      f"information. Skipping...")
                continue

            # Reuse add_trip() method, as it already raises a SakayDBError
            # for duplicate trips
            try:
                trip_id = self.add_trip(**trip)
            # from Specs "If a trip is already in the database, 
            # skip it and print"
            except SakayDBError:
                print(f"Warning: trip index {i} is already in the database. "
                      f"Skipping...")
                continue

            # Other "invalid" information
            except (TypeError, ValueError, AttributeError):
                print(f"Warning: trip index {i} has invalid or incomplete "
                      f"information. Skipping...")
                continue

            trip_ids.append(trip_id)

        return trip_ids

    def delete_trip(self, trip_id):
        """Delete a trip from the database.

        Parameters
        ----------
        trip_id : int
            The trip_id of the trip to remove.

        Raises
        ------
        SakayDBError
            If no trip with that trip_id exists (including when
            trips.csv doesn't exist yet).
        """
        df_trips = self._load_trips()

        match_mask = df_trips["trip_id"] == trip_id
        if not match_mask.any():
            raise SakayDBError(f"Trip id {trip_id} not found.")

        # drop the matching trip and keep the rest
        # Invert match_mask, make everything True
        # Except for the input trip ID
        df_trips = df_trips.loc[~match_mask]
        df_trips.to_csv(self._path("trips.csv"), index=False)


    def _validate_search_value(self, key, kind, value):
        """Check a single (non-range) search value is the right type.

        Parameters
        ----------
        key : str
            The keyword argument name (used only for the error message).
        kind : str
            One of "int", "float", "datetime" -- from SEARCHABLE_COLUMNS.
        value : object
            The value to check.

        Raises
        ------
        SakayDBError
            If the value doesn't match the expected type/format.
        """
        # isinstance(value, bool) is checked separately because in
        # Python, bool is a subclass of int -- without this, someone
        # passing True/False would silently pass an "int" check.
        if kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise SakayDBError(
                    f"Invalid value for '{key}': {value} is not an int.")

        elif kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SakayDBError(
                    f"Invalid value for '{key}': {value} is not a "
                    f"float.")

        elif kind == "datetime":
            # strptime raises ValueError
            # on anything that isn't a real date in the expected format,
            if not isinstance(value, str):
                raise SakayDBError(
                    f"Invalid value for '{key}': {value} is not a "
                    "datetime string."
                )
            try:
                datetime.strptime(value, DATETIME_FORMAT)
            except ValueError:
                raise SakayDBError(
                    f"Invalid value for '{key}': {value} does not "
                    f"match format {DATETIME_FORMAT}."
                ) 


    def _search_column(self, df_trips, key, kind):
        """Return a comparable version of a trips.csv column.

        Datetime columns are stored as strings and get parsed into
        actual pd.Timestamp values so that ">=" / "<=" comparisons
        during a range search reflect real chronological order
        (string comparison alone would sort "9-..." after "10-...").
        Non-datetime columns are returned unchanged.
        """
        if kind == "datetime":
            return pd.to_datetime(df_trips[key], format=DATETIME_FORMAT)
        return df_trips[key]


    def search_trips(self, **kwargs):
        """Search trips.csv by one or more keyword filters.

        Each keyword must be one of the keys in SEARCHABLE_COLUMNS:
        "driver_id", "pickup_datetime", "dropoff_datetime",
        "passenger_count", "trip_distance", "fare_amount".

        Per the Specs each value may be either:
          - a single exact value, or
          - a 2-tuple range (low, high), where either side may be None
            to mean "no lower/upper bound". Both bounds are inclusive.

        Rows must satisfy every keyword's condition (AND across all
        keywords passed in).

        Returns
        -------
        pd.DataFrame
            Matching trips, in trips.csv's columns. If any keyword used
            a range, the result is sorted by that column (chronologically
            for datetime columns, ascending otherwise).

        Raises
        ------
        SakayDBError
            If no keywords are passed, an unknown keyword is used, a
            range tuple doesn't have exactly 2 elements, or a value's
            type/format doesn't match what the keyword expects.
        """
        if not kwargs:
            raise SakayDBError("At least one search keyword is required.")

        # Check if the trips.csv already exist to satisy test cell requirement
        trips_exists = os.path.exists(self._path("trips.csv"))
        df_trips = self._load_trips()

        # Start with "keep everything", then AND in each keyword's
        # condition one at a time -- same boolean-mask approach used
        # in add_trip's duplicate check and delete_trip.
        mask = pd.Series(True, index=df_trips.index)

        # If a range filter is used, we remember its column/kind here
        # so we know what to sort the final result by afterward
        sort_key, sort_kind = None, None

        for key, value in kwargs.items():
            if key not in SEARCHABLE_COLUMNS:
                raise SakayDBError(f"Invalid search keyword: {key}")
            kind = SEARCHABLE_COLUMNS[key]

            if isinstance(value, tuple):
                if len(value) != 2:
                    raise SakayDBError(
                        f"Range for '{key}' must have exactly 2 "
                        f"elements, e.g. (value, None)."
                    )
                low, high = value
                column = self._search_column(df_trips, key, kind)

                if low is not None:
                    self._validate_search_value(key, kind, low)
                    if kind == "datetime":
                        low_bound = datetime.strptime(low, DATETIME_FORMAT)
                    else:
                        low_bound = low
                    mask &= column >= low_bound

                if high is not None:
                    self._validate_search_value(key, kind, high)
                    if kind == "datetime":
                        high_bound = datetime.strptime(
                            high, DATETIME_FORMAT
                        )
                    else:
                        high_bound = high
                    mask &= column <= high_bound

                sort_key, sort_kind = key, kind
            else:
                self._validate_search_value(key, kind, value)
                mask &= df_trips[key] == value

        result = df_trips.loc[mask]

        # Only re-sort if a range keyword was used - an exact-match-only
        # search (like driver_id=1) keeps trips.csv's original row order,
        # matching how the spec describes sorting only for range results.
        if sort_key is not None:
            sort_series = self._search_column(result, sort_key, sort_kind)
            result = result.loc[sort_series.sort_values().index]

        result = result.reset_index(drop=True)

        if not trips_exists:
            return []

        return result


    def export_data(self):
        """Export all trips as a single, Pandas dataframe.

        Joins trips.csv against drivers.csv and locations.csv (twice --
        once for pickup, once for dropoff) so that ids become readable
        names.

        Returns
        -------
        pd.DataFrame
            One row per trip, sorted by trip_id ascending, with columns
            "driver_lastname", "driver_givenname", "pickup_datetime",
            "dropoff_datetime", "passenger_count", "pickup_loc_name",
            "dropoff_loc_name", "trip_distance", "fare_amount".
        """
        EXPORT_COLUMNS = [
            "driver_lastname",
            "driver_givenname",
            "pickup_datetime",
            "dropoff_datetime",
            "passenger_count",
            "pickup_loc_name",
            "dropoff_loc_name",
            "trip_distance",
            "fare_amount",
        ]
        # load working trips.csv database
        # use as baseline
        df_trips = self._load_trips()
        if df_trips.empty:
            return pd.DataFrame(columns=EXPORT_COLUMNS)
        df_drivers = self._load_drivers()
        df_locations = self._load_locations()

        # Sort by trip ID
        df_trips = df_trips.sort_values("trip_id")

        # Merge trip and drivers database
        merged = df_trips.merge(df_drivers, on="driver_id", how="left")

        # Locations database uses a generic "location_id" as PK
        # Repurpose this to match with "pickup_loc_id" as FK in merged
        pickup_locations = df_locations.rename(columns={
            "location_id": "pickup_loc_id",
            "loc_name": "pickup_loc_name"})
        merged = merged.merge(pickup_locations, on="pickup_loc_id", 
                              how="left")

        # Do the same for drop-off
        dropoff_locations = df_locations.rename(columns={
            "location_id": "dropoff_loc_id",
            "loc_name": "dropoff_loc_name"})
        merged = merged.merge(dropoff_locations, on="dropoff_loc_id", 
                              how="left")

        # Rename "last_name" and "given_name" 
        # to match specs
        # Capitalize the first and last names of the drivers
        merged["driver_lastname"] = merged["last_name"].str.title()
        merged["driver_givenname"] = merged["given_name"].str.title()

        return merged[EXPORT_COLUMNS].reset_index(drop=True)

    def generate_statistics(self, stat):
        """Compute average daily trip counts by weekday.

        Parameters
        ----------
        stat : str
            One of "trip", "passenger", "driver", or "all".
            - "trip": {day_name: average trips that day, overall}
            - "passenger": {passenger_count: {day_name: average}}
            - "driver": {"Last name, Given name": {day_name: average}}
            - "all": {"trip": ..., "passenger": ..., "driver": ...}

        Returns
        -------
        dict

        Raises
        ------
        SakayDBError
            If `stat` isn't one of the four values above (case
            sensitive).
        """
        df_trips = self._load_trips()
        df_drivers = self._load_drivers()

        # Create a resuable nested function that averages trips per weekday
        # Later on, we will need to reuse this for passenger, driver, all
        def average_trips_per_weekday(df):
            """Average trips per day grouped by weekday.
            """
            if df.empty:
                return {}

            # pickup timestamp pull out both the:
            # plain calendar date (to count trips per specific day)
            # and the weekday name (to group those counts together).
            pickup = pd.to_datetime(
                df["pickup_datetime"], format=DATETIME_FORMAT)
            dates = pickup.dt.date
            day_names = pickup.dt.day_name()

             # First, how many trips happened on each individual date.
            daily_counts = (
                pd.DataFrame({"date": dates, "day_name": day_names})
                .groupby(["date", "day_name"])
                # for each (date, day_name) group, how many rows are in it?
                .size()
                # Convert the GroupBy object back into DataFrame
                # Columns: trip_count (from .size()), date, day_name
                .reset_index(name="trip_count")
            )

            # Second, average those daily counts across every date
            # with the same weekday name (like Monday or Tuesday)
            avg_by_day = (
                daily_counts.groupby("day_name")["trip_count"].mean())
            return avg_by_day.to_dict()

        def stat_trip():
            """
            Average trips per weekday, across the whole df_trips table
            """
            return average_trips_per_weekday(df_trips)

        def stat_passenger():
            """Average trips per weekday, split by passenger_count."""
            if df_trips.empty:
                return {}
            result = {}
            for count, group in df_trips.groupby("passenger_count"):
                result[int(count)] = average_trips_per_weekday(group)
            return result

        def stat_driver():
            """Average trips per weekday, split by driver."""
            if df_trips.empty:
                return {}
            merged = df_trips.merge(
                df_drivers, on="driver_id", how="left")
            # "Last name, Given name" label the spec asks to group by directly
            merged["driver_name"] = (
                merged["last_name"] + ", " + merged["given_name"])
            result = {}
            for name, group in merged.groupby("driver_name"):
                result[name] = average_trips_per_weekday(group)
            return result

        if stat == "trip":
            return stat_trip()
        elif stat == "passenger":
            return stat_passenger()
        elif stat == "driver":
            return stat_driver()
        elif stat == "all":
            return {"trip": stat_trip(),
                "passenger": stat_passenger(),
                "driver": stat_driver(),}
        else:
            raise SakayDBError(f"Unknown stat parameter")

    def plot_statistics(self, stat):
        """Plot the statistics computed by generate_statistics.
    
        Parameters
        ----------
        stat : str
            One of "trip", "passenger", "driver".
            - "trip": bar plot of average trips per weekday.
            - "passenger": line plot of average trips per weekday,
              one line per passenger_count.
            - "driver": 7x1 grid of horizontal bar plots, one per
              weekday, showing the top 5 drivers that day.
    
        Returns
        -------
        matplotlib.axes.Axes or matplotlib.figure.Figure
            An Axes for "trip"/"passenger"; a Figure for "driver".
    
        Raises
        ------
        SakayDBError
            If `stat` isn't one of the three values above (case
            sensitive).
        """
        # Calendar order for laying out day-of-week axes
        # and subplot grids consistently across all three plot types.
        DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday","Sunday",]

        def plot_trip():
            """Bar plot: overall average trips per weekday"""
            stats = self.generate_statistics("trip")
            values = [stats.get(day, 0) for day in DAY_ORDER]

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.bar(DAY_ORDER, values)
            ax.set_title("Average trips per day")
            ax.set_xlabel("Day of week")
            ax.set_ylabel("Ave Trips")
            return ax

        def plot_passenger():
            """Line plot: average trips per weekday, one line per
            passenger_count."""
            stats = self.generate_statistics("passenger")

            fig, ax = plt.subplots(figsize=(12, 8))
            # Sorted so the legend lists passenger counts in
            # ascending order (0, 1, 2, 3, ...).
            for count in sorted(stats.keys()):
                day_stats = stats[count]
                values = [day_stats.get(day, 0) for day in DAY_ORDER]
                ax.plot(
                    DAY_ORDER, values, marker="o", linestyle="-",
                    label=str(count)
                )
            ax.set_xlabel("Day of week")
            ax.set_ylabel("Ave Trips")
            ax.legend()
            return ax

        def plot_driver():
            """7x1 grid: top-5 drivers per weekday, horizontal bars."""
            stats = self.generate_statistics("driver")
             # Per Spec: x-ticks must be shared accross subplots
            fig, axes = plt.subplots(
                nrows=7, ncols=1, figsize=(8, 25), sharex=True
            )
            for ax, day in zip(axes, DAY_ORDER):
                # Every driver's average trip count for this specific
                # weekday (drivers with no trips that day are skipped)
                # stats: dict(driver_name1: dict(Monday: avg trips,
                # Tuesday: avg trips,...), driver_name2: ...)
                day_data = [(name, day_stats[day])
                    for name, day_stats in stats.items()
                    if day in day_stats
                ]
                # Spec: sort by decreasing average then alphabetically
                day_data.sort(key=lambda x: (-x[1], x[0]))
                top5 = day_data[:5]

                # Reverse so the #1 driver for the day ends up drawn
                # at the top of the horizontal bar chart, not the
                # bottom (barh draws its first item at the bottom).
                names = [name for name, avg in reversed(top5)]
                values = [avg for name, avg in reversed(top5)]

                ax.barh(names, values, label=day)
                ax.legend()

            # Per Spec: x-axis label only on the bottom-most subplot
            axes[-1].set_xlabel("Ave Trips")
            return fig

        if stat == "trip":
            return plot_trip()
        elif stat == "passenger":
            return plot_passenger()
        elif stat == "driver":
            return plot_driver()
        else:
            raise SakayDBError(f"Unknown stat parameter")

    def generate_odmatrix(self, date_range=None):
        """
        Build an origin-destination matrix of average daily trips.
 
        Parameters
        ----------
        date_range : tuple of (str or None, str or None), optional
            Filters trips by pickup_datetime, same inclusive-range
            rules as search_trips: (low, None) is "from low onward",
            (None, high) is "up to high", (low, high) is "between
            both, inclusive". Defaults to None, meaning no filtering
            -- every trip in trips.csv is included.
 
        Returns
        -------
        pd.DataFrame
            Square matrix indexed by every dropoff_loc_name (rows) and
            pickup_loc_name (columns) in locations.csv. Each cell is
            the average number of trips per day, for that specific
            pickup/dropoff pair, counted only over the days that
            *specific pair* actually had a trip.
 
        Raises
        ------
        SakayDBError
            If date_range isn't a 2-element tuple, or either side
            isn't a valid datetime string.
        """
        df_trips = self._load_trips()
        if df_trips.empty:
            return pd.DataFrame()
 
        df_locations = self._load_locations()
        pickup = pd.to_datetime(
            df_trips["pickup_datetime"], format=DATETIME_FORMAT
        )
 
        if date_range is not None:
            if not isinstance(date_range, tuple) or len(date_range) != 2:
                raise SakayDBError(
                    "date_range must be a 2-element tuple, e.g. "
                    "(low, None).")
            low, high = date_range
            # Create all True boolean mask with len(df_trips.index)
            mask = pd.Series(True, index=df_trips.index)
 
            # Filter by changing those elements in "mask"
            # that are True, to False based on date_range criteria
            # date_range: tuple(low, high)
 
            if low is not None:
                # Returns SakayDBError if datetime input 
                # does not follow format
                self._validate_search_value(
                    "pickup_datetime", "datetime", low)
                low_bound = datetime.strptime(low, DATETIME_FORMAT)
                mask &= pickup >= low_bound
 
            if high is not None:
                self._validate_search_value(
                    "pickup_datetime", "datetime", high
                )
                high_bound = datetime.strptime(high, DATETIME_FORMAT)
                mask &= pickup <= high_bound
 
            df_trips = df_trips.loc[mask]
            pickup = pickup.loc[mask]
 
        # Map location ids to their names, once, then attach the
        # readable names and the plain calendar date onto the trips
        # table for grouping.
        loc_names = df_locations.set_index("location_id")["loc_name"]
        pickup_names = df_trips["pickup_loc_id"].map(loc_names)
        dropoff_names = df_trips["dropoff_loc_id"].map(loc_names)
        pickup_dates = pickup.dt.date
 
        pair_table = pd.DataFrame({
            "pickup_loc_name": pickup_names,
            "dropoff_loc_name": dropoff_names,
            "pickup_date": pickup_dates,
        })
 
        # For each pickup/dropoff pair, the denominator is how many
        # distinct days THAT SPECIFIC PAIR had a trip on -- not the
        # number of days in the whole dataset. A pair that only ever
        # occurs on 4 different days averages over those 4 days, not
        # over every day the system has any data for.
        pair_groups = pair_table.groupby(
            ["pickup_loc_name", "dropoff_loc_name"]
        )
        trip_counts = pair_groups.size()
        pair_days = pair_groups["pickup_date"].nunique()
 
        # Every location in locations.csv gets a row and a column,
        # even ones with zero trips -- that's why the matrix is
        # always square with shape (n_locations, n_locations),
        # regardless of how many pairs actually have data.
        all_locations = df_locations["loc_name"].tolist()
        od_matrix = pd.DataFrame(
            0.0, index=all_locations, columns=all_locations
        )
        for (pickup_name, dropoff_name), count in trip_counts.items():
            n_days_pair = pair_days[(pickup_name, dropoff_name)]
            # Rows are dropoff_loc_name, columns are pickup_loc_name.
            od_matrix.loc[dropoff_name, pickup_name] = count / n_days_pair
 
        return od_matrix


# In[ ]:




