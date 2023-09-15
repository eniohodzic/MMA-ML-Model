import re
from datetime import datetime
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd


class Transformer():
    """
    Transformer class merges both tables into a final merged table while encoding non-ints into int/bool values

    Input raw datasets from Web scraper and merge on fighter column on fight table adding lifetime stats from fighter table

    Final Post-Processed dataset should have the following information:
    'date': Date of bout
    'fight_url': URL for fight information 
    'event_url': URL for event information
    'fighter_url': URL for fighter information
    'result': Result of fight relative to 'fighter' column, original is categorical, converted to int
    'fighter': Fighter name relative to current row
    'opponent': Fighter name that 'fighter' is against
    'division': Weightclass of current fight
    'stance': Stance of fighter, original is categorical, converted to int
    'dob': Date of birth, original is date, converted to int -> 'age'
    'method': Method of win, original is categorical, converted to int
    'round': Round of win
    'time': Time of win
    'time_format': Total rounds with time of each
    'referee': Name of ref
    'height': Height of fighter converted to absolute inch
    'reach': Reach of fighter converted to absolute inch 
    'knockdowns': Total knockdowns 
    'sub_attempts': Total submission attempts
    'reversals': Total reversals 
    'control': Total time of control 
    'takedowns_landed': Total takedowns successfully landed
    'takedowns_attempts': Total takedowns attempted
    'sig_strikes_landed': Total significant strikes successfully landed
    'sig_strikes_attempts': Total significant strikes attempted
    ...
    include all between
    ...
    'ground_strikes_landed': Total ground strikes successfuly landed
    'ground_strikes_attempts': Total ground strikes attempted

    """

    def __init__(self) -> None:
        self.fight_stats = pd.read_csv(abspath(join(dirname(dirname(dirname(__file__))), 'data/', 'raw/', 'RAW_fight_stats.csv')))
        self.fighter_stats = pd.read_csv(abspath(join(dirname(dirname(dirname(__file__))), 'data/', 'raw/', 'RAW_fighter_stats.csv')))
        self.fighter_stats.rename(columns={'name': 'fighter'}, inplace=True)
        self.transformed_data = pd.DataFrame()

    def write_transform(self) -> None:
        self.__merge_tables()
        self.__transfer_columns()
        self.__convert_dtypes()

        self.transformed_data.to_csv(abspath(join(dirname(dirname(dirname(__file__))), 'data/', 'interim/', 'INTERIM1_transform.csv')), index=False)
    
    def __merge_tables(self) -> None:
        """Merge both tables along fighter name"""
        self.data = self.fight_stats.merge(self.fighter_stats, on='fighter')
        self.data.to_csv(abspath(join(dirname(dirname(dirname(__file__))), 'data/', 'raw/', 'RAW_merged_stats.csv')), index=False)

    def __transfer_columns(self) -> None:
        """Transfer columns of merged table to post-processed dataframe"""
        
        # Various fight descriptor column names, not fight dependent
        cols = ['date',
                'fight_url',
                'event_url',
                'fighter_url',
                'result',
                'fighter',
                'opponent',
                'division',
                'stance',
                'dob',
                'method',
                'round',
                'time',
                'time_format',
                'referee',
                'height',
                'reach']
        self.transformed_data[cols] = self.data[cols]

        # Various fight statistic column names, fight dependent
        cols = self.data.loc[:, 'knockdowns':'ground_strikes_attempts'].columns
        self.transformed_data[cols] = self.data[cols]

        # Convert strings with -- to NaN and drop
        self.transformed_data.replace('--', np.nan, inplace=True)
        self.transformed_data.dropna(inplace=True)

        # Transformed_data now contains proper columns from merged_stats and is ready for dtype conversion and processing for future feature engineering
    
    def __convert_dtypes(self) -> None:
        """
        Convert columns in string format to integer/float values 
        """

        """
        Convert each height and reach of each fight row into integer inch value
        Ex. 6' 1" = 73
        Ex. 70" = 70

        Convert control time into float value of minutes
        Ex. 4:30 = 4.5

        Convert date of bout into DateTime dtype

        Convert DoB to DateTime dtype    
        """
        cols = ['height', 
                'reach',
                'control',
                'date',
                'dob']

        for col in cols:
            for idx, row in self.transformed_data.iterrows():
                stat = row[col]

                if stat == '--':
                    continue

                if col == 'height':
                    pattern = re.compile(r"""(\d+)' *(\d+)(?:"|'')?""")
                    match = re.match(pattern, stat)

                    feet, inches = map(int, match.groups())
                    height = 12 * feet + inches 

                    self.transformed_data.loc[idx, 'height'] = height

                if col == 'reach':
                    reach = int(re.findall(r'\d+', stat)[0])

                    self.transformed_data.loc[idx, 'reach'] = reach

                if col == 'control':
                    m, s = stat.split(':')
                    
                    self.transformed_data.loc[idx, 'control'] = int(m) + int(s) / 60

                if col == 'date':
                    date = datetime.strptime(stat, '%B %d, %Y')

                    self.transformed_data.loc[idx, 'date'] = date

                if col == 'dob':
                    date = datetime.strptime(stat, '%b %d, %Y')

                    self.transformed_data.loc[idx, 'dob'] = date

        
