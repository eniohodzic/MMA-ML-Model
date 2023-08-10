import re
from copy import deepcopy
from datetime import datetime
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
from tqdm import tqdm

from .elo_system import EloSystem


class Feature_Extractor():
    """
    Feature_Extractor serves to create new columns through various operations

    This differes from Transformer in that these formualate additional features based on the transformers output 

    Future work can be done for intuitive feature design based on my knowledge of MMA. Such as calculations implemententing a lot of the ground
    game of the sport, such as functions including ground strikes, ground control, etc.

    Dimensionality reduction may be useful for converting some of these down into some latent variables and using those for model input, but that requires 
    some visualization building. 
    """


    def __init__(self, predict_df=None) -> None:
        self.transformed_data = pd.read_csv(abspath(join(dirname(dirname(dirname(__file__))), 'data/', 'interim/', 'transformed_stats.csv')))

        if predict_df is None:
            self.extracted_data = deepcopy(self.transformed_data)
        else:
            self.extracted_data = predict_df

    def extract(self, shift=1):
        self.feature_conversion()
        self.feature_absorbed_defended()
        self.feature_accuracy()
        self.feature_per_min()
        self.feature_elo()
        self.feature_differential()
        self.feature_historical(shift=shift)
        self.feature_diff_opponent()
        
    def write(self):
        self.extracted_data.to_csv(abspath(join(dirname(dirname(dirname(__file__))), 'data/', 'processed/', 'extracted_stats.csv')), index=False)

    def feature_conversion(self):
        # Coverting some of these various columns into useable features

        # Converting DoB into Age using date of fight - dob
        for idx, row in tqdm(self.extracted_data.iterrows()):

            stat = row['dob']
            dob = datetime.strptime(stat, '%Y-%m-%d %H:%M:%S') 
            date = datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S') 
            time_delta = date - dob
            age = time_delta.days / 365

            self.extracted_data.loc[idx, 'age'] = age

        # Converting round + time into total fight length
        for idx, row in tqdm(self.extracted_data.iterrows()):

            round = row['round']
            time = row['time']

            if pd.isnull(round) or pd.isnull(time):
                continue

            m, s = time.split(':')

            self.extracted_data.loc[idx, 'total_time'] = (int(round) - 1) * 5 + int(m) + int(s) / 60

    def feature_absorbed_defended(self):
        """
        Calculating fight stat absorbed and defended during fight, the landed stat of the opponent        
        """
        
        strike_type = ['takedowns',
                       'sig_strikes',
                       'total_strikes',
                       'head_strikes',
                       'body_strikes',
                       'leg_strikes',
                       'distance_strikes',
                       'clinch_strikes',
                       'ground_strikes']


        # Take value from each landed strike from fighter and apply it as absorbed strike to opponent row
        # Take value from each landed and attempted strike from fighter and apply it as defended strike to opponent row
        generator = zip((strike + '_landed' for strike in strike_type), (strike + '_attempts' for strike in strike_type))
        for col in tqdm(generator):
            col_landed = col[0]
            col_attempts = col[1]

            for idx, row in self.extracted_data.iterrows():
                opponent = row['opponent'] 
                fight_url = row['fight_url'] # in case fighters fought more than once 

                # Get index of same fight relative to the other fighter
                index = self.extracted_data.index[(self.extracted_data.fighter == opponent) & (self.extracted_data.fight_url == fight_url)]

                if index.empty:
                    continue

                # Absorbed is landed from other fighter 
                self.extracted_data.loc[index, col_landed[:-7] + '_absorbed'] = row[col_landed] 

                # Defended is attempts - landed from other fighter
                self.extracted_data.loc[index, col_landed[:-7] + '_defended'] = row[col_attempts] - row[col_landed]
                
    def feature_accuracy(self):
        """
        Calculating various strike type accuracies
        """
        strike_type = ['takedowns',
                       'sig_strikes',
                       'total_strikes',
                       'head_strikes',
                       'body_strikes',
                       'leg_strikes',
                       'distance_strikes',
                       'clinch_strikes',
                       'ground_strikes']
        

        generator = zip((strike + '_landed' for strike in strike_type), (strike + '_attempts' for strike in strike_type))
        for col in tqdm(generator):
            col_landed = col[0]
            col_attempts = col[1]

            for idx, row in self.extracted_data.iterrows():
                landed = row[col_landed]
                attempts = row[col_attempts]

                # Accuracy is landed / attempts 
                if attempts == 0 or pd.isnull(attempts):
                    value = np.nan
                else:
                    value = np.divide(landed, attempts)

                self.extracted_data.loc[idx, col_landed[:-7] + '_accuracy'] = value

    def feature_per_min(self):
        """
        Calculating stat per min of each fighter 

        Calculation: stat / total_time
        """

        cols = self.extracted_data.loc[:, 'height':].columns

        for col in tqdm(cols):

            for idx, row in self.extracted_data.iterrows():
                stat = row[col]
                total_time = row['total_time']

                if total_time == 0 or pd.isnull(total_time) or pd.isnull(stat):
                    value = np.nan
                else:
                    value = np.divide(stat, total_time)

                self.extracted_data.loc[idx, col + '_pM'] = value

    def feature_custom(self):
        """
        Calculating custom feature stats that may be useful for the model

        Calculation Ex. ground_strikes_landed / takedowns_landed
        """
        pass

    def feature_elo(self):
        """
        Calculate simulated ELO rating system
        """
        elo = EloSystem(base_rating=800)

        # Sort by date occuring so ELO changes in proper time and get unique fight_urls from 
        fights = self.extracted_data.sort_values(by='date', ascending=True)['fight_url'].unique()
        for fight in fights:

            fight_info = self.extracted_data[self.extracted_data['fight_url'] == fight].filter(items=['fighter','opponent','method','round','result'], axis=1).head(1) # Grab first row containing all information needed for elo calc

            # Custom K parameter based on fight outcome, implementing something based on chess with average-below average fighters being more susceptible to changes
            """
            Cases:  
                    If KO/TKO, SUB: Rnd Calc, K = 15
                    Rnd 1, K*1.5
                    Rnd 2, K*1.4
                    Rnd 3, K*1.3
                    Rnd 4, K*1.2
                    Rnd 5, K*1.1
                
                    If Decision or others: K*1
            """
            fighter1, fighter2, method, round, result = fight_info.values[0]

            if pd.isnull(result): # In case prediction where result is not provided
                continue

            # K calculation
            if method == 'KO/TKO' or method == 'SUB':
                k = 32 * (2.0 - 0.2 * (round - 1))
            else:
                k = 32

            elo.add_match(fighter1=fighter1, fighter2=fighter2, winner=result, k=k)
            
            fighter1_index = self.extracted_data[(self.extracted_data.fighter == fighter1) & (self.extracted_data.fight_url == fight)].index
            fighter2_index = self.extracted_data[(self.extracted_data.fighter == fighter2) & (self.extracted_data.fight_url == fight)].index
    
            self.extracted_data.loc[fighter1_index, 'elo'] = elo.fighters[fighter1]
            self.extracted_data.loc[fighter2_index, 'elo'] = elo.fighters[fighter2]


    def feature_differential(self):
        """
        Calculating differential between two fighters of certain stat

        Calculation: fighter1_stat / fighter2_stat
        """

        # Doing all stats calculated so far after 'height'
        cols = self.extracted_data.loc[:, 'height':].columns
        
        for col in tqdm(cols):

            for idx, row in self.extracted_data.iterrows():
                opponent = row['opponent'] 
                fight_url = row['fight_url'] # in case fighters fought more than once 
                index = self.extracted_data.index[(self.extracted_data.fighter == opponent) & (self.extracted_data.fight_url == fight_url)]
                                        
                if index.empty:
                    continue

                stat_fighter1 = row[col]
                stat_fighter2 = self.extracted_data.loc[index.values[0], col]

                if stat_fighter2 == 0:
                    value = np.nan
                else:
                    value = np.divide(stat_fighter1, stat_fighter2)

                self.extracted_data.loc[idx, col + '_differential'] = value

    def feature_historical(self, shift=1):
        """
        Using groupby per fighter to calculate various stats across their fights
        I'll denote t=0 as the current fight. For example, say fighter1 and fighter2 are fighting. Fighter1's previous 
        fight will be fight (-1) and fight(-2) the one prior to that one. Functions applied to a fighters stats that include 
        fight(0) will be disingenous to the model as they include stats from the fight it is predicting. Fighters with 3 or more prior
        fights will have this calculated. Columns = [ knockdowns : end ]

        Based on Dan's naming convention, I'll use the prefix "precomp_" for all stats calculated not using fight (0)

        Ones with a * will include a precomp_ version or only make sense as being precomp

        One issue is with the NaNs populating some columns, nanmean obviously works but that leaves some issues with the feature analysis

        Calculations:

        Prior*: Previous fights (fight (-1)) stats                                              | precomp_*_prior
        Mean*: Mean of the stat across prior fights, min 3 fights needed: t-1                   | precomp_*_avg
        Window_Mean: Windowed average across prior 3 fights: t-3, t-2, t-1                      | precomp_*_windowavg 
        Var*: Variance of the stat across prior fights: t0 & t-1                                | precomp_*_var
        Window_Var: Windowed variance of the stat across prior 3 fights: t-3, t-2, t-1          | precomp_*_windowvar
        Peak*: Peak stat across prior fights                                                    | precomp_*_peak
        Low*: Minimum stat across prior fights                                                  | precomp_*_low
        Delta*: Change in stat across last two fights: t-2 and t-1                              | precomp_*_delta 

        """

        # Grouped object by fighter_url sorted by date to assure shifting is correct
        group = self.extracted_data.sort_values(by='date', ascending=True).groupby(by='fighter_url', group_keys=False)

        # Each column that we will be calculating (all columns after height)
        cols = self.extracted_data.loc[:, 'height':].columns

        for col in tqdm(cols):

            # Prior stat from last fight, first fight is set as NaN by default
            self.extracted_data['precomp_' + col + '_prior'] = group[col].shift(shift)

            # Mean stat from prior fights, t0 fight includes t-1,t-2,...., t-1 fight includes t-2, t-3
            self.extracted_data['precomp_' + col + '_avg'] = group[col].apply(lambda x : x.shift(shift).expanding(min_periods=2).mean())

            # Windowed mean from prior 3 fights, will make this adjustable in the future
            self.extracted_data['precomp_' + col + '_windowavg'] = group[col].apply(lambda x : x.shift(shift).rolling(window=3, min_periods=2).mean())

            # Variance stat
            self.extracted_data['precomp_' + col + '_var'] = group[col].apply(lambda x : x.shift(shift).expanding(min_periods=2).var())

            # Windowed variance 
            self.extracted_data['precomp_' + col + '_windowvar'] = group[col].apply(lambda x : x.shift(shift).rolling(window=3, min_periods=2).var())

            # Peak stat across prior fights
            self.extracted_data['precomp_' + col + '_peak'] = group[col].apply(lambda x : x.shift(shift).expanding().max())

            # Min stat across prior fights 
            self.extracted_data['precomp_' + col + '_low'] = group[col].apply(lambda x : x.shift(shift).expanding().min())

            # Delta between last fight and fight before that 
            self.extracted_data['precomp_' + col + '_delta'] = group[col].apply(lambda x : x.shift(shift).diff())

    def feature_diff_opponent(self):
        """
        Basically take every stat and find the absolute difference between 
        """

        group = self.extracted_data.sort_index(ascending=True).groupby(by='fight_url', group_keys=False)

        cols = self.extracted_data.loc[:, 'height':].columns
        for col in tqdm(cols): 

            # Do diff relative to second fighter in group, row2 - row1
            second_stat = group[col].apply(lambda x : x.diff(1))

            # Do diff relative to first fighter in group, row1 - row2
            first_stat = group[col].apply(lambda x : x.diff(-1))

            # Combine them based on NaN values 
            stat_vs_opp = first_stat.fillna(second_stat)

            self.extracted_data[col + '_vs_opp'] = stat_vs_opp
