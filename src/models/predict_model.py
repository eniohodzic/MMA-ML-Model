import numpy as np
import pandas as pd

from ..features.Feature_Extractor import Feature_Extractor


def predict_future_fight(model, df, info, columns):
    """
    For predicting fights that have not happened yet, precomp_stats need to be recalculated using all of that fighter's fights
    I.e, shift(1) -> shift(0)

    Note: Predicting relative to fighter1 

    Input is Transformed Dataset from /interim rerunning feature extraction with blank Future Fight row 

    Info as list:   
    
    Fight Date as string 'YYYY-MM-DD'
    Fighter1 (Fighter) as url
    Fighter2 (Opponent) as url
    """

    date, fighter1_url, fighter2_url = info

    df_f1_index = df.index[(df.fighter_url == fighter1_url)]
    df_f2_index = df.index[(df.fighter_url == fighter2_url)]

    height1, reach1, dob1, fighter1_name = df.loc[df_f1_index[0], ['height','reach','dob', 'fighter']].values
    height2, reach2, dob2, fighter2_name = df.loc[df_f2_index[0], ['height','reach','dob', 'fighter']].values

    df_subset = pd.concat([df.loc[df_f1_index,:], df.loc[df_f2_index,:]])

    row1 = {'date': date, 'fight_url': '?', 'event_url': '?', 'fighter_url': fighter1_url, 'fighter': fighter1_name, 'opponent': fighter2_name, 'height': height1, 'reach': reach1, 'dob': dob1}
    row2 = {'date': date, 'fight_url': '?', 'event_url': '?', 'fighter_url': fighter2_url, 'fighter': fighter2_name, 'opponent': fighter1_name, 'height': height2, 'reach': reach2, 'dob': dob2}

    rows = []
    rows.insert(0, row1)
    rows.insert(0, row2)

    col_order = df_subset.columns
    df_subset = pd.concat([pd.DataFrame(rows), df_subset], ignore_index=True)
    df_subset = df_subset[col_order]

    Extractor = Feature_Extractor(predict_df=df_subset)
    Extractor.extract(shift=1)

    final_df = Extractor.extracted_data

    prediction_df = final_df.iloc[0:2,:]

    results = model.predict_proba(prediction_df[columns])

    print(f'Result of {fighter1_name}: {results[1]}')
    print(f'Result of {fighter2_name}: {results[0]}')
