"""
USECASE DESCRIPTION
The task involves using clinical data as raw input to perform Mortality Prediction. Considering the last 2000 days of
clinical records, the model will predict for each patent whether they will die 30 days from the query date. The problem
is treated as a classification problem. The clinical data is used to train the model to predict the mortality of unseen
examples.

INPUT DATA
events.csv
Each line of this file consists of a tuple with the format (patient id, event id, event description, timestamp, value).

mortality_events.csv
The data provided in mortality_events.csv contains the patient ids of only the deceased
people. They are in the form of a tuple with the format (patient id, timestamp, label).

event_feature_map.csv
A straight-forward dictionary used for label encoding of the independent variables.

EXTRACT-TRANSFORM-LOAD PIPELINE
This file contains the methods that ingest the clinical data in .csv format, perform transformations on the data,
and output the features in SVMLight format. These will be loaded in models.py and used for training the classification
models.
"""

import pandas as pd
from datetime import datetime


def read_csv(filepath):

    # Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    # Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    # Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, output_path):
    """
    The index date is the day on which mortality is to be predicted.
    For deceased patients, the index date is 30 days prior to the death date.
    For alive patients, the index date is the last event date in data/train/events.csv for each alive patient.
    """

    # get the patient ids of dead and alive
    dead = set(mortality['patient_id'])
    alive = set(events['patient_id']) - dead
    dead = list(dead)
    alive = list(alive)
    
    # split events by dead or alive
    events['timestamp'] = events['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date())
    events_d = events.loc[events['patient_id'].isin(dead)]
    events_a = events.loc[events['patient_id'].isin(alive)].reset_index()
    
    # index date for dead
    index_d = mortality['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date()) - pd.to_timedelta(30, unit='d')
    index_d = pd.concat([mortality['patient_id'], index_d], axis=1, sort=False)
    index_d.set_index(['patient_id'], inplace=True)
    
    # index for alive
    index_a = events_a.pivot_table(index=['patient_id'], values=['timestamp'], aggfunc=max)
    
    # combine index dates for dead and alive
    indx_date = pd.concat([index_a, index_d])
    indx_date.rename({'timestamp': 'indx_date'}, axis=1, inplace=True)
    
    return indx_date


def filter_events(events, indx_date, output_path):
    """
    The observation window is 2000 days and prediction window is 30 days. That means that on the query date, we'll
    consider all clinical events over the past 2000 days and make a prediction for 30 days later.
    The method removes the events that occur outside the observation window.
    """

    # merge indx_date with events on patient_id
    events_indx = pd.merge(events, indx_date, on='patient_id')
    # calculate begining of observation period
    events_indx['start'] = events_indx['indx_date'].astype('datetime64[D]') - pd.to_timedelta(2000, unit='d')
    # determine which rows need to be dropped
    events_indx['drop'] = (events_indx['timestamp'].astype('datetime64[D]') < events_indx['start'].astype('datetime64[D]')) | (events_indx['timestamp'].astype('datetime64[D]') > events_indx['indx_date'].astype('datetime64[D]'))
    # drop rows with events outside the obs period
    filtered_events = events_indx.drop(events_indx[events_indx['drop'] == True].index)
    filtered_events = filtered_events[['patient_id', 'event_id', 'value']]
    
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df, feature_map_df, output_path):

    """
    Sum values for diagnostics and medication events (event id starting with DIAG and DRUG).
    Count occurences for lab events (event id starting with LAB).
    """

    filtered_events_df.dropna(subset = ['value'], inplace = True)
    # separate df for different agg func
    agg_sum = filtered_events_df.loc[filtered_events_df['event_id'].str.contains("DIAG|DRUG")]
    agg_count = filtered_events_df.loc[filtered_events_df['event_id'].str.contains("LAB")]
    # aggregate
    agg_count = agg_count.pivot_table(index=['patient_id', 'event_id'], values=['event_id'], aggfunc='count')
    agg_sum = agg_sum.pivot_table(index=['patient_id', 'event_id'], values=['event_id'], aggfunc='sum')
    aggregated_events = pd.concat([agg_count, agg_sum]).reset_index()
    # replace event_id's with index available in event_feature_map.csv
    aggregated_events['event_id'] = aggregated_events['event_id'].map(feature_map_df.set_index('event_id')['idx'])
    aggregated_events.rename({'event_id': 'idx'}, axis=1, inplace=True)

    # min-max normalization
    max_val = aggregated_events.pivot_table(index=['idx'], values=['value'], aggfunc='max')
    max_val.rename({'value': 'max'}, axis=1, inplace=True)
    aggregated_events = pd.merge(aggregated_events, max_val, on='idx')
    aggregated_events['value'] = aggregated_events['value']/aggregated_events['max']
    aggregated_events = aggregated_events.drop('max', 1)
    aggregated_events.rename({'idx': 'feature_id', 'value': 'feature_value'}, axis=1, inplace=True)
    
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    output_path = '../output/'

    # Calculate index date
    indx_date = calculate_index_date(events, mortality, output_path)

    # Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  output_path)
    
    # Aggregate the event values for each patient
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, output_path)

    aggregated_events['feature_id'] = aggregated_events['feature_id'].astype(int)
    aggregated_events['tuples'] = aggregated_events.apply(lambda row: (row['feature_id'], row['feature_value']), axis=1)
    patient_features = aggregated_events.groupby(['patient_id'])['tuples'].apply(lambda x: x.tolist()).to_dict()

    events['dead'] = 0
    events.loc[events['patient_id'].isin(mortality['patient_id']), 'dead'] = 1
    events['dead'] = events['dead'].astype(int)
    
    mortality_df = events[['patient_id', 'dead']]
    mortality = mortality_df.set_index('patient_id')['dead'].to_dict()

    return patient_features, mortality


def save_svmlight(patient_features, mortality, op_file, op_output):
    """
    Save in SVMLight format as feature matrix is large (we have around 3000 features) and sparse (there are many zero
    values).
    """
    features1 = open(op_file, 'wb')
    features2 = open(op_output, 'wb')
    
    for key in sorted(patient_features):
    
        line1 = "%d" % (mortality[key])
        line2 = "%d %d" % (key, mortality[key])
        for value in sorted(patient_features[key]):
            merged = "%d:%.6f" % (value[0], value[1])
            line1 = line1 + " " + merged
            line2 = line2 + " " + merged
        features1.write(bytes((line1 + " " + "\n"),'UTF-8'))
        features2.write(bytes((line2 + " " + "\n"),'UTF-8'))


def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../output/features_svmlight.train', '../output/features.train')


if __name__ == "__main__":
    main()