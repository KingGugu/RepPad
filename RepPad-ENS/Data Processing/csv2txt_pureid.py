# -*- coding: utf-8 -*-

import tqdm
import pandas as pd


def get_org_rank_users(dataframe):
    users = []
    for user in dataframe['user_id:token'].unique():
        users.append(user)
    return users


def convert_unique_idx(dataframe, column_name):
    column_dict = {x: i for i, x in enumerate(dataframe[column_name].unique())}
    dataframe[column_name] = dataframe[column_name].apply(column_dict.get)
    dataframe[column_name] = dataframe[column_name].astype('int')
    assert dataframe[column_name].min() == 0
    assert dataframe[column_name].max() == len(column_dict) - 1
    return dataframe, column_dict


def main(dataset, date_file_name):
    df = pd.read_csv(date_file_name, sep='\t', engine='python')
    df, user_mapping = convert_unique_idx(df, 'user_id:token')
    df, item_mapping = convert_unique_idx(df, 'item_id:token')
    items = []
    for item in df['item_id:token']:
        items.append(item)
    if 0 in items:
        df['item_id:token'] = df['item_id:token'].apply(lambda x: x + 1)
    users = []
    for user in df['user_id:token']:
        users.append(user)
    if 0 in users:
        df['user_id:token'] = df['user_id:token'].apply(lambda x: x + 1)

    df.sort_values(by='user_id:token', axis=0, inplace=True)
    df_copy = df.copy()
    rank_users = get_org_rank_users(df_copy)
    grouped = df_copy.groupby(['user_id:token'])
    item_file = open(dataset + '.txt', mode='w')

    for user in tqdm.tqdm(rank_users):
        temp = grouped.get_group(user)
        temp = temp.sort_values(by=['timestamp:float'])
        temp_copy = temp.copy()
        items = []
        for i in range(temp_copy.shape[0]):
            row = temp_copy.iloc[i]
            items.append(int(row['item_id:token']))
        item_line = str(user) + "\t" + "\t".join([str(i) for i in items])
        item_file.write(item_line + '\n')
    item_file.close()


dataset = 'Beauty'
date_file_name = 'Beauty_5.csv'
main(dataset, date_file_name)
