import cv2
import pandas as pd
import numpy as np


train_data = pd.read_csv('../train_file_with_location_full.csv')

df3 = train_data.groupby('landmark_id')['id'].nunique().sort_values(ascending = False).reset_index(name = 'count')


df = pd.read_csv('train_file_with_class_wise_near_distant_matchesv1.csv')

df4 = df.copy()

df = df.fillna(0)

df2 = df.groupby('landmark_id')['id'].nunique().sort_values(ascending = False).reset_index(name = 'count')

df["pick"] = 0

print("df")

for index, row1 in df2.iterrows():
    if row1['count'] >= 10:

        print("in the first for loop")
        df_landmarkId =  df.loc[df['landmark_id'] == row1['landmark_id'] ]

        df_landmarkId["nearValue"] = df_landmarkId["nearValue"].astype(int)

        df_landmarkId = df_landmarkId.sort_values('nearValue', ascending = False)

        uniqueSeries = df_landmarkId['nearID'].drop_duplicates()
        #df_landmarkCopy = df_landmarkId.copy()

        if uniqueSeries.size >= 10:
            uniqueSeries = uniqueSeries[0:10]

        df2 = df2.loc[df2['id'] != uniqueSeries[:]]


