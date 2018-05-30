import pandas as pd
import numpy as np
from collections import defaultdict

train_data = pd.read_csv("matches.csv")

train_data = train_data.fillna(0)

i = 0

numColumns = len(train_data.columns)

landMarkDict = defaultdict(int)

for index, row in train_data.iterrows():

    i = 0

    my_dict = defaultdict(int)


    for i in range(numColumns):

        if i == 0:
            continue

        if row[i]  == 0:
            continue
        else:

            j = row[i] // 500
            my_dict[j] += 1

    #k1,vr1 = max(my_dict, key=my_dict.get)
    #kr,vr = max(my_dict.items(), key=lambda k: k[1])
    inverse = [(value, key) for key, value in my_dict.items()]
    print( max(inverse)[1])
    #print(kr)
    #print(vr)
    landMarkDict[row[0]] = max(inverse)[1]



    #maximum = max(my_dict, key=my_dict.get)  # Just use 'min' instead of 'max' for minimum.
    #print(maximum, my_dict[maximum])

