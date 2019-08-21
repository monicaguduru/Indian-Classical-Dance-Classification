import csv
import os
import numpy as np
def get_data():
        """Load our data from file."""
        with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)
            # print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
            # print(data)

        return data

data = get_data()
# print(data)

seq_length = 30
class_limit = 6
for video in data:
    # print(video)
    path = os.path.join('data', 'sequences', video[1]+video[2] + '-' + str(seq_length) + '-features'+'.npy')
    path1 = os.path.join('data1', 'sequences', video[1]+video[2] + '-' + str(seq_length) + '-features'+'.npy')
    outpath = os.path.join('data2', 'sequences', video[1]+video[2] + '-' + str(seq_length) + '-features')
    filenames = [path, path1]
    sequence = []
    # with open(outpath, 'w') as outfile:
    for fname in filenames:
        dataArray = np.load(fname)
        sequence.append(dataArray)
    np.save(outpath, sequence)