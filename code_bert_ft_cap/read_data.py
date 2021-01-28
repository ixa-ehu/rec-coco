import numpy as np


def readDATA(fileDir):
#Input: a csv with first column containing words and the rest numerical data
#Output: 1.numerical matrix (np.array) with row vectors of words. 2.list of words (i.e., first column)
    words,vectors = [], []

    with open(fileDir) as infile:
        for line in infile:
            line = line.strip()
            line = line.split(",")
            word = line[0]
            words.append(word)
            vect = line[1:len(line)]
            vect = [float(t) for t in vect]
            vectors.append(vect)
        vectors = np.array(vectors)

    return (words, vectors)


def load_dict_data(fileDir):
    # Opens the visual genome parsed data and gets a dictionary for EACH SAMPLE for all variables
    DICT_DATA, i = [], 0

    with open(fileDir) as f:
        for line in f:
            i = i + 1
            splits = line.rstrip('\n').split(',')
            if i == 1:
                var_names = splits
                var_names = [var_names[j].replace('\r', '') for j in range(len(var_names))]
            else:
                new_example = {}
                if len(var_names) == len(splits):
                    for j in range(len(var_names)):
                        new_example[var_names[j]] = splits[j]
                    DICT_DATA.append(new_example)

    return DICT_DATA


def load_training_data(fileDir):
    #INPUT: fileDir: is the filename that the function load_dict_data function above needs
    #OUTPUT: a vector (of samples) of each of the columns (variables) of the training data
    dict_data = load_dict_data(fileDir)
    var_names = [key for key in dict_data[0]]
    VECTORS = {}
    for k in range(len(var_names)):
        VECTORS[var_names[k]] = []
    for i in range(len(dict_data)):
        for j in range(len(var_names)):
            VECTORS[var_names[j]].append( dict_data[i][var_names[j]] )
    return VECTORS



def readWordlist(wordsDir):
    words = open(wordsDir).read().splitlines()
    return words
