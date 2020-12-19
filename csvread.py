# reads in data of any csv, formatted as "label, data1, data2..."
# returns a list of label values, and a list of lists of data values
def read_csv(filename):
    labels = []
    data = []
    file = open(filename, "r")
    for line in file.readlines():
        row = line.rstrip().split(",")
        labels.append(float(row[0]))
        # could be int, but will be normalized later
        data.append([float(i) for i in row[1:]])
    return labels, data


# returns result of read_csv() on "mnist_[type]_0_[maxlabel].csv"
def read_mnist(type, maxlabel):
    return read_csv("mnist_"+type+"_0_"+str(maxlabel)+".csv")


# normalizes read-in data values
#  e.g. [0, 1, 2, 3, 4] -> [0, 0.25, 0.5, 0.75, 1]
# also works on 2D-array, normalizing each row individually
#  e.g. [[1, 2, 3, 4],      [[.25, .5, .75, 1],
#        [2, 2, 3, 4],  ->   [.5, .5, .75, 1],
#        [1, 1, 1, 2]]       [.5, .5, .5, 1]]
def normalize(data, max_val=None):
    if not isinstance(data[0], list):
        if not max_val:
            max_val = max(data)
        return [float(d / max_val) for d in data]
    else:  # data is 2D array, normalize every row
        result = []
        for row in data:
            result.append(normalize(row))
        return result


# returns result of read_csv() on "mnist_[type]_0_[maxlabel].csv"
# with the data rows all normalized
def read_norm_mnist(type, maxlabel):
    labels, data = read_mnist(type, maxlabel)
    data = normalize(data)
    return labels, data
