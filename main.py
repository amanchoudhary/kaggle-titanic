import csv as csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

def check(string):
    sum = 0
    for i in range(0, len(string)):
        if string[i] >= '0' and string[i] <= '9':
            sum += 1
            continue
        if string[i] == '.':
            continue
        return True
    if sum != 0:
        return False
    return True


def normalize_features(feature):
    columns = len(feature[0])
    min_value = [10000.0]*columns
    max_value = [-10000.0]*columns

    for row in feature:
        for i in range(0, len(row)):
            min_value[i] = min(min_value[i], row[i])
            max_value[i] = max(max_value[i], row[i])

    for i in range(0, len(feature)):
        for j in range(0, len(feature[i])):
            feature[i][j] = (feature[i][j]-min_value[j])/(max_value[j]-min_value[j])

    return feature

def prepare_data(data):
    feature = []
    label = []


    default_value = [0]*20
    count_value = [0]*20
    for row in data:
        for i in range(0, len(row)):
            if i == 2 or i == 5 or i == 6 or i == 9:
                if check(row[i]):
                    continue
                default_value[i] += float(row[i])
                count_value[i] += 1

    for i in range(0, 20):
        if count_value[i] == 0:
            continue
        default_value[i] /= count_value[i]


    for row in data:
        now = []
        for i in range(0, len(row)):
            if i == 1:
                label.append(int(row[1]))
                continue

            if i == 2 or i == 5 or i == 6 or i == 9:
                if check(row[i]):
                    now.append(default_value[i])
                else:
                    now.append(float(row[i]))
                continue

            if i == 4:
                if row[4] == 'male':
                    now.append(1.0)
                else:
                    now.append(0.0)
                continue

        feature.append(now)

    feature = normalize_features(feature)
    return feature, label

def get_features_vectors(csv_file):
    csv_file_object = csv.reader(open(csv_file, 'rb'))
    header = csv_file_object.next()
    data=[]

    for row in csv_file_object:
        data.append(row)

    feature, label = prepare_data(data)
    return feature, label


def prepare_test_data(data):
    feature = []
    label = []


    default_value = [0]*20
    count_value = [0]*20
    for row in data:
        for i in range(0, len(row)):
            if i == 1 or i == 4 or i == 5 or i == 8:
                if check(row[i]):
                    continue
                default_value[i] += float(row[i])
                count_value[i] += 1

    for i in range(0, 20):
        if count_value[i] == 0:
            continue
        default_value[i] /= count_value[i]


    for row in data:
        now = []
        for i in range(0, len(row)):
            if i == 1 or i == 4 or i == 5 or i == 8:
                if check(row[i]):
                    now.append(default_value[i])
                else:
                    now.append(float(row[i]))
                continue
            if i == 3:
                if row[3] == 'male':
                    now.append(1.0)
                else:
                    now.append(0.0)
                continue
        feature.append(now)

    feature = normalize_features(feature)
    return feature


def get_test_features_vectors(csv_file):
    csv_file_object = csv.reader(open(csv_file, 'rb'))
    header = csv_file_object.next()
    data=[]

    for row in csv_file_object:
        data.append(row)

    feature = prepare_test_data(data)
    return feature


feature, label = get_features_vectors('train.csv')
test_feature = get_test_features_vectors('test.csv')

# forest = RandomForestClassifier(n_estimators = 100)
forest = svm.SVC()
forest.fit(feature, label)
output = forest.predict(test_feature)

final_result = []
current = 892
for row in output:
    temp = []
    temp.append(current)
    temp.append(row)
    final_result.append(temp)
    current += 1
print output




with open('output.csv', 'w') as csvfile:
    fieldnames = ['PassengerId', 'Survived']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in final_result:
        writer.writerow({'PassengerId': row[0], 'Survived': row[1]})

# data = np.array(data)
#
# print data[0::,1::]
# print data[0::,0]
#
# forest = RandomForestClassifier(n_estimators = 100)
#
# # Fit the training data to the Survived labels and create the decision trees
# forest = forest.fit(data[0::,1::],data[0::,0])
#
# # Take the same decision trees and run it on the test data
# output = forest.predict(test_data)