import csv as csv
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from feature_similarity import feature_correlation
import re

def check(string):
    sum = 0
    for i in range(0, len(string)):
        if string[i] >= '0' and string[i] <= '9':
            sum += 1
            continue
        if string[i] == '.':
            continue
        return False
    if sum != 0:
        return True
    return False

def extract_name_feature(name):
    m = re.search(r"([a-zA-z( )]+), ([a-zA-z]+). ([a-zA-z(.)]+)", name)
    result = list(m.groups())
    return result[0], result[1]

# Normalization features
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

def prepare_data(input):
    del input[0]
    feature = []
    for i in range(0, len(input)):
        feature_example = [1]*7
        feature.append(feature_example)

    # Default value to be calculated for null values
    default_value = [0]*9
    default_count = [0]*9

    abbrev_map = {}
    abbrev_count = 0.0

    surname_map = {}
    surname_count = 0.0

    for i in range(0, len(input)):
        # Default value for age parameter
        if check(input[i][4]):
            default_count[4] += 1
            default_value[4] += float(input[i][4])
        # Default value for siblings travelling
        if check(input[i][5]):
            default_count[5] += 1
            default_value[5] += float(input[i][5])
        # Default value for parents/ children travelling
        if check(input[i][6]):
            default_count[6] += 1
            default_value[6] += float(input[i][6])
        # Default value for fare
        if check(input[i][8]):
            default_count[8] += 1
            default_value[8] += float(input[i][8])


    for i in range(0, len(input)):
        # Fare feature
        if check(input[i][8]):
            feature[i][0] = float(input[i][8])
        else:
            feature[i][0] = float(default_value[8])/float(default_count[8])
        # Feature for gender
        if input[i][3] == 'male':
            feature[i][1] = 1.0
        else:
            feature[i][1] = 0.0
        # Age feature
        if check(input[i][4]):
            feature[i][2] = float(input[i][4])
        else:
            feature[i][2] = float(default_value[4])/float(default_count[4])
        # Siblings count + Children/Parents feature
        siblings_value = 0
        if check(input[i][5]):
            siblings_value = float(input[i][5])
        else:
            siblings_value = float(default_value[5])/float(default_count[5])
        if check(input[i][6]):
            feature[i][3] = siblings_value + float(input[i][6])
        else:
            feature[i][3] = siblings_value + float(default_value[6])/float(default_count[6])
        # Embarked feature
        if input[i][10] == 'S':
            feature[i][4] = 0.0
        elif input[i][10] == 'C':
            feature[i][4] = 1.0
        else:
            feature[i][4] = 2.0
        # Name feature
        surname, abbrev = extract_name_feature(input[i][2])
        # Abbrev feature
        if abbrev_map.has_key(abbrev):
            feature[i][5] = abbrev_map[abbrev]
        else:
            abbrev_map[abbrev] = abbrev_count
            feature[i][5] = abbrev_count
            abbrev_count += 1.0
        # Surname + cousin count value feature
        value = surname + str(feature[i][3])
        if surname_map.has_key(value):
            feature[i][6] = surname_map[value]
        else:
            surname_map[value] = surname_count
            feature[i][6] = surname_count
            surname_count += 1.0
    return feature

def get_features_vectors(csv_file, test_data=False):
    csv_file_object = csv.reader(open(csv_file, 'rb'))
    data=[]
    for row in csv_file_object:
        data.append(row)
    label = []
    if not test_data:
        i = 1
        while i < len(data):
            label.append(data[i][1])
            del data[i][1]
            i += 1
    feature = prepare_data(data)
    if test_data:
        return feature
    return feature, label



# Main function starts here
feature, label = get_features_vectors('train.csv')
test_feature = get_features_vectors('test.csv', True)

total_features = feature + test_feature
total_features = normalize_features(total_features)

# feature_correlation(total_features)

feature = total_features[:891]
test_feature = total_features[-418:]

# GradientBoostingClassifier
forest = GradientBoostingClassifier()
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

