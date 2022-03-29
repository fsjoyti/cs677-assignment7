import math
import operator

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid
from Point import Point


def calculate_point_distances(test_point, X_trainPoint):
    point_distances = []
    for point in X_trainPoint:
        distance = math.dist(test_point.coordinate, point.coordinate)
        point_distances.append((point, distance))
    point_distances = sorted(point_distances, reverse=False, key=lambda x: x[1])
    return point_distances


def kNeighboringPoint(test_point, train_point, k):
    distance = calculate_point_distances(test_point, train_point)
    neighbors = []
    for i in range(k):
        neighbors.append(distance[i][0])
    return neighbors


def point_hyperplane(test_point, neighbor_point, training_points):
    test_coordinate = test_point.coordinate
    neighbor_coordinate = neighbor_point.coordinate
    test_x, test_y = test_coordinate[0], test_coordinate[1]
    neighbor_x, neighbor_y = neighbor_coordinate[0], neighbor_coordinate[1]
    test_neighbor_vector = np.array([test_x - neighbor_x, test_y - neighbor_y])
    negative_points = []
    for point in training_points:
        point_coordinate = point.coordinate
        x_coordinate = point_coordinate[0]
        y_coordinate = point_coordinate[1]
        train_neighbor_vector = np.array([x_coordinate - neighbor_x, y_coordinate - neighbor_y])
        dot = np.dot(test_neighbor_vector, train_neighbor_vector)
        if dot < 0:
            negative_points.append(point.label)
    if np.sum(negative_points == 'Green') >= len(negative_points) / 2:
        return 1
    else:
        return 0


def k_hyperPlane(testing_points, training_points, k):
    predicted_y = []
    for point in testing_points:
        k_neighbors = kNeighboringPoint(point, training_points, k)
        neighbor_label = []
        for neighbor in k_neighbors:
            neighbor_label.append(point_hyperplane(point, neighbor, training_points))
        if sum(neighbor_label) >= len(neighbor_label) / 2:
            predicted_y.append('Green')
        else:
            predicted_y.append('Red')

    return predicted_y


def trade_with_labels(df, labels, title):
    calculated_prices = [100]
    for i in range(len(labels)):
        if labels[i] == 'Green':
            calculated_price = calculated_prices[i] * pow((1 + df['mean_return'].iloc[i] / 100), 5)
            calculated_prices.append(calculated_price)
        else:
            calculated_prices.append(calculated_prices[i])
    week_number_np = df["Week_Number"].unique()
    week_number_np_additional_week_appended = np.append(week_number_np, 53)
    week_number_list = list(week_number_np_additional_week_appended)
    print(max(calculated_prices))
    plt.xlabel('Week Number')
    plt.ylabel('Account Balance')
    plt.title(title)
    plt.plot(week_number_list, calculated_prices)
    plt.show()


def get_yearly_data(df_year1, df_year2):
    y_year1 = df_year1['Label']
    y_year2 = df_year2['Label']
    X_year1 = df_year1[["mean_return", "volatility"]]
    X_year2 = df_year2[["mean_return", "volatility"]]
    scaler = StandardScaler()
    X_year1 = scaler.fit_transform(X_year1)
    X_year2 = scaler.fit_transform(X_year2)
    return X_year1, y_year1, X_year2, y_year2


def calculate_distances(testInstance, X_train):
    pairs = []
    for (mean_return, volatility) in zip(X_train['mean_return'], X_train['volatility']):
        pairs.append((mean_return, volatility))

    distances = []
    for pair in pairs:
        distance = math.dist(testInstance, pair)
        distances.append((pair[0], pair[1], distance))

    distances = sorted(distances, reverse=False, key=lambda x: x[2])
    return distances


def get_kneighboring_points(testInstance, X_train, k):
    distances = calculate_distances(testInstance, X_train)
    neighboring_points = []
    for i in range(k):
        node = distances[i]
        neighboring_points.append((node[0], node[1]))
    return neighboring_points


def predict_kneighbor(testInstance, X_train, y_train, k):
    neighboring_point = get_kneighboring_points(testInstance, X_train, k)
    knn = KNeighborsClassifier(n_neighbors=k)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(neighboring_point)
    return y_pred


def kpn(X_train, y_train, X_test, k):
    predicted_y = []
    for (f1, f2) in zip(X_test['mean_return'], X_test['volatility']):
        neighboring_point = predict_kneighbor((f1, f2), X_train, y_train, k)
        neighboring_point_np = np.array(neighboring_point)
        green_label_count = np.sum(neighboring_point_np == 'Green')
        if green_label_count >= k / 2:
            predicted_y.append('Green')
        else:
            predicted_y.append('Red')

    return predicted_y

'''
reference euclidean method knn to compare with the other distances
'''
def euclidean_distance_calculations(df_year1, df_year2):
    X_year1, y_year1, X_year2, y_year2 = get_yearly_data(df_year1, df_year2)
    scores_list = []
    k_range = range(3, 13, 2)
    X_train, X_test, y_train, y_test = train_test_split(X_year1, y_year1, test_size=0.5, random_state=0)
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))
    plt.plot(k_range, scores_list, color='red', linestyle='dashed', marker='o', markerfacecolor='black',
             markersize=10)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing accuracy')
    plt.show()
    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(X_train, y_train)
    predicted_labels = knn.predict(X_year2)
    print("Original KNN accuracy ", metrics.accuracy_score(y_year2, predicted_labels))
    trade_with_labels(jpm_df_year2, predicted_labels, 'Euclidean distance')

'''
method for manhattan distance knn
'''
def manhattan_distance(df_year1, df_year2):
    X_year1, y_year1, X_year2, y_year2 = get_yearly_data(df_year1, df_year2)
    X_train, X_test, y_train, y_test = train_test_split(X_year1, y_year1, test_size=0.5, random_state=0)
    k_range = range(3, 13, 2)
    scores_list = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, p=1)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))
    plt.plot(k_range, scores_list, color='red', linestyle='dashed', marker='o', markerfacecolor='black',
             markersize=10)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing accuracy')
    plt.title('Manhattan Distance knn')
    plt.show()
    knn = KNeighborsClassifier(n_neighbors=9, p=1)
    knn.fit(X_year1, y_year1)
    predicted_labels = knn.predict(X_year2)
    accuracy = metrics.accuracy_score(y_year2, predicted_labels)
    print("Manhattan distance accuracy: ", accuracy)
    confusion_matrix = metrics.confusion_matrix(y_year2, predicted_labels)
    print(metrics.confusion_matrix(y_year2, predicted_labels))
    print("No this is the same value of k I used regular knn")
    true_positive_rate = (confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1]))
    true_negative_rate = (confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1]))
    print(f"True positive rate for year 2: {true_positive_rate}, true negative rate for year2 : {true_negative_rate}")
    trade_with_labels(jpm_df_year2, predicted_labels, 'Manhattan Distance')
    print("Yes this gives higher accuracy than euclidean distance for predicting labels")
    return metrics.accuracy_score(y_year2, predicted_labels)

'''
method for doing calculating the minkowski distance knn
'''
def minkowski_distance(df_year1, df_year2):
    X_year1, y_year1, X_year2, y_year2 = get_yearly_data(df_year1, df_year2)
    X_train, X_test, y_train, y_test = train_test_split(X_year1, y_year1, test_size=0.5, random_state=0)
    k_range = range(3, 13, 2)
    scores_list = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, p=1.5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))
    plt.plot(k_range, scores_list, color='red', linestyle='dashed', marker='o', markerfacecolor='black',
             markersize=10)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing accuracy')
    plt.title('Minkowski Distance knn')
    plt.show()
    knn = KNeighborsClassifier(n_neighbors=9, p=1.5)
    knn.fit(X_year1, y_year1)
    predicted_labels = knn.predict(X_year2)
    accuracy = metrics.accuracy_score(y_year2, predicted_labels)
    print("Minkowski distance accuracy: ", accuracy)
    confusion_matrix = metrics.confusion_matrix(y_year2, predicted_labels)
    print(confusion_matrix)
    print("No this is the same value of k I used regular knn")
    true_positive_rate = (confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1]))
    true_negative_rate = (confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1]))
    print(f"True positive rate for year 2: {true_positive_rate}, true negative rate for year2 : {true_negative_rate}")
    trade_with_labels(df_year2, predicted_labels, 'Minkowski Distance')
    print("No this gives same accuracy as euclidean distance for predicting labels")

'''
method for calculating the nearest_centroid knn
'''
def nearest_centroid(df_year1, df_year2):
    X_year1, y_year1, X_year2, y_year2 = get_yearly_data(df_year1, df_year2)
    df_year1_green = df_year1[df_year1['Label'] == 'Green']
    green_centroid = (df_year1_green['mean_return'].mean(), df_year1_green['volatility'].mean())
    print("Green centroid: ", green_centroid)
    df_year1_red = df_year1[df_year1['Label'] == 'Red']
    red_centroid = (df_year1_red['mean_return'].mean(), df_year1_red['volatility'].mean())
    print("Red centroid: ", red_centroid)
    green_distance = []
    for (mean, volatility) in zip(df_year1_green['mean_return'], df_year1_green['volatility']):
        t = (mean, volatility)
        green_distance.append(math.dist(t, green_centroid))
    print('Average and Median distance for red labels: ', np.mean(green_distance), np.median(green_distance))
    red_distance = []
    for (mean, volatility) in zip(df_year1_red['mean_return'], df_year1_red['volatility']):
        t = (mean, volatility)
        red_distance.append(math.dist(t, red_centroid))
    print('Average and Median distance for green labels: ', np.mean(red_distance), np.median(red_distance))
    clf = NearestCentroid()
    clf.fit(X_year1, y_year1)
    predicted_labels = clf.predict(X_year2)
    accuracy = metrics.accuracy_score(y_year2, predicted_labels)
    print("Nearest centroid accuracy: ", accuracy)
    confusion_matrix = metrics.confusion_matrix(y_year2, predicted_labels)
    true_positive_rate = (confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1]))
    true_negative_rate = (confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1]))
    print(f"True positive rate for year 2: {true_positive_rate}, true negative rate for year2 : {true_negative_rate}")
    trade_with_labels(jpm_df_year2, predicted_labels, 'Nearest Centroid')
    print("Yes this gives higher accuracy than euclidean distance for predicting labels")

'''
method for calculating the domain transformation knn
'''
def domain_transformation(df_year1, df_year2):
    df_year1_transformed = df_year1
    df_year1_transformed = df_year1_transformed.assign(xx=df_year1_transformed['mean_return'] ** 2)
    df_year1_transformed = df_year1_transformed.assign(
        xy=df_year1_transformed['mean_return'] * df_year1_transformed['volatility'] * math.sqrt(2))
    df_year1_transformed = df_year1_transformed.assign(yy=df_year1_transformed.volatility ** 2)
    df_year2_transformed = df_year2
    df_year2_transformed = df_year2_transformed.assign(xx=df_year1_transformed['mean_return'] ** 2)
    df_year2_transformed = df_year2_transformed.assign(
        xy=df_year2_transformed['mean_return'] * df_year2_transformed['volatility'] * math.sqrt(2))
    df_year2_transformed = df_year2_transformed.assign(yy=df_year2_transformed.volatility ** 2)
    k_range = range(3, 13, 2)
    X_year1, y_year1, X_year2, y_year2 = get_yearly_data(df_year1_transformed, df_year2_transformed)
    X_train, X_test, y_train, y_test = train_test_split(X_year1, y_year1, test_size=0.5, random_state=0)
    scores_list = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))
    plt.plot(k_range, scores_list, color='red', linestyle='dashed', marker='o', markerfacecolor='black',
             markersize=10)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing accuracy')
    plt.title('Domain Transformation')
    plt.show()
    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(X_train, y_train)
    predicted_labels = knn.predict(X_year2)
    accuracy = metrics.accuracy_score(y_year2, predicted_labels)
    print("Domain transformation accuracy: ", accuracy)
    confusion_matrix = metrics.confusion_matrix(y_year2, predicted_labels)
    print(confusion_matrix)
    true_positive_rate = (confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1]))
    true_negative_rate = (confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1]))
    print(f"True positive rate for year 2: {true_positive_rate}, true negative rate for year2 : {true_negative_rate}")
    trade_with_labels(df_year2, predicted_labels, 'Domain Transformation')

'''
method for calculating the k_predicted_neighbors knn
'''
def k_predicted_neighbors(df_year1, df_year2):
    X_year1 = df_year1[["mean_return", "volatility"]]
    y_year1 = df_year1["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X_year1, y_year1, test_size=0.5, random_state=0)
    scores_list = []
    k_range = range(3, 13, 2)
    for k in k_range:
        y_pred = kpn(X_train, y_train, X_test, k)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))
    plt.plot(k_range, scores_list, color='red', linestyle='dashed', marker='o', markerfacecolor='black',
             markersize=10)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing accuracy')
    plt.title('K-predicted neighbors')
    plt.show()
    X_year2 = df_year2[["mean_return", "volatility"]]
    y_year2 = df_year2["Label"]
    predicted_labels = kpn(X_year1, y_year1, X_year2, 7)
    accuracy = metrics.accuracy_score(y_year2, predicted_labels)
    print("K-predicted neighbors accuracy: ", accuracy)
    confusion_matrix = metrics.confusion_matrix(y_year2, predicted_labels)
    print(confusion_matrix)
    true_positive_rate = (confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1]))
    true_negative_rate = (confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1]))
    print(f"True positive rate for year 2: {true_positive_rate}, true negative rate for year2 : {true_negative_rate}")
    trade_with_labels(df_year2, predicted_labels, 'K-predicted Neighbors')

'''
method for calculating the k_hyperplane knn
'''
def k_hyperplane_method(df_year1, df_year2):
    X_year1 = df_year1[["mean_return", "volatility"]]
    y_year1 = df_year1["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X_year1, y_year1, test_size=0.5, random_state=0)
    scores_list = []
    training_points = []
    testing_points = []
    k_range = range(3, 13, 2)
    for (p1, p2, label) in zip(X_train['mean_return'], X_train['volatility'], y_train):
        training_points.append(Point((p1, p2), label))
    for (p1, p2, label) in zip(X_test['mean_return'], X_test['volatility'], y_test):
        testing_points.append(Point((p1, p2), label))
    for k in k_range:
        y_pred = k_hyperPlane(testing_points, training_points, k)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))
    plt.plot(k_range, scores_list, color='red', linestyle='dashed', marker='o', markerfacecolor='black',
             markersize=10)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing accuracy')
    plt.title('K Hyperplane')
    plt.show()
    X_year1 = df_year1[["mean_return", "volatility"]]
    y_year1 = df_year1["Label"]
    training_points = []
    for (p1, p2, label) in zip(X_year1['mean_return'], X_year1['volatility'], y_year1):
        training_points.append(Point((p1, p2), label))
    X_year2 = df_year2[["mean_return", "volatility"]]
    y_year2 = df_year2["Label"]
    testing_points = []
    for (p1, p2, label) in zip(X_year2['mean_return'], X_year2['volatility'], y_year2):
        testing_points.append(Point((p1, p2), label))
    predicted_labels = k_hyperPlane(testing_points, training_points, 3)
    accuracy = metrics.accuracy_score(y_year2, predicted_labels)
    print("K hyperplane accuracy: ", accuracy)
    confusion_matrix = metrics.confusion_matrix(y_year2, predicted_labels)
    true_positive_rate = (confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1]))
    true_negative_rate = (confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1]))
    print(f"True positive rate for year 2: {true_positive_rate}, true negative rate for year2 : {true_negative_rate}")
    trade_with_labels(df_year2, predicted_labels, 'K-Hyperplane')

if __name__ == '__main__':

    jpm_df = pd.read_csv('data/JPM_weekly_return_volatility.csv')
    jpm_df_year1 = jpm_df[(jpm_df['Year']) == 2020]
    jpm_df_year2 = jpm_df[(jpm_df['Year']) == 2021]

    euclidean_distance_calculations(jpm_df_year1, jpm_df_year2)

    manhattan_distance(jpm_df_year1, jpm_df_year2)

    minkowski_distance(jpm_df_year1, jpm_df_year2)

    nearest_centroid(jpm_df_year1, jpm_df_year2)

    domain_transformation(jpm_df_year1, jpm_df_year2)

    k_predicted_neighbors(jpm_df_year1, jpm_df_year2)

    k_hyperplane_method(jpm_df_year1, jpm_df_year2)
