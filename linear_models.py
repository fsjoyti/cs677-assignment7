# This is a sample Python script.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from tabulate import tabulate


def show_correlation_matrices(surviving_df, deceased_df):
    surviving_df_columns = surviving_df.iloc[:, :-1]
    surviving_df_correlation = surviving_df_columns.corr()
    surviving_df_correlation = surviving_df_correlation.abs()
    _, ax = plt.subplots(figsize=(12, 6))
    plt.title('Correlation matrix of surviving patients')
    sns.heatmap(surviving_df_correlation, annot=True, linewidths=.5, ax=ax)
    plt.savefig('plots/correlation_matrix_surviving_patients.png')
    plt.show()

    deceased_df_columns = deceased_df.iloc[:, :-1]
    deceased_df_correlation = deceased_df_columns.corr()
    deceased_df_correlation = deceased_df_correlation.abs()
    _, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(deceased_df_correlation, annot=True, linewidths=.5, ax=ax)
    plt.savefig('plots/correlation_matrix_deceased_patients.png')
    plt.show()


def compare_linear_models(surviving_df, deceased_df):
    X = surviving_df['platelets']
    y = surviving_df['serum_creatinine']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    loss_functions_surviving = []
    print()
    print("For surviving patients")
    for i in range(1, 6):
        loss_functions_surviving.append(linear_model(X_train, y_train, X_test, y_test, i, 0))
    print()

    X = deceased_df['platelets']
    y = deceased_df['serum_creatinine']
    loss_functions_deceased = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    print()
    print("For deceased patients")
    for i in range(1, 6):
        loss_functions_deceased.append(linear_model(X_train, y_train, X_test, y_test, i, 1))

    return loss_functions_surviving, loss_functions_deceased


def linear_model(X_train, y_train, X_test, y_test, equation, event_type):
    models = ['simple linear regression', 'quadratic', 'cubic spline', 'GLM', 'GLM2']
    events = ['surviving', 'deceased']
    X = X_train
    y = y_train
    exponent = equation

    if equation == 4:
        exponent = 1
        X = np.log(X)
        X_test = np.log(X_test)

    if equation == 5:
        exponent = 1
        X = np.log(X)
        y = np.log(y)
        X_test = np.log(X_test)

    weights = np.polyfit(X, y, exponent)
    # Question 2 part b
    print("weights for " + models[equation - 1] + ": " + str(weights))

    # Question 2 part c
    model = np.poly1d(weights)
    predicted_labels = (model(X_test))

    if equation == 5:
        predicted_labels = np.exp(predicted_labels)

    # Question 2 part d
    plt.figure()
    plt.title(events[event_type] + " patients " + models[equation - 1])
    plt.scatter(X_test, y_test, color='b')
    plt.scatter(X_test, predicted_labels, color='r')
    plt.show()
    loss_function = round(((y_test - predicted_labels) ** 2).sum(), 2)
    print("Loss function (SSE for model) ", loss_function)
    return loss_function


if __name__ == '__main__':

    # Question 1 part 1
    hearts_df = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
    hearts_df = hearts_df[['creatinine_phosphokinase', 'serum_creatinine', 'serum_sodium', 'platelets', 'DEATH_EVENT']]
    df_0 = hearts_df[(hearts_df['DEATH_EVENT']) == 0]
    df_1 = hearts_df[(hearts_df['DEATH_EVENT']) == 1]

    # Question 1 part 2
    show_correlation_matrices(df_0, df_1)

    # Question 1 part 3
    print("Serum sodium and serum creatinine have the highest correlation for surviving patients")
    print("Platelets and Serum Sodium have the lowest correlation for surviving patients")

    print("Serum Sodium and Creatinine Phosphokinase have the highest correlation for deceased patients")
    print("Platelets and Serum Creatinine have the lowest correlation for deceased patients")

    print("No the results are not same for both cases")

    # Question 2
    print("Using group 4 for the next questions")

    sse_surviving, sse_deceased = compare_linear_models(df_0, df_1)

    # Question 3
    equations = ['y = ax + b', 'y = ax2 + bx + c', 'y = ax3 + bx2 + cx + d', 'y = a log x + b', 'log y = a log x + b']
    headers = ['Model', 'SSE (death event=0)', '(death event=1)']
    table_rows = []
    for e, s, d in zip(equations, sse_surviving, sse_deceased):
        row = [e, s, d]
        table_rows.append(row)

    print(tabulate(table_rows, headers=headers, tablefmt='orgtbl'))

    # Question 3 part 1
    print("Model 1(y= ax+b) has the smallest SSE for surviving patients "
          "and Model 5(log y = a log x + b) has the smallest SSE for surviving patients")

    # Question 3 part 2
    print("Model 3 (y= ax3 + bx2 + cx + d) has the largest SSE for both surviving and deceased patients")
