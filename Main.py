from sklearn.preprocessing import LabelEncoder, OneHotEncoder, QuantileTransformer, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import xgboost as xgb
import pandas as pd
import numpy as np
import requests
import numpy
import time
import sys
import re
import os

numpy.set_printoptions(threshold=sys.maxsize)


def plot_value_counts(df, column_name):
    """
    Plot bar charts of val_count
    """
    value_counts = df[column_name].value_counts()
    value_counts.plot(kind='bar')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.title('Count of Unique Values in {}'.format(column_name))
    plt.show()


def info_print(col_num):
    """
    Print information on each columns.
        - Describe
        - 0
        - Nan
    """
    print(df_main[df_main.columns[col_num]].head())
    print(df_main.columns[col_num], "\n", df_main[df_main.columns[col_num]].describe())
    print("N° of 0", "\t", sum(df_main[df_main.columns[col_num]] == 0))
    print("N° of nan", "\t", sum(df_main[df_main.columns[col_num]].isnull()))
    print("--------------------------------")


def fetch_population():
    """
    Fetch additional information on the regional population
    """
    url = "https://en.wikipedia.org/wiki/Regions_of_Tanzania"

    # Send a GET request to the URL and get the response
    response = requests.get(url)

    # Parse the HTML content of the response using pandas
    dfs = pd.read_html(response.content)

    # Find the table that contains the regions and populations
    df = dfs[1]

    # Extract the names of regions and populations into separate lists
    regions = df["Region"].tolist()
    populations = df["Population(2022)[5]"].tolist()
    data = dict(zip(regions, populations))

    new_dict = {}
    pattern = re.compile(r"\sRegion$")
    for key in data.keys():
        new_key = re.sub(pattern, "", key)
        new_dict[new_key] = data[key]
    return new_dict


def clean_dataset(df):
    """
    Clean the training set and add features
    """
    # c: Labels  --> Categorical encoder
    label_dict = {"functional": 2, "functional needs repair": 1, "non functional": 0}
    df[df.columns[1]] = df[df.columns[1]].map(label_dict)

    # c: Dates  --> Format to datetime & create the age column
    df[df.columns[3]] = pd.to_datetime(df[df.columns[3]], format='%Y-%m-%d')
    df["age"] = df[df.columns[24]].apply(lambda y: (2013 - y) if y != 0 else 55)

    # c: Funders --> Replace nan and 0 with Unknown & replace rare names by "Other"
    df[df.columns[4]] = df[df.columns[4]].fillna("Unknown")
    df[df.columns[4]].replace(str(0), 'Unknown', inplace=True)

    name_counts = df[df.columns[4]].value_counts()
    rare_names = name_counts[name_counts < 50].index
    # df[df.columns[6]].replace(rare_names, 'Other', inplace=True)  --> takes 1.264s to exec
    df[df.columns[4]] = np.where(df[df.columns[4]].isin(rare_names), 'Other',
                                 df[df.columns[4]])  # --> takes 0.001s to exec

    # c: Installer --> Replace nan and 0 with Unknown & replace rare names by "Other"
    df[df.columns[6]] = df[df.columns[6]].fillna("Unknown")
    df[df.columns[6]].replace(str(0), 'Unknown', inplace=True)
    name_counts = df[df.columns[6]].value_counts()
    rare_names = name_counts[name_counts < 200].index
    df[df.columns[6]] = np.where(df[df.columns[6]].isin(rare_names), 'Other',
                                 df[df.columns[6]])

    # c: Regional Population --> Fetch and add regional population to the dataset
    population_dict = fetch_population()
    df["Population_region"] = df[df.columns[13]].map(population_dict)

    # c: Public meeting --> Replace True by 2 - False by 0 and nan by 1
    replace_dict = {True: 2, False: 0, np.nan: 1}
    df[df.columns[19]] = df[df.columns[19]].map(replace_dict)

    # c: Scheme management --> Replace nan by Unknown
    df[df.columns[21]] = df[df.columns[21]].fillna("Unknown")

    # c: Permit --> Replace True by 2 - False by 0 and nan by 1
    replace_dict = {True: 2, False: 0, np.nan: 1}
    df[df.columns[23]] = df[df.columns[23]].map(replace_dict)

    # c: Payment type --> Categorical encoding
    replace_dict = {"annually": 6, "monthly": 5, "never pay": 0, "on failure": 2, "other": 4, "per bucket": 1,
                    "unknown": 3}
    df[df.columns[31]] = df[df.columns[31]].map(replace_dict)

    # c: Quality group --> Categorical encoding
    replace_dict = {"colored": 2, "fluoride": 3, "good": 5, "milky": 4, "salty": 1, "unknown": 0}
    df[df.columns[33]] = df[df.columns[33]].map(replace_dict)

    # c: Quantity group --> Categorical encoding
    replace_dict = {"dry": 0, "enough": 4, "insufficient": 1, "seasonal": 3, "unknown": 2}
    df[df.columns[35]] = df[df.columns[35]].map(replace_dict)

    # c: Drop undesired columns
    df_main = df.drop(['id', 'amount_tsh', 'wpt_name', 'date_recorded', 'gps_height', 'longitude', 'latitude',
                       'num_private', 'subvillage', 'region', 'region_code', 'district_code', 'ward',
                       'recorded_by', 'scheme_management', 'scheme_name', 'construction_year',
                       'extraction_type_group', 'extraction_type_class', 'management_group', 'payment',
                       'water_quality', 'quantity', 'source_type', 'source_class', 'waterpoint_type'], axis=1)
    return df_main


def encode_columns(df, onehot_col, excluded_col, SCALER):
    """
    Encode columns with OneHotEncoder or LabelEncoder
    """
    # c: OneHot encoding of some columns
    if ONEHOT:
        encoder = OneHotEncoder()
        # Fit and transform the specified columns
        onehot_columns = encoder.fit_transform(df[onehot_col])
        # Create a new dataframe to store the one-hot encoded columns
        onehot_df = pd.DataFrame(onehot_columns.toarray(), columns=encoder.get_feature_names(onehot_col))
        # Concatenate the one-hot encoded columns with the original dataframe
        df = pd.concat([df.drop(columns=onehot_col), onehot_df], axis=1)

        # c: Scale non-OneHotEncoded columns
        df = scale_numerical_data(df, excluded_col[1:], SCALER)

    # c: Categorical encoding of all columns
    elif not ONEHOT:
        for c in df.columns:
            # Label encode the column
            label_encoder = LabelEncoder()
            df[c] = label_encoder.fit_transform(df[c].astype(str))

        # c: scale all columns
        df = scale_numerical_data(df, df.columns[1:], SCALER)

    return df


def scale_numerical_data(df, columns_to_scale, SCALER):
    """
    Scale columns with StandardScaler or QuantileTransformer
    """
    # Select only the specified columns
    X = df[columns_to_scale].astype(float)

    if SCALER == "Quantile":
        # Apply QuantileTransformer
        X = QuantileTransformer(output_distribution="normal").fit_transform(X)
    elif SCALER == "Standard":
        X = StandardScaler().fit_transform(X)

    # Replace the specified columns in the original dataframe with the scaled values
    for i, col in enumerate(columns_to_scale):
        df[col] = X[:, i]

    return df


def create_sets(df):
    """
    Split the Training data into TrainingSet and TestingSet
    """
    label = df[df.columns[0]]
    data = df[df.columns[1:]]

    X_train, X_test, y_train, y_test = train_test_split(data, label,
                                                        random_state=1, stratify=label,
                                                        test_size=0.33)
    # print(X_train.shape, y_train.shape)
    # print(X_test.shape, y_test.shape)

    # c: Over Sample minority classes
    oversample = SMOTE(random_state=1)
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_test = xgb.DMatrix(X_test, label=y_test)

    return X_train, X_test, y_train, y_test, xgb_train, xgb_test


def train_models(X_train, X_test, y_train, y_test, xgb_train, xgb_test, model_types, SAVE_REPORT):
    """
    Train different classifiers
    """
    for i, model_type in enumerate(model_types):
        print(f"{i+1}/{len(model_types)}\t", model_type)
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            rf_classifier = RandomForestClassifier()
            rf_classifier.fit(X_train, y_train)
            y_predicted = rf_classifier.predict(X_test)

        elif model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            lr_classifier = LogisticRegression()
            lr_classifier.fit(X_train, y_train)
            y_predicted = lr_classifier.predict(X_test)

        elif model_type == "decision_tree":
            from sklearn.tree import DecisionTreeClassifier
            dt_classifier = DecisionTreeClassifier()
            dt_classifier.fit(X_train, y_train)
            y_predicted = dt_classifier.predict(X_test)

        elif model_type == "svm":
            from sklearn.svm import SVC
            svm_classifier = SVC()
            svm_classifier.fit(X_train, y_train)
            y_predicted = svm_classifier.predict(X_test)

        elif model_type == "naive_bayes":
            from sklearn.naive_bayes import GaussianNB
            nb_classifier = GaussianNB()
            nb_classifier.fit(X_train, y_train)
            y_predicted = nb_classifier.predict(X_test)

        elif model_type == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            knn_classifier = KNeighborsClassifier()
            knn_classifier.fit(X_train, y_train)
            y_predicted = knn_classifier.predict(X_test)

        elif model_type == "neural_network":
            from sklearn.neural_network import MLPClassifier
            nn_classifier = MLPClassifier()
            nn_classifier.fit(X_train, y_train)
            y_predicted = nn_classifier.predict(X_test)

        elif model_type == "XGBoost":
            params = {
                'max_depth': 3,
                'eta': 0.1,
                'objective': 'multi:softmax',
                'eval_metric': 'error',
                'num_class': 3
            }
            num_rounds = 100
            bst = xgb.train(params, xgb_train, num_boost_round=num_rounds)
            y_predicted = bst.predict(xgb_test)

        report = classification_report(y_test, y_predicted, zero_division=True)

        if SAVE_REPORT:
            file_path = f"Models_Report/{model_type}_classification_report.txt"
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    f.write(report)
            else:
                print(f"File {file_path} already exists.")


def hyperparam_tuning(X_train, X_test, y_train, y_test):
    param_grid = {
        'n_estimators': [301, 401, 501],
        'max_depth': [31, 41, 51],
        'max_features': ['sqrt', 'log2']
    }

    rf_classifier = RandomForestClassifier()
    cv = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=3)
    cv.fit(X_train, y_train.values.ravel())
    print(cv.best_params_)


def train_best_model(X_train, X_test, y_train, y_test):
    """
    Train Best model with best Hyper-parameters

        {'max_depth': 41, 'max_features': 'sqrt', 'n_estimators': 501}
        [CV 5/5] END max_depth=41, max_features=sqrt, n_estimators=501;, score=0.863 total time= 2.1min
    """

    rf_classifier_bst = RandomForestClassifier(n_estimators=501, n_jobs=-1, max_depth=41, max_features='sqrt',
                                               bootstrap=True, criterion='gini')
    rf_classifier_bst.fit(X_train, y_train.values.ravel())
    y_predict_bst = rf_classifier_bst.predict(X_test)

    print(classification_report(y_test, y_predict_bst))
    cnf_matrix = confusion_matrix(y_test, y_predict_bst)
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

    # ----------------------------------------
    # c: Plot confusion matrix
    # ----------------------------------------
    plt.figure(figsize=(12, 7))
    sns.set(font_scale=1.2)
    cnf_matrix_norm = cnf_matrix / cnf_matrix.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cnf_matrix_norm, annot=True, cmap=plt.cm.Blues, linewidths=0.2)
    class_names = ['Functional', 'Needs Repair', 'Non-Functional']
    plt.xticks(np.arange(len(class_names)), class_names, rotation=20)
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    plt.xlabel('Predicted Status')
    plt.ylabel('True Status')
    plt.title('Confusion Matrix: Prediction of Water-pump statuses')
    plt.tight_layout()
    plt.show()

    # ----------------------------------------
    # c: Plot confusion matrix
    # ----------------------------------------
    feature_importance = rf_classifier_bst.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, [X_train.columns[i] for i in sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.tight_layout()
    plt.show()


def make_figures(df_pump):
    def get_age_group(x):
        if x <= 10:
            return '0-10'
        elif x > 10 and x <= 20:
            return '10-20'
        elif x > 20 and x <= 30:
            return '20-30'
        elif x > 30 and x <= 40:
            return '30-40'
        else:
            return '40+'

    df_pump['age_group'] = df_pump['age'].apply(get_age_group)

    # Visualize Relation
    plt.subplots(figsize=(12, 8))
    sns.countplot(x=df_pump['age_group'], hue=df_pump['status_group'], order=['0-10', '10-20', '20-30', '30-40', '40+'])
    plt.title('Age Group Vs Water-pump Status')
    plt.tight_layout()
    plt.show()





    # --------------------------------------------------------------
    column = "status_group"
    unique_counts = df_main[column].value_counts()

    # Rename the index to match the desired labels
    unique_counts.index = ["Functioning", "Needs repairs", "Broken"]

    # Calculate the total count to calculate the proportion
    total_count = unique_counts.sum()

    # Plot the bar chart
    plt.bar(unique_counts.index, unique_counts.values, color=['green', 'orange', 'red'])
    plt.ylabel("Count")
    plt.title("Proportion of Pump Status")

    # Add the proportion percentage for each column
    for i, value in enumerate(unique_counts.values):
        plt.text(i, value, str(round(value / total_count * 100, 2)) + "%", ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



    # --------------------------------------------------------------
    unique_counts = df_main[column].value_counts()

    # Rename the index to match the desired labels
    unique_counts.index = ["Functioning", "Needs repairs", "Broken"]

    # Plot the pie chart
    fig, ax = plt.subplots()
    ax.pie(unique_counts.values, labels=unique_counts.index, colors=['green', 'orange', 'red'], autopct='%1.1f%%',
           shadow=False)
    ax.set_title("Proportion of Pump Status")

    ax.axis('equal')
    plt.tight_layout()

    # Add the box for the labels
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    unique_counts = df_main[column].value_counts()

    unique_counts = df_main[column].value_counts()

    # Rename the index to match the desired labels
    unique_counts.index = ["Functioning", "Needs repairs", "Broken"]

    # Plot the pie chart
    plt.pie(unique_counts.values, labels=unique_counts.index, colors=['green', 'orange', 'red'], autopct='%1.1f%%',
            shadow=False)
    plt.title("Proportion of Pump Status")

    # Add the percentage for each column to the legend
    plt.legend(title="Status", labels=['{} - {}%'.format(i, j) for i, j in
                                       zip(unique_counts.index, unique_counts.values / unique_counts.sum() * 100)])

    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    print()
    # --------------------------------------------------------------


if __name__ == '__main__':
    PRINT_INFO = False
    CORREL = True
    ONEHOT = False
    # SCALER = "Standard"
    SCALER = "Quantile"
    TEST_MODELS = False
    HYPER_T = False
    SAVE_REPORT = True

    # ----------------------------------------
    # c: Load datasets
    # ----------------------------------------
    df_label_set = pd.read_csv("Labels.csv")
    df_test_set = pd.read_csv("Testing_set.csv")
    df_train_set = pd.read_csv("Training_set.csv")

    # c: Merge Train and Labels
    df_main = pd.merge(df_label_set, df_train_set, on=["id"])

    # c: Print info on each column
    if PRINT_INFO:
        for i in range(len(df_main.columns)):
            info_print(i)
        """
        SUMMARY:
            # c: 0//    ID
            # c: 1//    Labels     3 uniques
            # c: 2//    Water available     Lots of 0 41639
            # c: 3//    Dates
            # c: 4//    Founder's names     1897 uniques    3635 nan     TODO: encode ?
            # c: 5//    Heights     neg and lots of 0 20438
            # c: 6//    Installer   55745 uniques   3655 nan    TODO: encode ?
            # c: 7//    Longitude   0 1812
            # c: 8//    Latitude
            # c: 9//    Water point name    37400 uniques   TODO: remove num at id=70359
            # c: 10//   ???     0 58643     TODO: remove col
            # c: 11//   Basin   9 uniques   TODO: OneHot
            # c: 12//   Sub village     19287 uniques       371 nan
            # c: 13//   Region      21 uniques
            # c: 14//   Region code     624 uniques
            # c: 15//   District code   563 uniques
            # c: 16//   IGA  Geographic location     125 uniques
            # c: 17//   Ward Geographic location     2092 uniques
            # c: 18//   Population      0 21381
            # c: 19//   Public_meeting  bool    3334 nan
            # c: 20//   Recorded_by     1 unique    TODO: remove
            # c: 21//   Scheme_management   3877 nan    12 uniques
            # c: 22//   Scheme_name     28166 nan   2696 uniques
            # c: 23//   Permit      bool    3056 nan
            # c: 24//   Construction_year   0 20709
            # c: 25//   Extraction_type     18 uniques
            # c: 26//   Extraction_type_group   13 unique
            # c: 27//   Extraction_type_class   7 uniques
            # c: 28//   Management  12 uniques
            # c: 29//   Management_group    5 uniques
            # c: 30//   Payment     7 uniques   TODO: duplicates
            # c: 31//   Payment_type    7 uniques
            # c: 32//   Water_quality   8 uniques
            # c: 33//   Quality_group   6 uniques
            # c: 34//   Quantity    5 uniques
            # c: 35//   Quantity_group      5 uniques   TODO: duplicates
            # c: 36//   Source      10 uniques
            # c: 37//   Source_type     7 uniques
            # c: 38//   Source_class    3 uniques
            # c: 39//   Waterpoint_type     7 uniques
            # c: 40//   Waterpoint_type_group   6 uniques
        """

    # ----------------------------------------
    # c: Correct columns
    # ----------------------------------------
    df_main = clean_dataset(df_main)
    # plot_value_counts(df_main, df_main.columns[1])

    print()

    # ----------------------------------------
    # c: Encode and scale columns
    # ----------------------------------------
    # c: Columns to OneHotEncode
    onehot_col = ['funder', 'installer', 'basin', 'lga', 'extraction_type', 'management', 'source',
                  'waterpoint_type_group']
    # c: Columns not to OneHotEncode
    excluded_col = [c for c in df_main.columns if c not in onehot_col]

    # c: Encode and Scale
    df_main = encode_columns(df_main, onehot_col, excluded_col, SCALER)

    # c: Plot features Correlation Matrix
    if CORREL:
        corrMatrix = df_main[df_main.columns[1:]].corr()
        plt.subplots(figsize=(18, 12))
        sns.heatmap(corrMatrix, annot=True, fmt='.1g')
        plt.title('Correlation of Features Matrix')
        plt.show()

    # ----------------------------------------
    # c: Create Training & Testing set
    # ----------------------------------------
    X_train, X_test, y_train, y_test, xgb_train, xgb_test = create_sets(df_main)

    # ----------------------------------------
    # c: Train classifiers
    # ----------------------------------------

    if TEST_MODELS:
        model_types = ["XGBoost", "random_forest", "logistic_regression", "decision_tree", "naive_bayes", "knn",
                       "neural_network"]
        train_models(X_train, X_test, y_train, y_test, xgb_train, xgb_test, model_types, SAVE_REPORT)



    # ----------------------------------------
    # c: Optimize best model
    # ----------------------------------------

    if HYPER_T:
        hyperparam_tuning(X_train, X_test, y_train, y_test)

    # ----------------------------------------
    # c: Train and assess best model
    # ----------------------------------------
    train_best_model(X_train, X_test, y_train, y_test)


