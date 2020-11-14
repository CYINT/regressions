import numpy as np
import pandas as pd
import statsmodels.api as sm
from warnings import simplefilter
from statsmodels.tsa.stattools import acf
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, bartlett, levene, shapiro, normaltest, boxcox, PearsonRConstantInputWarning


def calculate_residuals(model, X, y):
    predictions = model.predict(X)
    residuals = y - predictions
    return residuals 

def has_multicolinearity(X, colinearity_threshold=0.6, ignore_nan=True):
    columns = X.columns
    for column_under_test in columns:
        for column in columns:
            if column_under_test == column:
                continue

            simplefilter("ignore", PearsonRConstantInputWarning)
            result = pearsonr(X[column_under_test], X[column])
            simplefilter("default", PearsonRConstantInputWarning)
            if np.isnan(result[0]) and not ignore_nan:
                return True
            elif np.isnan(result[0]):
                continue 

            if abs(result[0]) >= colinearity_threshold:
                return True

    return False

def model_score_acceptable(model, threshold):
    return True

def normal_test(residuals, ha_threshold=0.05):
    result = shapiro(residuals)
    if ha_threshold >= result[1]:
        return False

    return True

def errors_autocorrelate(residuals, autocorrelation_threshold=0.6):
    result = acf(residuals, nlags=40, fft=False)
    test = abs(result[1:]) >= autocorrelation_threshold
    if True in test:
        return True

    return False

def error_features_correlate(residuals, X, correlation_threshold=0.6):
    for column in X.columns:
        a = X[column].to_numpy()
        if (a[0] == a).all():
            continue
        result = pearsonr(residuals, X[column])
        if abs(result[0]) >= correlation_threshold:
            return True
    
    return False 

def is_homoscedastic(residuals, y, ha_threshold=0.05):
    result = bartlett(residuals, y)
    if ha_threshold >= result[1]:
        return False
    
    return True

def select_best_features(dataset, train_model_type, alpha=0.05, max_feature_row_ratio=0.25, threshold=0.05, cv=5):
    X_train, y_train, X_test, y_test = dataset
    feature_names = X_train.columns
    model_candidates = []
    for column in feature_names:
        a = X_train[column].to_numpy()
        if (a[0] == a).all():
            X_train.drop(column, axis='columns', inplace=True)
            X_test.drop(column, axis='columns', inplace=True)
    
    train_indices = X_train.index   
    test_indices = X_test.index
    feature_max = int(np.ceil(y_train.shape[0] * max_feature_row_ratio))
    if feature_max > X_train.shape[1]:
        feature_max = X_train.shape[1]

    for i in range(1, feature_max):
        X_train_fs, X_test_fs, fs = select_features(X_train.copy(), y_train.copy(), X_test.copy(), i)
        if X_train_fs.shape[0] == 0:
            continue
        X_train_fs = pd.DataFrame(X_train_fs)
        X_test_fs = pd.DataFrame(X_test_fs)
        X_train_fs.index = train_indices
        X_test_fs.index = test_indices
        indices = fs.get_support(indices=True)
        selected_features = feature_names[indices]
        X_train_fs.columns = selected_features
        model, dataset = train_model_type(X_train_fs, y_train, X_test_fs, y_test)
        if model_score_acceptable(model, threshold):
            model_candidates.append({
                'model': model,
                'dataset': dataset,
                'features': selected_features
            })

    return select_winning_model(model_candidates)

def select_features(X_train, y_train, X_test, k):
    fs = SelectKBest(score_func=f_regression, k=k)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def join_dataset(X_train, X_test, y_train, y_test):
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])
    X.sort_index(inplace=True)
    y.sort_index(inplace=True)
    return [X, y]

def boxcox_transform(y, min_translation=0.01):
    a = min_translation - y.min()
    y_transformed, y_lambda = boxcox(y+a)
    return [y_transformed, y_lambda, a]

def detect_overfitting(model, dataset, cv=5, overfit_threshold=0.5, scorer=None):
    X_train, y_train, X_test, y_test = dataset
    training_score = cross_val_score(model, X_train, y_train, scoring=scorer, cv=cv).mean()
    test_score = cross_val_score(model, X_test, y_test, scoring=scorer, cv=cv).mean()
    if training_score > test_score * overfit_threshold:
        return True
    
    return False

def satisfies_gauss_markov(model, dataset):
    X_train, _, y_train, _ = dataset
    residuals = calculate_residuals(model, X_train, y_train)
    no_multicolinearity = not has_multicolinearity(X_train)
    normal_errors = normal_test(residuals)
    no_autocorrelation = not errors_autocorrelate(residuals)
    no_error_feature_correlation = not error_features_correlate(residuals, X_train)
    homoscedasticity = is_homoscedastic(residuals, y_train)
    return [homoscedasticity, no_multicolinearity, normal_errors, no_autocorrelation, no_error_feature_correlation]

def print_verbose(message, verbose=True):
    if verbose:
        print(message)

def select_non_overfit(model_candidates, cv=5, overfit_threshold=0.5, scorer=None):
    not_overfit = []
    for model_set in model_candidates:
        model = model_set['model']
        dataset = model_set['dataset']
        if not detect_overfitting(model, dataset, cv, overfit_threshold, scorer):
            not_overfit.append(model_set)

    return not_overfit

def select_satisfies_gauss_markov(candidate_list, transform_heteroscedastic=False, boxcox_translation=0.01):
    passed_gauss_markov = []
    for model_set in candidate_list:
        gauss_markov_conditions = satisfies_gauss_markov(model_set['model'], model_set['dataset'])
        if not False in gauss_markov_conditions:
            passed_gauss_markov.append(model_set)
            continue
    
        homoscedasticity, no_multicolinearity, normal_errors, no_autocorrelation, no_error_feature_correlation = gauss_markov_conditions
        if not homoscedasticity and no_multicolinearity and normal_errors and no_autocorrelation and no_error_feature_correlation and transform_heteroscedastic:
            X, y = join_dataset(X_train, X_test, y_train, y_test)
            model, dataset, transform_vars = boxcox_transform(y, boxcox_translation)
            X_train, X_test, y_train, y_test = dataset
            residuals = calculate_residuals(model, X_train, y_train)
            homoscedasticity = is_homoscedastic(residuals, y_train)
            new_model_set = {
                'model': model,
                'dataset': dataset,
                'features': model_set['features'],
                'transform': transform_vars
            }

            if homoscedasticity:
                passed_gauss_markov.append(new_model_set)
    
    return passed_gauss_markov


def select_passed_accuracy_test(candidate_list, accuracy_tests=[0.25,0.5,0.95]):
    passed_accuracy_test = []

    for model_set in candidate_list:
        #TODO implement accuracy testing
        passed_accuracy_test.append(model_set)

    return passed_accuracy_test

def select_best_score(candidate_list, cv=5, scorer=None):
    best_score = -9999
    for model_set in candidate_list:
        model = model_set['model']
        X_train, _, y_train, _ = model_set['dataset']
        score = cross_val_score(model, X_train, y_train, scoring=scorer, cv=cv)
        if score > best_score:  
            best_score = score
            winning_model = model_set
    
    return winning_model

def select_winning_model(model_candidates, cv=5, overfit_threshold=0.5, accuracy_tests=[0.25,0.5,0.95], transform_heteroscedastic=True, boxcox_translation=0.01, scorer=None, verbose=False):
    winning_model = None
    candidate_list = model_candidates
    candidate_list = select_non_overfit(candidate_list, cv, overfit_threshold, scorer)
    candidate_list = select_satisfies_gauss_markov(candidate_list, transform_heteroscedastic, boxcox_translation)
    candidate_list = select_passed_accuracy_test(candidate_list, accuracy_tests)
    winning_model = select_best_score(candidate_list, cv, scorer)
    return winning_model