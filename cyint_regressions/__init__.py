import numpy as np
import pandas as pd
import statsmodels.api as sm
import math
from warnings import simplefilter
from statsmodels.tsa.stattools import acf
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, bartlett, levene, shapiro, normaltest, boxcox, PearsonRConstantInputWarning
from scipy.special import inv_boxcox

def walk_forward_test_split(X,y,train_size=0.8, lags=1):
    length = X.shape[0]
    training_length = math.ceil(length * train_size)
    test_length = length - training_length
    X_walk_train = []
    X_walk_test = []
    y_walk_train = []
    y_walk_test = []
    for i in range(training_length-1, length-(lags-1)):
        X_train, X_test = X.iloc[0:i], X.iloc[i+lags-1:(i+(lags-1))+1]
        y_train, y_test = y[0:i], y[i+lags-1:(i+(lags-1))+1]
        X_walk_train.append(X_train)
        X_walk_test.append(X_test)
        y_walk_train.append(y_train)
        y_walk_test.append(y_test)
    
    return X_walk_train, X_walk_test, y_walk_train, y_walk_test

def calculate_residuals(model, X, y, y_substitute=None):
    predictions = model.predict(X)
    residuals = y - predictions
    return residuals 

def has_multicolinearity(X, colinearity_threshold=0.6, ignore_nan=True, verbose=False):
    columns = X.columns
    print_verbose(f"Testing for multicolinearity with a threshold of: {str(colinearity_threshold)}.", verbose=verbose)
    for column_under_test in columns:
        for column in columns:
            if column_under_test == column:
                continue

            simplefilter("ignore", PearsonRConstantInputWarning)
            result = pearsonr(X[column_under_test], X[column])
            simplefilter("default", PearsonRConstantInputWarning)
            if np.isnan(result[0]) and not ignore_nan:
                print_verbose('Constant detected, and ignore_nan is False. Model', verbose=verbose)
                return True
            elif np.isnan(result[0]):
                continue 

            if abs(result[0]) >= colinearity_threshold:
                print_verbose('Multicolinearity detected.', verbose=verbose)
                return True
    
    print_verbose('No multicolinearity within the threshold detected.', verbose=verbose)
    return False

def model_score_acceptable(model, threshold, cv=5, ):
    return True

def normal_test(X, ha_threshold=0.05, verbose=False):
    print_verbose(f"Testing the null hypothesis that the input is normally distributed with an alpha of {str(ha_threshold)}", verbose=verbose)
    result = shapiro(X)
    if ha_threshold >= result[1]:
        print_verbose(f"The p-value of the result is {str(result[1])}, at or below the threshold of {str(ha_threshold)}, therefore we reject the null hypothesis and accept the alternate hypothesis that the input is not normally distributed.", verbose=verbose)
        return False

    print_verbose(f"The p-value of the result is {str(result[1])}, above the threshold of {str(ha_threshold)}, therefore we cannot reject the null hypothesis and accept that the input is normally distributed.", verbose=verbose)
    return True

def errors_autocorrelate(residuals, autocorrelation_threshold=0.6, nlags=40, fft=False, verbose=False):
    print_verbose(f"Testing if errors are autocorrelated with a threshold of {str(autocorrelation_threshold)} for up to {str(nlags)}.", verbose=verbose)
    result = acf(residuals, nlags=nlags, fft=fft)
    test = abs(result[1:]) >= autocorrelation_threshold
    if True in test:
        print_verbose(f"Autocorrelation at or above the threshold detected.", verbose=verbose)
        return True

    print_verbose(f"Autocorrelation at or above the threshold not detected.", verbose=verbose)
    return False

def error_features_correlate(residuals, X, correlation_threshold=0.6, verbose=False):
    print_verbose(f"Testing that errors and features are not correlated with a threshold of {str(correlation_threshold)} or higher.", verbose=verbose)
    for column in X.columns:
        a = X[column].to_numpy()
        if (a[0] == a).all():
            continue
        result = pearsonr(residuals, X[column])
        if abs(result[0]) >= correlation_threshold:
            print_verbose(f"Correlation between errors and features at or above the treshold detected in column: {column}", verbose=verbose)
            return True
    
    print_verbose(f"No correlation detected between errors and features.", verbose=verbose)
    return False 

def is_homoscedastic(residuals, y, ha_threshold=0.05, verbose=False):
    print_verbose(f"Testing for homoscedasticity with an alpha of: {str(ha_threshold)}. The null hypothesis is that the errors are homoscedastic.", verbose=verbose)
    result = bartlett(residuals, y)
    if ha_threshold >= result[1]:
        print_verbose(f"P-value for Bartlett test is {str(result[1])} which is at or below the threshold. We therefore reject the null hypothesis and accept the errors are heteroscedastic.", verbose=verbose)
        return False
    
    print_verbose(f"P-value for Bartlett test is {str(result[1])} which is greater than the threshold. We therefore do not reject the null hypothesis and accept the errors are homoscedastic.", verbose=verbose)
    return True

def select_best_features(dataset, model_type, alpha=0.05, max_feature_row_ratio=0.25, threshold=0.05, cv=5, overfit_threshold=0.5, accuracy_tests=[0.25,0.5,0.95], transform_heteroscedastic=True, boxcox_translation=0.01, scorer=None, verbose=False):
    print_verbose(f"Analyzing {model_type['name']}...", verbose=verbose)
    X_train, y_train, X_test, y_test = dataset
    feature_names = X_train.columns
    model_candidates = []
    for column in feature_names:
        a = X_train[column].to_numpy()
        if (a[0] == a).all():
            print_verbose(f"{column} is a constant. Dropping", verbose=verbose)
            X_train.drop(column, axis='columns', inplace=True)
            X_test.drop(column, axis='columns', inplace=True)
    
    train_indices = X_train.index   
    test_indices = X_test.index
    feature_max = X_train.shape[1]
    if model_type['linearity'] == 'linear':
        feature_max = int(np.ceil(y_train.shape[0] * max_feature_row_ratio))
        if feature_max > X_train.shape[1]:
            feature_max = X_train.shape[1]

    print_verbose(f"Model is {model_type['type']}, analyzing {str(feature_max)} features.", verbose=verbose)

    for i in range(1, feature_max):
        print_verbose(f"Analyzing {str(i)} feature(s)...", verbose=verbose)
        X_train_fs, X_test_fs, fs = select_features(X_train.copy(), y_train.copy(), X_test.copy(), i)
        if X_train_fs.shape[0] == 0:
            print_verbose(f"No features selected.", verbose=verbose)
            continue
        X_train_fs = pd.DataFrame(X_train_fs)
        X_test_fs = pd.DataFrame(X_test_fs)
        X_train_fs.index = train_indices
        X_test_fs.index = test_indices
        indices = fs.get_support(indices=True)
        selected_features = feature_names[indices]
        print_verbose(f"Features selected: {str(selected_features)}", verbose=verbose)
        X_train_fs.columns = selected_features
        X_test_fs.columns = selected_features
        model = model_type['estimator']().fit(X_train_fs, y_train)
        if True in (fs.pvalues_ <= threshold):
            print_verbose(f"F-test contains p-values less than threshold of {threshold}. Selecting model as a candidate.", verbose=verbose)
            model_candidates.append({
                'model': model,
                'dataset': [X_train_fs, X_test_fs, y_train, y_test],
                'features': selected_features,
                'type': model_type['type'],
                'name': model_type['name'],
                'linearity': model_type['linearity']
            })
            continue
        print_verbose(f"F-test contains NO p-values less than threshold of {threshold}. Model rejected as a candidate.", verbose=verbose)

    return select_winning_model(
        model_candidates,
        cv=cv,
        overfit_threshold=overfit_threshold,
        accuracy_tests=accuracy_tests,
        transform_heteroscedastic=transform_heteroscedastic,
        boxcox_translation=boxcox_translation,
        scorer=scorer,
        verbose=verbose
    )

def select_features(X_train, y_train, X_test, k):
    fs = SelectKBest(score_func=f_regression, k=k)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def join_dataset(dataset):
    X_train, X_test, y_train, y_test = dataset
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])
    X.sort_index(inplace=True)
    y.sort_index(inplace=True)
    return [X, y]

def boxcox_transform(y, min_translation=0.01):
    a = min_translation - y.min()
    y_transformed, y_lambda = boxcox(y+a)
    return [y_transformed, y_lambda, a]

def detect_overfitting(model, dataset, cv=5, overfit_threshold=0.5, scorer=None, verbose=False):
    X_train, X_test, y_train, y_test = dataset
    training_score = cross_val_score(model, X_train, y_train, scoring=scorer, cv=cv).mean()
    test_score = cross_val_score(model, X_test, y_test, scoring=scorer, cv=cv).mean()
    if np.isnan(training_score) or np.isnan(test_score):
        print_verbose(f"Training or test score is NaN. Rejecting model..")
        return True

    print_verbose(f"Training score: {str(training_score)}", verbose=verbose)
    print_verbose(f"Test score: {str(test_score)}", verbose=verbose)
    print_verbose(f"Overfit threshold: {str(overfit_threshold)}", verbose=verbose)
    print_verbose(f"Cross validations: {str(cv)}", verbose=verbose)
    print_verbose(f"Scorer: {str(scorer)}", verbose=verbose)
    if training_score > (test_score * overfit_threshold):
        print_verbose(f"Model is overfit.", verbose=verbose)
        return True
    
    print_verbose(f"Model is not overfit.", verbose=verbose)
    return False

def satisfies_gauss_markov(model, dataset, verbose=False):
    X_train, _, y_train, _ = dataset
    residuals = calculate_residuals(model, X_train, y_train)
    no_multicolinearity = not has_multicolinearity(X_train, verbose=verbose)
    normal_errors = normal_test(residuals, verbose=verbose)
    no_autocorrelation = not errors_autocorrelate(residuals, verbose=verbose)
    no_error_feature_correlation = not error_features_correlate(residuals, X_train, verbose=verbose)
    homoscedasticity = is_homoscedastic(residuals, y_train, verbose=verbose)
    return [homoscedasticity, no_multicolinearity, normal_errors, no_autocorrelation, no_error_feature_correlation]

def print_verbose(message, verbose=True):
    if verbose:
        print(message)

def select_non_overfit(model_candidates, cv=5, overfit_threshold=0.5, scorer=None, verbose=False):
    not_overfit = []
    print_verbose(f"Testing for fitness...",verbose)
    for model_set in model_candidates:
        print_verbose(f"Evaluating {str(model_set['name'])} model for fitness..", verbose=verbose)
        print_verbose(f"(Note: a scorer of None uses the default scorer of the estimator)", verbose=verbose)
        if not detect_overfitting(model_set['model'], model_set['dataset'], cv, overfit_threshold, scorer, verbose=verbose):
            not_overfit.append(model_set)
            continue
    
    return not_overfit

def select_satisfies_gauss_markov(candidate_list, transform_heteroscedastic=False, boxcox_translation=0.01, verbose=False, random_state=None):
    passed_gauss_markov = []
    for model_set in candidate_list:
        if 'type' in model_set and model_set['linearity'] == 'non-linear':
            print_verbose('Model is non-linear. Skipping Gauss Markov checks.', verbose=verbose)
            passed_gauss_markov.append(model_set)
            continue

        gauss_markov_conditions = satisfies_gauss_markov(model_set['model'], model_set['dataset'])
        if not False in gauss_markov_conditions:
            print_verbose('All Gauss Markov conditions satisfied...', verbose=verbose)
            passed_gauss_markov.append(model_set)
            continue
    
        homoscedasticity, no_multicolinearity, normal_errors, no_autocorrelation, no_error_feature_correlation = gauss_markov_conditions
        if not homoscedasticity and no_multicolinearity and normal_errors and no_autocorrelation and no_error_feature_correlation and transform_heteroscedastic:
            print_verbose('Attempting Box-Cox transform to correct heteroscedasticity.', verbose=verbose)
            X, y = join_dataset(model_set['dataset'])
            transform_vars = boxcox_transform(y, boxcox_translation)
            y_transformed = transform_vars[0]
            dataset = train_test_split(X, y_transformed, train_size=0.8, random_state=random_state)
            X_train, X_test, y_train, y_test = dataset
            y_train = pd.DataFrame(y_train, index=X_train.index)
            y_test = pd.DataFrame(y_test, index=X_train.index)
            dataset = [X_train, X_test, y_train, y_test]
            model = model_set['model'].fit(X_train, y_train)
            residuals = calculate_residuals(model, X_train, y_train)
            homoscedasticity = is_homoscedastic(residuals, y_train)            
            new_model_set = {
                'model': model,
                'dataset': dataset,
                'features': model_set['features'],
                'transform': transform_vars,
                'type': model_set['type'],
                'name': model_set['name'],
                'linearity': model_set['linearity']
            }

            if homoscedasticity:
                print_verbose('Transformation applied. Model is now homoscedastic.', verbose=verbose)
                passed_gauss_markov.append(new_model_set)
                continue 

        print_verbose('Model did not satisfy Gauss Markov conditions.', verbose=verbose)
    return passed_gauss_markov


def select_passed_accuracy_test(candidate_list, accuracy_tests=[0.25,0.5,0.95], verbose=False):
    passed_accuracy_test = []

    for model_set in candidate_list:
        #TODO implement accuracy testing
        passed_accuracy_test.append(model_set)

    return passed_accuracy_test

def select_best_score(candidate_list, cv=5, scorer=None, verbose=False):
    best_score = -9999
    winning_model = None
    for model_set in candidate_list:
        print_verbose(f"Scoring model: {model_set['name']} using {str(scorer)}", verbose=verbose)
        model = model_set['model']
        X_train, _, y_train, _ = model_set['dataset']
        score = cross_val_score(model, X_train, y_train, scoring=scorer, cv=cv).mean()
        print_verbose(f"Score: {str(score)}.", verbose=verbose)
        print_verbose(f"Cross-validations: {str(cv)}.", verbose=verbose)
        print_verbose(f"Scorer: {str(scorer)}.", verbose=verbose)
        if score > best_score:  
            print_verbose(f"The model beats the high score of: {str(best_score)}.", verbose=verbose)
            best_score = score
            winning_model = model_set
    
    return winning_model

def select_winning_model(model_candidates, cv=5, overfit_threshold=0.5, accuracy_tests=[0.25,0.5,0.95], transform_heteroscedastic=True, boxcox_translation=0.01, scorer=None, verbose=False, random_state=None):
    candidate_list = model_candidates
    candidate_list = select_non_overfit(candidate_list, cv, overfit_threshold, scorer, verbose=verbose)
    candidate_list = select_satisfies_gauss_markov(candidate_list, transform_heteroscedastic, boxcox_translation, verbose=verbose, random_state=random_state)
    candidate_list = select_passed_accuracy_test(candidate_list, accuracy_tests, verbose=verbose)
    winning_model = select_best_score(candidate_list, cv, scorer, verbose=verbose)
    if not winning_model == None:
        print_verbose(f"The winning model is: {winning_model['name']}.")
    else:
        print_verbose(f"No models selected as a winner.")
    
    return winning_model

def random_forest_prediction_intervals(model, X, alpha = 0.05):
    percentile = (1 - alpha)*100
    err_down = []
    err_up = []
    preds = []
    X = np.array(X)
    for item in X:            
        item = np.array(item).reshape(1, -1)
        for pred in model.estimators_:
            preds.append(pred.predict(item))
        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))                
    return err_up, err_down

def model_predict(model, X):
    y = model['model'].predict(X)
    if 'transform' in model:
        y_lambda = model['transform'][1]
        y = inv_boxcox(y, y_lambda) - model['transform'][2]

    return y


#modified from https://saattrupdan.github.io/2020-03-01-bootstrap-prediction/
def bootstrap_prediction_intervals(model_set, X_train, y_train, x0, alpha =  0.05, random_seed=None):
    model = model_set
    # Number of training samples
    n = X_train.shape[0]

    # The authors choose the number of bootstrap samples as the square root 
    # of the number of samples
    nbootstraps = np.sqrt(n).astype(int)
    # Compute the m_i's and the validation residuals

    if random_seed != None:
        np.random.seed(random_seed)
        
    bootstrap_preds, val_residuals = np.empty(nbootstraps), []
    for b in range(nbootstraps):
        train_idxs = np.random.choice(range(n-1), size = n, replace = True)
        val_idxs = np.array([idx for idx in range(n) if idx not in train_idxs])
        model['model'].fit(X_train.iloc[train_idxs], y_train.iloc[train_idxs])
        preds = model_predict(model_set, X_train.iloc[val_idxs])
        val_residuals.append(y_train.iloc[val_idxs] - preds)
        bootstrap_preds[b] = model_predict(model, x0)
    bootstrap_preds -= np.mean(bootstrap_preds)
    val_residuals = np.concatenate(val_residuals)

    # Compute the prediction and the training residuals
    model['model'].fit(X_train, y_train)
    preds = model_predict(model_set, X_train)
    train_residuals = y_train - preds

    # Take percentiles of the training- and validation residuals to enable 
    # comparisons between them
    val_residuals = np.percentile(val_residuals, q = np.arange(100))
    train_residuals = np.percentile(train_residuals, q = np.arange(100))

    # Compute the .632+ bootstrap estimate for the sample noise and bias
    no_information_error = np.mean(np.abs(np.random.permutation(y_train) - \
    np.random.permutation(preds)))
    generalisation = np.abs(val_residuals - train_residuals)
    no_information_val = np.abs(no_information_error - train_residuals)
    relative_overfitting_rate = np.mean(generalisation / no_information_val)
    weight = .632 / (1 - .368 * relative_overfitting_rate)
    residuals = (1 - weight) * train_residuals + weight * val_residuals

    # Construct the C set and get the percentiles
    C = np.array([m + o for m in bootstrap_preds for o in residuals])
    qs = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    percentiles = np.percentile(C, q = qs)

    return percentiles[0], percentiles[1]