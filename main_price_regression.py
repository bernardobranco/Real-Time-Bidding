import numpy as np
# Evaluation metric:
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import sklearn.linear_model as linear_model
import timeit
import math
import pickle
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
import util.persist
import util.predict


def rmse_cv(model, train_df, y_train):
    rmse = np.sqrt(-cross_val_score(model, train_df, y_train, scoring="neg_mean_squared_error", cv=3, n_jobs=4, pre_dispatch="1*n_jobs"))
    return rmse.mean()


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def eval_validation(X, model, y_pred):
    valid_rmse = rmse(y_pred, model.predict(X))
    print('Model obtained rmse score of: %f' % (valid_rmse))
    return valid_rmse


def model_CV(model_name, X, Y, X_val, Y_val):
    # Lasso Model
    print("Lets run the lasso model")
    t0 = timeit.default_timer()

    best_score = math.inf
    best_model = None
    best_parameters = None

    # Lasso Model:
    if model_name == 'Lasso':
        alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
        for alpha in alphas:
            model = linear_model.Lasso(alpha=alpha, max_iter=50000)
            score = rmse_cv(model, X, Y)
            if score < best_score:
                best_score = score
                best_model = model
            print("Best alpha was {:.5f}s".format(model.alpha_))

        # getting score of validation set:
        eval_validation(X_val, best_model, Y_val)

    # Elastic Net:
    elif model_name == 'EN':
        l1_ratios = [.1, .5, .7, .9, .95, .99, 1]
        for l1 in l1_ratios:
            model = linear_model.ElasticNet(l1_ratio=l1, max_iter=50000)
            score = rmse_cv(model, X, Y)
            if score < best_score:
                best_score = score
                best_model = model
            print("Best alpha was {:.5f}s".format(model.alpha_))

        # getting score of validation set:
        eval_validation(X_val, best_model, Y_val)

    # XGB
    elif model_name == 'XGB':
        # XGB initial parameters
        gbm = xgb.XGBRegressor(
            max_depth=5,
            subsample=0.5,
            colsample_bytree=0.2,
            silent=1,
        )

        # parameters to do grid search on
        gbm_params = {
            'max_depth': [2, 3, 4, 5, 6],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
            'reg_alpha': [0, 0.0001, 0.0005, 0.001, 0.01, 0.1],
            'reg_lambda': [0, 0.0001, 0.0005, 0.001, 0.01, 0.1]
        }
        best_model = GridSearchCV(gbm, param_grid=gbm_params, cv=3, n_jobs=4, pre_dispatch="1*n_jobs")
        best_model.fit(X, Y)

        best_parameters, score, _ = max(best_model.grid_scores_, key=lambda x: x[1])
        print(score)
        for param_name in sorted(best_parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))

        # getting score of validation set:
        eval_validation(X_val, best_model, Y_val)

    else:
        print('That isnt a valid model!')
        exit(1)

    file_name = "./generated/price_regression_model_{}.npz".format(model_name)
    print("Saving model to {}".format(file_name))
    util.persist.save_with_pickle(best_model, file_name)

    print("Training took took {:.3f}s".format(timeit.default_timer() - t0))
    return best_model, best_parameters, best_score


def ensemble(X_val, Y_val, model_en_name, model_lasso_name, model_xgb_name):
    model_en = pickle.loads(model_en_name)
    y_pred_en = model_en.predict(X_val)

    model_lasso = pickle.loads(model_lasso_name)
    y_pred_lasso = model_lasso.predict(X_val)

    model_xgb = pickle.loads(model_xgb_name)
    y_pred_xgb = model_xgb.predict(X_val)

    y_pred = (y_pred_en + y_pred_lasso + y_pred_xgb) / 3

    print("Ensmble obtained rmse score of: %f" % (rmse(Y_val, y_pred)))


if __name__ == "main":
    X_train = util.persist.load_processed_data("train", include=['all'])["all"]
    y_train = pd.DataFrame(X_train['bidid'].to_dense()).merge(pd.read_csv("./data/train.csv", usecols=['bidid', 'payprice']), on='bidid')['payprice']
    print("got train y")

    X_train.drop(["bidid", "click"], axis=1, inplace=True)
    print("dropped unwanted train cols")
    X_train = util.predict.sparse_df_to_array(X_train)
    print("got sparse train")

    X_val = util.persist.load_processed_data("val", include=['all'])["all"]
    y_val = pd.DataFrame(X_val['bidid'].to_dense()).merge(pd.read_csv("./data/validation.csv", usecols=['bidid', 'payprice']), on='bidid')['payprice']
    print("got val y")

    X_val.drop(["bidid", "click"], axis=1, inplace=True)
    print("dropped unwanted val cols")
    X_val = util.predict.sparse_df_to_array(X_val)
    print("got sparse val")

    for name in ["Lasso", "XGB"]:
        model, params, score = model_CV(name, X_train, y_train, X_val,y_val)
        print("{}: {} -- {}".format(name, score, params))
