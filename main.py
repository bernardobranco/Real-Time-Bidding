#!/usr/local/bin/python3

import argparse
import multiprocessing
import timeit

import numpy as np
import pandas as pd
import xgboost
from sklearn import model_selection, linear_model, svm
import matplotlib

matplotlib.use("Agg")

import util.persist
import util.plots
import util.predict
import util.bidding


def process_raw_data(nrows, combine):
    n_train, n_val, n_test = nrows

    train_data = pd.read_csv("./data/train.csv", nrows=n_train)
    train_data_processed, values_filter = util.predict.pipeline(train_data, {}, combine)
    util.persist.store_generated_data(train_data_processed, "train")
    del train_data
    del train_data_processed

    val_data = pd.read_csv("./data/validation.csv", nrows=n_val)
    val_data_processed, _ = util.predict.pipeline(val_data, values_filter, combine)
    util.persist.store_generated_data(val_data_processed, "val")
    del val_data
    del val_data_processed

    test_data = pd.read_csv("./data/test.csv", nrows=n_test)
    test_data_processed, _ = util.predict.pipeline(test_data, values_filter, combine)
    util.persist.store_generated_data(test_data_processed, "test")
    del test_data
    del test_data_processed


def fit_logistic_regression(X, y):
    parameters = {'class_weight': [None, "balanced"], 'C': np.logspace(-5, 4, 10)}
    log_reg = linear_model.LogisticRegression(tol=1e-10, n_jobs=4)
    model = model_selection.GridSearchCV(log_reg, parameters, cv=3, verbose=True, n_jobs=multiprocessing.cpu_count(),
                                         pre_dispatch="1*n_jobs", scoring="roc_auc")
    # model = log_reg  # TODO testing
    model.fit(X, y)
    return model


def fit_svc(X, y):
    parameters = {'C': np.logspace(-5, 1, 7), "kernel": ["rbf", "linear"], "class_weight": [None]}
    svc = svm.SVC(probability=True, tol=1e-10)
    model = model_selection.GridSearchCV(svc, parameters, cv=3, verbose=True, n_jobs=multiprocessing.cpu_count(),
                                         pre_dispatch="1*n_jobs", scoring="roc_auc")
    # model = svc  # TODO testing
    model.fit(X, y)
    return model


def fit_xgboost(X, y):
    param_test1 = {
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 6, 2)
    }
    xgb = xgboost.XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                                min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                seed=27)

    model = model_selection.GridSearchCV(xgb, param_grid=param_test1, pre_dispatch="1*n_jobs", scoring='roc_auc',
                                         n_jobs=multiprocessing.cpu_count(), cv=3)
    # model = xgb  # TODO testing
    model.fit(X, y)
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Let's do this")
    parser.add_argument('--dev', default=False, choices=['True', 'False'],
                        help='Turn on dev mode to reduce dataset size')
    parser.add_argument('--process_data', default=False, choices=['True', 'False'],
                        help='True to generate data, false to load from files')
    parser.add_argument('--combine_data', default=False, choices=['True', 'False'],
                        help='Load per advertiser data and write to a combined dataframe')
    parser.add_argument('--train_models', default=False, choices=['True', 'False'],
                        help='True to train models, False to load them')
    parser.add_argument('--predict', default=False, choices=['True', 'False'],
                        help='Run predictions')
    parser.add_argument('--evaluate_models', default=False, choices=['True', 'False'],
                        help='True to evaluate models and generate ctr predictions')
    parser.add_argument('--train_strategies', default=None, choices=['val', 'train', 'all'],
                        help='val/train to train bidding strategies on respective datasets')
    parser.add_argument('--validate_strategy', default=None, choices=['val', 'train', 'all'],
                        help='Validate the best strategy from training/validation set on the validation set')
    parser.add_argument('--bid_on_testset', default=None, choices=['val', 'train', 'all'],
                        help='Bid on the test set using the best model from training/validation set or both')

    args = parser.parse_args()

    args.dev = bool(args.dev)
    args.process_data = bool(args.process_data)
    args.combine_data = bool(args.combine_data)
    args.train_models = bool(args.train_models)
    args.predict      = bool(args.predict)
    args.evaluate_models = bool(args.evaluate_models)

    # budgets used to train the strategies and find the best performing ones
    budgets = [None] #[None, 6250]

    models_conf = {
        "LogisticRegression": fit_logistic_regression,
        "XGBoost": fit_xgboost,
        "SVC": fit_svc
    }

    n_train, n_val, n_test = (200000, 50000, 50000) if args.dev else (None, None, None)

    if args.process_data:
        process_raw_data((n_train, n_val, n_test), False)

    if args.combine_data:
        process_raw_data((n_train, n_val, n_test), True)

    if args.train_models:
        # Train individual models first (memory reasons)
        processed_train_data = util.persist.load_processed_data("train", exclude=["all"])
        models = util.predict.train_models_per_advertiser(processed_train_data, models_conf)

        # And then throw away the data and train combined one (memory reasons)
        processed_train_data = util.persist.load_processed_data("train", include=["all"])
        models.update(util.predict.train_models_per_advertiser(processed_train_data, models_conf))
        util.persist.save_with_pickle(models, "./generated/models.npz")
    else:
        models = util.persist.load_from_pickle("./generated/models.npz")

    util.persist.save_all_figures()
    if args.predict:
        for dataset_name, raw_ds_name, n_rows in [("train", "./data/train.csv", n_train),
                                                  ("val", "./data/validation.csv", n_val),
                                                  ("test", "./data/test.csv", n_test)]:
            processed_data = util.persist.load_processed_data(dataset_name)
            use_cols = ["bidid", "advertiser"] if dataset_name == "test" else ["bidid", "advertiser", "click"]
            true_data = pd.read_csv(raw_ds_name, usecols=use_cols,
                                    dtype={"bidid": 'object', "advertiser": np.int32, "click": np.int8}, nrows=n_rows)

            for model_name in models_conf.keys():
                predictions = util.predict.predict_all_with_models(processed_data, true_data, models, model_name)
                predictions.to_csv("./generated/predictions_{}_{}.csv".format(dataset_name, model_name), index=False)

    util.persist.save_all_figures()
    if args.evaluate_models:
        for dataset_name, raw_ds_name, n_rows in [("train", "./data/train.csv", n_train),
                                                  ("val", "./data/validation.csv", n_val)]:

            roc_auc_scores = pd.DataFrame(columns=["advertiser"])
            for model_name in models_conf.keys():
                predictions = pd.read_csv("./generated/predictions_{}_{}.csv".format(dataset_name, model_name))

                if dataset_name != "test":
                    scores = util.plots.plot_roc_auc(predictions, model_name)

                    advertisers = [x for x in scores.keys()]
                    combined_model_roc = [x["combined"] for x in scores.values()]
                    advertiser_model_roc = [x["advertiser"] for x in scores.values()]
                    scores_df = pd.DataFrame({
                        "advertiser": advertisers,
                        "roc_auc_combined_model_{}".format(model_name): combined_model_roc,
                        "roc_auc_advertiser_model_{}".format(model_name): advertiser_model_roc})

                    util.plots.plot_confusion_matrix(predictions, model_name)
                    roc_auc_scores = roc_auc_scores.merge(scores_df, on="advertiser", how="outer")

            roc_auc_scores.to_csv("./generated/model_roc_auc_scores_{}.csv".format(dataset_name))

    util.persist.save_all_figures()
    if args.train_strategies:
        for dataset_name, raw_ds_name in [("train", "./data/train.csv"), ("val", "./data/validation.csv")]:
            if dataset_name != args.train_strategies and args.train_strategies != "all":
                continue

            print("Training strategies on {}".format(dataset_name))
            for model_name, _ in models_conf.items():
                pool = multiprocessing.Pool(multiprocessing.cpu_count())
                train_df = pd.read_csv("./generated/predictions_{}_{}.csv".format(dataset_name, model_name))
                train_df = train_df.merge(pd.read_csv(raw_ds_name, usecols=['bidid', 'payprice']), on='bidid',
                                          how='inner')

                advertiser_ctr = {x: y["click"].mean() for x, y in train_df.groupby("advertiser")}
                advertiser_ctr.update({"all": train_df["click"].mean()})

                strategy_param_grid = {
                    "combined": {"c": np.linspace(45, 65, 10), "alpha": np.linspace(1e-10, 1e-6, 20), "per_advertiser": [True], "avg_ctr": [advertiser_ctr]},
                    "random": {"lower_bound": np.linspace(1,50, 10), "upper_bound": np.linspace(200, 300, 10)},
                    "linear": {"base_bid": np.linspace(0.1, 350, 100), "avg_ctr": [advertiser_ctr], "per_advertiser": [True]},
                    "quadratic": {"base_bid": np.linspace(0.1, 350, 100), "avg_ctr": [advertiser_ctr], "per_advertiser": [True]},
                    "ortb1": {"c": np.linspace(45, 65, 10), "alpha": np.linspace(1e-10, 1e-6, 20), "per_advertiser": [True]},
                    "ortb2": {"c": np.linspace(45, 65, 10), "alpha": np.linspace(1e-10, 1e-6, 20), "per_advertiser": [True]},
                    "constant": {"const_bid": np.linspace(1, 350, 100)},
                }

                all_strategy_builders = util.bidding.build_cv_strategies(strategy_param_grid)

                for budget in budgets:
                    advertiser_results = {}
                    for advertiser, group in train_df.groupby("advertiser"):
                        print("Strategy training for advertiser {} - model {} budget: {}".format(advertiser, model_name, budget))
                        futures = list()
                        for strategy_builder in all_strategy_builders:
                            futures.append(pool.apply_async(util.bidding.async_validate, (group, strategy_builder, budget)))

                        results = list()
                        t0 = timeit.default_timer()
                        for f in futures:
                            results.append(f.get())
                            if len(results) % 100 == 0:
                                print("Training strategies: {}/{} - {:.2f}s".format(len(results), len(futures), timeit.default_timer() - t0))
                                t0 = timeit.default_timer()

                        advertiser_results.update({advertiser: results})
                        title = "{}({}): Clicks vs money spent ({}) budget: {}".format(model_name, dataset_name, advertiser, budget)
                        util.plots.plot_bidding_strategy_results(util.bidding.strategy_results_to_df(results), group['click'].sum(), title)
                        util.persist.save_all_figures()

                    util.persist.save_with_pickle(advertiser_results, "./generated/strategy_{}_results_{}_budget_{}.npz".format(dataset_name, model_name, budget))

    util.persist.save_all_figures()
    if args.validate_strategy:
        for dataset_name, raw_ds_name in [("train", "./data/train.csv"), ("val", "./data/validation.csv")]:
            if dataset_name != args.validate_strategy and args.validate_strategy != "all":
                continue

            for model_name, _ in models_conf.items():
                val_df = pd.read_csv("./generated/predictions_val_{}.csv".format(model_name))
                val_df = val_df.merge(pd.read_csv("./data/validation.csv", usecols=['bidid', 'payprice']), on='bidid', how='inner')

                for budget in budgets:
                    strategy_train_results = util.persist.load_from_pickle("./generated/strategy_{}_results_{}_budget_{}.npz".format(dataset_name, model_name, budget))

                    best_strategies = {}
                    for advertiser, strategy in strategy_train_results.items():
                        scores = [x[2]["score"] for x in strategy]
                        best_strategy = strategy[scores.index(max(scores))]
                        best_strategies[advertiser] = util.bidding.Strategies.new(best_strategy[0], best_strategy[1])

                    print("Evaluating strategies trained on {} - {} with budget {}".format(dataset_name, model_name, budget))
                    res = util.bidding.validate_strategy(val_df, best_strategies, budget_limit=6250, single_strategy=False)
                    print("Evaluation result for strategies trained on {} with budget {}: {}".format(dataset_name, budget, res))

    util.persist.save_all_figures()
    if args.bid_on_testset:
        for dataset_name, raw_ds_name in [("train", "./data/train.csv"), ("val", "./data/validation.csv")]:
            if dataset_name != args.bid_on_testset and args.bid_on_testset != "all":
                continue

            for model_name, _ in models_conf.items():
                test_df = pd.read_csv("./generated/predictions_val_{}.csv".format(model_name))

                for budget in budgets:
                    strategy_train_results = util.persist.load_from_pickle("./generated/strategy_{}_results_{}_budget_{}.npz".format(dataset_name, model_name, budget))

                    best_strategies = {}
                    for advertiser, strategy in strategy_train_results.items():
                        scores = [x[2]["score"] for x in strategy if x[0]]
                        best_strategy = strategy[scores.index(max(scores))]
                        best_strategies[advertiser] = util.bidding.Strategies.new(best_strategy[0], best_strategy[1])

                    tmp_df = pd.concat([test_df, test_df.apply(lambda row: best_strategies[row["advertiser"]](row), axis=1).rename("bidprice")], axis=1)
                    tmp_df.to_csv("./generated/testing_bidding_price_{}_{}_{}.csv".format(dataset_name, model_name, budget), columns=["bidid", "bidprice"], index=False)
                    print("Bidding done with strategy trained on {} with budget {}".format(dataset_name, budget))


util.persist.save_all_figures()
