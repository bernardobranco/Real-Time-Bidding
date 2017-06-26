# COMPGW02/M041: the Web Economics Project

##Authors
- Bernardo Branco
- David Agababyan
- Christoph Sitter

##Running the project
There are two main files in this project, 
 - main.py which does all the data preprocessing, CTR prediction and strategy training
 - main_price_regression.py which predicts the payprice for given impressions (never finished running)

### Using main.py
The main script is implemented to accept command line arguments. Each operation stores the results generated in the folder generated and subsequent commands can then load this data and operate on it. The available commands are:
available command line arguments for main.py:
- --help: print help text
- --dev={True/False}: use reduced size datasets to for development
- --process_data={True,False}: generate per advertiser one-hot encoded datasets including combined features
- --combine_data={True,False}: generate one-hot encoded full dataset with advertisers being a categorical feature
- --train_models={True,False}: train the models on all generated datasets (per advertiser as well as combined)
- --predict={True,False}: use all the models that were trained to predict the CTR and write results to CSV files
- --evaluate_models={True,False}: evaluate all models predictions (roc_auc) and generate table comparing the results
- --train_strategies={val,train,all}: train all strategies as defined in the param_grid in main.py (val = use strategies trained on the validation set, train=trained on training set, all=both)
- --validate_strategy={val,train,all}: validate all trained strategies on the validation set and print performance of best performing model
- --bid_on_testset={val,train,all}: use the best performing model to generate bid prices on the test set

Inside the main.py file there are two things that can/should be modified before running:
```
models_conf = {
    "LogisticRegression": fit_logistic_regression,
    "XGBoost": fit_xgboost,
    "SVC": fit_svc
}
```
Comment out lines for models that should not be trained on the dataset, or add more with respective training functions 

```
strategy_param_grid = {
    "linear": {"base_bid": np.linspace(0.1, 350, 100), "avg_ctr": [advertiser_ctr], "per_advertiser": [True]},
    "quadratic": {"base_bid": np.linspace(0.1, 350, 100), "avg_ctr": [advertiser_ctr], "per_advertiser": [True]},
    "ortb1": {"c": np.linspace(45, 65, 10), "alpha": np.linspace(1e-10, 1e-6, 20), "per_advertiser": [True]},
    "ortb2": {"c": np.linspace(45, 65, 10), "alpha": np.linspace(1e-10, 1e-6, 20), "per_advertiser": [True]},
    "constant": {"const_bid": np.linspace(1, 350, 100)},
}
```
Change the ranges and/or add/remove models. In case new models are added, they'll have to be added to the util/bidding/Strategies class as well
