import numpy as np
import pandas as pd
import scipy
import sklearn.preprocessing as preprocessing


def sparse_df_to_array(df):
    num_rows = df.shape[0]

    data = []
    row = []
    col = []

    for i, col_name in enumerate(df.columns):
        if isinstance(df[col_name], pd.SparseSeries):
            column_index = df[col_name].sp_index
            if isinstance(column_index, pd.sparse.array.BlockIndex):
                column_index = column_index.to_int_index()

            ix = column_index.indices
            data.append(df[col_name].sp_values)
            row.append(ix)
            col.append(len(df[col_name].sp_values) * [i])
        else:
            data.append(df[col_name].values)
            row.append(np.array(range(0, num_rows)))
            col.append(np.array(num_rows * [i]))

    data_f = np.concatenate(data)
    row_f = np.concatenate(row)
    col_f = np.concatenate(col)

    arr = scipy.sparse.coo_matrix((data_f, (row_f, col_f)), df.shape, dtype=np.int8)
    return arr.tocsr()


def create_usertag_cols(data, column_name='usertag', prefix='ut_', separator=',', inplace=True):
    result_frame = data if inplace is True else pd.DataFrame()

    for idx, row in data.iterrows():
        usertags_string = row[column_name]
        if usertags_string != 'null':
            for tag in usertags_string.split(separator):
                result_frame.set_value(idx, prefix + tag, np.int8(1))

    cols = [x for x in result_frame.columns if prefix in x]
    result_frame.loc[:, cols] = result_frame.loc[:, cols].fillna(0, axis=1).astype(np.int8).to_sparse(
        fill_value=np.int8(0))

    if inplace is True:
        return None
    else:
        return result_frame


def bin_numeric_col(data, col_name, bins, inplace=True):
    bin_names = [x for x in range(0, len(bins) - 1)]
    binned_col = pd.cut(data[col_name], bins, labels=bin_names, include_lowest=True).astype(np.int8)

    if inplace is True:
        data[col_name] = binned_col
        return None
    else:
        return binned_col


def pipeline(data, filter_dict={}, combined=True):
    EXCLUDED_COLS = ['logtype', 'userid', 'url', 'urlid', 'slotid', 'payprice', 'bidprice'] + \
                    ['usertag', 'city', 'weekday', 'hour', 'slotwidth', 'slotheight', 'useragent',
                     'slotformat', 'slotvisibility'] + \
                    ['IP']
    CLICK_COL_NAME = 'click'
    BIDID_COL_NAME = 'bidid'
    CONTINUOUS_COLS = []

    print("Preprocessing, usertags")
    create_usertag_cols(data)
    print("Preprocessing, dayhour")
    data['dayhour'] = data['weekday'] * 24 + data['hour']
    print("Generify IPs")
    data['gen_ip'] = data['IP'].apply(lambda x: ".".join(x.split(".")[:2]) + ".*")
    print("Combine width and height")
    data['slotsize'] = data.slotwidth.astype(str).str.cat(data.slotheight.astype(str), sep='x')

    print("Preprocessing, slotter")
    data['slotter'] = data['useragent'].astype(str).str.cat(data['slotvisibility'].astype(str), sep=',')
    data['slotter'] = data['slotter'].astype(str).str.cat(data['slotformat'].astype(str), sep=',')

    print("Preprocessing, slotprice")
    if "slotprice" not in CONTINUOUS_COLS:
        data['slotprice_bin'] = bin_numeric_col(data, "slotprice", [0, 1, 11, 51, 101, np.inf], inplace=False)
        EXCLUDED_COLS += ["slotprice"]

    data.drop([c for c in EXCLUDED_COLS if c in data.columns], axis=1, inplace=True)
    print("Using columns {}".format(list(data.columns)))

    if CLICK_COL_NAME in data:
        data[CLICK_COL_NAME] = data[CLICK_COL_NAME].astype(np.int8)

    NO_DUMMY_COLS = [CLICK_COL_NAME, BIDID_COL_NAME] + CONTINUOUS_COLS + (["advertiser"] if combined else [])
    processed_data = {}
    for advertiser, group in (zip(["all"], [data]) if combined else data.groupby('advertiser')):
        print("One-Hot data for {}".format(advertiser))
        X = group.reset_index(drop=True)
        processed_data[advertiser] = pd.get_dummies(X, columns=[c for c in group if
                                                                c not in NO_DUMMY_COLS and 'ut_' not in c],
                                                    sparse=True).to_sparse(fill_value=np.int8(0))

    if "columns" not in filter_dict:
        filter_dict["columns"] = {}
        for advertiser, df in processed_data.items():
            filter_dict["columns"][advertiser] = list(sorted(set(df.columns)))
    else:
        for advertiser, X in processed_data.items():
            print("Fix up columns {}".format(advertiser))
            cols_to_delete = [c for c in X if c not in filter_dict['columns'][advertiser]]
            cols_to_add = [c for c in filter_dict['columns'][advertiser] if c not in X]
            print("Deleting {} columns".format(len(cols_to_delete)))
            X.drop(cols_to_delete, axis=1, inplace=True)

            print("Adding {} columns".format(len(cols_to_add)))
            for col in cols_to_add:
                X[col] = pd.SparseSeries(np.int8(0), index=np.arange(len(X)), dtype=np.int8, fill_value=np.int8(0))

    print("Reordering columns")
    for advertiser, X in processed_data.items():
        processed_data[advertiser].sort_index(axis=1, inplace=True)

    if "scaler" not in filter_dict and len(CONTINUOUS_COLS) > 0:
        filter_dict["scaler"] = preprocessing.StandardScaler()
        filter_dict["scaler"].fit(data[CONTINUOUS_COLS])

    for advertiser, data in processed_data.items():
        if len(CONTINUOUS_COLS) > 0:
            data[CONTINUOUS_COLS] = filter_dict["scaler"].transform(data[CONTINUOUS_COLS])

    print("Pipeline done")
    return processed_data, filter_dict


def train_models(X, y, models_conf, how='all'):
    models = {}
    for model_name, model_generator in models_conf.items():
        print("{}: Training {} with {}".format(how, model_name, X.shape))
        model = model_generator(sparse_df_to_array(X), y)
        models[model_name] = model
    return models


def train_models_per_advertiser(all_data, models_conf):
    models = {}
    for advertiser, advertiser_data in all_data.items():
        X, y, _ = prepare_advertiser_frame_for_sklearn(advertiser_data)
        if len(np.unique(y)) != 2:
            print("{}: Can't train, not enough y labels".format(advertiser))
        else:
            models[advertiser] = train_models(X, y, models_conf, advertiser)

    return models


def prepare_advertiser_frame_for_sklearn(dataframe):
    y = dataframe['click'].astype(np.int8).to_dense() if 'click' in dataframe else None
    bidids = dataframe['bidid'].to_dense()
    X = dataframe.drop([x for x in ['click', 'advertiser', 'bidid'] if x in dataframe.columns], axis=1)
    return X, y, bidids


def predict_all_with_models(processed_data, true_data, trained_models, model_name):
    res_df_adv = pd.DataFrame(columns=["bidid", "p_ctr_adv", "p_click_adv"])
    res_df_all = pd.DataFrame(columns=["bidid", "p_ctr_all", "p_click_all"])
    for advertiser, models in trained_models.items():
        if advertiser not in processed_data:
            print("We've got a model, but no data for advertiser {}".format(advertiser))
            continue
        print("Predicting for {} with {}".format(advertiser, model_name))
        model = models[model_name]
        X, _, bidids = prepare_advertiser_frame_for_sklearn(processed_data[advertiser])
        X = sparse_df_to_array(X)
        proba = list(model.predict_proba(X)[:, 1])
        click = list(model.predict(X))

        if advertiser == "all":
            res_df_all = pd.DataFrame({"bidid": bidids, "p_ctr_all": proba, "p_click_all": click})
        else:
            res_df_adv = res_df_adv.append(pd.DataFrame({"bidid": bidids, "p_ctr_adv": proba, "p_click_adv": click}))

    print("Predicting done, merging results")
    res_df = res_df_adv.merge(res_df_all, on='bidid', how='outer')
    return true_data.merge(res_df, on="bidid", how="outer").fillna(0)
