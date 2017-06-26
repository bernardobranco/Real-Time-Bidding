import multiprocessing
import random

import numpy as np
import pandas as pd


class Strategies:
    def __init__(self, round_bids=False):
        self.round_bids = round_bids

    def linear(self, base_bid, avg_ctr, per_advertiser):
        def ctr(row):
            return row["p_ctr_adv"] if per_advertiser else row["p_ctr_all"]

        def advertiser(row):
            return row["advertiser"] if per_advertiser else "all"

        def bid(row):
            res = base_bid * ctr(row) / avg_ctr[advertiser(row)]
            return np.rint(res) if self.round_bids else res

        return bid

    def quadratic(self, base_bid, avg_ctr, per_advertiser):
        def ctr(row):
            return row["p_ctr_adv"] if per_advertiser else row["p_ctr_all"]

        def advertiser(row):
            return row["advertiser"] if per_advertiser else "all"

        def bid(row):
            res = base_bid * (ctr(row) / avg_ctr[advertiser(row)])**2
            return np.rint(res) if self.round_bids else res

        return bid

    def ortb1(self, c, alpha, per_advertiser):
        def ctr(row):
            return row["p_ctr_adv"] if per_advertiser else row["p_ctr_all"]

        def bid(row):
            res = np.sqrt((c / alpha) * ctr(row) + c ** 2) - c
            return np.rint(res) if self.round_bids else res

        return bid

    def ortb2(self, c, alpha, per_advertiser):
        def ctr(row):
            return row["p_ctr_adv"] if per_advertiser else row["p_ctr_all"]

        def bid(row):
            res = c * (np.power(((ctr(row) + np.sqrt(c ** 2 * alpha ** 2 + ctr(row) ** 2)) / (c * alpha)), 1 / 3) -
                       np.power((c * alpha) / (ctr(row) + np.sqrt(c ** 2 * alpha ** 2 + ctr(row) ** 2)), 1 / 3))
            return np.rint(res) if self.round_bids else res

        return bid

    def constant(self, const_bid):
        def bid(row):
            return np.rint(const_bid) if self.round_bids else const_bid

        return bid

    def random(self, lower_bound, upper_bound):
        def bid(row):
            return random.randint(int(lower_bound), int(upper_bound))

        return bid

    def combined(self, c, alpha, per_advertiser, avg_ctr):
        def ctr(row):
            return row["p_ctr_adv"] if per_advertiser else row["p_ctr_all"]

        def advertiser(row):
            return row["advertiser"] if per_advertiser else "all"

        def bid(row):
            res = (np.sqrt((c / alpha) * ctr(row) + c ** 2) - c) * ctr(row) / avg_ctr[advertiser(row)]
            return np.rint(res) if self.round_bids else res

        return bid

    @classmethod
    def new(cls, strategy_name, strategy_params, round_bids=False):
        if strategy_name == "linear":
            return Strategies(round_bids).linear(**strategy_params)
        if strategy_name == "quadratic":
            return Strategies(round_bids).linear(**strategy_params)
        if strategy_name == "ortb1":
            return Strategies(round_bids).ortb1(**strategy_params)
        if strategy_name == "ortb2":
            return Strategies(round_bids).ortb2(**strategy_params)
        if strategy_name == "constant":
            return Strategies(round_bids).constant(**strategy_params)
        if strategy_name == "random":
            return Strategies(round_bids).random(**strategy_params)
        if strategy_name == "combined":
            return Strategies(round_bids).combined(**strategy_params)

        raise Exception("No such strategy", strategy_name)


def _build_cv_param_permutations(head, tail, accumulator):
    if len(tail) == 0:
        accumulator.append({x[0]: x[1] for x in head})
    else:
        p_name, p_values = tail[0]
        for val in p_values:
            new_head = head + [(p_name, val)]
            _build_cv_param_permutations(new_head, tail[1:], accumulator)


def build_cv_strategies(strategy_cv_param_dict):
    strategies = list()
    for strategy, params in strategy_cv_param_dict.items():
        params_list = [x for x in params.items()]
        accumulator = list()
        _build_cv_param_permutations(list(), params_list, accumulator)
        for conf in accumulator:
            strategies.append((strategy, conf))

    return strategies


def validate_strategy(predictions, strategy, max_clicks=None, budget_limit=None, clicks_weight=2, single_strategy=True):
    max_spend = predictions['payprice'].sum() if budget_limit is None else budget_limit
    max_clicks = predictions['click'].sum() if max_clicks is None else max_clicks

    if single_strategy: 
        res = pd.concat([predictions, predictions.apply(strategy, axis=1).rename("bidprice")], axis=1)
    else:
        res = pd.concat([predictions, predictions.apply(lambda row: strategy[row["advertiser"]](row), axis=1).rename("bidprice")], axis=1)
    res['won'] = res["bidprice"] > res["payprice"]
    res['paid_cumsum'] = (res['won'] * res['payprice']).cumsum(axis=0)

    if budget_limit is not None:
        res = res[res['paid_cumsum'] <= budget_limit * 1000]

    won_clicks = res[res['won'] == True]['click'].sum()
    money_spent = res[res['won'] == True]['payprice'].sum()
    score = 1 / np.sqrt(
        np.power(clicks_weight * (1 - won_clicks / max_clicks), 2) +
        np.power(0 - money_spent / max_spend, 2))

    return {"won": len(res[res['won'] == True]),
            "lost": len(res[res['won'] == False]),
            "lost_clicks": max_clicks - won_clicks,
            "won_clicks": won_clicks,
            "total_spent": money_spent / 1000,
            "score": score}


def async_validate(train_df, strategy, budget_limit):
    return strategy[0], strategy[1], validate_strategy(train_df, Strategies.new(strategy[0], strategy[1]),
                                                       budget_limit=budget_limit)


def strategy_results_to_df(all_results, ignore_params=["avg_ctr"]):
    res_df = None
    for name, params, res in all_results:
        df_dict = {"name": [name]}
        df_dict.update(res)

        fringe = list(params.items())
        while fringe:
            key, val = fringe.pop()
            if key not in ignore_params:
                if isinstance(val, dict):
                    [fringe.append((k, v)) for k, v in val.items()]
                else:
                    df_dict.update({key: val})

        df = pd.DataFrame(df_dict)
        res_df = df if res_df is None else pd.concat([res_df, df], axis=0)

    return res_df.reset_index()
