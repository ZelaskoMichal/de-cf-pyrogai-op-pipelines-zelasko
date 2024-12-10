"""Module for the implementation of the Stock Portfolio Optimization problem."""

import logging
from typing import Dict, List, NamedTuple, Union

import pandas as pd
import xpress as xp

from dnalib.optimization_models.common.base_formulation import BaseFormulation
from dnalib.optimization_models.xpress_utils.dataframe_helpers import (
    build_bool_constraints,
    build_constraint,
    build_flex_constraints_indexed,
    build_sum_expression,
    build_variable,
)
from dnalib.optimization_models.xpress_utils.formulation_helpers import (
    add_items,
    build_flex_constraints,
    build_named_constraints,
    build_objective_function_with_penalties,
)

logger = logging.getLogger(__name__)


class StockPortfolioOptimization(BaseFormulation):
    """Formulation builder for portfolio stocks optimization."""

    def __init__(
        self,
        parameters: dict,
        stocks: pd.DataFrame,
        xpress_problem: xp.problem,
    ):
        """Formulation builder for portfolio stocks formulation."""
        self._save_dict_values_as_attributes(parameters)
        self.stocks = stocks
        self.xpress_problem = xpress_problem
        self.variables: List[Dict[NamedTuple, xp.var]] = []
        self.constraints: List[
            Union[Dict[NamedTuple, xp.constraint], List[xp.constraint], xp.constraint]
        ] = []
        self.penalty: xp.expression = 0

        logger.info(f"Data loaded. {len(stocks)} rows.")

    def _save_dict_values_as_attributes(self, the_dict: dict):
        # puts the dictionary into the base object, e.g.
        # self.param1 rather than self.params["param1"]
        for k, v in the_dict.items():
            setattr(self, k, v)

    def _define_sets(self):
        self.regions = pd.DataFrame(self.stocks["region"].drop_duplicates(), columns=["region"])

    def _define_variables(self) -> None:
        self.stocks, self.fraction = build_variable(
            xpress_problem=self.xpress_problem,
            df=self.stocks,
            var_name="fraction",
            index_cols=["name"],
            return_vars_object=True,
            vartype=xp.continuous,
        )
        add_items(target_list=self.variables, items_to_add=self.fraction)

        self.stocks, self.buy = build_variable(
            xpress_problem=self.xpress_problem,
            df=self.stocks,
            var_name="buy",
            index_cols=["name"],
            return_vars_object=True,
            vartype=xp.binary,
        )
        add_items(target_list=self.variables, items_to_add=self.buy)

    def _define_constraints(self) -> None:
        # Invest all of the portfolio in these 20 options
        # sum([fraction[s] for s in stocks]) == 1 noqa: E800
        invest_all_sum = xp.Sum(self.stocks["fraction"])
        invest_all_constraint = build_named_constraints(
            xpress_problem=self.xpress_problem,
            constraint_name="invest_all",
            lhs=invest_all_sum,
            constraint_sense=xp.eq,
            rhs=1,
        )
        add_items(target_list=self.constraints, items_to_add=invest_all_constraint)

        # tie the fraction and buy variables together (they are not independent)
        # using min_ratio_per_stock for little_m also enforces a minimum ratio on stocks we are buying
        # fraction[s] >= min_ratio_per_stock*buy[s] noqa: E800

        # using max_ratio_per_stock for big_m also enforces a maximum ratio on stocks we are buying
        # fraction[s] <= max_ratio_per_stock*buy[s] noqa: E800
        self.stocks, tie_buy_and_fraction_constraint = build_bool_constraints(
            xpress_problem=self.xpress_problem,
            df=self.stocks,
            constraint_name="tie_buy_and_fraction",
            index_cols=["name"],
            bool_var_col="buy",
            base_var_col="fraction",
            big_m=self.max_ratio_per_stock,
            little_m=self.min_ratio_per_stock,
        )
        add_items(target_list=self.constraints, items_to_add=tie_buy_and_fraction_constraint)

        # Maximum number risky stocks constraint
        # sum(is_risky[s]*buy[s]) <= max_risky_stocks noqa: E800
        max_number_risky_sum = xp.Sum(self.stocks[self.stocks["is_risky"] == 1]["buy"])

        (
            max_number_risky_sum_constraint,
            max_number_risky_penalty,
            max_number_risky_slack_vars,
        ) = build_flex_constraints(
            xpress_problem=self.xpress_problem,
            constraint_name="max_number_risky_sum",
            lhs=max_number_risky_sum,
            constraint_sense=xp.leq,
            rhs=self.max_risky_stocks,
            soft_constraint_penalties=self.penalty_max_number_risky,
            constraint_activation=self.max_number_risky_sum_activation,
        )
        self.penalty += max_number_risky_penalty
        add_items(target_list=self.variables, items_to_add=max_number_risky_slack_vars)
        add_items(target_list=self.constraints, items_to_add=max_number_risky_sum_constraint)

        # Maximum fraction risky stocks constraint
        # sum(is_risky[s]*fraction[s]) <= max_risky_stocks_ratio noqa: E800
        max_fraction_risky_sum = xp.Sum(self.stocks[self.stocks["is_risky"] == 1]["fraction"])
        max_fraction_risky_constraint = xp.constraint(
            constraint=max_fraction_risky_sum <= self.max_risky_stocks_ratio,
            name="max_fraction_risky",
        )
        self.xpress_problem.addConstraint(max_fraction_risky_constraint)
        add_items(target_list=self.constraints, items_to_add=max_fraction_risky_constraint)

        # Sum fraction and buy per region
        # sum(fraction[s|s.region==region[r]])  and  sum(buy[s|s.region==region[r]])  noqa: E800
        vars_summed_by_region = build_sum_expression(
            df=self.stocks,
            sum_cols=["fraction", "buy"],
            group_by_cols=["region"],
        )
        self.regions = pd.merge(self.regions, vars_summed_by_region, how="left", on="region")
        self.regions = self.regions.rename(
            columns={"fraction": "fraction_per_region", "buy": "buy_per_region"}
        )

        # Minimum fraction of investment per region
        # for all regions, sum(fraction[s|s.region==region[r]]) >= min_ratio_per_region noqa: E800
        self.regions["min_ratio_per_region"] = self.min_ratio_per_region
        self.regions, min_fraction_per_region_constraint = build_constraint(
            xpress_problem=self.xpress_problem,
            df=self.regions,
            constraint_name="min_fraction_per_region",
            index_cols=["region"],
            lhs_col="fraction_per_region",
            constraint_sense=xp.geq,
            rhs_col="min_ratio_per_region",
        )
        add_items(target_list=self.constraints, items_to_add=min_fraction_per_region_constraint)

        # Minimum Number of investments per region
        # for all regions, sum(buy[s|s.region==region[r]]) >= min_stocks_per_region noqa: E800
        self.regions["min_stocks_per_region"] = self.min_stocks_per_region
        (
            self.regions,
            min_number_per_region_constraints,
            min_number_per_region_penalty,
            min_number_per_region_slack_vars,
        ) = build_flex_constraints_indexed(
            xpress_problem=self.xpress_problem,
            df=self.regions,
            constraint_name="min_number_per_region",
            index_cols=["region"],
            lhs_col="buy_per_region",
            constraint_sense=xp.geq,
            rhs_col="min_stocks_per_region",
            soft_constraint_penalties=self.penalty_min_number_per_region,
            constraint_activation=self.min_number_per_region_activation,
        )

        # penalties must be extended, not appended
        self.penalty += min_number_per_region_penalty
        add_items(target_list=self.variables, items_to_add=min_number_per_region_slack_vars)
        add_items(target_list=self.constraints, items_to_add=min_number_per_region_constraints)

        # Maximum number of stocks
        # sum(buy[s]) <= max_total_stocks noqa: E800
        max_number_stocks_constraint = xp.constraint(
            constraint=xp.Sum(self.stocks["buy"]) <= self.max_total_stocks,
            name="max_number_stocks",
        )
        self.xpress_problem.addConstraint(max_number_stocks_constraint)
        add_items(target_list=self.constraints, items_to_add=max_number_stocks_constraint)

    def _define_objective_function(self) -> None:
        # Define portfolio expected return (objective function)
        objective_function = xp.Sum(self.stocks["fraction"] * self.stocks["expected_return"])
        objective_function = build_objective_function_with_penalties(
            xpress_problem=self.xpress_problem,
            objective_function=objective_function,
            penalties=self.penalty,
            sense=xp.maximize,
        )
