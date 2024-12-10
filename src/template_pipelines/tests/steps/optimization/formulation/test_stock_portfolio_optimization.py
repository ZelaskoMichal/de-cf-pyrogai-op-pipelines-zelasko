"""Tests for formulation."""

from pathlib import Path

import pandas as pd
import pytest
import xpress as xp

from dnalib.optimization_models.xpress_utils.testing import (
    assert_value,
    assert_xpress_constraints_df,
    assert_xpress_variable_cols,
)
from template_pipelines.steps.optimization.formulation.stock_portfolio_optimization import (
    StockPortfolioOptimization,
)
from template_pipelines.tests.steps.optimization.constants import (  # noqa I202
    DATA_PARAMETERS,
    RUNTIME_PARAMETERS,
)
from template_pipelines.tests.steps.optimization.formulation.utils import assert_df
from template_pipelines.utils.optimization.io_utils import cast_numeric_runtime_parameters


@pytest.fixture(scope="function")
def fixture_formulation():
    """Fixture for StockPortfolioOptimization formulation."""
    test_data_dir = Path(__file__).parent.parent / "test_data"
    portfolio_stocks = pd.read_parquet(test_data_dir / "input_data" / "portfolio_stocks.parquet")
    xpress_problem = xp.problem(name="portfolio")
    xpress_problem.setControl("miprelstop", 1e-4)

    formulation = StockPortfolioOptimization(
        {**cast_numeric_runtime_parameters(dict(RUNTIME_PARAMETERS)), **DATA_PARAMETERS},
        portfolio_stocks,
        xpress_problem,
    )

    yield formulation


def test_formulation_regions_df_columns_correctness(fixture_formulation):
    """Testing generation of dataframe."""
    # execute
    fixture_formulation.build_formulation()

    # assert
    assert_df(
        fixture_formulation.regions,
        [
            "region",
            "fraction_per_region",
            "buy_per_region",
            "min_ratio_per_region",
            "min_fraction_per_region",
            "min_stocks_per_region",
            "min_number_per_region",
        ],
        expected_df_len=3,
        check_nulls=True,
    )


def test_formulation_regions_df_expressions_correctness(fixture_formulation):
    """Testing if values from columns are xpress expressions."""
    # In case of our test data - all values in given columns must be expressions.
    # execute
    fixture_formulation.build_formulation()

    # assert
    assert all(
        map(
            lambda element: isinstance(element, xp.expression),
            fixture_formulation.regions["fraction_per_region"],
        )
    )
    assert all(
        map(
            lambda element: isinstance(element, xp.expression),
            fixture_formulation.regions["buy_per_region"],
        )
    )


def test_formulation_regions_df_expression_values_correctness(fixture_formulation):
    """Testing if xpress expressions values are correct."""
    # prepare
    # order list of stocks related to NA region from test data
    stock_names = [
        "treasury",
        "hardware",
        "highways",
        "bank",
        "energy",
        "analytics",
        "music",
        "grocery",
    ]
    expected_fraction_per_region_value = (
        "  "
        + " + ".join(  # heading spaces are presented because of df formatting
            [f"fraction(idx(name='{stock_name}'))" for stock_name in stock_names]
        )
    )
    expected_buy_per_region_value = (
        "  "
        + " + ".join(  # heading spaces are presented because of df formatting
            [f"buy(idx(name='{stock_name}'))" for stock_name in stock_names]
        )
    )

    # execute
    fixture_formulation.build_formulation()

    # assert
    regions_df = fixture_formulation.regions
    assert_value(regions_df["fraction_per_region"][0], expected_fraction_per_region_value)
    assert_value(regions_df["buy_per_region"][0], expected_buy_per_region_value)


def test_formulation_regions_df_parameters_values_correctness(fixture_formulation):
    """Testing if parameters from config was passed to dataframe."""
    # execute
    fixture_formulation.build_formulation()

    # assert
    assert_value(fixture_formulation.regions["min_ratio_per_region"][0], 0.2)
    assert_value(fixture_formulation.regions["min_stocks_per_region"][0], 2)


def test_formulation_stocks_df_columns_correctness(fixture_formulation):
    """Testing generation of dataframe."""
    # execute
    fixture_formulation.build_formulation()

    # assert
    assert_df(
        fixture_formulation.stocks,
        [
            "name",
            "expected_return",
            "is_risky",
            "region",
            "fraction",
            "buy",
            "tie_buy_and_fraction_lb",
            "tie_buy_and_fraction_lb_rhs",
            "tie_buy_and_fraction_ub",
            "tie_buy_and_fraction_ub_rhs",
        ],
        expected_df_len=20,
        check_nulls=True,
    )


def test_formulation_stocks_df_variables_correctness(fixture_formulation):
    """Testing generation of variables columns in dataframe."""
    # execute
    fixture_formulation.build_formulation()

    # assert
    assert_xpress_variable_cols(
        fixture_formulation.stocks,
        ["fraction", "buy"],
        expected_df_len=20,
        check_nulls=True,
        check_type=True,
        check_all_zero=True,
    )


def test_formulation_stocks_df_constraints_correctness(fixture_formulation):
    """Testing generation of constraints columns in dataframe."""
    # execute
    fixture_formulation.build_formulation()

    # assert
    assert_xpress_constraints_df(
        fixture_formulation.stocks,
        [
            "tie_buy_and_fraction_lb",
            "tie_buy_and_fraction_ub",
        ],
        expected_df_len=20,
        check_type=True,
        check_all_null=True,
    )


def test_formulation_stocks_df_xp_objs_correctness(fixture_formulation):
    """Testing if values from columns are expected xpress objects."""
    # In case of our test data - all values in given columns must be expressions.
    # prepare
    linterm_type = type(2 * xp.var("dummy_var"))

    #  execute
    fixture_formulation.build_formulation()

    # assert
    assert all(
        map(
            lambda element: isinstance(element, xp.constraint),
            fixture_formulation.stocks["tie_buy_and_fraction_lb"],
        )
    )
    assert all(
        map(
            lambda element: isinstance(element, linterm_type),
            fixture_formulation.stocks["tie_buy_and_fraction_lb_rhs"],
        )
    )
    assert all(
        map(
            lambda element: isinstance(element, xp.constraint),
            fixture_formulation.stocks["tie_buy_and_fraction_ub"],
        )
    )
    assert all(
        map(
            lambda element: isinstance(element, linterm_type),
            fixture_formulation.stocks["tie_buy_and_fraction_ub_rhs"],
        )
    )


def test_formulation_stocks_df_constraints_rhs_values_correctness(fixture_formulation):
    """Testing tie_buy_and_fraction constraint lhs values."""
    # prepare
    # heading spaces are presented because of df formatting
    expected_lb_rhs = "0.3*buy(idx(name='treasury'))"
    expected_ub_rhs = "0.01*buy(idx(name='treasury'))"

    # execute
    fixture_formulation.build_formulation()

    # assert
    assert_value(fixture_formulation.stocks["tie_buy_and_fraction_lb_rhs"][0], expected_lb_rhs)
    assert_value(fixture_formulation.stocks["tie_buy_and_fraction_ub_rhs"][0], expected_ub_rhs)


def test_formulation_variables_amount_correctness(fixture_formulation):
    """Testing if amount of variables is correct."""
    # execute
    fixture_formulation.build_formulation()

    # assert
    assert len(fixture_formulation.fraction) == 20
    assert len(fixture_formulation.buy) == 20


penalty_1 = 0
penalty_2 = (
    " 0.1 max_number_risky_sum_excess_violation_qty"
    " +0.1 min_number_per_region_gap_violation_qty(idx(region='NA'))"
    " +0.1 min_number_per_region_gap_violation_qty(idx(region='EMEA'))"
    " +0.1 min_number_per_region_gap_violation_qty(idx(region='LAC'))"
)
penalty_3 = (
    " 0.1 min_number_per_region_gap_violation_qty(idx(region='NA'))"
    " +0.1 min_number_per_region_gap_violation_qty(idx(region='EMEA'))"
    " +0.1 min_number_per_region_gap_violation_qty(idx(region='LAC'))"
)
penalty_4 = 0


@pytest.mark.parametrize(
    (
        "max_number_risky_sum_activation,min_number_per_region_activation,"
        "max_number_risky_sum_added,min_number_per_region_added,len_constraints,"
        "penalty_str"
    ),
    [
        ("hard", "hard", True, True, 50, penalty_1),
        ("soft", "soft", True, True, 50, penalty_2),
        ("off", "soft", False, True, 49, penalty_3),
        ("hard", "off", True, False, 47, penalty_4),
    ],
)
def test_formulation_constraints_existence(
    max_number_risky_sum_activation,
    min_number_per_region_activation,
    max_number_risky_sum_added,
    min_number_per_region_added,
    len_constraints,
    penalty_str,
    fixture_formulation,
):
    """Testing if all constraints were added directly in _define_constraints.

    What's more that test check various scenarios in terms of flex constraints activation .
    """
    # prepare
    fixture_formulation.max_number_risky_sum_activation = max_number_risky_sum_activation
    fixture_formulation.min_number_per_region_activation = min_number_per_region_activation

    # execute
    fixture_formulation.build_formulation()

    # assert
    constraints_names = [c.name for c in fixture_formulation.xpress_problem.getConstraint()]
    # assert - constraints
    assert_value(fixture_formulation.penalty, penalty_str)
    assert "invest_all" in constraints_names
    assert "tie_buy_and_fraction_lb(name='treasury')" in constraints_names
    assert "tie_buy_and_fraction_ub(name='grocery')" in constraints_names
    assert "max_fraction_risky" in constraints_names
    assert "max_number_stocks" in constraints_names
    assert "min_fraction_per_region(region='NA')" in constraints_names
    # assert - flex constraints
    assert ("max_number_risky_sum" in constraints_names) == max_number_risky_sum_added
    assert (
        "min_number_per_region(region='NA')" in constraints_names
    ) == min_number_per_region_added
    assert len(constraints_names) == len_constraints
