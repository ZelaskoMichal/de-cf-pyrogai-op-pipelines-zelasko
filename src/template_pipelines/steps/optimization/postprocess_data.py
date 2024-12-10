"""Step produces enriched output table."""

import pandas as pd

from aif.pyrogai.steps.step import Step
from template_pipelines.steps.optimization.preprocessing.sdm import StocksPortfolioSDM
from template_pipelines.utils.optimization.io_utils import load_tables, save_tables


class PostprocessData(Step):
    """PostprocessData Step."""

    def run(self) -> None:
        """Generate output table."""
        # get solution
        solution_fraction_df = load_tables(
            self.ioctx.get_fns(f"{self.config['solution_tmp_dir']}/fraction.parquet")
        )[0]["fraction"]

        # get portfolio_stocks
        sdm = StocksPortfolioSDM()
        sdm.load_stored_sdm(path=self.config["sdm_tmp_dir"], ioctx=self.ioctx)
        portfolio_stocks_df = sdm.sdm_data["portfolio_stocks"]

        # get raw data (stocks, industries, regions)
        data_path = f"{self.config['input_tmp_dir']}"
        file_paths = self.ioctx.get_fns(f"{data_path}/*.parquet")
        raw_data, _ = load_tables(file_paths)
        stocks_df = raw_data["stocks"].drop_duplicates(["name"])
        industries_df = raw_data["industries"].drop_duplicates(["industry"])
        regions_df = raw_data["regions"].drop_duplicates(["region"])

        # transform and save output
        output_df = self.create_output(
            solution_fraction_df, portfolio_stocks_df, stocks_df, industries_df, regions_df
        )
        self.save_output(output_df)

        # pretty print of the solution
        self.logger.info(self.interpret_solution(output_df))

    def create_output(
        self,
        solution_fraction_df: pd.DataFrame,
        portfolio_stocks_df: pd.DataFrame,
        stocks_df: pd.DataFrame,
        industries_df: pd.DataFrame,
        regions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transforms input, sdm and solution tables into pipeline output dataframe."""
        output_df = solution_fraction_df[solution_fraction_df["fraction"] > 0]
        output_df = output_df.merge(portfolio_stocks_df, on="name", how="left")
        output_df = output_df.merge(stocks_df[["name", "industry"]], on="name", how="left")
        output_df = output_df.merge(industries_df, on="industry", how="left")
        output_df = output_df.merge(regions_df, on="region", how="left")

        output_df = output_df.rename(
            columns={"fraction": "recommended portfolio percent", "avg_return": "industry average"}
        )
        output_df["recommended portfolio percent"] = (
            output_df["recommended portfolio percent"] * 100
        ).round()
        output_df = output_df[
            [
                "name",
                "recommended portfolio percent",
                "is_risky",
                "expected_return",
                "region",
                "region_index",
                "industry",
                "industry average",
            ]
        ]
        return output_df

    def save_output(self, output_df: pd.DataFrame) -> None:
        """Save output_df to the output_tmp_dir in ioctx."""
        path = self.ioctx.get_output_fn(f"{self.config['output_tmp_dir']}")
        save_tables(data={"output": output_df}, path=path, file_format="parquet")

    def interpret_solution(self, solution: pd.DataFrame) -> str:
        """Writes the solution as a human interpreatable string."""
        output_str = "Used items and their quantities: "
        for _, row in solution.iterrows():
            output_str = output_str + (
                f"\n\tSTOCK: {row['name']} \t\tREGION: {row['region']} \t\tRISKY: {row['is_risky']} \t\tPORTFOLIO: {row['recommended portfolio percent']}%"  # NOQA: E501, W505
            )
        return output_str
