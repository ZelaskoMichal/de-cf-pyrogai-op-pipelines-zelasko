"""Loading data from BigQuery."""
from aif.pyrogai.steps.step import Step  # noqa: E402


class BQIo(Step):
    """Step to interact with BigQuery methods.

    It was created to help people better understand how to use BQ IOSlots.
    """

    def run(self):
        """Run a sample step.

        This step serves the purpose of showcasing what features are available within each step.
        """
        self.logger.info("Running BQ IOSlot functions ...")
        bq_input = self.inputs["bq_slot"]
        # Remember to configure bq_dataset
        query = f"select * from `{self.config['bq_io']['gcp_project']}.{self.config['bq_io']['bq_dataset']}.campaign`"

        # Returns the first n_head rows from query as a pandas dataframe.
        self.logger.info("Running head -> return first n rows as pandas dataframe")
        df = bq_input.head(query=query, n_head=5)
        self.logger.info(df)

        # Count the elements returned by query.
        self.logger.info("Running count -> return nr of rows from query as pandas dataframe")
        df = bq_input.count(query=query)
        self.logger.info(df)

        # Counts the number of distinct col in query.
        self.logger.info(
            "Running count_distinct -> return nr of distinct values by column"
            "from query as pandas dataframe"
        )
        df = bq_input.count_distinct(col="campaign_id", query=query, precise=False)
        self.logger.info(df)

        # Returns the query results as a pandas data frame.
        self.logger.info("Running to_pandas that runs query and saves it to pandas Dataframe.")
        df = bq_input.to_pandas(query=query)
        self.logger.info(df)

        # Runs the query and return results.
        self.logger.info("Running run_query that runs query and prints result.")
        res = bq_input.run_query(query=query)
        self.logger.info(res)

        # Save the results of the query to a permanent table.
        self.logger.info("Running save_to_table which runs query and saves it as BigQuery table.")
        res = bq_input.save_to_table(
            query=query,
            table_id=f"{self.config['bq_io']['gcp_project']}.{self.config['bq_io']['bq_dataset']}.campaign_test",
            write_disposition="WRITE_TRUNCATE",
        )
        self.logger.info(res)

        # Extracts the results of query and saves them to a gcs location
        self.logger.info(
            "Running extract_to_gcs which queries BQ table and saves it to Cloud Storage."
        )
        res = bq_input.extract_to_gcs(
            query=query,
            uri=(
                f"gs://{self.config['bq_io']['bucket_name']}/bq_io"
                f"/campaign_from_{self.config['bq_io']['bq_dataset']}.csv"
            ),
            out_format="CSV",
        )
        self.logger.info(res)
        self.logger.info("BQ IOSlot functions run is done")
