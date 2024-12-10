"""Helper BigQuery Class."""
import logging
from datetime import datetime, timedelta

from google.cloud import bigquery
from google.cloud.bigquery.schema import SchemaField

logging.basicConfig()


class BigQuery:
    """Helper class to streamline common big query operations."""

    def __init__(self, location="us-east4"):
        """Helper class to streamline common big query operations."""
        # hardcoded location they do not differ between instances of BQ clients
        self.client = bigquery.Client(location=location)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def to_pandas(self, query):
        """Returns the query results as a pandas data frame.

        Args:
            query (str): SQL query

        Returns:
            pandas dataframe
        """
        self.logger.debug(query)
        return self.client.query(query).to_dataframe()

    def run_query(self, query):
        """Runs the query and return results.

        Args:
            query (str): SQL query

        Returns:
            Query results as an iterator over rows
        """
        self.logger.debug(query)
        return self.client.query(query).result()

    def save_to_table(self, query, table_id, write_disposition="WRITE_EMPTY"):
        """Save the results of the query to a permanent table.

        Args:
            query (str): SQL query
            table_id (str): The destination table in the form:
                            "your-project.your_dataset.your_table_name"
            write_disposition (str): determines if the dest table is overwritten or appended to.
                 by default, the data is written onnly if the destination table does not exist.
                 for details, see
                 https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.write_disposition
        """
        self.logger.debug(query)
        job_config = bigquery.QueryJobConfig(
            destination=table_id, write_disposition=write_disposition
        )

        # Start the query, passing in the extra configuration.
        q = self.client.query(query, job_config=job_config)
        q.result()  # Wait for the job to complete.

    def extract_to_gcs(self, query, uri, out_format=bigquery.DestinationFormat.CSV):
        """Extracts the results of query and saves them to a gcs location.

        Args:
            query (str): SQL query
            uri (str):  The destination URI in GCS, in the form 'gs://bucket/path'.
                 Can be a pattern if the file output needs to be split across multiple files,
                 e.g. 'gs://bucket/path-*.csv'
            out_format (bigquery.DestinationFormat): The format of the destination file
        """
        q = self.client.query(query)
        destination = q.destination
        q.result()  # wait for the query to complete
        config = bigquery.ExtractJobConfig()
        config.destination_format = out_format
        self.client.extract_table(destination, uri, job_config=config).result()

    def create_temp_table(self, table_id, schema_info, expires_in_hours=1):
        """Create fully functional temporary table - DELETE existing table if found.

        Args:
            table_id (str): FQN of BQ table to be created
                eg. 'dbce-smart-aud-app-dev-939b.test_dataset.test_bigquery_create_tmp_table'
            schema_info (list of str tuples): BQ SchemaFields in form of tuples
                eg. [("segment_id", "STRING", "REQUIRED"), ("segment_name", "STRING", "REQUIRED")]
            expires_in_hours (int): number of hours after which table will be automatically deleted

        Returns:
            table_location (str): complete table id of created table returned from client
        """
        schema = []
        for col in schema_info:
            schema.append(bigquery.SchemaField(col[0], col[1], mode=col[2]))

        table = bigquery.Table(table_id, schema=schema)
        table.expires = datetime.now() + timedelta(hours=expires_in_hours)

        # delete previous instance of table if still exists
        self.client.delete_table(table_id, not_found_ok=True)
        table_created = self.client.create_table(table)
        table_location = (
            f"{table_created.project}.{table_created.dataset_id}.{table_created.table_id}"
        )

        return table_location

    def insert_rows(self, table, data):
        """Inserts data into specified table in Big Query.

        Args:
            table (Union[str, bigquery.Table]): Table definition.
            You can provide Table object or full name as string
                eg: your-project.your_dataset.your_table.
            data (List[dict]): Data to insert. List of dictionaries.

        """
        if type(table) == str:
            table = self.client.get_table(table)
        self.logger.info(f"Inserting data into: {table}. Rows to insert: {len(data)}.")
        errors = self.client.insert_rows(table, data)
        if len(errors) > 0:
            raise ValueError(f"Problem with uploading data to Big Query. Errors: {errors}")
        self.logger.info(f"Finished inserting data into: {table}.")

    def delete_table(self, table_name):
        """Deletes specified table.

        Args:
            table_name (str): Full table name as string eg: your-project.your_dataset.your_table.

        """
        self.logger.info(f"Deleting table: {table_name}")
        self.client.delete_table(table_name, not_found_ok=True)
        self.logger.info(f"Done deleting table: {table_name}")

    def create_table(self, table_name, schema):
        """Creates table with given name and schema.

        Args:
            table_name (str): Full table name as string eg: your-project.your_dataset.your_table.
            schema (Dict[str, str]): Schema of the table.
                                     Keys are field names, values are field types.

        """
        bq_schema = [SchemaField(name=k, field_type=v) for k, v in schema.items()]
        table = bigquery.Table(table_ref=table_name, schema=bq_schema)
        self.logger.info(f"Creating table: {table_name}")
        self.client.create_table(table)
        self.logger.info(f"Done creating table: {table_name}")
        return table
