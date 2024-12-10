"""Ml training tester class."""
import datetime
import os
import random
from logging import Logger

import pandas as pd
from compatibility_matrix_controller import CompatibilityMatrixController

from template_pipelines.tests.compatibility_matrix.pipeline_testers.base_tester import (
    IPipelineTester,
)

# seed set so that it does not generate errors in the future due to constant data changes
random.seed(26072000)


class MlTrainingTester(IPipelineTester):
    """Ml training tester class."""

    def __init__(self, logger: Logger, matrix_controller: CompatibilityMatrixController) -> None:
        """Init."""
        super().__init__(logger, matrix_controller)

        self._already_created_files = False
        self._data_to_create_files = self.generate_random_data()

    def prepare_env_to_run_pipeline(self):
        """Preparation env."""
        self.logger.info(
            f"Start preparing environment and files for {self.matrix_controller.pipeline_name}"
        )

        # Create "CouponData" folder if it doesn"t exist
        os.makedirs("CouponData", exist_ok=True)

        data = self._data_to_create_files

        # Process and create files for each dataset separately
        for dataset_name, dataset_content in data.items():
            df = pd.DataFrame(dataset_content)
            # Ensure correct data types
            if "campaign_id" in df.columns:
                df["campaign_id"] = pd.to_numeric(df["campaign_id"], errors="coerce").astype(
                    "Int64"
                )
            file_path = os.path.join("CouponData", f"{dataset_name}.parquet")
            # Save DataFrame as Parquet file
            df.to_parquet(file_path, index=False)
            self.logger.info(f"Created file {file_path}")

        self.logger.info(
            f"Finish preparing environment and files for {self.matrix_controller.pipeline_name}"
        )

    def random_date(self) -> datetime.date:
        """Generate random date."""
        start_date = datetime.date(2012, 1, 1)
        end_day = random.randint(1, 29)
        end_date = datetime.date(2012, 2, end_day)
        delta = end_date - start_date
        delta_days = delta.days
        if delta_days > 0:
            random_days = random.randrange(delta_days)
            return start_date + datetime.timedelta(days=random_days)
        else:
            return start_date

    def generate_random_data(self) -> dict:
        """Generate random dict pretending CouponData."""
        num_items = 1000
        num_coupons = 500
        num_customers = 500

        data = {
            "campaigns": {
                "campaign_id": [i for i in range(25)],
                "campaign_type": [random.choice(["X", "Y"]) for _ in range(25)],
                "start_date": [self.random_date().isoformat() for _ in range(25)],
                "end_date": [self.random_date().isoformat() for _ in range(25)],
            },
            "coupon_item_mapping": {
                "coupon_id": [random.randint(1, num_coupons) for _ in range(1000)],
                "item_id": [random.randint(1, num_items) for _ in range(1000)],
            },
            "coupon_redemption": {
                "id": [i for i in range(1000)],
                "redemption_status": [random.choice([0, 1]) for _ in range(1000)],
                "campaign_id": [random.randint(0, 24) for _ in range(1000)],
                "coupon_id": [random.randint(1, num_coupons) for _ in range(1000)],
                "customer_id": [random.randint(1, num_customers) for _ in range(1000)],
            },
            "customer_demographics": {
                "customer_id": [i for i in range(1, num_customers + 1)],
                "age_range": [
                    random.choice(["18-25", "26-35", "36-45", "46-55", "56-65", "65+"])
                    for _ in range(num_customers)
                ],
                "marital_status": [
                    random.choice([None, "Married", "Single"]) for _ in range(num_customers)
                ],
                "rented": [random.choice([0, 1]) for _ in range(num_customers)],
                "family_size": [
                    random.choice(["1", "2", "3", "4", "5+"]) for _ in range(num_customers)
                ],
                "no_of_children": [
                    random.choice([None, "1", "2", "3"]) for _ in range(num_customers)
                ],
                "income_bracket": [random.randint(1, 10) for _ in range(num_customers)],
            },
            "customer_transactions": {
                "date": [self.random_date().isoformat() for _ in range(10000)],
                "customer_id": [random.randint(1, num_customers) for _ in range(10000)],
                "item_id": [random.randint(1, num_items) for _ in range(10000)],
                "quantity": [random.randint(1, 5) for _ in range(10000)],
                "selling_price": [round(random.uniform(10, 1000), 2) for _ in range(10000)],
                "other_discount": [round(random.uniform(-50, 0), 2) for _ in range(10000)],
                "coupon_discount": [0.0 for _ in range(10000)],
            },
            "items": {
                "item_id": [i for i in range(1, num_items + 1)],
                "brand": [random.randint(1, 100) for _ in range(num_items)],
                "brand_type": [random.choice(["Local", "Global"]) for _ in range(num_items)],
                "category": [
                    random.choice(["Grocery", "Clothes", "Electronics", "Fuel", "Furniture"])
                    for _ in range(num_items)
                ],
            },
        }
        return data
