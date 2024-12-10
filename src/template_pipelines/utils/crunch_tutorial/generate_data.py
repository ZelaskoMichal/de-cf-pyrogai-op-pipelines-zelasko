"""Helper module to generate random sales data."""
import csv
import os
import random
from pathlib import Path
from typing import Union

products = ["Head&Shoulders", "Pampers", "Ariel", "Old Spice"]


def generate_sales(n: int) -> list[dict]:
    """Generate n rows of random sales records."""
    return [
        {"Product": random.choice(products), "Sales": random.randint(-100, 100)} for _ in range(n)
    ]


def write_sales(output_path: Path, sales: list[dict]) -> None:
    """Write sales data to a file."""
    with open(output_path, "w") as output_file:
        writer = csv.DictWriter(output_file, delimiter=";", fieldnames=["Product", "Sales"])
        writer.writeheader()
        for row in sales:
            writer.writerow(row)


def create_sales_file(path: Path, row_count: int = 1000) -> None:
    """Create a sales file with random data."""
    sales = generate_sales(row_count)
    write_sales(path, sales)


def generate_dataset(path: Union[str, Path], file_count: int = 10) -> None:
    """Generate random sales data in multiple files."""
    path = Path(path)
    for i in range(file_count):
        filepath = path / f"part_{str(i)}"
        os.makedirs(filepath, exist_ok=True)
        create_sales_file(filepath / "data.csv")
