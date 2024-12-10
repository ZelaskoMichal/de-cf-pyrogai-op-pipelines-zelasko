"""Module to create simulated iris data."""

import datetime
import json
import random
import sys


def main():
    """Main function."""
    sepal_lth = random.uniform(4.0, 7.9)
    sepal_wth = random.uniform(2.2, 4.4)
    petal_lth = random.uniform(1.1, 6.7)
    petal_wth = random.uniform(0.1, 2.5)

    time_stamp = datetime.datetime.now().isoformat() + "Z"
    petal_length = {
        "tag": "Iris.PetalLength",
        "NumericValue": round(petal_lth, 1),
        "timeStamp": time_stamp,
    }
    petal_width = {
        "tag": "Iris.PetalWidth",
        "NumericValue": round(petal_wth, 1),
        "timeStamp": time_stamp,
    }
    sepal_length = {
        "tag": "Iris.SepalLength",
        "NumericValue": round(sepal_lth, 1),
        "timeStamp": time_stamp,
    }
    sepal_width = {
        "tag": "Iris.SepalWidth",
        "NumericValue": round(sepal_wth, 1),
        "timeStamp": time_stamp,
    }
    result = {}
    result["Items"] = []
    result["Items"].append(petal_length)
    result["Items"].append(petal_width)
    result["Items"].append(sepal_length)
    result["Items"].append(sepal_width)
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
