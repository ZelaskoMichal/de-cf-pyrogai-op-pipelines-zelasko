"""Utils EDA Toolkit."""

import dabl
import klib
import matplotlib.pyplot as plt
import missingno as msno
import sweetviz as sv


class EDAToolkit:
    """A toolkit for generating EDA reports using different visualization libraries."""

    def __init__(self, data, logger):
        """Initializes the EDAToolkit with a dataset and a logger.

        Args:
            data (DataFrame): The dataset to perform EDA on.
            logger (Logger): Logger instance to log messages.
        """
        self.data = data
        self.logger = logger

    def generate_report(self, tool="ydata_profiling"):
        """Generates an EDA report using the specified tool.

        Supported tools:
        - 'ydata_profiling': For comprehensive data profiling.
        - 'sweetviz': For visually appealing EDA reports.
        - 'klib': For dataset cleaning and correlation visualizations.
        - 'dabl': For automatic data visualization.
        - 'missingno': For missing data visualization.

        Args:
            tool (str): The tool to use for generating the report. Defaults to 'ydata_profiling'.
        """
        tool_methods = {
            "sweetviz": self._sweetviz_report,
            "klib": self._klib_report,
            "dabl": self._dabl_report,
            "missingno": self._missingno_report,
        }

        if tool in tool_methods:
            tool_methods[tool]()
        else:
            self.logger.warning(
                f"Invalid tool selected: {tool}. Supported tools: {list(tool_methods.keys())}"
            )

    def _sweetviz_report(self):
        """Generates a Sweetviz report and saves it as an HTML file."""
        report = sv.analyze(self.data)
        report.show_html("sweetviz_report.html")
        self.logger.info("Sweetviz report generated: 'sweetviz_report.html'")

    def _klib_report(self):
        """Generates a Klib report with summaries and visualizations."""
        self.logger.info("Generating Klib report...")
        klib.describe(self.data)
        klib.corr_plot(self.data)
        klib.missingno_matrix(self.data)
        self.logger.info("Klib report completed.")

    def _dabl_report(self):
        """Generates a Dabl report and displays it."""
        self.logger.info("Generating Dabl report...")
        X = self.data.drop("target", axis=1)
        y = self.data["target"]
        dabl.plot(X, y)
        plt.savefig("dabl_report.png")
        self.logger.info("Dabl report displayed.")

    def _missingno_report(self):
        """Generates visualizations of missing data using Missingno."""
        self.logger.info("Generating Missingno visualizations...")
        msno.matrix(self.data)
        msno.bar(self.data)
        self.logger.info("Missingno visualizations displayed.")
