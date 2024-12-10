"""InitialDqStep class."""
from aif.pyrogai.steps.base_dq_step import BaseDqStep


class InitialDqStep(BaseDqStep):
    """InitialDqStep."""

    def run(self):
        """Runs step."""
        self.logger.info("Start initial dq after data loading step")
        # Run checks which we implement in our .yaml file under steps
        self.run_checks(step_name=self.step_name)

        # generates Great Expectations' HTML reports known as Data Docs, for all the executed suites
        self.generate_data_docs()

        # prints all the executed expectation suites statuses so far,
        # and finally raises a `CriticalDataQualityError` in case
        # at least one suite whose name ends with .critical failed
        self.raise_for_status()

        # If data validation was not successful, maybe it's worth taking a look here?
        # https://developerportal.pg.com/docs/default/component/pyrogai/notifications/#email-notification
        # This is link to email notafication which you can configure in `config.json` and
        # it sends e-mails automatically

        self.logger.info("Finish initial dq after data loading step")
