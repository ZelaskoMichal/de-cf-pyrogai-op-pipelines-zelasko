<!---
Add Jira item and provide a concise summary of your changes in the title above.
-->

# Description

### Context
**Why are these changes required?**  
Provide a brief explanation of the necessity of these changes. Describe the problem you are solving and the benefits of your solution.

### Changes
**What has been changed?**  
Detail the modifications you've made. Include a clear description of what each file change accomplishes and any logical alterations that were necessary.

### Testing
**How have these changes been tested?**  
Describe the testing approach you took. Include:
- The environments (operating system, device, etc.) in which the tests were conducted.
- The edge cases you considered and how they were addressed.
- Any automated testing frameworks or methodologies used.

### Test Recommendations for Reviewers
**How should reviewers test your changes?**  
Provide clear, step-by-step instructions for reviewers to follow to verify the changes. Suggest additional test cases or environments that you might not have access to but would be valuable for thorough review.


### For the reviewers:

- [ ] I have tested the code and ran it locally

- [ ] I have checked the code throughly

- [ ] I have checked the results of CI/CD workflows

- [ ] I have checked Sonarqube for any problems or issues

- [ ] I have reviewed the documentation and ensured it is up to date.

## QA Environment Creation Guide

This guide details the steps to create a Databricks environment for QA testing. It provides instructions on how to specify the size of the cluster based on predefined configurations.

<details>
<summary><strong>Databricks Environment Configuration</strong></summary>
<p>

Initiate a Databricks environment for QA testing directly within your PR comments. Use the following commands to specify the size of your cluster:

```
DBR <size of your cluster>
DBR small
DBR large
```

The <size of your cluster> parameter corresponds to the configurations defined in provider_dbr.yml. For instance, if computes: in your configuration file includes a size named ultra_big, you can initialize a QA environment with that specific setup by replacing <size of your cluster> with ultra_big.

</p>
</details>

