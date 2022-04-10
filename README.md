# Temperature Map Prediction
## Project Summary
In this project, we aim to analyze and extrapolate the trends in land surface temperature over the past 20 years to accurately forecast surface temperature
maps in near future.

## Development
### Auto-formatting
Run the following command for auto-reformatting:
```
bash ./auto_format.sh
```
**Every commit made to the repository also goes through the same auto-reformatting on GitHub.**
If your remote branch is auto-reformatted, your new local commits may be rejected. 
Make sure to "_pull the remote branch with rebase to your local branch_" before pushing new local commits.

### Update Dependencies
**Add new dependencies to `requirements.in`.**

`requirements.in` keeps track of only the top level dependencies.

`requirements.txt` is generated with the following command using the `pip-tools` package:
```
pip-compile requirements.in
```
Run this command whenever a dependency in `requirements.in` is updated. It automatically writes all the necessary 
and compatible packages to `requirements.txt`.