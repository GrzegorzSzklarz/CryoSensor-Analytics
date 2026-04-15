# Cryogenic Data Analysis (CryoSensor-Analytics)

A system for automated correction and statistical analysis of data from resistance sensors.

## Project Description
The script processes raw resistance measurements by applying three correction methods depending on the sensor type and temperature range:
1.  **ITS-90**: For platinum sensors.
2.  **S-Factor**: Polynomial integration of sensitivity (dR/dT).
3.  **In-Situ Sensitivity**: Linear regression for points without sensivity function.

## Requirements
To run the script, Python 3.8+ and the following libraries are required:
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `numba`