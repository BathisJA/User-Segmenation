# User Profiling and Segmentation

This Streamlit dashboard analyzes user profiles for advertising purposes and segments users based on their demographics and online behavior. Below is a breakdown of the functionality and visualizations provided:

#### Dataset Information

- The sidebar displays information about the dataset, including the count of null values in each column.

#### Demographic Distributions

- Visualizations depicting the distribution of key demographic variables, including age, gender, education level, and income level.

#### Device Usage Distribution

- A graphical representation of device usage distribution among users.

#### User Online Behavior and Ad Interaction Metrics

- Histograms displaying various user online behavior and ad interaction metrics, such as time spent online on weekdays and weekends, likes and reactions, click-through rates, conversion rates, and ad interaction time.

#### Top 10 User Interests

- A bar chart illustrating the top 10 user interests based on frequency.

#### User Segmentation

- Utilizes K-means clustering to segment users into distinct groups based on features such as age, gender, income level, and online behavior metrics.
- Displays the mean values of numerical features for each cluster.
- Presents a radar chart profiling each user segment based on their online behavior and interaction metrics.

#### Technologies Used:

- **Streamlit**: Used for building the interactive dashboard.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib** and **Seaborn**: For creating various visualizations.
- **Scikit-learn**: For preprocessing data and performing K-means clustering.
- **Plotly**: For generating interactive radar charts.

### Instructions for Running the Dashboard:

1. Ensure you have all the necessary Python libraries installed, including Streamlit, Pandas, Matplotlib, Seaborn, Scikit-learn, and Plotly.
2. Download the `user_profiles_for_ads.csv` dataset and place it in the `data` directory relative to the location of the script.
3. Execute the provided Python script (`dasboard.py`) using a Python interpreter.
4. The Streamlit dashboard will open in your default web browser, allowing you to interact with the visualizations and explore the user profiling and segmentation analysis.

## Notes:

- This dashboard is intended for exploratory data analysis purposes and provides insights into user demographics and behavior for targeted advertising strategies.
- Feel free to modify the script and dashboard layout to suit your specific requirements or add additional features and visualizations as needed.