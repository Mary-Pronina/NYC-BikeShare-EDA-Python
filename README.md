# NYC Bike Share EDA

  In this project, I perform Exploratory Data Analysis (EDA) on the New York City Bike Share dataset using Python. My focus is on understanding trip durations, user behavior, and data quality, using a combination of statistical analysis, visualizations, and data cleaning techniques to uncover actionable insights.
  
  I begin with descriptive statistics like mean, median, mode, standard deviation, and range, which highlight key trends—such as the fact that most trips are short, with a typical duration between 4 and 7 minutes. However, the presence of extreme values (trips lasting several days or even months) suggests data entry errors or unreturned bikes.
  
  To improve data quality, I apply the Interquartile Range (IQR) method to identify and remove outliers, significantly reducing skewness and bringing the dataset closer to real-world usage patterns. Visual tools like histograms and box plots help me understand the distribution of trip durations before and after cleaning.

I also explore:
- How trip frequency varies by user type (Subscriber vs Customer), 
- Differences in usage patterns by gender,
- Time-based patterns (e.g., peak trip hours or popular months),
- Correlations between variables using heatmaps.

Techniques Used:
- Descriptive statistics,
- Outlier detection and removal (IQR),
- Histograms & box plots,
- Grouped bar charts,
- Time series trends,
- Correlation analysis.

Goal:
To create a clean and insightful dataset that shows how New Yorkers use bike sharing services. The aim is to understand who uses the service, when they ride, and how long their trips typically last. These insights can support better decision-making in user behavior modeling and urban planning.

This project was completed on Kaggle, where I performed all the analysis and visualizations using Python.
You can view the full notebook here: [NYC Bike Share EDA on Kaggle.](https://www.kaggle.com/code/marypronina/nyc-bike-share-eda/edit)

## Exploring the Dataset:
```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```
First, let's load our data into a DataFrame using pd.read_csv(). This function reads the data from the CSV file located in the specified path and stores it in the variable df. Now, df holds all the bike-share data from New York City, collected between 2015 and 2017.

```python
df= pd.read_csv("../input/new-york-city-bike-share-dataset/NYC-BikeShare-2015-2017-combined.csv")
```
Now I want to take a look at the first 10 rows of the dataset to get an initial sense of its structure and contents.

| Unnamed: 0 | Trip Duration | Start Time          | Stop Time           | Start Station ID | Start Station Name | Start Station Latitude | Start Station Longitude | End Station ID | End Station Name  | End Station Latitude | End Station Longitude | Bike ID | User Type | Birth Year | Gender | Trip Duration in min |
|-------------|---------------|---------------------|---------------------|------------------|--------------------|------------------------|-------------------------|----------------|-------------------|----------------------|-----------------------|---------|-----------|------------|--------|----------------------|
| 0           | 376           | 2015-10-01 00:16:26 | 2015-10-01 00:22:42 | 3212             | Christ Hospital     | 40.734786              | -74.050444              | 3207           | Oakland Ave       | 40.737604            | -74.052478             | 24470   | Subscriber| 1960.0     | 1      | 6                    |
| 1           | 739           | 2015-10-01 00:27:12 | 2015-10-01 00:39:32 | 3207             | Oakland Ave         | 40.737604              | -74.052478              | 3212           | Christ Hospital   | 40.734786            | -74.050444             | 24481   | Subscriber| 1960.0     | 1      | 12                   |
| 2           | 2714          | 2015-10-01 00:32:46 | 2015-10-01 01:18:01 | 3193             | Lincoln Park        | 40.724605              | -74.078406              | 3193           | Lincoln Park      | 40.724605            | -74.078406             | 24628   | Subscriber| 1983.0     | 1      | 45                   |
| 3           | 275           | 2015-10-01 00:34:31 | 2015-10-01 00:39:06 | 3199             | Newport Pkwy        | 40.728745              | -74.032108              | 3187           | Warren St         | 40.721124            | -74.038051             | 24613   | Subscriber| 1975.0     | 1      | 5                    |
| 4           | 561           | 2015-10-01 00:40:12 | 2015-10-01 00:49:33 | 3183             | Exchange Place      | 40.716247              | -74.033459              | 3192           | Liberty Light Rail| 40.711242            | -74.055701             | 24668   | Customer  | 1984.0     | 0      | 9                    |
| 5           | 365           | 2015-10-01 00:41:46 | 2015-10-01 00:47:51 | 3198             | Heights Elevator    | 40.748716              | -74.040443              | 3215           | Central Ave       | 40.746730            | -74.049251             | 24644   | Customer  | 1984.0     | 0      | 6                    |
| 6           | 139           | 2015-10-01 00:43:44 | 2015-10-01 00:46:03 | 3206             | Hilltop             | 40.731169              | -74.057574              | 3195           | Sip Ave           | 40.730743            | -74.063784             | 24482   | Subscriber| 1988.0     | 1      | 2                    |
| 7           | 1299          | 2015-10-01 01:10:10 | 2015-10-01 01:31:50 | 3197             | North St            | 40.752559              | -74.044725              | 3215           | Central Ave       | 40.746730            | -74.049251             | 24550   | Customer  | 1984.0     | 0      | 22                   |
| 8           | 647           | 2015-10-01 02:01:36 | 2015-10-01 02:12:24 | 3213             | Van Vorst Park      | 40.718489              | -74.047727              | 3190           | Garfield Ave Station| 40.710467          | -74.070039             | 24650   | Subscriber| 1988.0     | 1      | 11                   |
| 9           | 233           | 2015-10-01 04:43:33 | 2015-10-01 04:47:27 | 3194             | McGinley Square     | 40.725340              | -74.067622              | 3195           | Sip Ave           | 40.730743            | -74.063784             | 24584   | Subscriber| 1978.0     | 2      | 4                    |

Next, we're removing a column named 'Unnamed: 0'. This column is likely an unnecessary index column created during data export, so we don’t need it. We use df.drop() to remove it, specifying axis=1 to drop a column (not a row) and inplace=True to make the change directly to df.

```python
df.drop("Unnamed: 0",axis=1,inplace =True)
```
Finally, we use df.head() to display the first 5 rows of our DataFrame. This gives us a quick snapshot of our data, so we can see what it looks like and start understanding the structure of the dataset, such as column names and sample values.

| Trip Duration | Start Time           | Stop Time            | Start Station ID | Start Station Name | Start Station Latitude | Start Station Longitude | End Station ID | End Station Name | End Station Latitude | End Station Longitude | Bike ID | User Type  | Birth Year | Gender | Trip Duration in Min |
|---------------|----------------------|----------------------|------------------|--------------------|------------------------|-------------------------|----------------|------------------|----------------------|-----------------------|---------|------------|------------|--------|----------------------|
| 376           | 2015-10-01 00:16:26  | 2015-10-01 00:22:42  | 3212             | Christ Hospital     | 40.734786              | -74.050444              | 3207           | Oakland Ave      | 40.737604            | -74.052478            | 24470   | Subscriber | 1960.0     | 1      | 6                    |
| 739           | 2015-10-01 00:27:12  | 2015-10-01 00:39:32  | 3207             | Oakland Ave         | 40.737604              | -74.052478              | 3212           | Christ Hospital  | 40.734786            | -74.050444            | 24481   | Subscriber | 1960.0     | 1      | 12                   |
| 2714          | 2015-10-01 00:32:46  | 2015-10-01 01:18:01  | 3193             | Lincoln Park        | 40.724605              | -74.078406              | 3193           | Lincoln Park     | 40.724605            | -74.078406            | 24628   | Subscriber | 1983.0     | 1      | 45                   |
| 275           | 2015-10-01 00:34:31  | 2015-10-01 00:39:06  | 3199             | Newport Pkwy        | 40.728745              | -74.032108              | 3187           | Warren St        | 40.721124            | -74.038051            | 24613   | Subscriber | 1975.0     | 1      | 5                    |
| 561           | 2015-10-01 00:40:12  | 2015-10-01 00:49:33  | 3183             | Exchange Place      | 40.716247              | -74.033459              | 3192           | Liberty Light Rail | 40.711242          | -74.055701            | 24668   | Customer   | 1984.0     | 0      | 9                    |

The command df.info() is used to quickly get an overview of the dataset, including:

- The number of rows and columns.
- The data types of each column (e.g., integer, float, object).
- The number of non-null values in each column, which helps identify missing data.
This method provides a quick summary of the dataset's structure and allows me to spot potential issues, such as columns with missing values or incorrect data types.
```python
df.info()
```
| #   | Column                   | Non-Null Count   | Dtype  |
| --- | ------------------------ | ---------------- | ------ |
| 0   | Trip Duration            | 735502 non-null  | int64  |
| 1   | Start Time               | 735502 non-null  | object |
| 2   | Stop Time                | 735502 non-null  | object |
| 3   | Start Station ID         | 735502 non-null  | int64  |
| 4   | Start Station Name       | 735502 non-null  | object |
| 5   | Start Station Latitude   | 735502 non-null  | float64|
| 6   | Start Station Longitude  | 735502 non-null  | float64|
| 7   | End Station ID           | 735502 non-null  | int64  |
| 8   | End Station Name         | 735502 non-null  | object |
| 9   | End Station Latitude     | 735502 non-null  | float64|
| 10  | End Station Longitude    | 735502 non-null  | float64|
| 11  | Bike ID                  | 735502 non-null  | int64  |
| 12  | User Type                | 735502 non-null  | object |
| 13  | Birth Year               | 735502 non-null  | float64|
| 14  | Gender                   | 735502 non-null  | int64  |
| 15  | Trip_Duration_in_min     | 735502 non-null  | int64  |

**dtypes**: float64(5), int64(6), object(5)
**Memory usage**: 89.8+ MB

## Let`s start EDA:

```python
df.shape
```
(735502, 16)

There are 16 columns and 735502 rows in the dataset.

Let's first check if we have any duplicates. The first sep - is to clear our dataset from duplicates.
```python
df.drop_duplicates(inplace=True)
```
drop_duplicates(): This function removes duplicate rows from the DataFrame. inplace=True: This modifies the original DataFrame directly, so you don't have to reassign it.
```python
df.shape
```
(339620, 16)

Afer deduplication, we have just 339,620 records in the dataset left.

### Descriptive Statistics:

- Mean: The average value of a dataset.   
- Median: The middle value when the data is sorted.   
- Mode: The most frequently occurring value.   
- Standard Deviation: Measures how spread out the data is from the mean.   
- Range: The difference between the maximum and minimum values.

```python
mean_duration = df['Trip Duration'].mean()
median_duration = df['Trip Duration'].median()
mode_duration = df['Trip Duration'].mode()[0]
std_dev_duration = df['Trip Duration'].std()  # Standard Deviation
range_durtion = df['Trip Duration'].max() - df['Trip Duration'].min()  # Range
print("Mean:", mean_duration, "Median:", median_duration, "Mode:", mode_duration,"Std Dev:", std_dev_duration, "Range :", range_durtion)
```
**Mean**: 962.9629203227137 **Median**: 384.0 **Mode**: 244 **Std Dev**: 48685.701289240606 **Range** : 20260150

Based on the descriptive statistics of the Trip Duration data:

**Mean** (962.96 seconds): The average trip duration is about 16 minutes, suggesting that, on average, bike trips are relatively short.  
**Median** (384 seconds): The median trip duration is around 6.4 minutes, indicating that half of the bike trips are shorter than this time, showing that most trips are relatively brief.  
**Mode** (244 seconds): The most common trip duration is 244 seconds (about 4 minutes), reflecting a large number of quick bike trips.  
**Standard Deviation** (48,685.70 seconds): The data shows a high variability in trip durations, with some trips being significantly longer or shorter than the mean.  
**Range** (20,260,150 seconds): The enormous range indicates that there are still some extremely long trips, suggesting the presence of outliers that greatly extend the duration spread.   
**Overall Insights:**  
Most bike trips are short and consistent, but there are a few extremely long trips causing a large spread in the data.
It would be valuable to investigate these outliers further to understand their nature and decide if they should be treated differently in your analysis.
```python
df['Trip Duration'].max
```
| Index    | Trip Duration |
|----------|----------------|
| 0        | 376            |
| 1        | 739            |
| 2        | 2714           |
| 3        | 275            |
| 4        | 561            |
| ...      | ...            |
| 378143   | 717            |
| 378144   | 473            |
| 378145   | 237            |
| 378146   | 368            |
| 378147   | 2415           |
**Name**: Trip Duration, **Length**: 339620, **dtype**: int64>

Data Distribution Analysis Why We Do It:

The goal of data distribution analysis is to understand how the values of a variable, like trip duration, are spread across the dataset. This helps us identify patterns, such as whether most trips are short or if there are many long trips, and to detect any extreme values (outliers) that could affect our analysis.

**Histogram:**

Purpose: A histogram helps us see how frequently different values of trip duration occur. It shows whether the data is skewed (more data points on one side) or normally distributed.

```python
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Suppress FutureWarnings to keep the output clean
warnings.filterwarnings("ignore", category=FutureWarning)

# Replace infinite values with NaN to avoid plotting issues
df['Trip Duration'].replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop missing values (NaNs) from the Trip Duration column
df_clean = df['Trip Duration'].dropna()

# Plot a histogram of trip duration
plt.figure(figsize=(10, 6))
sns.histplot(df_clean, bins=30, kde=True, color='skyblue')
plt.title('Distribution of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```
![Destribution of Trip Duration](https://github.com/user-attachments/assets/84d934fe-558a-4b5f-8171-ad71ff0777f6)
