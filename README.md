# NYC Bike Share EDA

  In this project, I perform Exploratory Data Analysis (EDA) using Python on the [New York City Bike Share Dataset](https://www.kaggle.com/datasets/akkithetechie/new-york-city-bike-share-dataset) available on Kaggle. My focus is on understanding trip durations, user behavior, and data quality, using a combination of statistical analysis, visualizations, and data cleaning techniques to uncover actionable insights.
  
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

```python
# Example: Filtering out trip durations longer than a reasonable threshold
df_filtered = df[df['Trip Duration'] < 10000]  # Adjust the threshold as needed
sns.histplot(df_filtered['Trip Duration'], bins=30, kde=True)
plt.title('Filtered Distribution of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.show()
```
![Filtered Distribution of Trip Duration](https://github.com/user-attachments/assets/186bc69d-6de4-43b8-bb7f-83598294fb35)

```python
# Example: Filtering out trip durations longer than a reasonable threshold
df_filtered = df[df['Trip Duration'] < 3600]  # Adjust the threshold as needed
sns.histplot(df_filtered['Trip Duration'], bins=30, kde=True)
plt.title('Filtered Distribution of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.show()
```
![Filtered Distribution of Trip Duration (1)](https://github.com/user-attachments/assets/8bcc2917-dbe8-496a-8b07-294ed29310c0)

The goal of these plots is to better understand the distribution of trip durations by filtering out extreme values that may skew the visualization. In real-world data, outliers such as very long trips (e.g., over an hour or over 10,000 seconds) can distort the overall shape of the histogram and make it harder to observe typical patterns.

By applying thresholds like Trip Duration < 10,000 or Trip Duration < 3600, the plots focus on more reasonable, common trip durations. This makes the distribution easier to interpret and helps identify central tendencies and common trip lengths more clearly.

**Box Plot:**

Purpose: A box plot helps us visualize the spread of the data and detect outliers. It shows the median (middle value), quartiles, and outliers.

```python
sns.boxplot(x=df['Trip Duration'])
plt.title('Box Plot of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.show()
```
![Box Plot of Trip Durations](https://github.com/user-attachments/assets/c1905864-a926-41ff-9692-91f234d31baa)

```python
df_filtered = df[df['Trip Duration'] < 3600]  # Adjust the threshold as needed
sns.boxplot(x=df_filtered['Trip Duration'])
plt.title('Box Plot of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.show()
```
![Box Plot of Trip Durations (1)](https://github.com/user-attachments/assets/b7f849dc-2d3c-480b-81c8-df721b99759c)

The box plot shows where most of the data points lie and highlights outliers that may need further investigation. The "box" represents the middle 50% of the data (interquartile range), and the "whiskers" extend to show the rest of the data, except for outliers.

```python
# Example: Filtering out extremely long trips
df_filtered = df[df['Trip Duration'] < 10000]  # Adjust the threshold as needed
sns.boxplot(x=df_filtered['Trip Duration'])
plt.title('Box Plot of Trip Durations (Filtered)')
plt.show()
```
![Box Plot of Trip Durations (Filtered)](https://github.com/user-attachments/assets/fa88b1d2-fde9-4b10-aace-f9e045ca6234)

Let's take a look at the values sorted by Trip Duration in descending order.

```python
# Add a new column 'Trip Duration in Hours' by converting seconds to hours
df['Trip Duration(hrs)'] = df['Trip Duration'] / 3600

# Sort the DataFrame by 'Trip Duration' in descending order and select the top 30
top_50_max_duration = df.sort_values(by='Trip Duration', ascending=False).head(50)

# Select only the relevant columns, including the new 'Trip Duration in Hours'
top_50_max_duration = top_50_max_duration[['Trip Duration', 'Trip Duration(hrs)', 'Start Time', 'Stop Time']]

# Display the updated DataFrame
print(top_50_max_duration)
```
|        | Trip Duration  | Trip Duration (hrs)  | Start Time            | Stop Time             |
|--------|----------------|----------------------|-----------------------|-----------------------|
| 374349 | 20260211       | 5627.836389          | 2015-09-26 04:20:59   | 2016-05-17 16:11:10   |
| 109375 | 16329808       | 4536.057778          | 2016-03-22 07:02:10   | 2016-09-27 07:05:38   |
| 78142  | 6065936        | 1684.982222          | 2015-12-12 21:04:35   | 2016-02-21 02:03:32   |
| 71700  | 5366099        | 1490.583056          | 2015-11-27 13:49:07   | 2016-01-28 16:24:07   |
| 313891 | 4826890        | 1340.802778          | 2016-11-23 17:38:36   | 2017-01-18 14:26:46   |
| 95455  | 2104123        | 584.478611           | 2016-02-12 07:27:56   | 2016-03-07 15:56:40   |
| 95491  | 2100551        | 583.486389           | 2016-02-12 08:31:06   | 2016-03-07 16:00:18   |
| 95593  | 2071209        | 575.335833           | 2016-02-12 16:32:54   | 2016-03-07 15:53:03   |
| 128729 | 1837255        | 510.348611           | 2016-04-28 09:05:14   | 2016-05-19 15:26:09   |
| 8191   | 1620142        | 450.039444           | 2015-10-13 23:52:48   | 2015-11-01 16:55:11   |
| 87178  | 1569765        | 436.045833           | 2016-01-09 05:49:39   | 2016-01-27 09:52:25   |
| 243701 | 1532001        | 425.555833           | 2016-09-11 16:32:21   | 2016-09-29 10:05:42   |
| 7135   | 1471896        | 408.860000           | 2015-10-12 16:29:28   | 2015-10-29 17:21:05   |
| 226371 | 1258736        | 349.648889           | 2016-08-26 23:19:01   | 2016-09-10 12:57:58   |
| 235239 | 1120971        | 311.380833           | 2016-09-03 22:05:27   | 2016-09-16 21:28:18   |
| 266151 | 1021330        | 283.702778           | 2016-10-01 15:01:46   | 2016-10-13 10:43:57   |
| 9932   | 942374         | 261.770556           | 2015-10-16 09:59:46   | 2015-10-27 07:46:00   |
| 363996 | 871460         | 242.072222           | 2017-03-09 08:59:22   | 2017-03-19 12:03:42   |
| 335320 | 802101         | 222.805833           | 2017-01-10 08:49:55   | 2017-01-19 15:38:17   |
| 312564 | 721297         | 200.360278           | 2016-11-21 23:29:17   | 2016-11-30 07:50:55   |
| 247265 | 623780         | 173.272222           | 2016-09-14 12:54:47   | 2016-09-21 18:11:07   |
| 154321 | 622821         | 173.005833           | 2016-06-07 13:20:54   | 2016-06-14 18:21:15   |
| 4282   | 496680         | 137.966667           | 2015-10-08 14:49:02   | 2015-10-14 08:47:02   |
| 236155 | 488819         | 135.783056           | 2016-09-04 21:11:32   | 2016-09-10 12:58:32   |
| 361199 | 399412         | 110.947778           | 2017-03-03 19:08:11   | 2017-03-08 10:05:04   |
| 368395 | 390893         | 108.581389           | 2017-03-24 19:45:28   | 2017-03-29 08:20:22   |
| 290143 | 361889         | 100.524722           | 2016-10-25 10:34:43   | 2016-10-29 15:06:13   |
| 7807   | 357636         | 99.343333            | 2015-10-13 15:49:34   | 2015-10-17 19:10:11   |
| 328893 | 355757         | 98.821389            | 2016-12-22 13:00:20   | 2016-12-26 15:49:38   |
| 314068 | 353192         | 98.108889            | 2016-11-24 08:18:57   | 2016-11-28 10:25:29   |
| 111850 | 322385         | 89.551389            | 2016-03-27 01:10:51   | 2016-03-30 18:43:57   |
| 78139  | 310083         | 86.134167            | 2015-12-12 21:01:46   | 2015-12-16 11:09:50   |
| 1674   | 294734         | 81.870556            | 2015-10-05 07:42:52   | 2015-10-08 17:35:06   |
| 58136  | 264226         | 73.396111            | 2015-11-01 16:31:36   | 2015-11-04 17:55:23   |
| 96271  | 259489         | 72.080278            | 2016-02-17 10:33:53   | 2016-02-20 10:38:42   |
| 244959 | 251098         | 69.749444            | 2016-09-12 18:30:39   | 2016-09-15 16:15:38   |
| 263826 | 248376         | 68.993333            | 2016-09-28 18:17:15   | 2016-10-01 15:16:52   |
| 131542 | 237444         | 65.956667            | 2016-05-03 23:49:53   | 2016-05-06 17:47:17   |
| 105299 | 234181         | 65.050278            | 2016-03-12 13:11:30   | 2016-03-15 07:14:31   |
| 101557 | 234066         | 65.018333            | 2016-03-03 18:56:50   | 2016-03-06 11:57:57   |
| 89665  | 227014         | 63.059444            | 2016-01-16 17:16:13   | 2016-01-19 08:19:48   |
| 87985  | 222661         | 61.850278            | 2016-01-11 21:15:40   | 2016-01-14 11:06:42   |
| 364668 | 221604         | 61.556667            | 2017-03-10 17:39:21   | 2017-03-13 08:12:46   |
| 19260  | 203604         | 56.556667            | 2015-10-31 23:45:12   | 2015-11-03 07:18:37   |
| 73443  | 202474         | 56.242778            | 2015-12-02 18:45:13   | 2015-12-05 02:59:48   |
| 322619 | 199384         | 55.384444            | 2016-12-09 11:33:44   | 2016-12-11 18:56:48   |
| 351266 | 196255         | 54.515278            | 2017-02-16 07:45:17   | 2017-02-18 14:16:13   |
| 141839 | 194937         | 54.149167            | 2016-05-20 10:13:57   | 2016-05-22 16:22:54   |
| 274364 | 190956         | 53.043333            | 2016-10-10 02:13:37   | 2016-10-12 07:16:14   |
| 218222 | 188396         | 52.332222            | 2016-08-19 13:10:14   | 2016-08-21 17:30:11   |

Unusually Long Durations: The trip durations in the top 30 range from 496,680 seconds (about 5.7 days) to 20,260,211 seconds (about 234 days). For a bike-sharing service, this duration is extraordinarily long. Most bike trips typically last from a few minutes to a few hours, not multiple days. Suspicious Patterns: The start and stop times for these trips span multiple days, weeks, or even months. This suggests that these trips could be due to data errors, such as trips that were not properly checked back in or recorded.

```python
# Example: Filtering out extremely long trips
df_filtered = df[df['Trip Duration'] < 200000]  # Adjust the threshold as needed
sns.boxplot(x=df_filtered['Trip Duration'])
plt.title('Box Plot of Trip Durations (Filtered)')
plt.show()
```
![Box Plot of Trip Durations (Filtered 1)](https://github.com/user-attachments/assets/7f6bddb3-b5eb-4eb0-9725-8fa794b632a0)

**How to get rid of outlers?**

The Quartile Method, also known as the Interquartile Range (IQR) Method, is commonly used to identify and remove outliers from a dataset.  

**Q1 (First Quartile)**: Imagine you have all your trip durations lined up from shortest to longest. Q1 is the point where 25% of the trips are shorter, and 75% are longer. It's like the mark where the first quarter of your data ends.  
**Q3 (Third Quartile)**: Q3 is the point where 75% of the trips are shorter, and only 25% are longer. It's the mark where three-quarters of your data is below it, and only the top quarter is above.  
**IQR (Interquartile Range)**: The IQR is the space between Q1 and Q3. It tells us how spread out the middle half of the trips are. So, if the IQR is small, most trips are of similar duration. If it’s big, the trip durations are more spread out.  

Think of it like dividing your trip data into four equal parts: Q1 is the end of the first part, Q3 is the end of the third part, and the IQR is the chunk in between where the bulk of your data sits.

```python
Q1 = df['Trip Duration'].quantile(0.25)
Q3 = df['Trip Duration'].quantile(0.75)
IQR = Q3 - Q1
print("IQR:",IQR,"Q1:",Q1,"Q3:",Q3)
```
**IQR**: 409.0 **Q1**: 247.0 **Q3**: 656.0

Q1 (25th percentile): 247 seconds (about 4.1 minutes)
Q3 (75th percentile): 656 seconds (about 10.9 minutes)
IQR (Interquartile Range): 409 seconds (Q3 - Q1)
Using These Values to Identify Outliers: We can use the IQR to calculate the lower and upper bounds for detecting outliers:

Calculate the Bounds:
Lower Bound: Q1−1.5×IQR=247−1.5×409=247−613.5=−366.5 Since trip duration cannot be negative, the lower bound will be 0.
Upper Bound: Q3+1.5×IQR=656+1.5×409=656+613.5=1269.5

**Interpretation**:

Any trip duration below 0 seconds (which is impossible in this context) or above 1269.5 seconds (about 21.2 minutes) is considered an outlier.

```python
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['Trip Duration'] < lower_bound) | (df['Trip Duration'] > upper_bound)]
print("outliers: trips longer than",upper_bound)
```
**outliers**: trips longer than 1269.5

```python
# Example: Filtering out extremely long trips
df_filtered = df[df['Trip Duration'] <= 1269.5]  # Adjust the threshold as needed
sns.boxplot(x=df_filtered['Trip Duration'])
plt.title('Box Plot of Trip Durations (Filtered)')
plt.show()
```
![Box Plot of Trip Durations (Filtered 2)](https://github.com/user-attachments/assets/2cbda292-10b0-459f-a3cf-60f0b2577fdc)

```python
df_filtered.shape
```
(309029, 17)

```python
median = df_filtered['Trip Duration'].median()
mean = df_filtered['Trip Duration'].mean()
mode = df_filtered['Trip Duration'].mode()[0]
std_trip = df_filtered['Trip Duration'].std()
range_trip = df_filtered['Trip Duration'].max() - df_filtered['Trip Duration'].min()
print("Median:",median,"Mean:",mean,"Mode:" , mode,"Std:", std_trip,"Range:", range_trip)
```
**Median**: 353.0 **Mean**: 428.31488306922654 **Mode**: 244 **Std**: 257.8431412666783 **Range**: 1208

**Median:** `353.0` seconds (≈ 5.88 minutes)  
  The median represents the middle value of trip durations. Half of the trips are shorter than 353 seconds, and the other half are longer. This indicates that many bike trips are fairly short.  
**Mean:** `428.31` seconds (≈ 7.14 minutes)  
  The mean (average) is higher than the median, which suggests that some longer trips are pulling the average up. However, they are not as extreme as the original outliers before data cleaning.  
**Mode:** `244` seconds (≈ 4.07 minutes)  
  This is the most frequently occurring trip duration. Many rides last around 4 minutes, showing a popular pattern of short trips.  
**Standard Deviation:** `257.84` seconds  
  This reflects a moderate spread in trip durations. Most trips fall within roughly 4 minutes of the mean.  

**Insights**

- Most bike trips are **short**, typically between **4 to 6 minutes**.
- The **mean** being slightly higher than the **median** shows a few longer trips still influence the average.
- After outlier removal, the dataset gives a **clearer picture of user behavior**, confirming that **bike-sharing is mostly used for short rides**, with occasional longer ones.

```python
# Example: Filtering out trip durations longer than a reasonable threshold
#df_filtered = df[df['Trip Duration'] < 3600]  # Adjust the threshold as needed
sns.histplot(df_filtered['Trip Duration'], bins=30, kde=True)
plt.title('Filtered Distribution of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.show()
```
![Filtered Distribution of Trip Duration (2)](https://github.com/user-attachments/assets/8d28f3fe-3e7f-408a-87b7-1d0535a46c6a)

This histogram represents the filtered distribution of trip durations. Here's how to interpret it:

**Shape of the Distribution**:

The distribution is right-skewed, meaning that most bike trips are short in duration, and as the duration increases, the number of trips decreases. The peak of the histogram occurs around 200-300 seconds (3.3 to 5 minutes), indicating that most bike trips are concentrated around this duration. Frequency:

The y-axis shows the frequency of trip durations, with the highest frequency reaching around 30,000 trips. This indicates that a large number of bike trips are in the shorter duration range. As the duration increases beyond 300 seconds, the frequency steadily decreases, showing that longer trips are less common. Tail of the Distribution:

The right tail extends up to around 1200 seconds (20 minutes), with relatively fewer trips occurring in this range. This confirms that while some longer trips still occur, they are much less frequent compared to the shorter ones.

**Insights**:

The majority of bike trips are short, typically between 3 to 7 minutes, which is consistent with the nature of a bike-sharing service meant for quick rides. The right-skewed nature of the data suggests that there are still some longer trips, but they are not as extreme as the original outliers. This distribution helps us understand typical usage patterns and indicates that most users prefer short-duration rides.

## How does trip number vary on user type?

```python
df_filtered.shape
df = df_filtered
```

```python
# Count the total number of records and the number of 'Subscriber' user types
total_users = df.shape[0]
subscriber_count = df[df['User Type'] == 'Subscriber'].shape[0]
customer_count = df[df['User Type'] == 'Customer'].shape[0]

# Calculate the percentage of 'Subscriber' user types
subscriber_percentage = round((subscriber_count / total_users) * 100,2)
customer_percentage = round((customer_count / total_users) * 100,2)
print("Percentage of Subscribers:", subscriber_percentage, "%, while just", customer_percentage,"% is not subscribed")
```
**Percentage of Subscribers**: 96.06 %, while just 3.94 % is not subscribed

```python
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")
```

```python
px.pie(values = df['User Type'].value_counts(),
       names =df['User Type'].value_counts().index,
       title ="User Type Variation")
```
![User Type Variation](https://github.com/user-attachments/assets/29156ab7-e25b-491c-8e18-0a2de18b013d)

To better understand the user types in the dataset, I first checked the shape of the filtered DataFrame and reassigned it to df to continue working with a cleaned version of the data.  
Then, I calculated the total number of users and split them into two categories: Subscribers and Customers. Using this, I calculated the percentage of each group.  
The result showed that a significant 96.06% of users are Subscribers, while only 3.94% are Customers, meaning they are not subscribed.

## How does trip number vary based on gender?

```python
px.pie(values = df['Gender'].value_counts(),
       names =df['Gender'].value_counts().index,
       title ="Gender Variation")
```
![Gender Variation](https://github.com/user-attachments/assets/cde2f59d-2683-4f23-b90b-658a2ff51104)

```python
# Group the data by 'Gender' and calculate summary statistics for 'Trip Duration'
gender_summary = df.groupby('Gender')['Trip Duration'].agg(['mean', 'median', 'count'])

# Rename the columns for clarity
gender_summary.rename(columns={'mean': 'Average Trip Duration', 'median': 'Median Trip Duration', 'count': 'Number of Trips'}, inplace=True)

# Display the summary statistics
print(gender_summary)
```
**Trip Duration Statistics and Total Number of Trips by Gender**

| Gender | Average Trip Duration (sec) | Median Trip Duration (sec) | Number of Trips |
|--------|-----------------------------|-----------------------------|-----------------|
| 0      | 610.04                      | 562.0                       | 17,550          |
| 1      | 412.17                      | 336.0                       | 227,201         |
| 2      | 435.77                      | 372.0                       | 64,278          |
> **Note:**  
> Gender `0` refers to unspecified, `1` to male, and `2` to female users (based on dataset encoding).

```python
# Calculate average and median trip durations by gender
gender_trip_duration = df_filtered.groupby('Gender')['Trip Duration'].agg(['mean', 'median'])
print(gender_trip_duration)

# Visualize with a box plot
sns.boxplot(x='Gender', y='Trip Duration', data=df_filtered)
plt.title('Trip Duration by Gender')
plt.xlabel('Gender (0 = Not Specified, 1 = Male, 2 = Female)')
plt.ylabel('Trip Duration (seconds)')
plt.show()
```
**Average and Median Trip Duration by Gender**

| Gender | Mean Trip Duration (sec) | Median Trip Duration (sec) |
|--------|---------------------------|-----------------------------|
| 0      | 610.04                    | 562.0                       |
| 1      | 412.17                    | 336.0                       |
| 2      | 435.77                    | 372.0                       |

![Trip Duration by Gender](https://github.com/user-attachments/assets/c184872d-5c85-42d5-8c84-bdab00de3341)

```python
# Count the number of trips for each gender
trip_count_by_gender = df_filtered['Gender'].value_counts()
print(trip_count_by_gender)

# Visualize with a bar plot
sns.barplot(x=trip_count_by_gender.index, y=trip_count_by_gender.values)
plt.title('Trip Frequency by Gender')
plt.xlabel('Gender (0 = Not Specified, 1 = Male, 2 = Female)')
plt.ylabel('Number of Trips')
plt.show()
```
**Number of Trips by Gender**

| Gender | Number of Trips |
|--------|------------------|
| 1      | 227,201          |
| 2      | 64,278           |
| 0      | 17,550           |

![Trip Frequency by Gender](https://github.com/user-attachments/assets/dc6e4347-7294-455f-a9fd-f6b03fdcc09d)

The analysis of NYC Bike Share usage shows significant differences in how often and how long people ride, depending on their gender:

Male users (Gender = 1) take the most trips by far — over 227,000 rides.  
Female users (Gender = 2) are the second-largest group, with about 64,000 rides.  
Users who didn’t specify their gender (Gender = 0) take the fewest trips, around 17,500. 

Interestingly, even though users with unspecified gender take the least number of trips, their average and median trip durations are the longest. In contrast, male riders have the shortest trips on average, even though they ride most frequently. Female users fall in between — their trip durations are longer than male users but shorter than those with unspecified gender.

This suggests that gender may not only influence how frequently people use the service but also how long they ride when they do.

## Conclusion: EDA with Python

This project has demonstrated how Python can be an invaluable tool for conducting Exploratory Data Analysis (EDA). By leveraging powerful libraries such as Pandas, Seaborn, Matplotlib, and Plotly, we were able to efficiently clean, analyze, and visualize data, uncovering important trends and insights.

Through the process of data cleaning, aggregation, and visualization, we gained a deeper understanding of the patterns in bike trip data, such as the differences in trip durations and trip frequencies across gender. The use of groupby(), agg(), and various plotting techniques helped us quickly identify key statistics and make the data more accessible and interpretable.

**Key Takeaways**:

**Quick Insights**: Python allows for rapid exploration of datasets, providing immediate insights into key metrics and patterns.  
**Flexibility**: Python’s extensive libraries enable flexible data manipulation and analysis, making it easy to perform both simple and complex EDA tasks.  
**Effective Visualization**: Tools like Seaborn and Matplotlib helped to present the results in an engaging and informative way, making the data more comprehensible and actionable.

In conclusion, Python is a powerful tool for EDA, providing all the necessary capabilities to clean, explore, and visualize data, ultimately allowing us to make data-driven decisions efficiently and effectively.
