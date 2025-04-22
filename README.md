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
You can view the full notebook here: [NYC Bike Share EDA on Kaggle](https://www.kaggle.com/code/marypronina/nyc-bike-share-eda/edit)

```python
# In [1]:
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
