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

