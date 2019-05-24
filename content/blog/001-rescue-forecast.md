---
title: "RescueForest: Predicting Emergency Response with Random Forests"
date: 2019-05-23
draft: False
type: "Blog"
---

# RescueForest: Predicting Emergency Response with Random Forests

I'm finishing up the material in the fast.ai [Introduction to ML](http://course18.fast.ai/ml) course and it has been fantastic. The learning philosophy, which the instructor Jeremy describes as a top-down approach, worked really well for my learning style. The general idea is that you learn HOW to do ML before you learn WHY it works - the same way you learn HOW to play baseball before learning WHY you might use one particular strategy at a given time.

I wanted to do a small capstone project to practice what I learned, so I decided to study emergency response prediction - particularly for an organization called [King County Search and Rescue](http://www.kingcountysar.org/) (SAR), which I volunteer as a member of. We do a variety of rescue missions, but the most common call is to help injured or lost hikers in the wilderness surrounding Seattle. It's mostly a volunteer-run organization, so resources are inherently limited. Having a method to predict the likelihood of a call on any given day might help us prioritize resources and preparation to best serve the community.

I broke the project into three general steps covered in the course:
1. Data Exploration and cleaning
2. Building a simple model and using it to study feature importance
3. Feature engineering and tuning the model

This post will cover my general approach and findings. Detailed code and notebooks can be found in my project's [repo](https://github.com/afederation/SAR_predict).

## Data Exploration, Cleaning

Before building the model, I want to understand the nature of the data we have. There are several data sources we're using, and we'll be linking all of them together by the date. Python's [datetime](https://docs.python.org/2/library/datetime.html) library was key for this and made these operations much easier.

### Outcome Data

The goal of the project was to predict whether or not a SAR call happened on a particular day. I've downloaded this from the organization's internal database and removed some information for privacy. Loading the data into python and pandas allows for some simple plots to make sure we're focusing on the most relevant data. I had to narrow the dates into a range from 2002 to current date, since the data before this was incomplete.

Some exploratory plots show that we have calls on 29% of days, which means the data is somewhat imbalanced, but not horribly so. Calls happen most frequently on weekends and in the summer. We know this by intuition, and the data backs this up. This also confirms that there is probably some information within the date alone that may have some predictive power.

![calls per week](/images/post1/callsperweek.png)
![calls per day](/images/post1/callsperday.png)

But first, we need a tidy dataset. Essentially, we a table with every date in the range we're considering and give a boolean that reports on the presence/absence of a call on that day. A simple script to convert the raw `sar_data` into a `clean_table` accomplished this by taking advantage of the panda's date_range function.

```python
date_range = pd.date_range(start='1/1/2002', end='4/01/2019')
clean_table = []
for d in date_range:
    if sar_data.date.isin([d]).any(): # check if date in in table containing all calls
        clean_table.append([d,1])
    else:
        clean_table.append([d,0])               
sar_clean = pd.DataFrame(clean_table)
sar_clean.columns = ['date','mission']
```

### Features

Using the date column from the `clean_table` table, we can use the nifty fast.ai fonction that extracts information like day of week, year, month, etc. from a datetime object. After applying that to our tidy table, we have ~13 features to start with.

Along with these, we'll integrate some weather information from NOAA. There are a bunch of cool [tools](https://www.ncdc.noaa.gov/cdo-web/) they provide that data scientists will find useful, and I encourage anyone interested in integrating weather data to check it out.

Ideally, we'd be able to use weather forecast data for this project, rather than actual weather data. When we're planning ahead, we'll only have access to forecasts, which obviously differ from the actual weather quite often. All the resources I found that keep these data were paid services, so for I'll have to settle with the NOAA data for now.

I downloaded weather data from two local weather stations - Boeing Field, which is just south of Seattle, and Mount Gardner, which is in the mountain range where most of our missions occur. We get information on temperature, wind, sun and precip, which is a good picture of the weather on a given day. A couple sanity checks make sure the data are organized in the way we expect (hot in the summer, more rain in the winter). Merging these with the `clean_table` using `pandas.merge` give us our initial data set that contains the following features:

`[Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
       'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start',
       'Is_year_end', 'Is_year_start', 'Elapsed', 'DATE', 'AWND_x', 'PRCP_x',
       'TAVG_x', 'TMAX_x', 'TMIN_x', 'TSUN', 'WT01', 'WT02', 'WT03', 'WT05',
       'WT08', 'WT10', 'AWND_y', 'PRCP_y', 'SNWD', 'TAVG_y', 'TMAX_y',
       'TMIN_y', 'TOBS', 'WESD']``

## Feature Engineering

### Test, Train, Initial Model

Selecting an appropriate validation and test set was an important topic in the course. I'm taking a similar approach here as the Kaggle competitions for bulldozer auctions and grocery store sales predictions that Jeremy discussed. In these situations, we need to use data from the past to predict the future, so we sort the data by date and hold out the most recent 15-20% of the data for the validation set. We'll do the same thing here.

For the test set, I'll use a more 'real-life approach' and just test my model on this year in real-time.

For model evaluation, the built-in scoring functions probably aren't ideal. There's only a mission 29% of the time, so a naive model always predicting 'no call' will be correct with 71% accuracy. ROC curves are a decent approach for this situation, so I used these for model evaluation. These weren't extensively discussed in the fast.ai course, as they focused mostly on regression problems. As a refresher, I found [this article](https://medium.com/greyatom/lets-learn-about-auc-roc-curve-4a94b4d88152) to be helpful.

Next, a simple random forest built wirth scikit-learn gave some signal in the ROC curve (AUC=0.61), so I moved ahead with this to understand the data at hand a bit better.

![roc1](/images/post1/roc1.png)

When digging into the feature importances used in the model, things seemed to make sense. The day of week, days elapsed (just a running count of days since the beginning of the dataset) and day of year were the three most important features. The most important weather feature was the wind in Seattle, which may be reporting on the presence of a storm? The next few weather features were temps, which isn't a surprise.

In the course, Jeremy keeps all features that show an importance, even if they may be correlated. There's no doubt that temperatures in nearby places are correlated, but the model is supposed to deal with this with an ensemble approach. In the future, it may be worth doing some additional feature engineering to remove some redundancies. I did remove a few features that had very low importance without and didn't hurt performance after removal.

### Holiday Data

Will adding holiday data help the model? I used `pandas.tseries.holiday` which contains a database of US holidays and tried a few different approaches, summarized below:
- Add a column with a boolean for `is_holiday`
- Add columns for `days_until_next_holiday` and `days_since_last_holiday`
- Add a column for `days_away_from_nearest_holiday`

The last approach was most effective, which makes sense. The days surrounding holidays are most likely to be days off of work and correspond to an increase in hikers. The ROC improves from 61% to 65% by including this information.

### Google Trends

I discoverd a cool feature of google trends data during this project. There isn't an API, but just playing around with the web interface, I was able to get the search frequency on the general topic of 'hiking' in the King County region - our exact zone of response. The data are seasonal as expected, and I was hoping they might capture other events happening that may increase or decrease interest in hiking during the short-term.

![trend](/images/post1/trends.png)

Adding this feature gave a confusing result - the ROC doesn't improve, but when you look at feature importance, it's the second most important feature. Does this just mean that the trend data is redundant with some of the date features? I decided to keep it and move forward.

## Building the Final Model

Okay, so now the final model. I compared three tree-based algorithms with the same basic parameters and the results are summarized below.

| Model         | ROC AUC       |
|:------------|:-------------:|
| Random Forest     | 0.59 |
| AdaBoost      | 0.61      |  
| XGBoost |  0.63      |  

XGBoost did the best, so a tool the opportunity to learn about how to tune these models. The main approach I utilized was sklearn's build in [grid search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) functionality. This took a long time to run, but the results give me the best combination parameters the model could find. This brought our AUC up to 0.67.

## Model Evaluation

I'm going to monitor the model's performance all year, but as of writing this article, the model is predicting correctly 2/3 of the time. It's a hard problem, with a lot of randomness and a small amount of data. That being said, it has made a difference for my life so far, letting me have gear ready ahead of time on the days it predicts a high call probability.

Still, there's definitely room for improvement. I'm going to continue trying new models as I learn them and adding in additional data as I find it. I'm also excited to hear suggestions from anyone with other ideas for features or ways I can improve my approach. And thanks for Jeremy, Rachel and the fast.ai team for a great experience!
