# Alaa and Tristan's Epic Energy Modeling

---

## Problem Identification

# The Prediction Problem 
**Our goal** is to predict the Outage Duration using Regression Modeling. 

**The response variable:** Outage Duration

There are many benefits in having the response variable as outage duration. 
**The reasons:**
1. The ability to accurately predict outage duration given a set of features will help individuals that are in the effected area to plan accordingly. 
2. The abilitiy to accurately predict outage duration given a set of features will help the city institutions, such as the police, firemen, schools, etc. plan accordingly.
3. Knowing how long an outage might last will allow everyone to plan, and thus, will make it so the effects of the outage are not as strong and felt heavily across the city/affected region. 

**The independent variables:**
categorical:
1. nerc.region
2. month
3. cause.category
quantitative: 
1. population


**Justification:**
- It is easy ti see that nerc.region, month, and population would be known prior to the outage, and thus, prior to observing the power outage duration (when the duration ends).
- Regarding cause.category, one might say that the cause.category is not known before the outage duration is recorded. However, we believe that the cause.category must be known prior to the outage duration. First, outage duration is recorded only when the outage ends. In order for the outage to end, it must be fixed, and in order for it to be fixed, the cause (the root of the problem) must be identified. Therefore, cause.category is recorded before outage.duration is recorded.



**The model evaluation metric:** 
RMSE (Root Mean Squared Error)

**Why?** Since our model will be a regression model, this eliminates accuracy, precision, and recall as those are most suitable for classification models.
This leaves us with RMSE and R^2.
We chose RMSE because R^2 is most suitable for simple linear regression models, and our model is a MULTIPLE linear regression model. 
Therefore, the ideal evaluation metric is RMSE. 

# Data Cleaning
We start with the same data cleaning we did in **Project 3**, as seen below:
```py
df = get_data_with_correct_types()
df=df.sort_values(by=['YEAR','MONTH'])
df=df.reset_index().drop(columns='index')
```

We want to predict outage duration, but there are 58 missing values. They are missing by design. Every column where the outage restoration date was unknown, they put an NA in OUTAGE.DURATION, and every missing OUTAGE.DURATION has a missing OUTAGE.RESTORATION.DATE. We drop the rows where duration is missing because duration is the attribute we care to predict and it cannot be filled in with the other columns. We drop the rows before we perform the train-test split to preserve the test set size.

We will be training models to predict the future. We will sort the rows by date and reserve the last 20% as the test set. With the first 80% we will do all of our modeling and generalization testing/hyperparameter tuning. Then with our final models we will apply them to the test set, which contain the rows future to all of the rows the models have seen.
```py
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='OUTAGE.DURATION'), df['OUTAGE.DURATION'], test_size=0.2, shuffle=False)
```

## Baseline Model

# Model Description 
Multiple linear regression model that takes in two features: nerc.region and month

**Feature 1:** NERC.REGION 
- *data type:* nominal
- one hot encoding

**Feature 2:** MONTH 
- *data type:* nominal
- one hot encoding


**Model preformance:**

rmse = 5800.448462506011

## Final Model

# Model Description 
Multiple linear regression model that takes in four features: nerc.region, month, cause.category, population

**Feature 1:** NERC.REGION 
- *data type:* nominal
- one hot encoding

**Feature 2:** MONTH 
- *data type:* nominal
- one hot encoding

**Feature 3:** CAUSE.CATEGORY 
- *data type:* nominal
- one hot encoding

**Feature 4:** POPULATION 
- *data type:* quantitative
- standard scaling quantitative encoding



**Why did we choose Feature 3 and Feature 4?**
**Feature 3 reasoning:**
- We believe that the addition of Cause Category feature will decrease our model's RMSE, and improve its predictive power. 
- There is a strong correlation between the duration of a power outage and its cause because certain causes like severe weather and equipment failure often result in more extensive damage that takes longer to repair. On the other hand, causes like public appeal or intentional attacks may have shorter durations as they are more dependent on external factors or specific targeted actions.

**Feature 4 reasoning:**
- We believe that introducing the population of the affected region will decrease the RMSE and improve the predictive power of our model. 
- There is a strong correlation between the duration of a power outage and the population of the affected region because larger populations typically require more extensive repairs and logistical efforts to restore power to all affected areas. 
- Additionally, prioritizing power restoration in densely populated regions may take longer due to the complexity of the infrastructure and the number of affected customers.

**Model preformance:**

rmse = 5333.232830575117

For our training data: 
There is a **8.05%** decrease in RMSE :D

## Fairness Analysis

Did our model perform differently on high-population and low population areas? It would be interesting to see whether there appears to be a difference in prediction quality. This would indicate a bias for one type of energy grid (hihg population) versus another (low population).
We see the histogram of the population column below.

<iframe src="pop_dist.html" width=800 height=600 frameBorder=0></iframe>

We binarize, choosing 14M as the cutoff. For the high population outages we get a 5092 RMSE, and with the low population outages we get a 5343 RMSE. The model seems to be (much) better at predicting high population outage durations than low population outage durations.

Let's put this to the test with a hypothesis test. Here are the RMSE differences in the random samples.

<iframe src="high_low_diff_dist.html" width=800 height=600 frameBorder=0></iframe>

The observed difference was -250, which is right in the middle of the distribution, giving a p-value of 0.494. That's higher than alpha=0.05. Fail to reject the null; there is not enough evidence to conclude that the model is biased towards better high population or low population area predictions. So the model seems to be fair on predictions with different population sizes.