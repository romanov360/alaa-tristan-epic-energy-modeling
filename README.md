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