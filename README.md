# sber_ml_hack
a project of predicting gender of a person based on his/her transactions from the Sber ML Hackathon

## Team:
[Arina Nevolina](https://github.com/nevolinaa), [Artem Pospelov](https://github.com/artem-pospelov)

## Project stages
- Exploratory data analysis
- Features engineering
- Model training
- Sending the solution to a prepared container to calculate a quality metric

## Project outcomes
- There are the same number of men and women in the sample, and they spend about the same by categories generated using the transaction type feature  
![picture](https://github.com/nevolinaa/sber_ml_hack/blob/main/data/trans_types.png) 
- However, men and women spend differently by categories generated using the mcc code feature and information on broader cashback categories from the bank's website  
![picture](https://github.com/nevolinaa/sber_ml_hack/blob/main/data/mcc_codes.png)
- The time of the transaction also proved to be an important feature, specifically 1) the month 2) the time of day ( afternoon, evening, night, morning) 3) the day of the week
- Finally, with the generated features and the use of the XGBoost model, we succeeded in obtaining a value of the quality metric AUC ROC of 0.86 on the test part of the dataset
- The most important attributes based on the SHAP value are:  
![picture](https://github.com/nevolinaa/sber_ml_hack/blob/main/data/features_shap.png)
