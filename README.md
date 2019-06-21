# CEBD1160 : Prediction of Apparent Temperature using Local Atmospheric conditions

| Name | Date |
|:-------|:---------------|
|Nikhil R|June 14, 2019|

-----

### Resources
Your repository should include the following:

- Python script for your analysis: `Weather_20190613`
- Figures:  `Prediction_NR_20190614`
- Dataset for your experiment: `weatherHistory.csv`

-----

## Research Question

Is there a relation between humidity and temperature, if so is it possible to predict the apparent temperature using Humidity?

### Abstract

Coming from a tropical region of the world, it always felt that the humidity and temperature are highly interconnected but without any solid proof. Keeping this in mind, a data set was forked out from Kaggle which gives details of weather conditions of Szeged, Hungary.
The data was for a period of 10 years and had nearly 97000 data points.
Fortunately, the data was pretty much cleaned and consisted data of temperature, apparent temperature, humidity, wind speed, wind angle, visibility, pressure etc.
Using these data, we may be able to understand relationship between apparent temperature and other factors, which can then be used for predicting precipitation type (i.e will rain or not).
Here, we tried to predict the apparent temperature using humidity alone.
With the given condition of humidity alone it was found to be difficult to predict the apparent temperature and it was found all the other factors play an equally significant role in prediction.


### Introduction

The heatmap correlation gives an insight about the interdependencies of selected features. (https://github.com/NIKH0610/Final-Project/blob/master/HeatMap_Correlation_NR_20190614.jpeg). From this we can get a hint of which factors contribute most to the prediction of Apparent Tempertaure.

### Methods

The method used for modelling this data was the Linear Regression built into scikit-learn.
Method uses the training set to calculate the weights and bias for all the features that are used to predict the apparent temperature. 
Recursive Feature Extraction can then be added on the top to extract the features that give the maximum score which is calculated by subtracting the predict value and the actual output.
sklearn.linear_model.LinearRegression and sklearn.feature_selection.RFE are two main modules used for implementation.
This method is used in because of simplicity and imporved efficiency.

### Results

Humidity is an important factor in predicting the apparent temperature, but on seeing the prediction score we can conclude that humidity alone is not enough to make an accurate prediction of the apparent temperature. We need another factor of Temperature to get an accurate prediction always.

