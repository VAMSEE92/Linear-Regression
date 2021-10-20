
# Linear Regression
Before understanding and implementing linear regression, let us first understand what is a regression. Regression is a method of finding a model that explains target variables based on independent features. Regression techniques differ mostly based on the number of independent variables and the type of relation between independent and the dependent variables.
#### Representation of Linear regression model 
Linear regression model is the combination of the set of input values(x) and a predicted output(y) for the set of input values.
##### Linear equation assaigns one scale factor to each input column, called as coefficient and represented by Beta(β) or sometimes by 'm'. One additional value is also added which is often called as 'Intercept' or 'bias'. It is used for moving the line in two dimentional plot 
##### For example, a simple linear model has one input(x) and a predicted target(y), and the linear model would be represented as 
##### y=β0+β1X
#### simple Linear Regression
With simple linear regression when we have single input, we can estimate the coefficients.
##### This requires to calculate the mean, standard deviations, Correlation.
#### Ordinary Least Squares
When we have more than one input we use Ordinary Least Squares to estimate the coefficients. The coefficients and intercept is calculated as
##### β1=[n(Σxy)-(Σx)(Σy)]/[n(Σx^2)-(Σx)^2]
##### β0=[(Σy)(Σx^2)-(Σx)(Σxy)]/[n(Σx^2)-(Σx)^2] 
##### Note that just because independent and dependent variables are related to each other, it doesn't mean independent variable causes the dependent variable. 
#### Gradient Descent
We can also get the coefficients and the intercept at which the sum of squared error is minimum with the gradient descent. It works by starting with random values for each coefficient. The sum of squared error is calculated for each pair of input and output values. A learning rate is used as a scale factor and the coefficients are updated in the direction towards minimizing the error. The process is repeated until a minimum sum squared error is achieved or no further improvement is possible.
##### In this method we select a learning  rate(alpha) parameter that determines the size of the improvement step to take on each iteration of the procedure.
## Implementing Linear regression
We will construct a linear model that explains the relationship a car's mileage (mpg) has with its other attributes
### Preparing Data for Linear Regression
For our model to fit and perform well our dataset should be clean without any unrelated values or features, should not contain any missing values. We should also take care of outliers and categorical features
##### To achieve this we need to perform certain steps on our data which is also called as preprocessing of data
##### First is first, before starting with the preprocessing let us get more into the details of our dataset.
##### -> The dataset total has 9 features 1 dependent and 8 independent features
##### ->MPG(Miles per gallon) : It is the target feature which needs to be explained with the help of other independent features
##### ->cylinders : It has the information on number of cylinders present in the engine of that perticular car model
##### ->engine displacement: It explains total volume of all cylinders in an engine. Units 'cu. inches'
##### ->horsepower: Explains the power produced by a car engine
##### ->weight: Weight of the vehicle measured in 'lbs'
##### ->acceleration: It is the time taken to accelerate from 0 to 60mph and is measured in 'seconds' 
##### ->model year: Year in wich the model is manufactured
##### ->origin of car: Categorical feature. Though it is a categorical feature in the dataset it is represented as 1 for America 2 for Europian 3 for Japanese
##### ->car name
#### Data Preprocessing
Pandas is very extensively used for data analysis. It allows to import data from various file formats and perform various data manipulation operations
##### Read the data with the help of pandas which is in the csv format. It has 398 rows and 9 columns
##### Car name is feature that doesn't have much significance and it might lead to 'curse of dimentionality', it is a phenomina that occurs in high-dimentional space. In this the error ineeases with the increase in number of features.Hence we are dropping the car name
##### Origin is a categorical feature in which American, Europian and Japanese are represented as 1,2 and 3 respectively in the dataset. Replacing them with the actual categorical values
##### Lets check the null value count and the data types of all features in the dataset
![data type ](https://github.com/VAMSEE92/Linear-Regression/blob/main/Images/Data_types.JPG)
##### From non-null count we can see there are no null values. However the horse power column which actually should be a float datatype is having an object datatype. This means there might be some non-digit values present in the horse power column. With the isdigit() method we can find if there is any string made of non digit values. It simply returns 'True' or 'False'. 
![horsepower ](https://github.com/VAMSEE92/Linear-Regression/blob/main/Images/Horsepower_Q.JPG) 
##### Above we can see there are no horsepower values instead those rows were filled with question marks '?'. Replacing missing values with the median.
#### Checking outliers
Outliers are such data points  significantly different from the other observed datapoints. In a mathematical space outliers are located away from the observed datapoints.
![Outliers Example ](https://github.com/VAMSEE92/Linear-Regression/blob/main/Images/Outliers_example.png)
With the outliers our model will be skewed away from the actual underlying relationship and will perform terrible and also can cause a serious problems in statistical analysis.
##### Inter Quantile Range(IQR) is the most extensively used way of finding the outliers in the data. We can visualize the outliers with boxplot, which underlyingly used IQR technique to display outliers
![Box plot ](https://github.com/VAMSEE92/Linear-Regression/blob/main/Images/boxplots.jpg)   
From above boxplot we can see there are outliers present in 'horsepower' and 'acceleration' columns. Let us check the outliers count using statistical methods
![Outliers count ](https://github.com/VAMSEE92/Linear-Regression/blob/main/Images/Outliers_count.JPG)
Outliers are significantly less in number compared to the samples of the data. Hence removing the outliers
#### Checking the Linearity
We need to check the relationship between independent and the dependent features. For linear regression model to perform well it is very important to have a linear relationship between independent and dependent variables. Using pairplot we can plot all the variables against each other
![Pair plot ](https://github.com/VAMSEE92/Linear-Regression/blob/main/Images/Variables_relation.png) 
Above we can see there is an inverse correlation between dependent and the independent variables. We can calculate the strength of the correlation using pearson correlation 
![Correlation statics ](https://github.com/VAMSEE92/Linear-Regression/blob/main/Images/Correlation_statistics.JPG) 
From the above correlaiton coefficients and its respective p-values we can conclude that there is almost 0 probability of having no relationbetween dependent and independent features
#### Feature scaling
Since the range of values in the features is varing widely, It is very much important to bring all features into same scale 
for the better performance of the model. Before scaling the features it is important to understand the distribution of the data. 
Using histograms we can visualize how the data is distributed. 
![Histogram distribution](https://github.com/VAMSEE92/Linear-Regression/blob/main/Images/Histogram_datadist.jpg)
From the above histograms we can see the features not following the Guassian distribution. However we also confirm from statistical methods
![Q-Q plot](https://github.com/VAMSEE92/Linear-Regression/blob/main/Images/QQ_plots.jpg) 
From the above Shapiro Wilk test and the probability plot we can confirm that the dataset is not following the Guassian distribution.
Let us also deal with the categorical feature and split the data into train and test data before scaling the feature.
##### Values like 'america' cannot be read into an equation. Using substitutes like 1 for america, 2 for europe and 3 for asia would end up implying that european cars fall exactly half way between american and asian cars. We don't want to impose such an baseless assumption. Hence creating dummy variables
![Dummy variables](https://github.com/VAMSEE92/Linear-Regression/blob/main/Images/Dummy_variables.JPG) 
Now let us split the 80% as training data and remaining as testing.
##### Min-Max scaler scales down the feature in such a way that every value of the features are with in specific range e.g between zero and one. To be noted this scaler is sensitive to outliers however it responds well when a distribution is not Guassian.
##### Xnew=(X-Xmin)/(Xmax-Xmin)
We will fit the min-max scaler on training data and transform on both training and testing data.
Now our data has no missing values, outliers and features are also on same scale. So, it is completely ready for linear regression
### Building Linear Regression model
There are multiple libraries for building linear models among them statsmodels.regression.linear_model and sklearn.linear_model are very popularly used. Here let us build linear regression model with the help of sklearn.linear_model.
After fitting the data, generally we use a very popular method called R-square to check how much of our data has been explained by our model.
##### R-square=1-(Sum of squares of residuals(RSS)/Total som of squares(TSS))
##### RSS = Σ(y-f(x))^2 where y: observed value, f(x): predicted value
##### TSS= Σ(y-ȳ)^2 where y: observed value, ȳ:mean of observed value. Note: here y is the value at ith element where i ranges from (0 to n), n: total number of observations  
##### R-square ranges from 0 to 1, where '1' means our model is able to explain everything and it is the perfect fit on our data and '0' means our model has explained nothing and is not at all good for our data
![Score](https://github.com/VAMSEE92/Linear-Regression/blob/main/Images/Score.JPG)   
We can also visualize how the model fitted on train and test data using matplotlib
#### Model on training
![Model on training](https://github.com/VAMSEE92/Linear-Regression/blob/main/Images/Model_training.jpg)
#### Model on testing   
![Model on testing](https://github.com/VAMSEE92/Linear-Regression/blob/main/Images/Model_testing.jpg)
#### Evaluation Metrics
There are other evaluation metrics which are commonly used for analysing the performance of the regression.
##### Mean Absolute Error(MAE): We take the difference between each observed value and its respective predicted value and sum up all the differences. Depending on the observed and predicted values we get the differences both in negetive as well as positive values and cancels out each other. To avoid this we take absolute values of every difference, add them all and divide the result with number of observations to get mean absolute error.
##### MAE: (Σ|y-ŷ|)/n where y: observed value, ŷ: predicted value, n: number of observations 
##### Mean Square Error(MSE):  We take the difference between each observed value and its respective predicted value, square the difference, add them and find the mean we get mean squared error
##### MSE: (Σ(y-ŷ)^2)/n where y: observed value, ŷ: predicted value, n: number of observations
##### Root Mean Square Error(RMSE): It is the square root of the MSE
##### RMSE: sqrt((Σ(y-ŷ)^2)/n) 
![MAE,MSE,RMSE](https://github.com/VAMSEE92/Linear-Regression/blob/main/Images/Metrics.JPG)
### Conclusion
Now let us see what feature has high impact on the mileage of a car. This can be interpreted with the coefficients of the model
![Model coefficients](https://github.com/VAMSEE92/Linear-Regression/blob/main/Images/Coefficients.JPG) 
We can see weight co-efficient is -18.86 the negetive sign indicates that with the increase in weight the mileage decreases. Value after sign indicates the change in mileage with the unit change in weight
## We are done with it!
Thanks for reading, stay tuned!
Credits:https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a
