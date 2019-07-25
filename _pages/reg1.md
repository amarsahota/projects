---
layout: home
permalink: /reg1/
title: "Simple Linear Regression"
author_profile: true
toc: true
---

##  <a name="toc"/>
- [Introduction](#Introduction)
- [Assumptions of linear regression](#assumptions)
- [Example of linear regression](#example)


## Introduction <a name="Introduction"/> 
[Return to Top](#toc)

Simple linear regression is useful for finding relationships between two continuous variables. One is predictor or independent variable (X) and the other is the response or dependent variable (Y). It looks for a statistical relationship but not deterministic relationship. The relationship between two variables is said to be deterministic if one variable can be accurately expressed by the other. For example, using temperature in degree Celsius, it is possible to accurately predict Fahrenheit. A statistical relationship is not accurate in determining the relationship between two variables. For example, the relationship between height and weight. Linear regression models provide a simple approach towards supervised learning. They are simple yet effective. The linear equation for univariate linear regression is given below:

![lm](https://latex.codecogs.com/gif.latex?y%3D%5Cbeta_0x&plus;%5Cbeta_1)

![y](https://latex.codecogs.com/gif.latex?y) is the dependent variable, ![x](https://latex.codecogs.com/gif.latex?x) independent variable, ![b0](https://latex.codecogs.com/gif.latex?\beta_0), ![b1](https://latex.codecogs.com/gif.latex?\beta_1) are the intercept and slope of the best fit line, respectively.

## Assumptions of Linear Regression <a name="assumptions"/> 
[Return to Top](#toc)

1. **Linear Relationship between the features and target:** According to this assumption there is linear relationship between the features and target. Linear regression captures only the linear relationship. This can be validated by plotting a scatter plot between the features and the target. The left scatter plot tells us that as x increases y also increases linearly. However, the right scatter plot shows no significant linear relation between x and y.
<img src="{{ site.url }}{{ site.baseurl }}/images/reg/reg1.png"> 
2. **Little or no Multicollinearity between the features:** Multicollinearity is a state of very high inter-correlations or inter-associations among the independent variables. Pair plots and heatmaps(correlation matrix) can be used for identifying highly correlated features. If we have 2 features which are highly correlated we can drop one feature or combine the 2 features to form a new feature,which can further be used for prediction.

3. **Homoscedasticity Assumption:** Homoscedasticity describes a situation in which the error term (that is, the “noise” or random disturbance in the relationship between the features and the target) is the same across all values of the independent variables.A scatter plot of residual values vs predicted values is a goodway to check for homoscedasticity. There should be no clear pattern in the distribution and if there is a specific pattern,the data is heteroscedastic.
<img src="{{ site.url }}{{ site.baseurl }}/images/reg/reg2.png">

4. **Normal distribution of error terms:** Normal distribution of the residuals can be validated by plotting a q-q plot. The left graph shows a normal distribution, whereas, the right graph does not.
<img src="{{ site.url }}{{ site.baseurl }}/images/reg/reg3.png">

5. **Little or No autocorrelation in the residuals:** Autocorrelation occurs when the residual errors are dependent on each other.The presence of correlation in error terms drastically reduces model’s accuracy. This usually occurs in time series models where the next instant is dependent on the previous instant. Autocorrelation can be tested with the help of the Durbin-Watson test.The null hypothesis of the test is that there is no serial correlation. The Durbin-Watson test statistics is defined as:

	![DW](https://latex.codecogs.com/gif.latex?DW_%7Btest%7D%3D%5Cfrac%7B%5Csum_%7Bt%3D2%7D%5E%7BT%7D%28e_t-e_%7Bt-1%7D%29%5E2%7D%7B%5Csum_%7Bt%3D1%7D%5E%7BT%7De_t%5E2%7D)

	The test statistic is approximately equal to ![DW2](https://latex.codecogs.com/gif.latex?2%281-r%29) where r is the sample autocorrelation of the residuals. Thus, for r = 0, indicating no serial correlation, the test statistic equals 2. This statistic will always be between 0 and 4. The closer to 0 the statistic, the more evidence for positive serial correlation. The closer to 4, the more evidence for negative serial correlation.


## Example of linear regression <a name="example"/> 
[Return to Top](#toc)

In this post, I’ll walk you through how to build a linear regression in R and use built-in diagnostic plots for linear regression analysis. The data I will be using shows the height and weight of 50 random women. You can download the data here. The first step is to read the data and assign variables x and y to each column.
```r
data = read.csv(file="C:/Users/User/Desktop/GitHub/sasanmehrabian.github.io/data/women.csv")
x = data $ height
y = data $ weight
```
Use the **lm** function to build the model, and use the "summary" command line to see the coefficients and statistical parameters:
``` r
model = lm(y ~ x, data = data)
summary(model)
# Plot the raw data and the fitted linear line
plot(x, y, xlab = "Height (cm)", ylab = "Weight (kg)", pch = 19, col = "blue", cex = 2, cex.axis = 2, cex.lab = 2)
abline(model, lwd = 3, col = 2)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/reg/women1.png">
<img src="{{ site.url }}{{ site.baseurl }}/images/reg/women2.png">

The statistics in the above figure are:

1) [**Residuals**](#one)

2) [**Coefficients:**](#two)

3) [**Residual standard error (RSE)**, **R-squared (R2)** and the **F-statistic**](#three)

1) **Residuals:<a name="one"/>** Provide a quick view of the distribution of the residuals, which by definition have a mean zero. Therefore, the median should not be far from zero, and the minimum and maximum should be roughly equal in absolute value. One can also visualize these statistics by plotting the distribution of the residuals or ploting the box plot of the residuals. The black curve in the distribution is the normal distribution fit.
```r
# Assign the residuals into a variable
residuals = model $ residuals

# use the fitdistr to fit the data for a normal distribution
library(fitdistrplus)
fit <- fitdistr(residuals, "normal")

# assign the parameters of the fit (mean and standard deviation) to a variable
para <- fit $ estimate

# plot
windows()
par(mfrow = c(2, 1))
hist(residuals, prob = TRUE, cex.axis = 1.5, cex.lab = 1.5, col = 'grey', main = '')
curve(dnorm(x, para[1], para[2]), col = 1, lwd = 3, add = TRUE)
res_outliers = boxplot(residuals,col = "grey", horizontal = TRUE, cex.axis = 1.5, cex.lab = 1.5) $ out
```
<img src="{{ site.url }}{{ site.baseurl }}/images/reg/women3.png">

To view the QQ plot of the residuals, type:
```r
plot(model, 2, cex.axis=2, cex.lab=2,lwd=3,cex.lab=2)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/reg/women6.png">

2) **Coefficients:<a name="two"/>** Shows the regression beta coefficients and their statistical significance. The stars on the very right hand side shows the significance of how much ![x](https://latex.codecogs.com/gif.latex?x) and ![y](https://latex.codecogs.com/gif.latex?y) are related to each other. Mathematically, the beta coefficients (![b0](https://latex.codecogs.com/gif.latex?\beta_0), ![b1](https://latex.codecogs.com/gif.latex?\beta_1)) are determined so that the RSS is as minimal as possible. This method of determining the beta coefficients is technically called least squares regression or ordinary least squares (OLS) regression. Once, the beta coefficients are calculated, a t-test is performed to check whether or not these coefficients are significantly different from zero. A non-zero beta coefficients means that there is a significant relationship between the predictors (![x](https://latex.codecogs.com/gif.latex?x)) and the outcome variable (![y](https://latex.codecogs.com/gif.latex?y)). 

For a given predictor (![x](https://latex.codecogs.com/gif.latex?x)), the t-statistic (and its associated p-value) tests whether or not there is a statistically significant relationship between a given predictor and the outcome variable (![y](https://latex.codecogs.com/gif.latex?y)), that is whether or not the beta coefficient of the predictor is significantly different from zero.

The statistical hypotheses are as follow:
- **Null hypothesis (H0):** the coefficients are equal to zero (i.e., no relationship between x and y)
- **Alternative Hypothesis (Ha):** the coefficients are not equal to zero (i.e., there is some relationship between x and y)

Mathematically, for a given beta coefficient (![bi](https://latex.codecogs.com/gif.latex?\beta_i)), the t-test measures the number of standard deviations that beta is away from 0. Thus a large t-statistic will produce a small p-value. The higher the t-statistic (and the lower the p-value), the more significant the predictor. The symbols to the right visually specifies the level of significance. The line below the table shows the definition of these symbols; one star means 0.01 < p < 0.05. The more the stars beside the variable’s p-value, the more significant the variable. The t-statistic is a very useful guide for whether or not to include a predictor in a model. This will be useful in multivariate linear regression. High t-statistics (which go with low p-values near 0) indicate that a predictor should be retained in a model, while very low t-statistics indicate a predictor could be dropped.

In our example, both the p-values for the intercept and the predictor variable are highly significant, so we can reject the null hypothesis and accept the alternative hypothesis, which means that there is a significant association between the predictor and the outcome variables.

To verify the homoscedasticity assumption, the values of residuals are ploted against the predicted values. Their values are standardized. The distance of the point from 0 specifies how bad the prediction was for that value. If the value is positive, then the prediction is low. If the value is negative, then the prediction is high. 0 value indicates prefect prediction. Detecting residual pattern can improve the model. To see if the homoscedasticity of the residuals, simply type:
```r
plot(model, 1, cex.axis=2, cex.lab=2,lwd=3,cex.lab=2)
```

<img src="{{ site.url }}{{ site.baseurl }}/images/reg/women4.png">

**Standard errors and confidence intervals:** The standard error measures the variability/accuracy of the beta coefficients. It can be used to compute the confidence intervals of the coefficients. For example, the 95% confidence interval for the coefficient ![b0](https://latex.codecogs.com/gif.latex?\beta_0) is defined as ![se1](https://latex.codecogs.com/gif.latex?%5Cbeta_0%20%5Cpm%202%5Ctimes%20SE%28%5Cbeta_0%29), where

- The lower limit of  ![b1](https://latex.codecogs.com/gif.latex?\beta_1) = ![a](https://latex.codecogs.com/gif.latex?32.50287-2%5Ctimes%203.07760%3D26.34767)
- The higher limit of ![b1](https://latex.codecogs.com/gif.latex?\beta_1) =![b](https://latex.codecogs.com/gif.latex?32.50287&plus;2%5Ctimes%203.07760%3D38.65807)

That is, there is approximately a 95% chance that the interval [26.34767, 38.65807] will contain the true value of ![b0](https://latex.codecogs.com/gif.latex?\beta_0). To get these information, simply type:
```r
confint(model)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/reg/women5.png">

3) **Residual standard error (RSE)**, **R-squared (R2)** and the **F-statistic <a name="three"/>** are metrics that are used to check how well the model fits to our data.

- *Residual standard error (RSE):* The RSE (also known as the model sigma) is the residual variation, representing the average variation of the observations points around the fitted regression line. This is the standard deviation of residual errors.

	RSE provides an absolute measure of patterns in the data that can’t be explained by the model. When comparing two models, the model with the smallest RSE is a good indication that this model fits the data the best.

	Dividing the RSE by the average value of the outcome variable will give you the prediction error rate, which should be as small as possible. In our example, RSE = 2.458, meaning that the observed sales values deviate from the true regression line by approximately 2.5 units on average.

- *R-squared and Adjusted R-squared:* The R-squared (R2) ranges from 0 to 1 and represents the proportion of information (i.e. variation) in the data that can be explained by the model. The adjusted R-squared adjusts for the degrees of freedom.

	The R2 measures, how well the model fits the data. For a simple linear regression, R2 is the square of the Pearson correlation coefficient. A high value of R2 is a good indication. However, as the value of R2 tends to increase when more predictors are added in the model, such as in multiple linear regression model, you should mainly consider the adjusted R-squared, which is a penalized R2 for a higher number of predictors. An (adjusted) R2 that is close to 1 indicates that a large proportion of the variability in the outcome has been explained by the regression model. A number near 0 indicates that the regression model did not explain much of the variability in the outcome. The adjusted R2 will be explained more in the next section.

- *F-Statistic:* The F-statistic gives the overall significance of the model. It assess whether at least one predictor variable has a non-zero coefficient.

	In a simple linear regression, this test is not really interesting since it just duplicates the information given by the t-test, available in the coefficient table. In fact, the F test is identical to the square of the t test: 113.5 = (10.65)^2. This is true in any model with 1 degree of freedom. The F-statistic becomes more important once we start using multiple predictors as in multiple linear regressions. A large F-statistic will corresponds to a statistically significant p-value (p < 0.05). In our example the F-statistic equal 113.5, producing a p-value of 3.96e-14, is highly significant.
