# Data Preprocessing:
## Importing dataset:
* choose all cols but outputs for X, rest is y

## Missing Data: _sklearn.impute.SimpleImputer_
* avoid string cols
* missing_values = _np.nan_
* strategy = _'mean'_


## String Data: 
* 1 hot encode so no relation between strs
* do not scale these features afterwards
### Column Transformer: for X
* transform type = _'encoder'_
* encoding class = _OneHotEncoder()_
* col index = _[i]_
* remainder = _'passthrough'_
* transformers = [(transform type, encoding class, col index)]
### Label encoder: for y
* no/yes -> 0/1

## Split into Train and Test sets
* test_size = E[0, 1], usually ~0.2
* random_state = constant -> same split each time

## Feature Scaling
* after splitting, else test set gets biased, info leaks
* fit: get mean and SD
* fit_transform: standardization formula
* transform: apply the mean and SD
* not on dummy/1hot cols, else may remove 1hot
* X_test: Don't fit-> fresh data-> use training set scalar on test set

## Standardization
* (x - mean(x) )/ SD(x) 
* E [-3,3] since percentiles around mean
* works all the time, best choice

## Normalization
* (x - min(x) )/ (max(x) - min(x))
* E [0,1], good for normal distributions



# Regression
## Simple Linear
Pros | Cons
------ | ------
Works on any size of dataset, gives information about relevance of features |  Linear Regression Assumptions

Assumptions of Linear Regression: 
* Linearity
* Homoscedasticity (same variance for all values)
* Multivariate Normality (normally distributed across all independent vars)
* Independence of errors
* Lack of Multi-Collinearity (no relation between "independent" vars)
    *     when we have a set of string inputs, like states, then we can
          assign them Dummy Vars (one hot columns), but then D1 = 1 - D2 - D3..., so if we
          have n different entries in the column, only use n-1 dummy vars, 
          since the last term will be part of the bias term anyways)
          
P-value:
*   probability of getting a sample like ours, or more extreme than ours,
    IF the null hypothesis is true
    *       Assume the null hypothesis == true, determine how “strange” sample 
            really is. 
            If not that strange (a large p-value), then assumption ok.
            As the p-value gets smaller, we may reject the null hypothesis.

_LinearRegression_ auto avoids Dummy Var Trap and auto selects best P-value
    
    if we wanted to manually remove dummy var, doing the whole model in pure python, we would consider X = X[:,1:]
    
    

## Multiple Linear
5 methods to build a model (middle 3 are step-wise regression):

All-in | Backward Elimination | Forward Selection | Bidirectional Elimination | Brute Force
------ | -------------------- | ----------------- | ------------------------- | -----------
throw in all verifiable valid vars |select significance level (ie SL = 0.05) |select significance level (ie SL = 0.05)|Choose SL-Enter and SL-Stay|select criterion of goodness of fit
  | requires prior knowledge |fit the full model with all possible predictors | fit all basic models with all predictors (y~xn), choose the one with minimum P |  | construct 2^(numVars)-1 possible regression models, choose the best one
  | | while (vars in model): | while (vars NOT in model): | while (vars can be added to model): |
  | | consider prediction with highest P | keep this var and find next smallest P when vars added to current model | ForwardSelect(SL=SL-Enter) BackwardSelect(SL=SL-Stay)  
  | | if (P > SL): Remove the predictor, Fit model without this variable | if (P < SL): Add the predictor, Fit model with this variable
  | | else: model is ready, break | else: model is ready, break.
  
  
## Polynomial Linear
Pros | Cons
------ | ------
Works on any size of dataset, works very well on non linear problems | Need to choose the right polynomial degree for a good bias/variance trade-off

linear refers to the linear independence of the variable matrix

transforms vector into degree n matrix to get poly regression, use only on single feature sets
## Support Vector (SVR)
Pros | Cons
------ | ------
Easily adaptable, works very well on non linear problems, not biased by outliers | Compulsory to apply feature scaling, not well known, more difficult to understand

    Trend-line has epsilon-insensitive tube, ie a value of E where error less than this is negated, and otherwise the errors 
    are taken from the edge of the tube around the trend-line. ( sum of minimum squares on this tube to get best trend-line.
    The points outside the tube are the SVs, defining the shape of the E-I tube. The ones above are Ei*, and the ones below are Ei.

kernel = 'rbf', # radial basis function kernel (function measuring euclidian distances radially around some center, as a basis)
    used for SVR typically, (does not catch outliers well, but is better for polynomial trends than linear kernel)

Non Linear SVM = using RBF kernel to shift the nD plane to higher dimensionality ((n+1)D), we can cast a hyperplane
onto the shape, with a top and bottom hyperplane at distance epsilon away in which we negate error.

## Decision Tree
Pros | Cons
------ | ------
Interpretability, no need for feature scaling, works on both linear / nonlinear problems | Poor results on small dataset, over-fitting can easily occur

no feature scaling since it results on splits of the data, not an equation with a scale of data (thus no mean nor SD)

    We make n-D trees splitting leaves around some percent above and below a cutoff percentile.
    Different check for each point in decision tree.
    We assign each segment an avg'd value and assign any new points that meet the cutoff as such

## Random Forest
Pros | Cons
------ | ------
Powerful and accurate, good performance on many problems, including non linear | No interpretability, over-fitting can easily occur, need to choose the number of trees

When we want the output of a random input, assign it the avg output across all Decision Trees,
thus getting a very stable result not prone to extraneous outliers (higher likelihood of correctness)

    for i in range (number of N-Trees, ie 500):
      Pick at random K data points from Training Set
      DecisionTree(K)
      
sum of squares of residuals __SSres__:
    Total squared error off the trend-line,
        
        we want SSres=0
    
sum of squares of total __SStot__: 
    Total squared error off the avg
    
__R^2__: value estimating total error, E[0,1] if good trendline, ideally 1.
    
    R^2 = 1 - (SSres/SStot)

Adjusted R-Squared: adding new vars to model will never decrease R^2, potentially minimizing SSres,
    since we are adding a var with at least a slight random correlation to the model (non zero factor)

    n = sample size, p = numRegressors
    Adj R^2 = 1 - (1 - R^2)(n-1)/(n-p-1) thus inhibiting adding too many regressors
    
#### To adjust for over-correction: 
use hyper-tuning parameters to add factors to minimized errors
* Ridge Regression: adding the factor lambda*(SSmodel_coefficients)
* Lasso: adding the factor lambda*(Sum(abs(model_coefficients))
* Elastic Net: use both Ridge and Lasso with different lambdas, reducing over-fitting