# Object Oriented Machine Learning in Python
an amalgamation of all my machine learning training and OO Python to demonstrate functionality


# Data Preprocessing:
## Importing dataset:
* choose all cols but outputs for X, rest is y

## Missing Data: 
* _sklearn.impute.SimpleImputer_
* avoid string cols
* missing_values = _np.nan_
* strategy = _'mean'_


## String Data: 
* 1 hot encode so no relation between strs
* do not scale these features afterwards
### Column Transformer: for X
_sklearn.compose.ColumnTransformer_
* transform type = _'encoder'_
* encoding class = _OneHotEncoder()_
* col index = _[i]_
* remainder = _'passthrough'_
* transformers = [(transform type, encoding class, col index)]
### Label encoder: for y
_sklearn.preprocessing.LabelEncoder_
* no/yes -> 0/1

## Split into Train and Test sets
_sklearn.model_selection.train_test_split_
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
_sklearn.preprocessing.StandardScaler_
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

_sklearn.preprocessing.PolynomialFeatures_

transforms vector into degree n matrix to get poly regression, use only on single feature sets
## Support Vector (SVR)
Pros | Cons
------ | ------
Easily adaptable, works very well on non linear problems, not biased by outliers | Compulsory to apply feature scaling, not well known, more difficult to understand

    Trend-line has epsilon-insensitive tube, ie a value of E where error less than this is negated, and otherwise the errors 
    are taken from the edge of the tube around the trend-line. ( sum of minimum squares on this tube to get best trend-line.
    The points outside the tube are the SVs, defining the shape of the E-I tube. The ones above are Ei*, and the ones below are Ei.

kernel = 'rbf', # __Radial Basis Function Kernel__ 
* (function measuring euclidian distances radially around some center, as a basis)
* used for SVR typically, (does not catch outliers well, but is better for polynomial trends than linear kernel)
* points are considered proportional to the inverse exponential euclidean distance from the center, scaled by the inverse exponential variance
    * thus we can get a circumference as a cutoff hyperplane. (variance can be changed here)
* we can also add different RBF equations for more complex mapping

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
<<<<<<< HEAD


# Classification

## Logistic Regression
classifying into groups, ie yes and no, or cat vs dog, as a probability

Intuition: 

    we can thus apply a sigmoid function to the output, 
    and the n invert to get the input sigmoid (probability)
    then we project y^ to the closest classification (if y<0.5 y ->0)

classification boundary is linear due to logistic regression being linear

* __C__: Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
         regularization, counters over-fitting
     * can be as high as 1e5 if not worried about overfitting


## K nearest Neighbors
* _nonlinear_
* metric = _'minkowski'_ for distance metric to use for the tree. 
    * with _p=2_ is equivalent to the standard Euclidean metric
* n_neighbors = _5_ neighbors by default

Intuition: 
    
    Choose number of K neighbors (ie 5),
    take the K nearest neighbors of the new data point, by Euclidean distance.
    Count the classifications of these neighbors,
    assign data point to the closest group
    
        
## Support Vector Machine (SVM)
* _assume kernel is separable_
 Intuition: 
 
    finding best decision boundary,
        which has a Maximum margin between the closest points to the boundary
            (like the tube around the trendline), 
        we want the max distance between these closest points
    
__Maximum Margin Hyperplane__:
    maximum margin classifier line in nD, splits positive and negative hyperplane
    
## Kernel SVM
how to choose the right classification method
* sometimes linearly inseparable sets in nD are linearly separable in higher dimensions
    * (as lines, planes, or hyperplanes)
    * ie x->(x-5)^2
    
Types of Kernel Functions: __linear, rbf, poly, sigmoid__

Non Linear SVM: using RBF kernel
* shifts the nD plane to higher dimensionality ((n+1)D)
* then we can cast a hyperplane onto the shape,
    * with a top and bottom hyperplane at distance epsilon away in which we negate error.

## Naive Bayes
Intuition: 
    
    Posterior Probability: probability of y given X, P(y|X) = P(X|y) * P(y) / P(X)
    P(y) = prior probability = number in classification y / total classified
    P(X) = marginal likelihood = # similar Observations / Total observations, is the same for y and !y, can be factored out
    P(X|y) = Likelihood = # similar observations among classification y / Total classified as y
Naive since it has independence assumptions of features, otherwise it will over-correlate to much
* I.e. feature1 = 1 - all other features

* __priors__: _None_ so that there is no probability of one class over another predefined
* __var_smoothing__: Portion of the largest variance of all features that is added to 
                variances for calculation stability
## Decision Tree 
Intuition: 

    CART = Classification And Regression Trees
    The rest is the same as Regression Decision Tree
* __Criterion__:  The function to measure the quality of a split. Supported criteria are
    * _"gini"_ for the Gini impurity
    * _"entropy"_ for the information gain.

## Random Forest
Intuition: 

    The basic intuition is the same as Regression Random Forest

## Checking Correctness
__False Positives and Negatives__: easy with sigmoid around center

__Confusion Matrix__: determining the occurance of falses: 

    [[#neg , #falsePos],[#falseneg, #Pos]]

__Accuracy rate__: correct / total, __Error rate__ = 1 - accuracy rate

__Accuracy Paradox__: if y occurs much more than !y, then we can get a better model by always choosing y,
                  which defeats the purpose of  having a model

__CAP (Cumulative Accuracy Profile) Curve__: using specific demographics as the initial set of features to boost
performance, there is a peak operating point where the slope of the model is maximized (total performance/sample size)
Peak performance would be to include only those who choose y and nothing otherwise.

__CAP Analysis__: Random Model Line (__R__), Perfect Model Line (__P__), and Good model line (__G__) in between.

__Accuracy ratio__ = check ratio of R/P at 50% sample size,
* if output is less than 60%, then bad model, after which it improves diminishingly.
* If near 90-100%, then likely over-fitting

__ROC (Receiver Operating Characteristic)__ is not the same as CAP


# Clustering

## K-means Clustering
Intuition: how to identify clusters of features

    Choose the number of clusters, K
    Select any K random points (not necessarily from dataset) as the centroids
    Assign each data point to the closest centroid (forms K clusters)
    while(reassignment of cluster data):
        Compute and place the new centroid of each cluster
        Reassign data points if possible

* __Caveat__: Do we use Euclidean Distance or something else
* __Random Initialization Trap__: if we choose centroids poorly initially, the final clusters can be objectively incorrect
    * Solution: K-means++
* __wcss__: Within Cluster Sum of Squares, converges to 0 as numClusters -> numPoints, so we choose the optimal numClusters at the pivot point of the exponential relationship (elbow)

## Hierarchical Clustering
Intuition: 
    
    assign each point as a cluster
    while numClusters > 1:
        combine closest clusters (Euclidean distance or otherwise)
         (between centroids, closest or farthest points, avg dist)
         
__Dendrogram__: stores grouping memory, plotting the points as the values on the x-axis,
           connecting various points at the computed dissimilarity between them (the height of the bar)
* A threshold dissimilarity can be set to limit the minimum numClusters,
           best split is at largest single height in chart (greatest dissimilarity)

* __ward__: min variance method


# Associative Rule Learning
people who bought x also bought y

## Apriori
* I sort these by lifts

Intuition: 
    
    support(x) = #transactions containing x / #transactions
    confidence(x -> x_2) = #transactions containing x and x_2 / #transactions containing x
    lift(x->x_2) = confidence(x->x_2) / support(x), the improvement of the prediction

Steps: 
1. set a minimum support and confidence
1. take all subsets of transactions that have valid support
    1. and all the rules of these subsets with valid confidence 
1. sort by decreasing lift

## Eclat
simple apriori, determining sets of relations
* only considering supports, sorted decreasingly (no confidences or lifts)
* I sort these by supports


# Reinforcement Learning

## Upper Confidence Bound (UCB)
Intuition: 

    we have d devices to monitor, at each round n, where i gives reward r_i(n)E[0,1],
    with r_i(n)=1 if user took action, and r_i(n) = 0 if not. We want to maximize
     reward over many rounds

* Deterministic, Requires update at every round
* we have N rounds of data, but we want to
 see how few rounds we need to validly identify the best device

__N_i(n)__ : # times i was selected up to round n
__R_i(n)__ : sum of rewards of i up to round n
__r*_i(n)__ : avg rewards of i up to round n 
    
    = R_i(n) / N_i(n)
__delta_i(n)__ 

    = sqrt(3 log(n) / (2 N_i(n))
confidence interval
 
    [r*_i(n) - delta_i(n), r*_i(n) + delta_i(n)] (aka [LCB, UCB]), 
    we select i with max UCB

## Thompson Sampling
Same Intuition as UCB

We are creating distribution of where the expected values lie 
* probabilistic, not deterministic like UCB
* Can accommodate delayed feedback

__Bayesian Inference__: derives the posterior probability as a consequence of two antecedents:
  * a prior probability
  * a "likelihood function" derived from a statistical model for the observed data.
  
  Bayesian inference computes the posterior probability according to Bayes' theorem
   Ad i get reward y from Bernoulli distribution
    
        p(y|theta_i) ~ B(theta_i)
   theta_i is unknown, but we set its uncertainty assuming uniform distribution
    
        p(y|theta_i) ~ U([0,1])
   Bayes Rule: approach theta_i by the posterior distribution: p(theta_i|y),
   
        p(theta_i|y) = ( p(y|theta_i) * p(theta_i) ) / ( integral(p(y|theta_i) * p(theta_i)) d theta_i )
        p(theta_i|y) ~= ( p(y|theta_i) * p(theta_i) ), aka (likelihood function * prior distribution)
        p(theta_i|y) ~ B(numSuccess + 1, numFail + 1),
        at each round we take random theta_i(n) from p(theta_i|y), for each ad i
        At each round n, select i with highest theta_i(n).


# Natural Language Processing
__Bag of Words Model__:  tokenization
* preprocess text before classification
 * involving vocab of know words
 * measure of the presence of known words
 
 __stopwords__: words that dont add meaning, like the, I, etc.
 
 __quoting__: =3 to ignore quotes
 
__PorterStemmer__: only consider the stem of the word, replace punctuation with spaces

__max_features__:  how many words to include, thus letting us remove unnecessary words like names


# Deep Learning

## Artificial Neural Networks (ANN)
__activation functions__: sigmoid (used in output layer), rectifer(used in hidden layers), tanh, threshold.

__cost function__: measuring error between y (real) and y^ (predicted) = 

    .5(y^-y)^2
__batch gradient descent__ |__stochastic gradient descent__
---------------|------------------ 
deterministic | random
looks at weights after all data runs | updates the weights dynamically
| | stochastic gradient descent avoids local minimums and finds global minimums, is faster
__back-propagation__: adjusting all weights compared to the error matrix simultaneously for max speed

Example code with comments:
````python
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(6, activation='relu', input_dim=11, kernel_initializer='uniform'))  # rectifier activation func, input dim seems unnecessary

# Adding the second hidden layer
classifier.add(Dense(6, activation='relu', kernel_initializer='uniform'))  # uniform weight dist around 0

# Adding the output layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))  # sigmoid for output probability
# classifier.add(Dense(n, activation='softmax', kernel_initializer='uniform'))  # softmax for nD dependant output probability

# Compiling the ANN    
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# adam is a good stochastic gradient descent function, avoiding local min
# loss function is log loss func to account for sigmoid loss
# metrics uses accuracy criterion to improve model performance

# Part 3 - Training the ANN
# Training the ANN on the Training set
classifier.fit(X_train, y_train, batch_size=32, # batch_size operations before updating weights
                                    epochs=100) # epoch num of these
````   
## Convolutional Neural Networks (CNN)
__Intuition__: 
    
    Convolution -> Max Pooling -> Flattening -> Full Connection
__Convolution__: signal processing func we learned in Math 256, integral of two
              functions to get laplace easier. (f*g)(t)
              We use a standard stride of 2 pixels to analyze image
              Feature detector (3x3) maps input image(nxn) to a feature
              map ((n-3+1)^2) and reduces noise, feature map lists
              number of matching points on image section compared
              to feature detector. Many maps for detecting all
              features -- goes into convolution layer
              
__ReLU Layer__: rectifier linear operation allows the isolation of all 1s
              from zeros, hence focusing only on the nonlinear, positive
              portions of the image, reducing noise.
              
__Max Pooling__: AKA _Down Sampling_,
              How to understand features facing different directions, or
              existing in different parts of image (tilts, offsets, etc).
              we use a stride of two to get the max values from the
              feature map into a Pooled Feature Map, accounting for any
              distortions since we pull only the max features.
              (ok to go over image size limits)
              This reduces input info and over-fitting because of it
         sub-sampling: mean pooling instead of max pooling
         
__Flattening__: Converting pooled feature map into a flat column, acts as input
              layer of ANN
              
__Full Connection__: layers of the ANN which are fully connected (O(n^2) connections)
              multiple output neurons possible, thus we have NN of features
              matched to output at output layer (due to output weights),
              redundant and irrelevant auto-removed by NN backtracking of errors
              
__SoftMax__: normalized exponential function: kD vector -> E[0,1] for each output type,
          all probs add to 1 overall,
           
     f_j(z) = e^(z_j) / (sum_k(e^z_k))
            
__Cross-Entropy__: similar to reducing error (like mean-squared) but is better with
              small initial gradient descent errors due to log term
              
* __Loss Func__: 

        L_i = -log(e^(f_yi) / (sum_j(e^f_j)))) (minimize these)
* __Cost Func__: 
        
        H(p,q) = - Sum_x( (p(x))*log(q(x)) ) (p=actual val, q=predicted val)
        
Example Code with comments:
````python
# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32,  # the dimensionality of the output space
                               kernel_size=3,  # height and width of the 2D convolution window.
                               activation='relu',
                               padding='same',  # for convolution layer
                               input_shape=[64, 64, 3]))  # (batch_size, channels, rows, cols) 4d tensor
# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),  # max value over a 2x2 pooling window
                                  strides=2,  # 2 pixel stride
                                  padding='valid'))  # for pooling
# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu', kernel_initializer='uniform'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x=training_set, validation_data=test_set, epochs=25)
````

## Recurrent Neural Networks (RNN)


# Dimensionality Reduction

## Principal Component Analysis (PCA)
 dimensionality reduction algorithm by detecting correlation between vars.
 
    (sort the eigenvalue of covariance matrix in descending order, choose top k)
    create projection matrix W from k eigenvectors, transfer X via W to get
    k-dimensional feature subspace Y
* highly affected by data outliers
* unsupervised algorithm
* if 2 final components, then graphable
* do not fit X-test in order to avoid info leakage

https://setosa.io/ev/principal-component-analysis/

## Linear Discriminant Analysis (LDA)
 dimensionality reduction algorithm by detecting correlation between vars.
 
LDA differs from PCA by trying to maximize the separation between multiple classes (adjusting component axis)
* LDA is supervise due to relation to dependant variable
    * (uses scatter matrices -- in-between-class and within-class)
    * sorts eigen-vectors and similar steps to PCA

## Kernel PCA
Similar to PCA, but uses the kernels defined previously to assess correlations between features


# Model Selection: Boosting

## K-fold Cross Validation
creates multiple test folds to avoid any outlying data weighing results
* __cv__: if an integer, then specify the number of folds in a `(Stratified)KFold`,
    * `CV splitter`, - An iterable yielding (train, test) splits as arrays of indices

## Grid Search
test and find the best of many params all at once
* different C values, kernels, gammas
* own regularization parameters can be entered as sets inside array with arrays of desired values to test as seen here

       '[{'C': [0.25, 0.5, 0.75, 1],
        'kernel': ['rbf'],
        'gamma': [.1, .2, .3, .4, .5, .6, .7, .8, .9]}, ...]
## XGBoost
an estimator object (regression or classification) which can run many tests in parallel and return optimal model
* may be difficult to import library, thus the code is currently commented out
* otherwise acts like any other regressor or classifier from a high level perspective
