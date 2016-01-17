### 1. Classification vs Regression

Classifying students into graduate / non graduate classes is a classification problem. we are trying to classify students into two classes.

### 2. Exploring the Data

- Total number of students: 395
- Number of students who passed: 265
- Number of students who failed: 130
- Number of features: 30
- Graduation rate of the class: 67.09% (this is slightly unbalanced, although not terrible)

### 3. Preparing the Data

See ipython notebook.

### 4. Training and Evaluating Models

##### Regularized Logistic Regression
The default implementation in sklearn of logistic regression (LR) is regularlized version. Regularlized means that there is a penalty term added to the cost function, penalizing coefficients for being too large. This helps to make the fit model more generalizeable and also helps to deal with correlated features.

Logistic regression is a generalized linear model. These models have a theoretical complexity of O(n) and a constant space requirement for training and a constant time complexity for prediction, making them attractive learners due to their speed. These models are convenient, and particularly important, easily interpretable coefficients. This means, once can look at a model and make an interpretation about GPA or abscences. One of the disadvantages of logistic regression is that it assumes there is a linear decision boundary.  As we can see in the run times below, logistics regression scales well (to the extent going from 99 to 296 observations is "scaling"), with very little increasing in fitting time. The model that gets trained does not get any larger as there are more training cases because we only store the functional form with fit coefficients. In order to predict, we just calculate the odds by plugging in the prediciton values.

Overall, logistic regression looks promising because it is very fast and seems to predict reasonably well. We should compare it to other methods that are sometimes more accurate, although more complicated to fit.

Training Time | Prediciton Time | F1 Training | F1 Test | Training Size
--------------|-----------------|-------------|---------|--------------
0.001|0.000|0.82|0.72|99
0.002|0.000|0.83|0.78|199
0.003|0.000|0.85|0.78|296


##### SVM
There are two main types of SVMs for classification: linear and non-linear. Linear SVMs (implemented in sklearn with LinearSVC()) have theoretical complexity of O(n). Non-linear SVMs use a kernal function that have - depending on the source - between O(n^2) and O(n^3) theoretical complexity. SVMs are thought to be the "best" classifier out of the box. 

SVMs are similar to logistic regression in that they try to draw either a line or hyperplane to separate the data into the classes. SVMs try to maximize the margin between the classes by putting extra weight on the observations closes to the plane, while in logistic regression, there is equal weight on all points. Additionally, SVM can fit a non-linear decision plane using kernals. Logistic Regression always uses a linear decision boundary. Unfortunately for SVM, the extra parameters - particularly calibrating the kernal - adds computational complexity to the model.

Looking at the table below, we can see that the SVM takes considerably more time than Logistic Regression to learn the data (although to be fair, in absolute terms we're still talking sub second times). This indicates that an SVM may not scale well if the data set grows signficantly. However, just like logistic regression there is constant complexity in predicting the outcome. The F1 scores are comparable to Logistic Regression, but ultimately slight lower on the full training set.


Training Time | Prediciton Time | F1 Training | F1 Test | Training Size
--------------|-----------------|-------------|---------|--------------
0.008|0.000|0.82|0.73|99
0.017|0.000|0.84|0.79|199
0.023|0.000|0.83|0.77|296

##### Random Forest Classifier
Random forest classifiers (RF) are popular because of their ease of use, relative accuracy with minimal tuning, and ability to consider interactions between variables. The algorith works by estimating on decision tree using a randomly selected variable and subset of the data. After fitting many iterations of these individual classifiers, it attempts to average the predictions across features to get a more stable estimate. This technique is known as ensemble learning, and has the advantage of generating more stable and generalizeable results. Another interseting result of RFs, are that you can look through all of the trees in the forest, and find the features that were most correlated with explaining the variance of the outcome variable. Similarly to LR, we can examine the output and see which variables are important (although we can't necessarily see the "effect" on the outcome like in LR).

RFs tend to run in O(p * n log n) time, where p is the number of features. This is fairly slow compares to LR and SVM. These speeds are also reflected in the table below, as we scale from 100 to 300 training set observations. The algorithm is significantly slower than both LR and SVM for both learning and predicting values. However, given the limited amount of scaling below, the training time does not seem to increase.


-Theoretical space complexity
-General applications, strengths and weaknesses
-Why choose this model?

Training Time | Prediciton Time | F1 Training | F1 Test | Training Size
--------------|-----------------|-------------|---------|--------------
0.173|0.006|1.0|0.78|99
0.159|0.005|1.0|0.78|199
0.165|0.005|1.0|0.78|296


### 5. Choosing the Best Model
For this application, I'd suggest using the Logistic Regression for several reasons. First of all, it has similar accuracy to the other two more complex models - SVM and Random Forest. Secondly, the predictions are made in constant time. Third, the outcomes are interpretable. For example, we can see that having an other job, having failures, and being enrolled in the nursery increase the risk of student dropping out. On the other hand A,B, and C decrease the odds of dropping out. This is useful information to administrators, because now we can store students and also see factors that contribute to their risk scores. With this information, we can help tailor an intervention system for each student. We could not tailor this intervention without knowledge of how each risk factor affects the student. Furthermore, LR is extremely fast compared to the other methods and likely to scale well into the feature. In addition, the accuracy is comparable to other more complicated methods like SVM and Random Forests. However, we before recommending an action based on the model, we should check how severe multicollinearity might be affecting the coefficients. One way to check on the stability would be to fit several models, adding variables at each step. If coefficients start flipping, we should be careful in how we interpret the model. Alternatively, in or our preprocessing step, we could try to proactively remove one variable from pairs of highly correlated (>.95) variables.

How does LR work? LR regression will give us probabilities for students based on the factors we know in the system. These probabilities can be used to sort the students by risk and draw a decision boundary for whether we should intervene or not, for example - above 50% probability. We know who has graduated and who has dropped out, historically. We also observe information about those students. LR tries to search through the variables and find a model that most accurately fits the data historically, to predict who will graduate based on the available data. 

Finally, after fine tuning the model (on C - the regularization penalty, whether to fit a ridge or lasso penalty, and whether to fit an intercept or not), we can expect an F-1 score (similar to accuracy) of 0.79 using a model fit with C=1, penalty='l1', and fit_intercept=True.