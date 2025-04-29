# cse472assignment-1--logistic-regression-and-adaboost-for-classification-solved
**TO GET THIS SOLUTION VISIT:** [CSE472Assignment 1- Logistic Regression and AdaBoost for Classification Solved](https://www.ankitcodinghub.com/product/cse472-machine-learning-sessional-solved-2/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;112870&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSE472Assignment 1- Logistic Regression and AdaBoost for Classification Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
&nbsp;

Introduction

In ensemble learning, we combine decisions from multiple weak learners to solve a classification problem. In this assignment, you will implement a Logistic Regression (LR) classifier and use it within AdaBoost algorithm. For any query about this document, contact Saifur sir.

Programming Language/Platform

• Python 3 [Hard requirement]

Dataset preprocessing

You need to demonstrate the performance and efficiency of your implementation for the following three different datasets.

1. https://www.kaggle.com/blastchar/telco-customer-churn

2. https://archive.ics.uci.edu/ml/datasets/adult

3. https://www.kaggle.com/mlg-ulb/creditcardfraud

They are different in terms of size, number and types of attributes, data quality (missing attribute values), data descriptions (whether train and test data are separate, attribute description format etc.) etc. Your core implementation for both LR and Adaboost model must work for all three datasets without any modification. You can (possibly need to) add a separate dataset-specific preprocessing script/module/function to feed your learning engine a standardized data file in matrix format. On the day of submission, you are likely to be given another new (hopefully smaller) dataset for which you need to create a preprocessor. Any lack of understanding about your own code will severely hinder your chances to make it. Here are some suggestions for you,

1. Design and develop your own code. You can take help from tons of materials available on the web, but do it yourself. This is the only way to ensure that you know every subtle issue needed to be tweaked during customization.

2. Don’t assume anything about your dataset. Keep an open mind. Deal with their subtleties in preprocessing.

3. To get an idea about different data preprocessing tasks and techniques, specifically how to handle missing values and numeric features using information gain [AIMA 3rd ed.18.3.6] visit the following link http://www.cs.ccsu.edu/~markov/ccsu_courses/DataMining-3.html

4. Use Python library functions for common preprocessing tasks such as normalization, binarization, discretization, imputation, encoding categorical features, scaling etc. This will make your life easier and you will thank us for enforcing Python implementation. Visit http://scikit-learn.org/stable/modules/preprocessing.html for more information.

5. Go through the dataset description given in the link carefully. Misunderstanding will lead to incorrect preprocessing.

6. For the third dataset, don’t worry if your implementation takes long time. You can use a smaller subset (randomly selected 20000 negative samples + all positive samples) of that dataset for demonstration purpose. Do not exclude any positive sample, as they are scarce.

7. Split your preprocessed datasets into 80% training and 20% testing data when the dataset is not split already. All of the learning should use only training data. Test data should only be used for performance measurement. You can use Scikit-learn built-in function for train-test split. See https://developers.google.com/machinelearning/crash-course/training-and-test-sets/splitting-data for splitting guidelines.

Logistic Regression Tweaks for weak learning

1. Use information gain to evaluate attribute importance in order to use a subset of features.

2. Control the number of features using an external parameter.

3. Early terminate Gradient Descent if error in the training set becomes &lt; 0.5.

Parameterize your function to take the threshold as an input. [If you set it to 0, then Gradient Descent will run its own natural course, without early stopping]

4. Use tanh function (instead of sigmoid). You need to calculate the gradient and derive the update rules accordingly.

Adaboost implementation

1. Use the following pseudo-code for Adaboost implementation

2. As the weak/base learner use logistic regression. You can explore different ways to speed up the learning of the base models, sacrificing the accuracy, so long as the learning perform better than random guess (i.e. weak learner). For example, you can use a small subset of features or reduce the number of iterations in gradient descent etc. You can come up with your novel idea too.

3. Adaboost should treat the base learner as a black box (in this case a decision stump) and communicate with it via a generic interface that inputs resampled data and outputs a classifier.

5. In each round, resample from training data and fit current hypothesis (linear classifier) using resampled data but calculate the error over original (weighted) training data.

6. Use +1 for positive decision and -1 for negative decision, so that the sign of your combined majority hypothesis indicates decision.

7. After learning the ensemble classifier evaluate performance over test data. Don’t get confused over which dataset to use at which step.

Performance evaluation

1. Always use a constant seed for any random number generation so that each run produces same output.

2. Report the following performance measure of your logistic regression implementation on both training and testing data for each of the three datasets. Use the following table format for each dataset.

Performance measure Training Test

Accuracy

True positive rate (sensitivity, recall, hit rate)

True negative rate (specificity)

Positive predictive value (precision)

False discovery rate

F1 score

3. Report the accuracy of Adaboost implementation with logistic regression (K=5, 10, 15 and 20 rounds) on both training and testing data for each of the three datasets.

Number of boosting rounds Training Test

5

10

15

20

Submission

2. You need to submit a report file in pdf format containing the following items (No hardcopy is required.):

a. Clear instructions on how to run your script to train your model(s) and test them. (For example, which part needs to be comment out when training each dataset, how to run evaluation etc.) We would like to run the script in our computers before the session class.

b. The tables shown in the performance evaluation section with your experimental results.

c. Any observations.

3. Write code in a single *.py file, then rename it with your student id. For example, if your student id is 1605123, then your code file name should be “1605123.py” and the report name should be “1605123.pdf”.

4. Finally make a main folder, put the code and report in it, and rename the main folder as your student id. Then zip it and upload it.

Evaluation

1. You have to reproduce your experiments during in-lab evaluation. Keep everything ready to minimize delay.

2. You will likely be given online tasks during evaluation, which will require you to modify your code.

3. You will be tested on your understanding through viva-voce.

5. You are encouraged to bring your computer in the sessional to avoid any hassle. But in that case, ensure an internet connection as you have to instantly download your code from the Moodle and show it.

1. Don’t copy! We regularly use copy checkers.

2. First time copier and copyee will receive negative marking because of dishonesty. Their default is bigger than those who will not submit.

3. Repeated occurrence will lead severe departmental action and jeopardize your academic career. We expect fairness and honesty from you. Don’t disappoint us!
