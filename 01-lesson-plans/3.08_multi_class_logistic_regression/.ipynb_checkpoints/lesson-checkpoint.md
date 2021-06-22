# Lesson 3.8: Multi-class classification & Classification models

### Lesson Duration: 3 hours

> Purpose: The purpose of this lesson is to introduce multi-class classification problems and revisit the data analysis workflow with classification modeling using logistic regression.

---

### Setup

All previous set up

### Learning Objectives

After this lesson, students will be able to:
- Recognize classification problems
- Differentiate binary classification and multi-class classification problems
- Understand and being able to explain the idea behind the logistic regression model 
- Use logistic regression model for binary and multiclass  classification problems
- Check the accuracy of the model
---
### Lesson 1 key concepts

> :clock10: 20 min

- Introduce the classification problem.
  Important point: Difference between classification and regression problem.<br>
  Check this [example](classification-regression.png)
- Examples of classification problems. Think about the background of the students to add relevant examples
- Single class vs multiple class classification

---
#### :pencil2: Check for Understanding - Class activity/quick quiz

> :clock10: 10 min 
 
Suggested activities, choose one 
- Start a discussion where you ask a student the difference between a  classification and regression problem. Ask another student if he/she agrees and to explain why in both cases 
- Come up with a list of problems and ask the students if they would solve it with a regression of classification approach. Try to involve the students who have not participated in the previous discussion
- Ask one student to come up with a list of problems and ask different students which approach (regression/classification) would use to tackle the problem and why 
  
---

:coffee: **BREAK** 5 minutes

---

### Lesson 2 key concepts

> :clock10: 30 min

Introduce the logistic regression model for classification problems

  Some possible real world examples
  - Classify transactions as fraudolent/legitimate 
  - Classify bank client as likely to default or not
  - Sentiment analysis as multiclass classification problem
  - Email spam or ham
  - Object detection
  - Disease prediction 

Some important points 
  - The name of this algorithm has the word regression, which in this case is a false friend as it is used to solve classification problems
  - Outputs probability along with classification results and that provides with useful insights
  - It is a benchmark model vs more complex algorithms (Neural Networks)
  - It is a building block to understand Neural Networks
  - Widely used in business 
  
Useful [Reference](https://realpython.com/logistic-regression-python/)


----
Break pencil2: Check for Understanding - Class activity/quick quiz

> :clock10: 10 min 

- Ask a student to plot and explain the logistic function and the main difference with linear regression
- Ask different students if they agree with the explanation and why, what they would change.

-----------
:coffee: **BREAK** 5 minutes

-----------
### Lesson 3 key concepts

> :clock10: 20 min
 
Show how to use the logistic regression in practice. Show how to use this algorithm in Python with the Titanic dataset, use it to predict the survivors. Introduce also the following concepts
- Fitting the model
- Making predictions on the test data
- Check model accuracy

Consider using the attached notebook `Titanic_practical_survivors.ipynb` as a stepping stone
 
:exclamation: Note to instructor: When we work with multi class classification problem and use Logistic Regression method from `sklearn`, the argument "multi_class" can take different arguments. Discuss the one versus rest method and multinomial mehtod briefly, how they are different.


#### :pencil2: Check for Understanding - Class activity/quick quiz

> :clock10: 15 min (+ 10 min Review)

 Activity_3: Predict class 
 
- Activity3_titanic_predict_class.ipynb
- Link to [Activity3_titanic_predict_class_solutions.ipynb]().

---
:coffee: **BREAK** 5 minutes

---
### Lesson 4 key concepts

> :clock10: 20 min

The goal of this lesson is to take a step back and show the students how to complement and support the logistic regression analysis with exploration and plotting of important features of the dataset.

- More EDA/data cleaning
- Data plotting 
- Data pre processing

Consider using the attached notebook `Titanic_practical_plotting.ipynb` as a stepping stone


### :pencil2: Practice on key concepts - Lab_1

> :clock10: 30 minutes 

<details>
  <summary> Click for Instructions: Lab </summary>

- Link to the lab: [https://github.com/ironhack-labs/lab-predictions-logistic-regression](https://github.com/ironhack-labs/lab-predictions-logistic-regression)

</details>

<details>
  <summary> Click for Solution: Lab solutions </summary>

- Link to the [lab solution](https://gist.github.com/ironhack-edu/c3e7fba417de11bcf152ba6329acbbb4).

</details>

### :pencil2: Practice on key concepts - Bonus lab
Bonus_lab_titanic_plotting.ipynb

---
