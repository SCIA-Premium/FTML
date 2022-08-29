# FTML [![Profile][title-img]][profile]

[title-img]:https://img.shields.io/badge/-SCIA--PRIME-red
[profile]:https://github.com/bictole

## Objective

The goal of this project is work on the **theoretical fundamentals** of **machine learning** with different exercices.

We had to work on the **Bayes predictor** and the **Bayes risk** associated to some particular settings. Another part is dedicated to the **OLS estimator**. 


## Regression


In this part, we had to perform a **regression** on the dataset stored in `data/regression/`.
* The inputs x are stored in inputs.npy.
* The labels y are stored in labels.npy

We were free to choose the regression method. In the report, you can retrieve the explanation and the discussion of our approach.
For instance, you could find :
— the performance of several methods that we tried.
— the choice of the hyperparameters and the method to choose them.
— the optimization method

In a first part, we had to work on a dataset and perform **supervised learning** on it. We decided to work on mobile price **classification**, and try to find some relation between features of a mobile phone (RAM, number of cores, internal memory...) and its selling price. 

We found a dataset that instead of giving the actual price of each phone give a price range indicating how high the price is.

We could perform different analysis ont this dataset and try to visualize our data.

<img src="https://github.com/Pypearl/PTML/blob/main/readme_images/supervised_vis.png" alt="Supervised_Visualization">

We decided to test several algorithms with the **sklearn** and **xgboost** library.
First, we split our data to obtain a train dataset and a test dataset, then we iterate on the **classifier** to fit them and test them. The scoring is here a `R2 score`, which is evaluated by **cross validation**.

The results are the following :

<img src="https://github.com/Pypearl/PTML/blob/main/readme_images/supervised_res.png" alt="Supervised_Visualization">

## Classification

In this section, we decided to work on a dataset containing information about 167 **countries**, like the inflation, the net income by person and other **economical parameters**.

After loading the data we checked that there were no missing and duplicated data. And we decided to visualize it with **seaborn** library.

Our objectives in this unsupervised learning is to decide whether or not a country should receive a **Funding** for Development Aid.

For this **machine learning** part, we decided to use the **K-Means** algorithm from the **sklearn** library. The objective was to apply a **clustering** technique on our data to classify them and to know which one of them should receive a Funding for Development Aid.

We applied the algorithm on two datasets, the first one which was the result of a **PCA**, and the second one which was the result of a **MinMax Scaling**. The most optimized number of cluster is 3, we used the elbow method and the silouhette method from the sklearn library to find it.


<img src="https://github.com/Pypearl/PTML/blob/main/readme_images/unsupervised_map.png" alt="Supervised_Visualization">


## Authors

alexandre.lemonnier\
alexandre.poignant\
sarah.gutierez\
victor.simonin