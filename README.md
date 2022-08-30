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

We were free to choose the **regression method**. In the report, you can retrieve the explanation and the discussion of our approach.
For instance, you could find :
— the performance of several methods that we tried.
— the choice of the **hyperparameters** and the method to choose them.
— the **optimization** method

Here are the benchmark results :

<img src="https://github.com/Pypearl/FTML/blob/main/readme_images/bench_reg.png" alt="Regression Benchmark">


## Classification

For this exercice, we had to perform **classification** on a given dataset, with our inputs stored in a `inputs.npy` file and our labels in a `labels.npy` file as in the last part.
We were again free to choose our **classifier** model and free to implement what we wanted, so we decided to test several algorithms exposed by the **Sklearn Api** and compare them.

## Authors

alexandre.lemonnier\
alexandre.poignant\
sarah.gutierez\
victor.simonin