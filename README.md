# RatingsModel
A Bayesian hypothesis testing tool which determines the significance of user rating differences.


## Background
Suppose you have a user ratings distribution for a product you're looking to buy, or maybe the responses from a survey you've conducted. Given the counts for each category, how does one assess the significance of the differences between these counts? 

This tool helps infer the answer to that question! At a high level, RatingsModel smoothes the observed counts with a multinomial distribution and calculates the probability that the counts for the highest voted category are greater than the counts of all the other categories, assuming the event of assigning ratings is repeated indefinitely with the same (or about the same) frequencies. This is the p-value of the test with a null hypothesis that there are no significant differences.

For situations where the number of responses is low, the observed proportions for each rating will be crude approximations, therefore a Dirichlet prior (which is conjugate to the Multinomial distribution) on the proportions may be imposed to regularize the observed proportions, adding noise for each hypothetical event where ratings are assigned. You do this by using the monte carlo version of the test ``monte_carlo_test``, which uses a monte carlo approximation of the p-value, with ``sample_from_prop_prior = True``. 

For situations where the number of responses is high, the observed proportions will be good approximations, and the Dirichlet prior may be dropped. Either call the method ``monte_carlo_test`` with ``sample_from_prop_prior = False``, or use the ``exact_test``. Mathematically, the exact p-value calculation involves a summation over integer partitions, which is done by utilizing a fast and clever algorithm to calculate these partitions, as well as the ``multiprocessing`` module to distribute the workload over multiple cores. A fast computer with many cores or a good cloud computing instance is recommended for this. If you wish to calculate the exact p-value without a good machine, you may estimate or gauge it by trading imprecision over speed via ``monte_carlo_test``, setting ``sample_from_prop_prior = False`` and ``sample_from_count_prior = False``.

You may wish to also impose a count prior on the hypothetical events that ratings are assigned. For example, what if there is great incentive for every (or almost every) person to submit their response to a question in a survey? In that case, you'd want to use some heavy tailed discrete distribution, concentrated on the side of 100% turnout. For this purpose, ``RatingsModel`` comes with a ``RightGeometricCountPrior`` class, whose masses follow a geometric sequence. You may use it, or define your own count prior distribution class!* 

*your count prior class must have a ``count_rvs()`` method with an argument called ``size``, specifying the number of samples to draw. The input is an integer, and output must be a one-dimensional numpy array of random variables. 

## Dependencies

## Examples

![](images/ternary_contour.png)
