# RatingsModel
A Bayesian hypothesis testing tool which determines the significance of user rating differences.


## Background
Suppose you have a user ratings distribution for a product you're looking to buy, or maybe the responses from a survey you've conducted. Given the counts for each category, how does one assess the significance of the differences between these counts? 

This tool helps infer the answer to that question! At a high level, RatingsModel smoothes the observed counts with a multinomial distribution and calculates the probability that the counts for the highest voted category are greater than the counts of all the other categories, assuming the event of assigning ratings is repeated indefinitely with the same (or about the same) frequencies. This is the p-value of the test with a null hypothesis that there are no significant differences.

For situations where the number of responses is low, the observed proportions for each rating will be crude approximations, therefore a Dirichlet prior on the proportions (which is conjugate to the Multinomial distribution) may be imposed to regularize the observed proportions, adding noise for each hypothetical event where ratings are assigned. You do this by using the monte carlo version of the test ``monte_carlo_test``, which uses a monte carlo approximation of the p-value, with ``sample_from_prop_prior = True``. 

For situations where the number of responses is high, the observed proportions will be good approximations, and the Dirichlet prior could be dropped. Either call the method ``monte_carlo_test`` with ``sample_from_prop_prior = False``, or use the ``exact_test``. Mathematically, the exact p-value calculation involves a summation over integer partitions, which is done by utilizing a fast and clever algorithm to calculate these partitions, as well as the ``multiprocessing`` module to distribute the workload over multiple cores. A fast computer with many cores or a good cloud computing instance is recommended for this.


## Dependencies
