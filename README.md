# RatingsModel
A Bayesian hypothesis testing tool which determines the significance of user rating differences.


## Background
Suppose you have a user ratings distribution for a product you're looking to buy, or maybe the responses from a survey you've conducted. Given the counts for each category, how does one assess the significance of the differences between these counts? 

This tool helps infer the answer to that question! At a high level, RatingsModel smoothes the observed counts with a multinomial distribution and calculates the probability that the sampled counts for the (observed) highest voted category are greater than the sampled counts of all the other categories, assuming the event of assigning ratings is repeated indefinitely with the same (or about the same) frequencies. This is the p-value of the test with a null hypothesis that there are no significant differences.


## Details on Usage
For situations where the number of responses is low, the observed proportions for each rating will be crude approximations, therefore a Dirichlet prior (which is conjugate to the Multinomial distribution) on the proportions may be imposed to regularize the observed proportions, adding noise for each hypothetical event where ratings are assigned. You do this by using the monte carlo version of the test ``monte_carlo_test``, which uses a monte carlo approximation of the p-value, with ``sample_from_prop_prior = True``. 

For situations where the number of responses is high, the observed proportions will be good approximations, and the Dirichlet prior may be dropped. Either call the method ``monte_carlo_test`` with ``sample_from_prop_prior = False``, or use the ``exact_test``. Mathematically, the exact p-value calculation involves a summation over integer partitions, which is done by utilizing a fast and clever algorithm to calculate these partitions, as well as the ``multiprocessing`` module to distribute the workload over multiple cores. A fast computer with many cores or a good cloud computing instance is recommended for this. If you wish to calculate the exact p-value without a good machine, you may find it easier to estimate it instead, by trading imprecision over speed via ``monte_carlo_test``, setting ``sample_from_prop_prior = False`` and ``sample_from_count_prior = False``. 

You may also wish to impose a count prior on the total number of responses, to sample from it for each hypothetical event where ratings are assigned. You would want to do this if you think the turnout is expected to fluctuate according to some distribution, queue or process, time dependent or not, and you wish to include this information in your inference. 

For example, what if there is great incentive for every (or almost every) person to submit their response to a question in a survey, while accounting for the busyness of everyday life? In that case, you'd want to use some heavy tailed discrete distribution, concentrated on the side of 100% turnout. For this purpose, ``RatingsModel`` comes with a ``RightGeometricCountPrior`` class, whose masses follow a geometric sequence. A nice feature of this distribution is that parameters can be estimated given a confidence interval. Perhaps, you are 95% sure that the turnout will be somewhere within 180 and 200 (the max possible along the distribution's support)? The estimation works by finding a root via Newton's method, which starts at an inflection point and continues off with a consistent decreasing slope from right to left, guaranteeing quadratic convergence. Sampling from this distribution is also very efficient, done by applying a transformation to a uniform random variable, amounting to 100% acceptance and 0% rejection. You are free to use include it, or define your own count prior distribution class!*


*your count prior class must have a ``count_rvs()`` method with an argument called ``size``, specifying the number of samples to draw. The input is an integer, and output must be a one-dimensional numpy array of random variables. 

## Examples
#### Header
To import the model itself, as well as an optional count prior, which comes with the package.
```python
from Ratings import RatingsModel
from count_priors import RightGeometricCountPrior # optional
```
#### Number of Total Responses is Low
<p align="middle">
  <img src="images/product1.png" width="55%" height="60%" />
  <img src="images/product1ratings.png" width="40%" height="80%" /> 
</p>

#### Number of Total Responses is High

#### RightGeometricCountPrior

#### Custom Count Prior


## Recommended Dependencies
- Python (3.6.4 or greater)
  - Builtins [io](https://docs.python.org/3.6/library/io.html) and [functools](https://docs.python.org/3.7/library/functools.html?highlight=functools#module-functools)
- [NumPy](https://numpy.org/) (version 1.15.3 or greater)
- [SciPy](https://scipy.org/) (version 1.5.4 or greater)
- [Matplotlib](https://matplotlib.org/) (version 3.1.2 or greater)
- [Pillow](https://python-pillow.org/) (version 5.2.0 or greater)
- [Plotly](https://plotly.com/) (version 5.3.1 or greater)

## Underlying Mathematics
TODO


![](images/ternary_contour.png)
