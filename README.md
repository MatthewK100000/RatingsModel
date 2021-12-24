# RatingsModel
A Bayesian hypothesis testing tool which determines the significance of user rating differences.


## Background
Suppose you have a user ratings distribution for a product you're looking to buy, or maybe the responses from a survey you've conducted. Given the counts for each category, how does one assess the significance of the differences between these counts? 

This tool helps infer the answer to that question! At a high level, RatingsModel smoothes the observed counts with a multinomial distribution and calculates the probability that the sampled counts for the (observed) highest voted category are greater than the sampled counts of all the other categories, assuming the event of assigning ratings is repeated "in alternate, independent universes" with the same (or about the same) frequencies*. One minus this probability is the p-value of the test with a null hypothesis that there are no significant differences.

*that is, ratings given on an individual level are "drawn out of a box" with those frequencies.

## Details on Usage and Implementation
For situations where the number of responses is low, the observed proportions for each rating will be crude approximations, therefore a Dirichlet prior (which is conjugate to the Multinomial distribution) on the proportions may be imposed to regularize the observed proportions, adding noise to the proportions for each hypothetical event where ratings are assigned. You do this by using the monte carlo version of the test ``monte_carlo_test``, which uses a monte carlo approximation of the p-value, with ``sample_from_prop_prior = True``. 

For situations where the number of responses is high, the observed proportions will be good approximations, and the Dirichlet prior may be dropped. Either call the method ``monte_carlo_test`` with ``sample_from_prop_prior = False``, or use the ``exact_test``. Mathematically, the exact p-value calculation involves a summation over integer partitions, which is done by utilizing two fast and memory efficient algorithms to calculate these partitions, as well as the ``multiprocessing`` module to distribute the workload over multiple cores. A fast computer with many cores or a good cloud computing instance is recommended for this. Without a good machine, you may find it easier to estimate the p-value instead, by trading imprecision over speed via ``monte_carlo_test``, setting ``sample_from_prop_prior = False`` and ``sample_from_count_prior = False``. 

You may also wish to impose a count prior on the total number of responses, to sample from it for each hypothetical event where ratings are assigned. You would want to do this if you think the turnout is expected to fluctuate according to some distribution, queue or process, time dependent or not, and you wish to include this information in your inference. This is important since the turnout will influence the scale at which differences are significant*. 

For example, what if there is great incentive for every (or almost every) person to submit their response to a question in a survey, while accounting for the busyness of everyday life? In that case, you'd want to use some heavy tailed discrete distribution, concentrated on the side of 100% turnout. For this purpose, ``RatingsModel`` comes with a ``RightGeometricCountPrior`` class, whose masses follow a geometric sequence. A nice feature of this distribution is that parameters can be estimated given a confidence interval. Perhaps, you are 95% sure that the turnout will be somewhere within 180 and 200 (the max possible along the distribution's support)? The estimation works by finding a root via Newton's method, which starts at an inflection point and continues off with a consistent decreasing slope from right to left, guaranteeing quadratic convergence. Sampling from this distribution is also very efficient, done by applying a numpy-vectorized transformation to uniform random variables, amounting to 100% acceptance and 0% rejection. You are free to use include it, or define your own count prior distribution class!**

*For example, suppose you have a turnout of 100. A difference of 30 between the two highest voted categories seems significant. Now, what if the turnout is 500? It may be hard to tell if 30 is a big difference on this scale. 

**your count prior class must have a ``count_rvs()`` method with an argument called ``size``, specifying the number of samples to draw. The input is an integer, and output must be a one-dimensional numpy array of random variables. 

## Examples
### Header
To import the model itself, as well as an optional count prior, which comes with the package.
```python
from Ratings import RatingsModel
from count_priors import RightGeometricCountPrior # optional
```
<br> <br />
### Number of Total Responses is Low
Let's say we want to purchase some running gear and see the following user ratings distribution:
<p align="middle">
  <img src="images/product1.png" width="55%" height="70%" />
  <img src="images/product1ratings.png" width="43%" height="100%" /> 
</p>
Is there a strong, statistically significant, public consensus at 5 stars? This is the same as asking: how small is one minus the probability that people have a 5 star preference over the other stars?  
<br> <br />
Create an instance of the model:

```python
model = RatingsModel()(observed_counts = [14,5,0,1,1])

```
There's a total of only 21 responses, so we'll allow a Dirichlet prior on the observed proportions [14/21, 5/21, 0/21, 1/21, 1/21] and run a monte carlo approximation of the p-value, with the default ``num_samples = 1_000``, treating the total response/count as fixed. The default output is a 95% confidence interval (the p-value is asymptotically normal by the Central Limit Theorem):

```python
print(model.monte_carlo_test(sample_from_prop_prior = True, sample_from_count_prior = False)) 
# outputs (0.09707853036985647, 0.1369214696301435)
```
<br> <br />

#### Changing Defaults
We could always increase ``num_samples`` if we want a tighter estimate:

```python
print(model.monte_carlo_test(sample_from_prop_prior = True, sample_from_count_prior = False, num_samples = 10_000)) 
# outputs (0.10505451077446598, 0.10625948922553402)
```

Or if we want a 99% confidence interval instead:
```python
print(model.monte_carlo_test(sample_from_prop_prior = True, sample_from_count_prior = False, num_samples = 10_000, confidence = 0.99)) 
# outputs (0.09855417196237032, 0.11444582803762976)
```
<br> <br />
Conclusion: It looks like there isn't enough evidence to conclude that people will give the product 5 stars over other stars!
<br> <br />

> :warning:  **Running the exact test under low total scenarios will lead to conflicting results!**
> ```python
> print(model.exact_test()) # outputs to 0.01842333675483565
> ```

<br> <br />
#### Dirichlet Prior Visualization
The Dirichlet prior lies on a 4-[simplex](https://en.wikipedia.org/wiki/Simplex), which is hard to visualize. However, we can inspect a 2-simplex instead, by marginalizing over the other variables. The marginal distribution over the first and second proportion in the list is also a Dirichlet:
```python
model.render_figure(element_pair = [0,1], output = 'image')
```
<p align="middle">
  <img src="images/ternary_contour.png" width="40%" height="40%" />
</p>

It looks like there is some noticeable variability around the first and second proportion, which feature the highest counts.

<details><summary>How to read the ternary plot</summary>
<p>

>The alpha corresponding to each side of the triangle is read counterclockwise. The alpha0 axis is the left leg of the triangle while the alpha1 axis is the bottom leg. 
>
>You can look at the various points along the support of this distribution by connecting the lines intersecting from each axis. The numbers where those lines originate always add up to 1. For example, when alpha0 = 0.6 and alpha1 = 0.2 (consequently alpha_rest = 1 - 0.6 - 0.2 = 0.2) it's close to the hotspot, or mode of the dirichlet prior.

</p>
</details>
<br> <br />

You can modify or add to the plotting parameters of this method. Refer to the [full function definition](https://plotly.com/python-api-reference/generated/plotly.figure_factory.create_ternary_contour.html). You can also make the plot interactive by running it in your web browser:
```python
model.extra_plotting_params = {'colorscale':'Hot'}
model.render_figure(element_pair = [3,4], output = 'web browser') # renders a filled ternary plot of the last 2 proportions
```
<p align="middle">
  <img src="images/ternary_filled.png" width="50%" height="50%" />
</p>



### Number of Total Responses is High
This shampoo looks promising with its total number of ratings and ratings distribution!
<p align="middle">
  <img src="images/product2.png" width="55%" height="70%" />
  <img src="images/product2ratings.png" width="30%" height="10%" /> 
</p>
Once again, is there a strong, statistically significant, public consensus at 5 stars? This is the same as asking: how small is one minus the probability that people have a 5 star preference over the other stars?
<br> <br />

Create an instance of the model via an alternative constructor:

```python
model = RatingsModel().from_percentages_and_total(total = 100, 
                            observed_percentages = [0.62, 0.18, 0.11, 0.04, 0.05])

# replaced 32_852 with 100 since my macbook air can't handle the task in reasonable time
```
We'll drop the Dirichlet prior since almost all of the mass will concentrate around the mode, and it won't make much of a difference if we treat the observed proportions fixed. We'll also fix the total counts distribution parameter. We can then run the ``exact_test``:

```python
print(model.exact_test(parallel_processes = 4, chunksize = 200)) # outputs 6.942235009077535e-08
```
The ``parallel_processes`` specifies the number of process to spawn. I set this to 4 since I only have 4 cores on my machine. The ``chunksize`` parameter dictates the number of unique partitions (up to reordering) to allocate to each core. Each core will basically still go through each unique permutation of each partition however. 

Judging from how small the exact p-value is, it looks like there is enough statistical evidence to conclude that there are significant differences between the highest voted rating and all the other ratings. It is expected that as we increase the ``total``, the evidence will be even stronger. We can get a sense of what the true p-value would have been, have we had more computing power, via an approximation:

```python
model = RatingsModel().from_percentages_and_total(total = 32_852, 
                            observed_percentages = [0.62, 0.18, 0.11, 0.04, 0.05]) # new instance of the model, using the actual num of responses
                            
print(model.monte_carlo_test(sample_from_prop_prior = False, sample_from_count_prior = False, num_samples = 1_000_000))
```

You'll most likely end up seeing a ``DegenerateIntervalWarning`` raised. The warning comes with the following message:
```
Out of 1000000 samples, all cases were positive, so the estimated sampling variability is zero. 
Margin of error dictates that the actual p-value may be less than or equal to 1e-06.

(0, 0)
```



### RightGeometricCountPrior

### Custom Count Prior


## Recommended Dependencies
- Python (3.6.4 or greater)
  - Builtins [io](https://docs.python.org/3.6/library/io.html), [warnings](https://docs.python.org/3.6/library/warnings.html), and [functools](https://docs.python.org/3.7/library/functools.html?highlight=functools#module-functools)
- [NumPy](https://numpy.org/) (version 1.15.3 or greater)
- [SciPy](https://scipy.org/) (version 1.5.4 or greater)
- [Matplotlib](https://matplotlib.org/) (version 3.1.2 or greater)
- [Pillow](https://python-pillow.org/) (version 5.2.0 or greater)
- [Plotly](https://plotly.com/) (version 5.3.1 or greater)

## Underlying Mathematics
TODO

