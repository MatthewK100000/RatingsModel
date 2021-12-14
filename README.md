# RatingsModel
A Bayesian hypothesis testing tool which determines the significance of user rating differences.


## Background
Suppose you have a user ratings distribution for a product you're looking to buy, or maybe the responses from a survey you've conducted. Given the counts for each category, how does one assess the significance of the differences between these counts? 

This tool helps infer the answer to that question! At a high level, RatingsModel smoothes the observed counts with a multinomial distribution and calculates the probability that the counts for the highest voted category are greater than the counts of all the other categories, assuming the event of assigning ratings is repeated indefinitely with the same (or about the same) frequencies. 

For situations where the number of responses is low, the observed proportions for each rating will be crude approximations, 


