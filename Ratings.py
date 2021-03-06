from proportion_priors import DirichletProportionsPrior
from count_priors import RightGeometricCountPrior

import multiprocessing
import warnings

from scipy import stats
import numpy as np

warnings.simplefilter("always")


def _partitionfunc(n,k,l=0):
	'''n is the integer to partition, k is the length of partitions, l is the min partition element size'''
	if k < 1:
		return
	if k == 1:
		if n >= l:
			yield (n,)
		return
	for i in range(l,n+1):		
		for result in _partitionfunc(n-i,k-1,i):
			yield (i,)+result




def _unique_permutations(sub_permutation):
	# algorithm: https://en.wikipedia.org/wiki/Permutation#Generation_in_lexicographic_order

	a = sorted(sub_permutation)
	len_ = len(sub_permutation)

	keep_generating = True

	while keep_generating:
		yield tuple(a)

		k_found = 0
		for k in range(len_ - 2, -1, -1):
			if a[k] < a[k + 1]:
				k_found	= 1
				break

		if k_found == 0:
			keep_generating = False
			continue

		for l in range(len_ - 1, k, -1):
			if a[k] < a[l]:
				break

		a[k],a[l] = a[l],a[k]
		a[k + 1:] = a[k + 1:][::-1]




def _prob_adder(total_responses, observed_proportions, partition):
	max_entry = partition[-1]
	
	max_count = 0
	for entry in partition:
		if entry == max_entry:
			max_count += 1
	if max_count > 1:
		return 0

	sub_probability = 0
	for partition_reordering in _unique_permutations(partition[:-1]):
		sub_probability += stats.multinomial(n = total_responses, p = observed_proportions).pmf(partition_reordering+(partition[-1],))

	return sub_probability




def exact_test(self, parallel_processes = 4, chunksize = 10_000):
	observed_prop = sorted(self.observed_proportions)
	total_responses = self.total_responses
	num_categories = len(self.observed_counts)

	prob_of_event = 0

	partitions_and_extra = [(total_responses, observed_prop, partition) for partition in _partitionfunc(n = total_responses, k = num_categories)]
	with multiprocessing.Pool(processes = parallel_processes) as pool:
		for prob in pool.starmap(_prob_adder, partitions_and_extra, chunksize = chunksize):
			prob_of_event += prob

	return 1 - prob_of_event




class DegenerateIntervalWarning(UserWarning):
	pass



def monte_carlo_test(self, count_prior = None, sample_from_count_prior = False, sample_from_prop_prior = True, num_samples = 1_000, confidence = 0.95, details = False):

	if (not isinstance(sample_from_count_prior,bool)) or (not isinstance(sample_from_prop_prior,bool)):
		raise Exception("Arguments 'sample_from_count_prior' and 'sample_from_prop_prior' must be booleans.")
	
	if (not isinstance(num_samples,int)) or (num_samples <= 0):
		raise Exception("The number of samples used for estimating the p-value must be a positive integer.")

	if not isinstance(details,bool):
		raise Exception("Argument 'details', which results in the mean and standard deviation of the success proportions, must be boolean.")
	
	observed_proportions_sorted = sorted(self.observed_proportions)
	num_categories = len(self.observed_counts)
	total_resp = self.total_responses


	if sample_from_prop_prior + sample_from_count_prior == 0:
		# treat the proportions and total responses as fixed. Responses are sampled from a multinomial distribution with no priors.
		
		sampling_dist = stats.multinomial(n = total_resp, 
									  p = observed_proportions_sorted)

		successes = 0
		for rv in sampling_dist.rvs(size = num_samples):
			if np.sum(rv[-1] > rv[:-1]) == num_categories - 1:
				successes += 1

		success_prop = successes/num_samples


	elif (not sample_from_count_prior) and sample_from_prop_prior:
		# treat the total responses as fixed. Responses are sampled from a multinomial distribution with a Dirichlet prior on the proportions.

		successes = 0
		for rv_prop in self.proportions_rvs(size = num_samples, ordered = True):
			rv_outcome = stats.multinomial.rvs(n = total_resp,
								  			   p = rv_prop,
								  			   size = 1)
			if np.sum(rv_outcome[0][-1] > rv_outcome[0][:-1]) == num_categories - 1:
				successes += 1

		success_prop = successes/num_samples


	elif sample_from_count_prior and (not sample_from_prop_prior):
		# treat the observed proportions as fixed. Responses are sampled from a multinomial distribution with a 'count_prior' prior on the counts.
		if count_prior is None:
			raise Exception("You cannot set 'sample_from_count_prior' to True if 'count_prior' is not given.")

		successes = 0
		for rv_count in self.count_rvs(size = num_samples):
			rv_outcome = stats.multinomial.rvs(n = rv_count,
											   p = observed_proportions_sorted,
											   size = 1)
			if np.sum(rv_outcome[0][-1] > rv_outcome[0][:-1]) == num_categories - 1:
				successes += 1

		success_prop = successes/num_samples

	else:
		# nothing is fixed. Responses are sampled from a multinomial distribution with a 'count_prior' prior on the counts and Dirichlet prior on the proportions.
		if count_prior is None:
			raise Exception("You cannot set 'sample_from_count_prior' to True if 'count_prior' is not given.")

		successes = 0

		counts_and_props = [(rv_count, rv_prop) for rv_count, rv_prop in zip(self.count_rvs(size = num_samples), self.proportions_rvs(size = num_samples, ordered = True))]
	
		for rv_count, rv_prop in counts_and_props:
			rv_outcome = stats.multinomial.rvs(n = rv_count,
											   p = rv_prop,
											   size = 1)
			if np.sum(rv_outcome[0][-1] > rv_outcome[0][:-1]) == num_categories - 1:
				successes += 1

		success_prop = successes/num_samples



	statistics = {
			'p-value mean': 1 - success_prop,
			'p-value std': pow(success_prop*(1 - success_prop)/num_samples,0.5)
				}

	if details:
		return statistics

	elif confidence is not None:
		if success_prop == 0:
			warnings.warn("Out of {} samples, no positive case was encountered, so the estimated sampling variability is zero. Margin of error dictates that the actual p-value may be over {}.".format(num_samples,1 - 1/num_samples),
				DegenerateIntervalWarning)
			return (1,1)
		elif success_prop == 1:
			warnings.warn("Out of {} samples, all cases were positive, so the estimated sampling variability is zero. Margin of error dictates that the actual p-value may be less than or equal to {}.".format(num_samples,1/num_samples),
				DegenerateIntervalWarning)
			return (0,0)
		else:
			pass
		
		if (not isinstance(confidence,float)) or (confidence >= 1) or (confidence <= 0):
			raise Exception("Argument 'confidence' must be a float between 0 and 1 (exclusive).")

		asymptotic_dist = stats.norm(loc = statistics['p-value mean'], 
									scale = statistics['p-value std'])

		interval = ( (1 - confidence)/2 , confidence + (1 - confidence)/2 )
		return tuple(asymptotic_dist.ppf(interval)) # interval may be out of bounds (<=0 or >=1). In that case, increase 'num_samples'.

	else:
		raise Exception("Argument 'confidence' must either be not None or argument 'details' must be True.")






@classmethod
def from_percentages_and_total(cls, total, observed_percentages, **kwargs):
	counts_ = [int(perc*total) for perc in observed_percentages]
	return cls(observed_counts = counts_, **kwargs)





def RatingsModel_init(self, count_prior = None, observed_counts = None, **kwargs):

	if observed_counts is None or not isinstance(observed_counts,list):
		raise Exception("Argument 'observed_counts' must be a list of integers.")

	if not all([isinstance(i,int) for i in observed_counts]):
		raise Exception("Argument 'observed_counts' must be a list of integers.")


	self.observed_counts = observed_counts

	total = sum(observed_counts)
	self.total_responses = total

	self.observed_proportions = [i/total for i in observed_counts]
	DirichletProportionsPrior.__init__(self, alpha = [i + 1 for i in observed_counts])


	params = set(kwargs.keys())
	if count_prior == RightGeometricCountPrior:
		if params.issubset({'left_endpoint', 'right_endpoint', 'concentration', 'maxiter'}): # don't need concentration and maxiter since they have default values
			RightGeometricCountPrior.__init__(self,
							m = kwargs['right_endpoint'], 
							p = RightGeometricCountPrior._estimate_p_solver(**kwargs))
		
		elif params == {'m','p'}:
			RightGeometricCountPrior.__init__(self,m = kwargs['m'], p = kwargs['p'])

		elif params == {'right_endpoint','p'}:
			RightGeometricCountPrior.__init__(self,m = kwargs['right_endpoint'], p = kwargs['p'])

		else:
			raise Exception("Failed to provide sufficient arguments for RightGeometricCountPrior instantiation.")

	elif count_prior is None:
		pass # nothing else to be done
	else:
		count_prior.__init__(self, **kwargs)  # for custom count prior implementation
		# doesn't work: super(produced_cls, self).__init__(**kwargs)







def RatingsModel(count_prior = None):
	methods = {}

	methods['from_percentages_and_total'] = from_percentages_and_total
	methods['__init__'] = lambda self, count_prior = count_prior, observed_counts = None, **kwargs: RatingsModel_init(self, count_prior, observed_counts, **kwargs)

	methods['monte_carlo_test'] = lambda self, count_prior = count_prior, sample_from_count_prior = False, sample_from_prop_prior = True, num_samples = 1_000, confidence = 0.95, details = False: monte_carlo_test(self, count_prior, sample_from_count_prior, sample_from_prop_prior, num_samples, confidence, details)
	methods['exact_test'] = exact_test


	if count_prior is not None:
		produced_cls = type('RatingsModel',
				(DirichletProportionsPrior, count_prior),
				methods
				)
	else:
		produced_cls = type('RatingsModel',
				(DirichletProportionsPrior,),
				methods
				)

	return produced_cls
