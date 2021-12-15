from scipy.optimize import newton,bisect
import matplotlib.pyplot as plt
import numpy as np



class RightGeometricCountPrior:
	
	def __init__(self,m,p):
		if not isinstance(m,int):
			raise Exception("The maximum number of possible voters 'm' must be an integer.")
		else:
			self.m = m

		if not (0 < p < 1):
			raise Exception("The common ratio probability 'p' must be between 0 and 1.")
		else:
			self.p = p



	def pmf(self,k):
		if isinstance(k,int):
			if not (0 <= k <= self.m):
				raise Exception("The support of this distribition are the set of integers in the set [0,m].")

		elif isinstance(k,np.ndarray):
			if not np.all( np.logical_and(0 <= k, k <= self.m) ):
				raise Exception("The support of this distribition are the set of integers in the set [0,m].")
		else:
			raise Exception("Supported types for k either integer or np.ndarray")

		const = (1 - self.p) / (1 - self.p**(self.m + 1))
		return const * self.p**(self.m - k)

	# need to include count_ as a prefix for some of these function names, as they will be used via inheritance in the RatingsModel via self.count_rvs() or self.count_pmf
	# to remove ambiguity.
	# however, if you are just using the count prior as a standalone class, may just call instance.rvs() or instance.pmf()

	def count_pmf(self,k):
		return RightGeometricCountPrior.pmf(self,k) # calls the pmf method above (within this count prior)

	def rvs(self,size = 1):
		u = np.random.uniform(0,1,size)
		const = pow(self.p,-self.m-1) - 1
		return np.ceil( - np.log(const*u + 1)/np.log(self.p) - 1 )

	def count_rvs(self,size = 1):
		return RightGeometricCountPrior.rvs(self,size)



	@classmethod
	def from_interval(cls, left_endpoint, right_endpoint, concentration = 0.95, maxiter = 100):

		estimated_p = cls._estimate_p_solver(left_endpoint,
							right_endpoint,
							concentration,
							maxiter
							)

		return cls(right_endpoint, estimated_p)

	
	@staticmethod
	def _estimate_p(x,left_endpoint,right_endpoint,concentration):
		return x**(right_endpoint - left_endpoint + 1) - concentration*x**(right_endpoint + 1) - (1-concentration)

	@staticmethod
	def _estimate_p_prime(x,left_endpoint,right_endpoint,concentration):
		return (right_endpoint - left_endpoint + 1)*x**(right_endpoint - left_endpoint) - concentration*(right_endpoint + 1)*x**right_endpoint
		
	@staticmethod
	def _estimate_p_initial(left_endpoint,right_endpoint,concentration):
		return pow( (right_endpoint - left_endpoint + 1)*(right_endpoint - left_endpoint)/(concentration*(right_endpoint+1)*right_endpoint), 1/left_endpoint)

	@staticmethod
	def _estimate_p_solver(left_endpoint, right_endpoint, concentration = 0.95, maxiter = 100):
		if not (0 < left_endpoint < right_endpoint) or not isinstance(left_endpoint,int) or not isinstance(right_endpoint,int):
			raise Exception("'left_endpoint' and 'right_endpoint' must be integers satisfying: 0 < 'left_endpoint' < 'right_endpoint'.")

		if not (0 < concentration < 1):
			raise Exception("'concentration' must be a valid the probability over the interval [left_endpoint,m]")

		if not isinstance(maxiter,int) or (maxiter <= 0):
			raise Exception("maxiter is the maximum number of iterations for newton's method (or bisection method, if it fails), which is the root finder that estimates parameter 'p'.")

		x0 = RightGeometricCountPrior._estimate_p_initial(left_endpoint,right_endpoint,concentration)

		root, result = newton(func = RightGeometricCountPrior._estimate_p, 
					x0 = x0, 
					fprime = RightGeometricCountPrior._estimate_p_prime, 
					args = (left_endpoint, right_endpoint, concentration),
					maxiter = maxiter, 
					full_output = True, 
					disp = False
					)

		if not result.converged:
			# fallback to bisection method, which is a way slower root finder, but is guaranteed to work.
			root, result = bisect(f = RightGeometricCountPrior._estimate_p, 
						a = 0, 
						b = 1, 
						args = (left_endpoint, right_endpoint, concentration),
						maxiter = maxiter,
						full_output = True,
						disp = False
						)
			# in case it doesn't work...
			if not result.converged:
				raise Exception("Bisection method is the last resort after Newton's method, and the root finder did not converge.")

		return root


# Copyright 2021, Matthew Kulec, All rights reserved.
