import numpy as np
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import plotly.figure_factory as ff
from PIL import Image
import io



# to use self.function() defined outside mixin, not meant to be used as a standalone.

class TernaryPlotMixin:
	"""
	Ternary plot of the marginal distribution over two selected arguments in the parameter vector alpha.
	"""

	# default arguments for plotting, can be changed on instance level
	triangle_mesh_subdivisions = 6
	distribution_contours = 20
	extra_plotting_params = {'coloring':'lines'}

	def _produce_marginal_figure(self,element_pair = None):
		if element_pair is None:
			raise Exception("Must select which pair of arguments in alpha to plot as tuple or list, zero indexed.")

		if (not isinstance(element_pair,tuple)) and (not isinstance(element_pair,list)):
			raise Exception("Must select which pair of arguments in alpha to plot as tuple or list, zero indexed.")

		if len(element_pair) != 2:
			raise Exception("Must select which pair of arguments in alpha to plot as tuple or list, zero indexed.")

		xy_plane_projection_corners = np.array([[0, 0], [1, 0], [0,1]])

		triangle_outline = tri.Triangulation(x = xy_plane_projection_corners[:, 0], 
							y = xy_plane_projection_corners[:, 1]
							)

		refiner = tri.UniformTriRefiner(triangle_outline)
		triangle_mesh = refiner.refine_triangulation(subdiv = self.triangle_mesh_subdivisions)

		X = triangle_mesh.x
		Y = triangle_mesh.y
		Z = 1 - X - Y
		simplex_mesh = np.array([X,Y,Z])

		# temporarily alter the alpha attribute to call proper pdf
		old_alpha = self.alpha
		alpha_temp = [arg for idx,arg in enumerate(old_alpha) if idx in element_pair]
		self.alpha = alpha_temp + [sum(old_alpha) - sum(alpha_temp)]
		new_alpha = self.alpha
		dist_eval = [self.pdf(pt) for pt in simplex_mesh.T]
		self.alpha = old_alpha

		dist_eval = np.asarray(dist_eval)

		fig = ff.create_ternary_contour(
								coordinates = simplex_mesh,
								values = dist_eval,
								pole_labels = list('alpha'+str(arg) for arg in element_pair) + ['alpha_rest'], # referring to the above, it is technically same as ['X', 'Y', 'Z'],
								interp_mode = 'cartesian',
								ncontours = self.distribution_contours,
								showscale = True,
								title = 'Ternary plot of Dirichlet({},{},{}) distribution'.format(*new_alpha),
                                **self.extra_plotting_params
							)

		return fig


	def render_figure(self,element_pair,output = 'image'):
		if output == 'image':
			img_data = self._produce_marginal_figure(element_pair).to_image(format='png') # image data in bytes.
			image = Image.open(io.BytesIO(img_data)) # buffer for image byte data. Opens the image via Pillow.
			image.show()
			return
		elif output == 'web browser':
			self._produce_marginal_figure(element_pair).show()
			return
		else:
			raise Exception("Argument 'output' must either be 'image' or 'web browser'.")

	def _produce_conditional_figure(self,):
		return





class DirichletProportionsPrior(TernaryPlotMixin):

	def __init__(self,alpha):
		if not (isinstance(alpha,list) or isinstance(alpha,tuple)):
			raise Exception("Concentration parameter 'alpha' must be a list or tuple.")

		if not all([(x > 0) for x in alpha]):
			raise Exception("Every element of concentration parameter 'alpha' must be positive.")

		self.alpha = alpha

	def pdf(self,x):
		return dirichlet.pdf(x,self.alpha)

	def proportions_pdf(self,x):
		return DirichletProportionsPrior.pdf(self,x)

	def rvs(self, size = 1, seed = None, ordered = False):
		if not ordered:
			return dirichlet.rvs(self.alpha, size = size, random_state = seed)
		else:
			ordered_alpha = sorted(self.alpha)
			return dirichlet.rvs(ordered_alpha, size = size, random_state = seed)

	def proportions_rvs(self, size = 1, seed = None, ordered = False):
		if not ordered:
			return DirichletProportionsPrior.rvs(self, size, seed, ordered = False)
		else:
			return DirichletProportionsPrior.rvs(self, size, seed, ordered = True)

	def mode(self):
		return [(arg - 1)/(sum(self.alpha) - len(self.alpha)) for arg in self.alpha]


# Copyright 2021, Matthew Kulec, All rights reserved.