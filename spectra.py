from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sqrt, pi
from lmfit import Model
import seaborn as sns
from importlib import reload
import emcee
import corner

plt.rcParams['figure.figsize'] = [15, 8]

#gaussian function
def gaussian(x, amp, lambda0, sigma):
	return (amp / (sqrt(2*pi) * sigma)) * exp(-(x-lambda0)**2 / (2*sigma**2))

#mcmc methods which don't depend on individual spectrum
def delta_l_from_delta_v(delta_velocity, lambda_rest):
	return delta_velocity*lambda_rest/299792.458 #d_v is in km/s not in m/s

def lnL(theta, x, y, yerr, lambda_rest, c):
	amp,sigma,delta_velocity, c = theta
	delta_lambda = delta_l_from_delta_v(delta_velocity,lambda_rest)
	model = c + (amp / (sqrt(2*pi) * sigma)) * exp(-(x-delta_lambda-lambda_rest)**2 / (2*sigma**2))
	inv_sigma2 = 1.0/(yerr**2)
	return -0.5*(np.sum((y-model)**2*inv_sigma2))

def lnprior(theta):
	amp,sigma,delta_velocity,c = theta
	if 0 < sigma < 10 and -100 < delta_velocity < 100:
		return 0.0
	return -np.inf

def lnprob(theta, x, y, yerr, lambda_rest, c):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnL(theta, x, y, yerr, lambda_rest, c)

class Spectrum:    
	def __init__(self, filename):
		#properties for gfitting
		self.c = 299792458
		
		#reading .fits
		hdul = fits.open(filename)
		self.flux = hdul[0].data
		self.dflux = hdul[1].data #value for sigma
		self.header = hdul[0].header
		hdul.close()
		units = self.header['CUNIT1'].lower()
		#preferred unit is angstrom - so when given in meter, convert.
		if units == 'm':
			f = 10**10
		elif units == 'angstrom':
			f = 1
		else:
			f = 1 #default value - don't know in which other units spectra can come. I'll change this later
		self.wavelength = f*(np.arange(self.header['NAXIS1'])*self.header['CDELT1']+self.header['WSTART'])

	"""
	Public methods: 
	- plot (plots spectra in a given range)
	- fit_mcmc (runs MCMC using provided wavelengths - users don't need to insert H alpha and H beta)
	"""	
	
	#Plotting function given min and max wavelengths
	def plot(self, t='-', lmin=0, lmax=10000):
		wave_range = np.where((self.wavelength>=lmin) & (self.wavelength<=lmax))
		x = self.wavelength[wave_range]
		y = self.flux[wave_range]
		plt.plot(x,y, t, linewidth=.5)
		plt.xlabel('wavelength')
		plt.ylabel('Flux')
		plt.show()

	"""
	Options for kwargs:
		showGaussianPlots = boolean (whether you want to display the gaussian fits or not),
		showCornerPlots = boolean (if you want to see the corner plots,
		save = string (filepath to save corner plots)
	"""
	def fit_mcmc(self, lambda_rest_arr=[], **kwargs):
		#obtaining avg velocity for h alpha and h beta
		b_lines = [4861.297761, 6562.85175] #by default script finds velocity by analyzing H alpha and H beta lines (balmer lines)
		#result_balmer_lines = self.__get_avg_velocity_balmer(b_lines)	

		#any line which yields velocities outside this range will be ignored
		min_vel=-100
		max_vel=100

		velocities = []
		sigmas = []

		#concatenate lists
		lambda_rest_arr = b_lines + lambda_rest_arr

		for i in range(len(lambda_rest_arr)):
			lambda_rest = lambda_rest_arr[i]
			data = self.__fit_single_line(lambda_rest)
			amp = data['result'].params['amp'].value
			sigma = data['result'].params['sigma'].value

			if amp*sigma > 0: #if amp * sigma is positive, that means the gaussian approximation wasn't good (it fit a gaussian with positive amplitude) and this line should be ignored!
				continue
    
			lambdaObs = data['result'].params['lambda0'].value
			velocity = (self.c/1000)*(lambdaObs-lambda_rest)/lambda_rest
			#only ignore lines which are not H alpha and H beta (also ignore spectra with sigma/velocity >= .5)
			if ((velocity < min_vel or velocity > max_vel or sigma/velocity >= 0.3) and i > 1):
				continue

			y0 = data['y'][0]
			p_initial = [amp,sigma,velocity, y0]#values from result object
			sampler_gaussian, samples_gaussian = self.__run_emcee(data['x'], data['y'], data['sigma'], lambda_rest, y0, p_initial)
			v_med=np.quantile(samples_gaussian[:,2],.50)
			dv = 0.5*(np.quantile(samples_gaussian[:,2],.84)-np.quantile(samples_gaussian[:,2],.16))
			ps = np.quantile(samples_gaussian[:,2],[.16,.50,.84])
			dv=(ps[2]-ps[0])/2
			

			velocities.append(v_med)
			sigmas.append(dv)
			
			if 'showGaussianPlots' in kwargs and kwargs['showGaussianPlots'] == True:
				print(lambda_rest)
				self.__plot_best(lambda_rest,data['x'],data['y'],data['result'])

			if 'showCornerPlots' in kwargs:
				if 'save' in kwargs:
					filepath = kwargs['save']
				else:
					filepath = None

				fig, axes = self.__show_walkers(sampler_gaussian.chain, ['amp', 'sigma', 'velocity', 'c'],lambda_rest, filepath)
				fig = self.__show_corner_plot(samples_gaussian, ['amp','sigma','velocity','c'], p_initial,lambda_rest, filepath)

		return {'avg_velocity': self.__calculate_avg_velocity(velocities, sigmas), 'avg_uncertainty':self.__calculate_avg_uncertainty(sigmas)}

	"""
	Private methods: 
	- fit_single_line 
	- get_best_result 
	- fit
	- plot_best
	- calculate_avg_velocity
	- calculate_avg_uncertainty
	- run_emcee
	- show_walkers
	- show_corner_plot
	- get_avg_velocity_balmer
	"""	

	def __fit_single_line(self, lT):
		data = self.__get_best_result(lT)
		x = data['x']
		y = data['y']
		wave_range = data['wave_range']
		sigma = np.array(self.dflux[wave_range])

		return {'result':data['result'],'x':x,'y':y,'sigma':sigma}

	def __get_best_result(self, lT):
		init=3
		delta_left = 20
		delta_right = 20
		results=[]#list containing results for each iteration
		for i in range(delta_left, init-1, -1):
			for j in range(init, delta_right+1, 1):
				wave_range = np.where((self.wavelength>=round(lT)-i) & (self.wavelength<=round(lT)+j))#iterating between 20 and 5 on the left and 5 and 20 on the right
				x = np.array(self.wavelength[wave_range])
				y = np.array(self.flux[wave_range])

			result = self.__fit(x,y,1,lT,2)
			obj = {'x':x, 'y':y,'wave_range':wave_range, 'result':result}
			err = result.params['lambda0'].stderr
			if not (err==None or np.isnan(err) or err==0.0 or err==-0.0):
				results.append(obj)#only adding to the list valid results
				
		errs = [i['result'].params['lambda0'].stderr for i in results] 
		min_err = np.amin(errs)#getting minimum value for the std error on lambda0
		index = np.where((errs==min_err))#getting index for minimum result value
		data = results[index[0][0]] #variable that holds result object (and also x and y for plot)
		return data

	def __fit(self,wavelength, model, amp, lambda0, sigma):
		gmodel = Model(gaussian)
		result = gmodel.fit(model-model[0], x=wavelength, amp=amp, lambda0=lambda0, sigma=sigma)#subtracting y0 so that best fit can start with y0=0
		return result

	def __plot_best(self,lambda0, x, y, result):
		plt.plot(x, y, 'o')
		plt.axvline(x=lambda0,linestyle='--')
		plt.errorbar(x, result.best_fit+y[0], result.eval_uncertainty(), 0, label='best fit', elinewidth=2)#adding y0 to best fit data so that it can be shifted
		plt.legend(loc='best')
		plt.xlabel('Wavelength (Å)')
		plt.ylabel('Spectral flux (arbitrary units)')
		#plt.savefig(str(lambda0)+'.jpg')
		plt.show()

	def __calculate_avg_velocity(self, velocities, uncertainties):
		if len(velocities)==0 or len(uncertainties)==0:
			return ''
			
		a = 0
		b = 0
		for i in range(len(velocities)):
			a+= (velocities[i]/(uncertainties[i])**2)
		for i in range(len(velocities)):
			b+=(1/uncertainties[i]**2) 
		return a/b

	def __calculate_avg_uncertainty(self, uncertainties):
		if len(uncertainties)==0:
			return ''

		d = 0
		for i in range(len(uncertainties)):
			d+= (1/uncertainties[i]**2)
		return 1/sqrt(d)


	
	#MCMC methods
	def __run_emcee(self, x, y_obs, sigma, lambda_rest, c, ml_result): #ml_result is the result object from lmfit
		# Set up the properties of the problem.
		ndim, nwalkers = 4, 200
		# Setup a bunch of starting positions.
		pos = [ml_result + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]    
		# Create the sampler.
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y_obs, sigma, lambda_rest, c))
		sampler.run_mcmc(pos, 500)
		samples = sampler.get_chain(discard=300, flat=True)
		return sampler, samples



	def __show_walkers(self, chain, labels, wavelength, savefile=None):
		nwalkers, nreps, ndim = chain.shape
		xval = np.arange(0, nreps)
		fig, ax = plt.subplots(ncols=1, nrows=ndim, figsize=(14, 4))
		for i_dim in range(ndim):
			ax[i_dim].set_ylabel(labels[i_dim])
			for i in range(100):
				ax[i_dim].plot(xval, chain[i, :, i_dim], color='black', alpha=0.5)
		if savefile is not None:
			plt.savefig(savefile+'walkers_plot'+str(wavelength).split('.')[0]+'.png')

		return fig, ax


	def __show_corner_plot(self, samples,labels,truths, wavelength, savefile=None):
		fig = corner.corner(samples, labels=labels, truths=truths, quantiles=[0.16,0.5,0.84], show_titles=True)
		if savefile is not None:
			plt.savefig(savefile+'corner_plot'+str(wavelength).split('.')[0]+'.png')

		return fig



	def __get_avg_velocity_balmer(self, wavelengths):
		v = []
		dv = []
		sigmas = []
		for wavelength in wavelengths:
			data = self.__fit_single_line(wavelength)
			lambdaObs = data['result'].params['lambda0'].value
			uncertainty = data['result'].params['lambda0'].stderr
			velocity = (self.c/1000) * (lambdaObs-wavelength)/wavelength #km/s
			err_velocity = self.c*uncertainty/(wavelength*1000) #km/s
			v.append(velocity)
			dv.append(err_velocity)
			sigmas.append(data['result'].params['sigma'].value)

		avg_velocity = self.__calculate_avg_velocity(v, dv)
		avg_uncertainty = self.__calculate_avg_uncertainty(dv)
		return {'max': avg_velocity+2*avg_uncertainty, 'min':avg_velocity-2*avg_uncertainty,'velocities':v, 'sigmas':sigmas}
	