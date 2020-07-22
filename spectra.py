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
        self.velocities = []
        self.uncertainties = []

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


    #plotting function given min and max wavelengths
    def plot(self, t='-', lmin=0, lmax=10000):
        wave_range = np.where((self.wavelength>=lmin) & (self.wavelength<=lmax))
        x = self.wavelength[wave_range]
        y = self.flux[wave_range]
        plt.plot(x,y, t, linewidth=.5)
        plt.xlabel('wavelength')
        plt.ylabel('Flux')
        plt.show()

    def get_best_result(self, lT):
        init=3
        delta_left = 20
        delta_right = 20
        results=[]#list containing results for each iteration
        for i in range(delta_left, init-1, -1):
            for j in range(init, delta_right+1, 1):
                wave_range = np.where((self.wavelength>=round(lT)-i) & (self.wavelength<=round(lT)+j))#iterating between 20 and 5 on the left and 5 and 20 on the right
                x = np.array(self.wavelength[wave_range])
                y = np.array(self.flux[wave_range])

            result = self.fit(x,y,1,lT,2)
            obj = {'x':x, 'y':y,'wave_range':wave_range, 'result':result}
            err = result.params['lambda0'].stderr
            if not (err==None or np.isnan(err) or err==0.0 or err==-0.0):
                results.append(obj)#only adding to the list valid results
                
        errs = [i['result'].params['lambda0'].stderr for i in results] 
        min_err = np.amin(errs)#getting minimum value for the std error on lambda0
        index = np.where((errs==min_err))#getting index for minimum result value
        data = results[index[0][0]] #variable that holds result object (and also x and y for plot)
        return data

    def gfitting(self, lambdaT, plot=False, unique=False):
        avg_balmer=0
        avg_err= 0
        for lT in lambdaT:
            data = self.get_best_result(lT)
            lambdaObs = data['result'].params['lambda0'].value#index of min err
            err = data['result'].params['lambda0'].stderr
            v = (self.c/1000) * (lambdaObs-lT)/lT #km/s
            err_velocity = self.c*err/(lT*1000) #km/s
                
            #excluding from analysis wrong fits (which yield velocities that differ more then 50km/s than balmer lines)
            if len(self.velocities)==2 and avg_balmer==0:#only enter this if block after two balmer lines
                avg_balmer = self.calculate_avg_velocity(self.velocities, self.uncertainties)
                avg_err = self.calculate_avg_uncertainty(self.uncertainties)
                #print(avg_balmer, avg_err)
            if len(self.velocities)>=2:
                if v > avg_balmer+2*avg_err or v < avg_balmer-2*avg_err :
                    continue 
            #if requested (and if velocity is valid within constrains), plot fit and compare with theoretical value of lambda
            if plot: 
                print('Wavelength: '+str(lT))
                self.plot_best(lT, data['x'], data['y'], data['result'])
            self.velocities.append(v)
            self.uncertainties.append(err_velocity)

        avg_velocity = self.calculate_avg_velocity(self.velocities, self.uncertainties) 
        sigma_avg = self.calculate_avg_uncertainty(self.uncertainties)
        print(str(avg_velocity)+ " +/- "+str(sigma_avg)+" km/s")
        print(self.velocities)
        print(self.uncertainties)
        self.avg_velocity = avg_velocity
        self.uncertainty_avg_velocity = sigma_avg
        #returning result object if unique
        if unique:
            return data['result']


    def fit(self,wavelength, model, amp, lambda0, sigma):
        gmodel = Model(gaussian)
        result = gmodel.fit(model-model[0], x=wavelength, amp=amp, lambda0=lambda0, sigma=sigma)#subtracting y0 so that best fit can start with y0=0
        return result

    def plot_best(self,lambda0, x, y, result):
        plt.plot(x, y, 'o')
        plt.axvline(x=lambda0,linestyle='--')
        plt.errorbar(x, result.best_fit+y[0], result.eval_uncertainty(), 0, label='best fit', elinewidth=2)#adding y0 to best fit data so that it can be shifted
        plt.legend(loc='best')
        plt.xlabel('wavelength')
        plt.ylabel('Flux')
        #plt.savefig(str(lambda0)+'.jpg')
        plt.show()

    def calculate_avg_velocity(self, velocities, uncertainties):
        a = 0
        b = 0
        for i in range(len(velocities)):
            a+= (velocities[i]/(uncertainties[i])**2)
        for i in range(len(velocities)):
            b+=(1/uncertainties[i]**2) 
        return a/b

    def calculate_avg_uncertainty(self, uncertainties):
        d = 0
        for i in range(len(uncertainties)):
            d+= (1/uncertainties[i]**2)
        return 1/sqrt(d)


    def fit_unique(self, lT):
        data = self.get_best_result(lT)
        x = data['x']
        y = data['y']
        wave_range = data['wave_range']
        sigma = np.array(self.dflux[wave_range])

        return {'result':data['result'],'x':x,'y':y,'sigma':sigma}

    #MCMC methods
    def run_emcee(self, x, y_obs, sigma, lambda_rest, c, ml_result): #ml_result is the result object from lmfit
        # Set up the properties of the problem.
        ndim, nwalkers = 4, 200
        # Setup a bunch of starting positions.
        pos = [ml_result + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]    
        # Create the sampler.
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y_obs, sigma, lambda_rest, c))
        sampler.run_mcmc(pos, 500)
        samples = sampler.get_chain(discard=300, flat=True)
        return sampler, samples



    def show_walkers(self, chain, labels, savefile=None):
        nwalkers, nreps, ndim = chain.shape
        xval = np.arange(0, nreps)
        fig, ax = plt.subplots(ncols=1, nrows=ndim, figsize=(14, 4))
        for i_dim in range(ndim):
            ax[i_dim].set_ylabel(labels[i_dim])

        for i in range(100):
            ax[i_dim].plot(xval, chain[i, :, i_dim], color='black', alpha=0.5)

        if savefile is not None:
            plt.savefig(savefile)

        return fig, ax


    def show_corner_plot(self, samples,labels,truths, savefile=None):
        fig = corner.corner(samples, labels=labels, truths=truths, quantiles=[0.16,0.5,0.84], show_titles=True)
        if savefile is not None:
            plt.savefig(savefile)

        return fig

    def fit_mcmc(self, lambda_rest_arr, showPlots=False):
        velocities = []
        sigmas = []
        for lambda_rest in lambda_rest_arr:
            data = self.fit_unique(lambda_rest)
            amp = data['result'].params['amp'].value
            sigma = data['result'].params['sigma'].value
            lambdaObs = data['result'].params['lambda0'].value
            delta_velocity = 299792.458*(lambdaObs-lambda_rest)/lambda_rest
            y0 = data['y'][0]
            p_initial = [amp,sigma,delta_velocity, y0]#values from result object
            sampler_gaussian, samples_gaussian = self.run_emcee(data['x'], data['y'], data['sigma'], lambda_rest, y0, p_initial)
            if showPlots:
                print("Walker and Corner plots for wavelentgh: "+str(lambda_rest))
                fig, axes = self.show_walkers(sampler_gaussian.chain, ['amp', 'sigma', 'velocity', 'c'])
                fig = self.show_corner_plot(samples_gaussian, ['amp','sigma','velocity','c'], p_initial)
            v_med=np.quantile(samples_gaussian[:,2],.50)
            dv = 0.5*(np.quantile(samples_gaussian[2,:],.84)-np.quantile(samples_gaussian[2,:],.16))
            ps = np.quantile(samples_gaussian[:,2],[.16,.50,.84])
            dv=(ps[2]-ps[0])/2
            velocities.append(v_med)
            sigmas.append(dv)

        return {'avg_velocity': self.calculate_avg_velocity(velocities, sigmas), 'avg_uncertainty':self.calculate_avg_uncertainty(sigmas)}