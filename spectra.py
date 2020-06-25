from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import exp, sqrt, pi
from lmfit import Model
plt.rcParams['figure.figsize'] = [15, 8]

#gaussian function
def gaussian(x, amp, lambda0, sigma):
        return (amp / (sqrt(2*pi) * sigma)) * exp(-(x-lambda0)**2 / (2*sigma**2))

class Spectrum:    
    def __init__(self, filename):
        #properties for gfitting
        self.init=5
        self.delta_left = 15
        self.delta_right = 15
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
     
    #it's too slow right now. I have to fix it
    def gfitting(self, lambdaT, plot=False, unique=False):
        avg_balmer=0
        avg_err= 0
        for lT in lambdaT:
            results=[]#list containing results for each iteration
            for i in range(self.delta_left, self.init-1, -1):
                for j in range(self.init, self.delta_right+1, 1):
                    wave_range = np.where((self.wavelength>=round(lT)-i) & (self.wavelength<=round(lT)+j))#iterating between 20 and 5 on the left and 5 and 20 on the right
                    x = np.array(self.wavelength[wave_range])
                    y = np.array(self.flux[wave_range])

                    result = self.fit(x,y,1,lT,2)
                    obj = {'x':x, 'y':y, 'result':result}
                    err = result.params['lambda0'].stderr
                    if not (err==None or np.isnan(err) or err==0.0 or err==-0.0):
                        results.append(obj)#only adding to the list valid results
            
            errs = [i['result'].params['lambda0'].stderr for i in results] 
            min_err = np.amin(errs)#getting minimum value for the std error on lambda0
            index = np.where((errs==min_err))#getting index for minimum result value
            data = results[index[0][0]] #variable that holds result object (and also x and y for plot)
            #data['result'].plot() 
            lambdaObs = data['result'].params['lambda0'].value#index of min err
            v = (self.c/1000) * (lambdaObs-lT)/lT #km/s
            err_velocity = self.c*min_err/(lT*1000) #km/s
                
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
    
        
    def fit_unique(self, lT, i, j):
        wave_range = np.where((self.wavelength>=i) & (self.wavelength<=j))
        x = np.array(self.wavelength[wave_range])
        y = np.array(self.flux[wave_range])
        sigma = np.array(self.dflux[wave_range])
        return {'result':self.fit(x,y,1,lT,2),'x':x,'y':y,'sigma':sigma}