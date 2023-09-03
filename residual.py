from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from astropy import units as u
import healpy as hp
import numpy as np
import emcee
import warnings
from tqdm import tqdm

import sys 
sys.path.append('../marcia/')
from marcia import database
from marcia import Cosmology

class ResidualStat:
    """
    ResidualStat is a class that computes the residuals of a cosmological model fit
    to supernova data. It takes as input the cosmological model, its parameters,
    the chain file containing the MCMC samples, and the supernova data. 
    It provides methods to plot the positions of the supernovae in galactic coordinates, 
    compute the residuals, and plot the residual histogram. It also provides methods
    to compute the angular power spectrum of the residuals using different bootstraping methods.

    Attributes:
    ----------
    model: str - cosmological model
    parameters: dict - cosmological parameters
    chainfile: str - path to the chain file
    data: str - data to use
    zmin: float - minimum redshift
    zmax: float - maximum redshift
    """

    def __init__(self, model,parameters,chainfile,data='Pantheon_plus',zmin=None,zmax=None):

        self.model = Cosmology(model,parameters)
        self.sample = emcee.backends.HDFBackend(chainfile)
        self.samples = self.sample.get_chain(discard=50, flat=True)
        self.sample_median = np.median(self.samples, axis=0)
        db = database.Data(data)
        self.ra, self.dec = db.get_pantheon_plus(position=True)
        self.zcmb,self.mbs,mbs_cov = db.get_pantheon_plus()
        self.zhel = db.get_pantheon_plus(Zhel=True)
        self.mbs_err = np.sqrt(np.diag(mbs_cov))

        self.NSIDE = 16
        self.npix = hp.nside2npix(self.NSIDE)
        self.lmax = 3*self.NSIDE - 1
        
        if zmin is not None:
            mask = (self.zcmb >= zmin)
            self.ra = self.ra[mask]
            self.dec = self.dec[mask]
            self.zcmb = self.zcmb[mask]
            self.mbs = self.mbs[mask]
            self.mbs_err = self.mbs_err[mask]
            self.zhel = self.zhel[mask]


        if zmax is not None:
            mask = (self.zcmb <= zmax)
            self.ra = self.ra[mask]
            self.dec = self.dec[mask]
            self.zcmb = self.zcmb[mask]
            self.mbs = self.mbs[mask]
            self.mbs_err = self.mbs_err[mask]
            self.zhel = self.zhel[mask]
        
    
    @property
    def resolution(self):
        """
        Returns the resolution of the map in degrees
        """
        return hp.nside2resol(NSIDE, arcmin=True) / 60 
        
    def plot_SN(self):
        """
        Plots the positions of the supernovae in galactic coordinates
        """
        c = SkyCoord(ra = self.ra*u.degree,dec = self.dec*u.degree,frame = 'icrs')

        ra_rad = c.ra.radian
        ra_rad[ra_rad > np.pi] -= 2. * np.pi
        dec_rad = c.dec.radian

        c_gal = c.galactic
        l_rad = c_gal.l.radian
        l_rad[l_rad > np.pi] -= 2. * np.pi
        b_rad = c_gal.b.radian
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(1,1,1,projection="aitoff", aspect='equal')
        plt.grid(True)
        points = plt.scatter(-1.*l_rad, b_rad, marker = 'o',alpha = 0.8, s = 20,c=self.zcmb,cmap=plt.cm.RdBu_r)
        plt.colorbar(points,orientation="vertical", pad=0.02, label = 'Redshift')
        plt.title("SN Ia Positions in galactic Coordinates", y=1.08)
    
    def get_m(self):
        """
        Returns the pixel index of the supernovae
        """
        m = hp.ang2pix(nside=self.NSIDE,phi=self.dec ,theta=self.ra, lonlat=True)
        return m.tolist()
    
    def get_mask(self):
        """
        Returns the mask and the index of the supernovae
        """
        arr = np.array([0.]*self.npix)
        index = np.array([0.]*self.npix)
        return arr,index

    def get_residual(self,theory):
        """
        Returns the residual of the fit

        Parameters:
        ----------
        theory: array - array of the distance modulus of the supernovae
        """
        return (self.mbs - theory)/self.mbs_err
    
    def plot_residual_hist(self,param=None):
        """
        Plots the residual histogram
        """
        if param is None:
            param = self.sample_median
        theory = self.get_theory(param)
        ri = self.get_residual(theory)
        plt.hist(ri,bins=30)
        plt.xlabel('Residual')
        plt.ylabel('Counts')
        plt.title('Residual Histogram')
        plt.show()

    
    def get_residual_arr_ind(self,theory,bootstrap=False):
        """
        Returns the residual array and index

        Parameters:
        ----------
        theory: array - array of the distance modulus of the supernovae
        bootstrap: bool - if True, the residual array is shuffled
        """
        m = self.get_m()
        arr_,ind_ = self.get_mask()
        arr = arr_.copy()
        ind = ind_.copy()
        ri = self.get_residual(theory).copy()
        if bootstrap:
            ri = np.random.permutation(ri)
        for i in range(len(self.zcmb)):
            arr[m[i]] += ri[i]
            ind[m[i]] += 1.
        return arr,ind
    
    def get_theory(self,param=None):
        """
        Returns the distance modulus of the supernovae

        Parameters:
        ----------
        param: List - cosmological parameters
        """
        if param is None:
            param = self.sample_median
        theory = self.model.distance_modulus(param, self.zcmb, self.zhel)
        return theory
    
    def get_residual_map(self,param=None,bootstrap=False):
        """
        Returns the residual map

        Parameters:
        ----------
        param: List - cosmological parameters
        bootstrap: bool - if True, the residual array is shuffled
        """
        if param is None:
            param = self.sample_median
        theory = self.get_theory(param)
        arr,index = self.get_residual_arr_ind(theory,bootstrap=bootstrap)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            arri = np.nan_to_num(arr/index)
        arri = hp.ma(arri, badval=0, copy=True)
        return arri
    
    def get_residual_cls(self,param=None,bootstrap=False):
        """
        Returns the residual angular power spectrum
        
        Parameters:
        ----------
        param: List - cosmological parameters
        bootstrap: bool - if True, the residual array is shuffled
        """
        if param is None:
            param = self.sample_median
        sn_map = self.get_residual_map(param,bootstrap=bootstrap)
        #sn_map = hp.smoothing(sn_map, fwhm=np.radians(.5))
        cl = hp.anafast(sn_map, lmax=self.lmax,use_weights=True)
        return cl

    def random_nsamples(self, n):
        """
        Returns n random samples from the chain

        Parameters:
        ----------
        n: int - number of samples
        """
        samples = self.samples.copy()
        np.random.shuffle(samples)
        
        seen = set()
        count = 0
        for sample in tqdm(samples,desc='Monte Carlo Samples',unit='samples',total=n):
            sample_tuple = tuple(sample)
            if sample_tuple not in seen:
                yield sample
                seen.add(sample_tuple)
                count += 1
            if count > n:
                del (samples,seen)
                break
    
    def get_residual_cls_bootstrap(self,nsamples):
        """
        Performs bootstraping on the residual angular power spectrum

        Parameters:
        ----------
        nsamples: int - number of samples
        """
        cls = []
        for i in range(nsamples):
            cls.append(self.get_residual_cls(bootstrap=True))
        return np.array(cls).mean(axis=0)
    
    def bootstrap_position(self,mc_nsamples,b_nsamples):
        """
        Returns the residual angular power spectrum using bootstraping on the position of the supernovae

        Parameters:
        ----------
        mc_nsamples: int - number of Monte Carlo samples
        b_nsamples: int - number of bootstrap samples
        """
        n = len(self.samples)
        assert mc_nsamples < n, "mc_nsamples must be less than the number of samples"
        cls = []
        for sample in self.random_nsamples(mc_nsamples):
            cls.append(self.get_residual_cls_bootstrap(b_nsamples))
        return np.array(cls)
    
    def bootstrap_pixel(self,mc_nsamples,b_nsamples):
        """
        Returns the residual angular power spectrum using bootstraping on the pixel of the supernovae

        Parameters:
        ----------
        mc_nsamples: int - number of Monte Carlo samples
        b_nsamples: int - number of bootstrap samples
        """
        n = len(self.samples)
        assert mc_nsamples < n, "mc_nsamples must be less than the number of samples"
        cls = []
        for sample in self.random_nsamples(mc_nsamples):
            res_map = self.get_residual_map(sample)
            mask = res_map.mask
            sn_ = np.array(res_map)
            res_masked_indexes = np.where(mask == False)[0]
            cls_ = []
            for i in range(b_nsamples):
                shuffled_indexes = np.random.permutation(res_masked_indexes)
                sn_shuffled = np.zeros_like(sn_)
                sn_shuffled[res_masked_indexes] = sn_[shuffled_indexes]
                cl_arri_shuffled = hp.anafast(sn_shuffled,lmax=self.lmax,use_weights=True)
                cls_.append(cl_arri_shuffled)
            cls.append(np.array(cls_).mean(axis=0))
        return np.array(cls)
    
    def bootstrap(self,mc_nsamples,b_nsamples,method='position'):
        """
        Returns the residual angular power spectrum using bootstraping

        Parameters:
        ----------
        mc_nsamples: int - number of Monte Carlo samples
        b_nsamples: int - number of bootstrap samples
        method: str - method to use for bootstraping
        """
        if method == 'position':
            return self.bootstrap_position(mc_nsamples,b_nsamples)
        elif method == 'pixel':
            return self.bootstrap_pixel(mc_nsamples,b_nsamples)
        else:
            raise ValueError("method must be either 'position' or 'pixel'")


    
    def plot_cls(self,cls=None,nsamples=None):
        """
        Plots the residual angular power spectrum

        Parameters:
        ----------
        cls: array - residual angular power spectrum
        nsamples: int - number of samples
        """
        if cls is None:
            assert nsamples is not None, "nsamples must be specified or cls must be provided"
            cls = self.bootstrap_chain(nsamples)
        l = np.arange(len(cls[0]))
        plt.figure(figsize=(6,6))
        plt.plot(l,cls.T,alpha=0.1)
        plt.plot(l,np.mean(cls,axis=0),color='black',label='Mean')


class ResidualComp:
    """
    ResidualComp is a class that takes a dictionary of cosmological models
    and their parameters, and runs the bootstraping analysis on each model 
    using the specified method. It provides a method to plot the angular 
    power spectrum of the residuals for each model.

    Attributes:
    ----------
    dict: dict - dictionary of cosmological models, their parameters, and chain files
    method: str - method to use for bootstraping
    mc_nsamples: int - number of Monte Carlo samples
    b_nsamples: int - number of bootstrap samples
    """

    def __init__(self,dict,method,mc_nsamples=100,b_nsamples=100):
        self.models = list(dict.keys())
        self.mc_nsamples = mc_nsamples
        self.b_nsamples = b_nsamples
        self.method = method
        self.residuals = {}
        self.results = {}
        for model in self.models:
            self.residuals[model] = ResidualStat(model,
                                                 dict[model]['params'],
                                                 dict[model]['datafile'])

    def run_bootstraping(self):
        """
        Runs bootstraping on each model
        """
        for model in self.models:
            print(f'Running bootstraping on {self.method} for model: {model}')
            self.results[model] = self.residuals[model].bootstrap(self.mc_nsamples,self.b_nsamples,self.method)
    
    def plot(self):
        for model in self.models:
            plt.plot(self.results[model].mean(axis=0)[1:],label=model)
        plt.legend()
        plt.title(f'{self.method} bootstraping')
        plt.semilogy()