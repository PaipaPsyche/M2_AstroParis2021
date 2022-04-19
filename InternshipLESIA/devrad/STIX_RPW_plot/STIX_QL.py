#package imports
# UTILS
from datetime import datetime,time,timedelta
import os
## PLOT
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

## MATH
from scipy.optimize import curve_fit
import numpy as np

## GUI
from ipywidgets import interact, interactive, widgets, fixed,interact_manual,FloatSlider
try:
    from ipywidgets import Layout
except:
    pass 

## ASTROPY
from astropy.io import fits
from astropy.time.core import Time, TimeDelta
from astropy.table import Table, vstack, hstack
import astropy.units as u


        
def get_counts_data_per_energy(pathfile, is_bkg=False,time_arr=None):
    
    print("Extracting info: ")
    print("  File: ",os.path.basename(pathfile))
    print("  Type: STIX L1","BKG" if is_bkg else "")
    #output  dict
    return_dict = {}
    #fits info
    hdulist = fits.open(pathfile)
    header = hdulist[0].header
    data = Table(hdulist[2].data)
    
    #sum over all detectors and pixels (optional)
    data_counts = np.sum(data['counts'],axis=(1,2))
    # get cts_per_sec
    n_energies = len(hdulist[3].data["channel"])
    print ("  Energy channels extracted: ",n_energies)
    #normalise by time_bin duration ("timedel" keyword)
    if is_bkg and np.shape(data['timedel'])[0]>1:
        data_counts = np.mean(data_counts,axis=0)
        timedel = np.mean(data['timedel'])
        data_counts_per_sec = np.reshape(data_counts/timedel,(n_energies))
    else:
        data_counts_per_sec = np.reshape(data_counts/data['timedel'],(n_energies)) if is_bkg else data_counts/data['timedel'].reshape(-1,1)
    
    # for bakground create array of constant bkg cts/sec value per energy bin
    if is_bkg:
        bkg_arr = []
        for i in range(len(time_arr)):
            bkg_arr.append(data_counts_per_sec)
            
            
            
        return_dict = {"time":time_arr,
                       "counts_per_sec":bkg_arr
        }
    # for L1 images, return energy info , cts/sec/bin, time array
    else:
        energies = Table(hdulist[3].data)
        max_e = np.max(energies["e_low"])
        mean_energy = [(min(max_e+1,e_high)+e_low)/2 for chn,e_low,e_high in hdulist[3].data]
        
        data_time = Time(header['date_obs']) + TimeDelta(data['time'] * u.s)
        data_time = [t.datetime for t in data_time]
    # counts object, input  for plotting and spectral analysis routines
        return_dict = {"time":data_time,
                   "counts_per_sec":data_counts_per_sec,
                   "energy_bins":energies,
                   "mean_energy":mean_energy}
    return return_dict



def remove_bkg_counts(pathfile,pathbkg):
    
    #import L1 data
    data_L1 = get_counts_data_per_energy(pathfile)
    #import BKG data
    data_BKG = get_counts_data_per_energy(pathbkg,is_bkg=True, time_arr=data_L1["time"])
    
    #subtract background 
    data_counts_per_sec_nobkg = data_L1["counts_per_sec"]-data_BKG["counts_per_sec"] 
    
    # replace ctc/secinfo with corrected info
    return_dict = data_L1.copy()
    return_dict["counts_per_sec"] = data_counts_per_sec_nobkg
    
    return return_dict

def plot_spectrogram(counts_dict,savename=None,colorbar=True,
                      xfmt=" %H:%M",title=None,cmap="jet",fill_nan=True,
                      date_range=None,energy_range=None,x_axis=False,
                      logscale=True,**kwargs):
    # date_ranges param is used for visualizing delimiters for date range selection of the 
    # background and sample pieces (interactive plotting)
    # date_ranges = [[bkg_initial, bkg_final],[smpl_initial, smpl_final]]
    
    plot_time = counts_dict["time"]
    cts_per_sec = counts_dict["counts_per_sec"]
    energies = counts_dict ["energy_bins"]
    mean_e = counts_dict["mean_energy"]
    ax = plt.gca()
    
    
    myFmt = mdates.DateFormatter(xfmt)
    ax.xaxis.set_major_formatter(myFmt)

    cts_data = np.log10(cts_per_sec, out=np.zeros_like(cts_per_sec), where=(cts_per_sec!=0)).T if logscale else cts_per_sec.T
    if(fill_nan):
        cts_data=np.nan_to_num(cts_data,nan=0)
    cm= plt.pcolormesh(plot_time,mean_e,cts_data,shading="auto",cmap=cmap,vmin=0)
    if(colorbar):
        cblabel = "$Log_{10}$ Counts $s^{-1}$" if logscale else "Counts $s^{-1}$"
        plt.colorbar(cm,label=cblabel)
    if(x_axis):
        plt.xlabel("start time "+date_ranges[0],fontsize=14)
    plt.ylabel('Energy bins [KeV]',fontsize=14)
    if(energy_range):
        plt.ylim(*energy_range)
    if(title):
        plt.title(title)
    if(date_range):
        dt_fmt_ = "%d-%b-%Y %H:%M:%S"
        date_range=[datetime.strptime(x,dt_fmt_) for x in date_range]
        plt.xlim(*date_range)
    #return fig, axes
    if(savename):
        plt.savefig(savename,bbox_inch="tight")
    
    return ax