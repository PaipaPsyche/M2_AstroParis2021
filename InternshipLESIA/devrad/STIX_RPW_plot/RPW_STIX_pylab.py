# -*- coding: utf-8 -*-

# SOLAR ORBITER DATA ANALYSIS LAB
# RPW - STIX========================
# functionalities
# RPW-----
#   * create/plot psd from CDF
#   * frequency drift analysis/ beam veloicty estimation
# STIX -----------
#    * spectrogram creation / bkg removal
#    * spectrogram/lightcurve plotting
#    * combined views of RW/STIX



#package imports
# UTILS
import datetime as dt
from datetime import datetime,time,timedelta
import os
## PLOT
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
## MATH
from scipy.optimize import curve_fit as cfit
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

os.environ["CDF_LIB"] = "/home/localuser/Documents/CDF/src/lib"
from spacepy import pycdf



# CONSTANTS

#date
std_date_fmt = '%d-%b-%Y %H:%M:%S'
speed_c_kms = 299792.458
Rs_per_AU = 215.032
km_per_Rs = 695700.

# rpw indexes
#rpw_suggested_freqs_idx =[437,441,442,448,453,458,465,470,477,482,                     
#                      493,499,511,519,526,533,
#                      538,545,552,559,566,576,588,592,600,612,
#                      656,678,696,716,734,741,750,755,505,629,649,
#                      673,703,727]629,656,673
rpw_suggested_freqs_idx=[ 437,441,442,448,453,458,465,470,
                         477,482,493,499,511,519,526,533,
                         538,545,552,559,566,600,
                         612,678,696,#,576,588,592,649,629,656,673
                         703,716,727,734,741,750,755]
rpw_idx_hfr=436
rpw_suggested_indexes = np.array(rpw_suggested_freqs_idx)-rpw_idx_hfr


# SOLAR EVENTS CLASS
class solar_event:
    def __init__(self,event_type,times,color=None,linestyle="-",linewidth=2,hl_alpha=0.4,paint_in=None,date_fmt=std_date_fmt):
        self.type = event_type
        #interval,stix_flare,rpw_burst
        try:
            self.start_time = dt.datetime.strptime(times['start'],date_fmt)
        except:
            self.start_time = None
        try:
            self.end_time = dt.datetime.strptime(times['end'],date_fmt)
        except:
            self.end_time = None
        try:
            self.peak_time = dt.datetime.strptime(times['peak'],date_fmt)
        except:
            self.peak_time = None
            
           
        #    self.end_time = times['end'] if  times['end'] else None
        #self.peak_time = times['peak'] if  times['peak'] else None
        self.color = color
        self.linestyle=linestyle
        self.linewidth=linewidth
        self.hl_alpha=hl_alpha
        self.paint_in=paint_in
        if(self.paint_in==None):
            if self.type=="rpw_burst":
                self.paint_in = "rpw"
            elif self.type=="stix_flare":
                self.paint_in = "stix"
            else:
                self.paint_in = "both"
        
        
        
    def paint(self):
        if(self.type=="interval"and self.start_time and self.end_time):
            color = self.color if  self.color else "white"
            plt.axvspan(self.start_time, self.end_time, color=color, alpha=self.hl_alpha)
            plt.axvline(self.start_time,c=color,linestyle=self.linestyle,linewidth=self.linewidth)
            plt.axvline(self.end_time,c=color,linestyle=self.linestyle,linewidth=self.linewidth)
        elif(self.type=="stix_flare"):
            color = self.color if  self.color else "white"
            if(self.start_time):
                plt.axvline(self.start_time,c=color,linestyle="--",linewidth=self.linewidth)
            if(self.peak_time):
                plt.axvline(self.peak_time,c=color,linestyle="-",linewidth=self.linewidth)
        elif(self.type=="rpw_burst" and self.peak_time):
            color = self.color if  self.color else "orange"
            plt.axvline(self.peak_time,c=color,linestyle="-",linewidth=self.linewidth)
        elif(self.type=="marker" and self.peak_time):
            color = self.color if  self.color else "magenta"
            plt.axvline(self.peak_time,c=color,linestyle="-",linewidth=self.linewidth)
    


# RPW data rpw

def rpw_read_hfr_cdf(filepath, sensor=9, start_index=0, end_index=-99):
    
    
    #import datetime

    with pycdf.CDF ( filepath ) as l2_cdf_file:

        frequency = l2_cdf_file[ 'FREQUENCY' ][ : ]  # / 1000.0  # frequency in MHz
        nn = np.size ( l2_cdf_file[ 'Epoch' ][ : ] )
        if end_index == -99:
            end_index = nn
        frequency = frequency[ start_index:end_index ]
        epochdata = l2_cdf_file[ 'Epoch' ][ start_index:end_index ]
        sensor_config = np.transpose (
            l2_cdf_file[ 'SENSOR_CONFIG' ][ start_index:end_index, : ]
        )
        agc1_data = np.transpose ( l2_cdf_file[ 'AGC1' ][ start_index:end_index ] )
        agc2_data = np.transpose ( l2_cdf_file[ 'AGC2' ][ start_index:end_index ] )
        sweep_num = l2_cdf_file[ 'SWEEP_NUM' ][ start_index:end_index ]
        cal_points = (
            l2_cdf_file[ 'FRONT_END' ][ start_index:end_index ] == 1
        ).nonzero ()
    frequency = frequency[ cal_points[ 0 ] ]
    epochdata = epochdata[ cal_points[ 0 ] ]
    sensor_config = sensor_config[ :, cal_points[ 0 ] ]
    agc1_data = agc1_data[ cal_points[ 0 ] ]
    agc2_data = agc2_data[ cal_points[ 0 ] ]
    sweep_numo = sweep_num[ cal_points[ 0 ] ]
    ssweep_num = sweep_numo
    timet = epochdata

    # deltasw = sweep_numo[ 1:: ] - sweep_numo[ 0:np.size ( sweep_numo ) - 1 ]
    deltasw = abs ( np.double ( sweep_numo[ 1:: ] ) - np.double ( sweep_numo[ 0:np.size ( sweep_numo ) - 1 ] ) )
    xdeltasw = np.where ( deltasw > 100 )
    xdsw = np.size ( xdeltasw )
    if xdsw > 0:
        xdeltasw = np.append ( xdeltasw, np.size ( sweep_numo ) - 1 )
        nxdeltasw = np.size ( xdeltasw )
        for inswn in range ( 0, nxdeltasw - 1 ):
            # sweep_num[ xdeltasw[ inswn ] + 1:xdeltasw[ inswn + 1 ] ] = sweep_num[
            # xdeltasw[ inswn ] + 1:xdeltasw[ inswn + 1 ] ] + \
            # sweep_numo[ xdeltasw[ inswn ] ]
            sweep_num[ xdeltasw[ inswn ] + 1:xdeltasw[ inswn + 1 ] + 1 ] = sweep_num[
                                                                           xdeltasw[ inswn ] + 1:xdeltasw[
                                                                                                     inswn + 1 ] + 1 ] + \
                                                                           sweep_numo[ xdeltasw[ inswn ] ]
    sens0 = (sensor_config[ 0, : ] == sensor).nonzero ()[ 0 ]
    sens1 = (sensor_config[ 1, : ] == sensor).nonzero ()[ 0 ]
    print("  sensors: ",np.shape(sens0),np.shape(sens1))
    psens0 = np.size ( sens0 )
    psens1 = np.size ( sens1 )

    if (np.size ( sens0 ) > 0 and np.size ( sens1 ) > 0):
        agc = np.append ( np.squeeze ( agc1_data[ sens0 ] ), np.squeeze ( agc2_data[ sens1 ] ) )
        frequency = np.append ( np.squeeze ( frequency[ sens0 ] ), np.squeeze ( frequency[ sens1 ] ) )
        sens = np.append ( sens0, sens1 )
        timet_ici = np.append ( timet[ sens0 ], timet[ sens1 ] )
    else:
        if (np.size ( sens0 ) > 0):
            agc = np.squeeze ( agc1_data[ sens0 ] )
            frequency = frequency[ sens0 ]
            sens = sens0
            timet_ici = timet[ sens0 ]
        if (np.size ( sens1 ) > 0):
            agc = np.squeeze ( agc2_data[ sens1 ] )
            frequency = frequency[ sens1 ]
            sens = sens1
            timet_ici = timet[ sens1 ]
        if (np.size ( sens0 ) == 0 and np.size ( sens1 ) == 0):
            print('  no data at all ?!?')
            V = (321)
            V = np.zeros ( V ) + 1.0
            time = np.zeros ( 128 )
            sweepn_HFR = 0.0
    #           return {
    #               'voltage': V,
    #               'time': time,
    #               'frequency': frequency,
    #               'sweep': sweepn_HFR,
    #               'sensor': sensor,
    #           }
    ord_time = np.argsort ( timet_ici )
    timerr = timet_ici[ ord_time ]
    sens = sens[ ord_time ]
    agc = agc[ ord_time ]
    frequency = frequency[ ord_time ]
    maxsweep = max ( sweep_num[ sens ] )
    minsweep = min ( sweep_num[ sens ] )
    sweep_num = sweep_num[ sens ]

    V1 = np.zeros ( 321 ) - 99.
    V = np.zeros ( 321 )
    freq_hfr1 = np.zeros ( 321 ) - 99.
    freq_hfr = np.zeros ( 321 )
    time = 0.0
    sweepn_HFR = 0.0
    # ind_freq = [(frequency - 0.375) / 0.05]
    ind_freq = [ (frequency - 375.) / 50. ]
    ind_freq = np.squeeze ( ind_freq )
    ind_freq = ind_freq.astype ( int )
    for ind_sweep in range ( minsweep, maxsweep + 1 ):
        ppunt = (sweep_num == ind_sweep).nonzero ()[ 0 ]
        xm = np.size ( ppunt )
        if xm > 0:
            V1[ ind_freq[ ppunt ] ] = agc[ ppunt ]
            freq_hfr1[ ind_freq[ ppunt ] ] = frequency[ ppunt ]
            # print(frequency[ppunt])
        if np.max ( V1 ) > 0.0:
            V = np.vstack ( (V, V1) )
            freq_hfr = np.vstack ( (freq_hfr, freq_hfr1) )
            sweepn_HFR = np.append ( sweepn_HFR, sweep_num[ ppunt[ 0 ] ] )
        V1 = np.zeros ( 321 ) - 99
        freq_hfr1 = np.zeros ( 321 )  # - 99
        if xm > 0:
            time = np.append ( time, timerr[ min ( ppunt ) ] )
    # sys.exit ( "sono qui" )
    V = np.transpose ( V[ 1::, : ] )
    time = time[ 1:: ]
    sweepn_HFR = sweepn_HFR[ 1:: ]
    freq_hfr = np.transpose ( freq_hfr[ 1::, : ] )
    return {
        'voltage': V,
        'time': time,
        'frequency': freq_hfr,
        'sweep': sweepn_HFR,
        'sensor': sensor,
    }


# RPW get data object
def rpw_get_data(file,sensor=9):
    # data read
    print("Extracting info:")
    print("  File: ",os.path.basename(file))
    print("  Type: RPW L2")
    return rpw_read_hfr_cdf(file,sensor=sensor)

    

# RPW filter frequencies
def rpw_select_freq_indexes(frequency,**kwargs):#,freq_col=0,proposed_indexes=None):
    #indexes of frequencies different from 0 or -99 (column 0 in frequency matrix)
    
    fcol = kwargs["freq_col"]
    freq_nozero = np.where(frequency.T[fcol]>0)[0]
    
    selected_freqs = freq_nozero
    
    dfreq = np.array(frequency[selected_freqs,fcol])
    dfreq = dfreq[1:]-dfreq[:-1]
    
    #print("nz",dfreq)
    if kwargs["which_freqs"]=="both":
        selected_freqs = [ j  for j in kwargs["proposed_indexes"] if j in freq_nozero]
        
    if(not kwargs["freq_range"]==None):
        #print(frequency[selected_freqs,fcol])
        selected_freqs = [selected_freqs[j] for j in range(len(selected_freqs)) if np.logical_and(frequency[selected_freqs[j],fcol] <= kwargs["freq_range"][1], frequency[selected_freqs[j], fcol]>=kwargs["freq_range"][0])]
    
    return selected_freqs,frequency[selected_freqs,fcol],dfreq

# create PSD from rpw data object
def rpw_create_PSD(data,freq_range=None,date_range=None,freq_col=0,proposed_indexes=rpw_suggested_indexes,which_freqs="both",rpw_bkg_interval=None):
    # return,x,y,z
    
    time_data = data["time"]
    date_idx = np.arange(len(time_data))
    start_date,end_date = time_data[0],time_data[-1]
    # when date range provided,select time indexes
    if(date_range):
        
        start_date = dt.datetime.strptime(date_range[0], std_date_fmt)
        end_date = dt.datetime.strptime(date_range[1], std_date_fmt)

        date_idx = np.array( np.where( np.logical_and(time_data<=end_date,time_data>=start_date))[0] ,dtype=int)

        if(len(date_idx)==0):
            print("  RPW Error! no data in between provided date range")
            return
    print("  data cropped from ",start_date," to ",end_date)
    # define time axis
    date_idx = np.array(date_idx)
    t_axis = time_data[date_idx]
    
    #define energy axis
    freq_ = data['frequency']
  
    freq_idx,freq_axis,dfreq = rpw_select_freq_indexes(freq_,freq_col=freq_col,freq_range=freq_range,
                                         proposed_indexes=proposed_indexes,which_freqs=which_freqs)
    freq_idx = np.array(freq_idx)
    print("  Selected frequencies [kHz]: ",*freq_axis.astype(int))
    
    # selecting Z axis (cropping)
    z_axis= data["voltage"][:,date_idx]
    z_axis = z_axis[freq_idx,:]
    
# BKG subtraction (approx) if needed
    if rpw_bkg_interval :
        print("  Creating mean bkg from ",rpw_bkg_interval[0]," to ",rpw_bkg_interval[1],"...")
        start_bkg = dt.datetime.strptime(rpw_bkg_interval[0],std_date_fmt)
        end_bkg = dt.datetime.strptime(rpw_bkg_interval[1],std_date_fmt)
        idx_in = [j for j in range(len(t_axis)) if np.logical_and(t_axis[j]>=start_bkg,t_axis[j]<=end_bkg)]

        mn_bkg = np.mean(z_axis[:,idx_in],axis=1)
        mn_bkg = np.array([mn_bkg for i in range(np.shape(z_axis)[1])]).T
        
        z_axis=np.clip(z_axis-mn_bkg,1e-16,np.inf)
        print("  bkg done.")

    
    
    
    return_dict = {
        "t_idx":date_idx,
        "freq_idx":freq_idx,
        "time":t_axis,
        "frequency":freq_axis,
        "v":z_axis,
        "df":dfreq
    }
    
    return return_dict
# plot PRW psd object
def rpw_plot_psd(psd,logscale=True,colorbar=True,cmap="jet",t_format="%H:%M:%S",
            axis_fontsize=13,xlabel=True,frequency_range=None):

    t,f,z=psd["time"],psd["frequency"],psd["v"]
    if(logscale):
        z = np.log10(z)


    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter(t_format))

    #mn = np.mean(z[:,0:1500],axis=1)
    #bckg_ = np.array([mn for i in range(np.shape(z)[1])]).T
    #z_=np.clip(z-bckg_,1e-16,np.inf)

    cm= plt.pcolormesh(t,f,z,shading="auto",cmap=cmap)
    if(colorbar):
        plt.colorbar(cm,label="$Log_{10}$ PSD (V)")

    #plt.axvline(t[np.argmax(z[-1,:])],c="w")
    
    
    ax.set_yscale('log')
    ax.set_yticks([], minor=True)
    ax.set_yticks([x  for x in [0,100,500,1000,2500,5000,10000,15000] if np.logical_and(x<=f[-1],x>=f[0])])
    #ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    #plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y/1000.),1)))).format(y/1000.)))
    
    if(frequency_range):
        plt.ylim(max(frequency_range[0],np.min(f)),min(frequency_range[1],np.max(f)))
    
    0
    if(xlabel):
        plt.xlabel("start time: "+t[0].strftime(std_date_fmt),fontsize=axis_fontsize)
    plt.ylabel("Frequency [MHz]",fontsize=axis_fontsize)



 # STIX data read

def stix_create_counts(pathfile, is_bkg=False,time_arr=None,correct_flight_time=False):
    
    print("Extracting info: ")
    print("  File: ",os.path.basename(pathfile))
    print("  Type: STIX L1","BKG" if is_bkg else "")
    #output  dict
    return_dict = {}
    #fits info
    hdulist = fits.open(pathfile)
    header = hdulist[0].header
    earth_sc_delay = header["EAR_TDEL"] if correct_flight_time else 0
    
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
        
        data_time = Time(header['date_obs']) + TimeDelta(data['time'] * u.s)+TimeDelta(earth_sc_delay * u.s)
        data_time = [t.datetime for t in data_time]
    # counts object, input  for plotting and spectral analysis routines
        return_dict = {"time":data_time,
                   "counts_per_sec":data_counts_per_sec,
                   "energy_bins":energies,
                   "mean_energy":mean_energy}
    return return_dict



def stix_remove_bkg_counts(pathfile,pathbkg,correct_flight_time = False):
    
    #import L1 data
    data_L1 = stix_create_counts(pathfile)
    #import BKG data
    data_BKG = stix_create_counts(pathbkg,is_bkg=True, time_arr=data_L1["time"],correct_flight_time =correct_flight_time )
    
    #subtract background 
    data_counts_per_sec_nobkg = data_L1["counts_per_sec"]-data_BKG["counts_per_sec"] 
    
    # replace ctc/secinfo with corrected info
    return_dict = data_L1.copy()
    return_dict["counts_per_sec"] = data_counts_per_sec_nobkg
    
    return return_dict

def stix_plot_spectrogram(counts_dict,savename=None,colorbar=True,
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

    cts_data = np.log10(cts_per_sec, out=np.zeros_like(cts_per_sec), where=(cts_per_sec>0)).T if logscale else cts_per_sec.T
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
    
    
    
    
# Combined views

def rpw_stix_combined_view(stx_cts,rpw_psd,date_range=None,dt_fmt=std_date_fmt,figsize=[15,9],
                          rpw_freq_range=None,stix_energy_range=None,invert_rpw_axis=True,markers={},markerwidth=1.5,cbars=True,
                          common_interval=True,stix_cmap="jet",rpw_cmap="jet",events=[]):

    
    # select time range datetime
    time_interval = [dt.datetime.strptime(x,dt_fmt) for x in date_range]
    
    common_interval = [max(np.min(stx_cts["time"]),np.min(rpw_psd["time"])),
                      min(np.max(stx_cts["time"]),np.max(rpw_psd["time"]))]
    if(common_interval):
        time_interval = [max(common_interval[0],time_interval[0]),
                        min(common_interval[1],time_interval[1])]
        print("Time axis constrained to common time interval...")
    
    new_date_range = [dt.datetime.strftime(x,dt_fmt) for x in time_interval]
    print("Time interval from",new_date_range[0]," to ",new_date_range[1])
    
    

    fig=plt.figure(figsize=figsize)
    #plt.title("start time "+date_range[0])
    fig.subplots_adjust(hspace=0.09)
    
    # RPW
    ax = plt.subplot(2,1,1)
    plt.title("start time "+new_date_range[0])
    rpw_plot_psd(rpw_psd,xlabel=False,colorbar=cbars,frequency_range=rpw_freq_range,cmap=rpw_cmap)
    if(invert_rpw_axis):
        plt.gca().invert_yaxis()
    plt.xlim(*time_interval)
    for mk in markers:
        ax.axvline(dt.datetime.strptime(mk,dt_fmt),c=markers[mk],lw=markerwidth)
    for ev in events:
        if ev.paint_in in ["both","rpw"]:
            ev.paint()


    #STIX
    ax2 = plt.subplot(2,1,2)
    _=stix_plot_spectrogram(stx_cts,colorbar=cbars,energy_range=stix_energy_range,cmap=stix_cmap)
    for mk in markers:
        _.axvline(dt.datetime.strptime(mk,dt_fmt),c=markers[mk],lw=markerwidth)
    for ev in events:
        if ev.paint_in in ["both","stix"]:
            ev.paint()
    plt.xlim(*time_interval)

    
    
    
### RPW frequency drift

#format model for fit functions
# param 1: factor
# param 2: time shift (avg of distribution)
# param 3: width of distribution
# param 4: additive (free)
def fit_func_gaussian(x,a,b,c,d):
    return (10**a) * np.exp(-(x-b)**2/( c**2)) + (10**d)
    
def estimate_rmse(x,y,model,params):
    rmse = 0
    for i in range(len(x)):
        rmse += (y-model(x,*params))**2
        
    return np.sqrt(np.mean(rmse))

def dt_to_sec_t0(time_data,t0=None):
    if(t0==None):
        t0=time_data[0]
    time_dts = [time_data[i]-t0 for i in range(len(time_data))]
    t_0 = np.array([t.seconds + t.microseconds/1e6 for t in time_dts])
    
    return [t_0,time_data[0]]

def sec_t0_to_dt(secs, t0):
    tmes = [t0 + dt.timedelta(seconds=secs[j]) for j in range(len(secs))]
    return tmes
    
def rpw_fit_freq_peaks(rpw_psd,peak_model,date_range,frequency_range=None,initial_pos_guess=None,excluded_freqs=[],dt_fmt="%d-%b-%Y %H:%M:%S"):
    time_data = rpw_psd["time"]
    v_data = rpw_psd["v"]
    freq_data = rpw_psd["frequency"]
    
    
    #estimate error
    t_err = np.mean(time_data[1:]-time_data[:-1]).seconds/2.
    f_err = (freq_data[1:]-freq_data[:-1])/2.
    print("Estimated uncertainty:")
    print("  Time: {} s".format(t_err))
    print("  Freq: between {}-{} kHz".format(np.min(f_err),np.max(f_err)))
    print("Defining time reference...")
    # selecting fit intervals
    date_range_dt = [dt.datetime.strptime(x,dt_fmt) for x  in date_range]

    idx_sel_time = np.logical_and(time_data<=date_range_dt[1],time_data>=date_range_dt[0])
    
    time_data = time_data[idx_sel_time]
    v_data = v_data[:,idx_sel_time]
    
    if(frequency_range):
        idx_sel_freq = np.logical_and(freq_data<=frequency_range[1],freq_data>=frequency_range[0])
        freq_data = freq_data[idx_sel_freq]
        v_data = v_data[idx_sel_freq,:]
        
    # convering time to seconds
    time_sec,t0 = dt_to_sec_t0(time_data)
    time_span = time_sec[-1]-time_sec[0] 
    print(" t0 = ",dt.datetime.strftime(t0,dt_fmt))
    print("Fitting peaks for {} frequencies between {} kHz and {} kHz".format(len(freq_data),freq_data[0],freq_data[-1]))
    curve_fits = {}
    curve_fits_meta = {}
    #asume that peak of max freq. is close to the beggining of the timespan
    prev_center = None
    if(initial_pos_guess):
        prev_center = dt_to_sec_t0([dt.datetime.strptime(pos_guess,dt_fmt)],t0)[0][0]
    # starting point (time,freq)
    starting_point =[]
    for i in range(len(freq_data)-1,-1,-1): 
        
        if(freq_data[i] in excluded_freqs):
            print("[{}] {:.0f} kHz   : Excluded!   omitted.".format(i,freq_data[i]))
            continue
        
        
        #if not defined, use approx position in timespan (lineal)
        if(not prev_center):
            prev_center= time_sec[0]    
        
             
        x_ = time_sec
        y_ = v_data[i,:]  #V[freqn,date_idx]
        curve_fits_meta={
                         "t":x_,
                         "y":y_,
                         "t0":t0,
                         "dt":t_err,
                         "time_interval":date_range,
                         "excluded_f":excluded_freqs
                        }
        #fit_bounds = [(1e-18,1e-11),(0,np.max(t_sec0)),(1,1000),(1e-18,1)]
        #fit_bounds = ((-18.,0.,1.,-18.),(-14.,np.max(x_),time_span,-14.))
        try:
            #if(p0):
            
            init_guess = [np.log10(np.max(y_)),prev_center,60.,-16.]
            popt,pcov = cfit(peak_model,x_,y_,p0=init_guess, method="lm")#,bounds=fit_bounds)
       
            # for cases where aprameters were found
            if (len(popt)>0) :
                # diference with previous found point
                dif = popt[1]-prev_center
                if(i==len(freq_data)-1):
                    dif = 0
                # discard if center out of bounds
                if(popt[1]<x_[0]*0.8 or popt[1]>x_[-1]*1.2):
                    print("[{}] {:.0f} kHz   : Not in bounds! omitted.".format(i,freq_data[i]))
                    #popt= []
                    #pcov = []
                else:
                    if(len(starting_point)==0):
                        starting_point = [freq_data[i],sec_t0_to_dt([popt[1]],t0)[0]]
                        sp_t=dt.datetime.strftime(starting_point[1],dt_fmt)
                        #curve_fits_meta={
                         #"t":x_,
                         #"y":y_,
                         #"t0":t0,
                         #"dt":t_err,
                         #"df":f_err[i],
                         #"time_interval":date_range,
                         #"excluded_f":excluded_freqs
                        #}
                        print("Starting point ---------- frequency: {:.0f}+-{:.0f} kHz   time: {}".format(freq_data[i],f_err[i],sp_t))
                    
                    
                    # FITTING ERROR
                    rmse = estimate_rmse(x_,y_,peak_model,popt)
                    snr = 10**(popt[0]-popt[3])
                    
                    ## ADD CURVE FIT TO SOLUTIONS
                    curve_fits[freq_data[i]] = {
                     "params":popt,
                     "covar":pcov,
                     "rmse":rmse,
                     "snr":snr,
                     "df":f_err[i]}
                    
                    
                    print("[{}] {:.0f} kHz: Fit found!   t-t0: {:.2f} s   Dif.: {:.2f} s  Log10(RMSE): {:.2f}  Log10(S/N): {:.2f}".format(i,freq_data[i],popt[1],dif,np.log10(rmse),np.log10(snr)))
                    
                    prev_center = popt[1]#np.mean([time_span * (1-i/len(freq_data))**2,popt[1]])
            
        except:
            print("[{}] {:.0f} kHz   : Not found".format(i,freq_data[i]))
            #popt= []
            #pcov=[]
        
    #print(curve_fits.keys()) 
    fit_results ={
        "freq_fits":curve_fits,
        "metadata":curve_fits_meta
    }
    
    return fit_results
def rpw_plot_fit_results(fit_results,rpw_psd,cmap="jet",fit_limits=False):
    dt_fmt="%d-%b-%Y %H:%M:%S"
    cmap = mpl.cm.get_cmap(cmap)
    
    c_fits=fit_results["freq_fits"]
    meta = fit_results["metadata"]
    
    flist = list(c_fits.keys())
    if(fit_limits):
        frequency_range=[int(flist[0]),int(flist[-1])]
        dt_range=sec_t0_to_dt(meta["t"],meta["t0"])
        dt_range = [dt_range[0]-dt.timedelta(seconds=20),dt_range[-1]+dt.timedelta(seconds=20)]
        dt_range = [dt.datetime.strftime(x,dt_fmt)for x in dt_range]
        
        rpw_plot_psd(rpw_psd,cmap="binary")#,frequency_range=frequency_range)
        plt.gca().invert_yaxis()
        solar_event(event_type="interval",times={'start':dt_range[0],'end':dt_range[1]},color="blue",linewidth=0.5,hl_alpha=0.2).paint()
        #plt.xlim(dt_range)
    else:
        rpw_plot_psd(rpw_psd,cmap="binary")
    #print("frnge",frequency_range)
    
    for i in range(len(flist)):
        695700
        params = c_fits[flist[i]]["params"]
        covars = c_fits[flist[i]]["covar"]
        if(len(params)>0):
            
            t0 = meta["t0"]
            ctime = sec_t0_to_dt([params[1]],t0=t0)[0]
            times_sigma1= sec_t0_to_dt([params[1]-params[2],params[1]+params[2]],t0=t0)
            times_sigma1 = times_sigma1[1]-times_sigma1[0]
            ydat = meta["y"]
            f_sigma = c_fits[flist[i]]["df"]
        
            rgba = cmap(i/len(flist))
            #delays.append(c_fits[flist[i]]["params"][1])
            #freqs.append(flist[i])
            lbl = "{} MHz".format(round(flist[i]/1000.,2))
            if(len(flist)>20 and i%2!=0 and i!=len(flist)-1):
                lbl=None
            plt.errorbar(ctime,int(flist[i]),xerr=times_sigma1,yerr=f_sigma,color=rgba,
                         label=lbl,marker="o",markersize=3)
    plt.legend(ncol=3,fontsize=8)
    plt.xlabel("Date")
    plt.ylabel("Frequency [MHz]")
    
    

def rpw_freq_drifts(fit_results,excluded_freqs=[],):
    peak_fits =fit_results["freq_fits"]
    meta = fit_results["metadata"]
    flist = np.array(list(peak_fits.keys()))
    # peak times
    delays = []
    # frequencies
    freqs = []
    # peak time unceertainty
    devs = []
    # freq uncertainty
    dfs = []
    #time uncertainty695700
    dts = []
    
    
    for i in range(len(flist)):
        params_ =peak_fits[flist[i]]["params"]
        covs_ =peak_fits[flist[i]]["covar"]
        if(len(params_)>0 and not int(flist[i]) in excluded_freqs ):
            delays.append(params_[1])
            freqs.append(flist[i])
            #error
            #devs.append(np.sqrt(np.diag(covs_)[1]))
            dts.append(np.mean(meta["dt"])/2.)
            devs.append(np.sqrt(np.abs(params_[2])))#(2.*np.mean(peak_fits[flist[i]]["dt"])))

            dfs.append(peak_fits[flist[i]]["df"])
            #plt.scatter(int(flist[i]),cf[flist[i]]["params"][1], label=i)
    
    delays = np.array(delays)
    devs = np.abs(np.array(devs))
    dfs = np.abs(np.array(dfs))
    freqs = np.array(freqs)
   
    #f drift estimationsololab.
    dif_freqs = freqs[1:]-freqs[:-1]
    dif_delays = delays[1:]-delays[:-1]
    f_drifts = dif_freqs/dif_delays

    err_delays = devs[:-1]#np.sqrt((devs[1:]**2) + (devs[:-1]**2))
    err_freqs =  dfs[:-1]#np.sqrt((dfs[1:]**2) + (dfs[:-1]**2))
    
    #print(err_delays[:],dif_delays[:],err_freqs[:],dif_freqs[:])
    err_fdrift = np.abs( f_drifts[:]*np.sqrt((err_delays[:]/dif_delays[:])**2 + (err_freqs[:]/dif_freqs[:])**2) )

    return_dict = {"frequencies":flist,
                   "conv_frequencies": freqs,
                   "delays":delays,
                   "freq_drifts":f_drifts,
                   "sigma_dfdt":err_fdrift,
                   "sigma_f" :err_freqs,
                   "sigma_tpeak":err_delays,
                   "sigma_t":dts}
    #print([(len(return_dict[x]),x) for x in list(return_dict.keys())])
    
    #print(f_drifts/1000)
    return  return_dict
def rpw_plot_freq_drift(freq_drifts,errorbars=False,limit_cases=True):
    
    freqs = freq_drifts["conv_frequencies"]
    f_drifts = freq_drifts["freq_drifts"]
    
    maxyerr=np.mean([t for t in freq_drifts["sigma_dfdt"] if np.abs(t)!=np.inf ])
    ax = plt.gca()
    interval_centers=(freqs[1:]+freqs[:-1])/2
    fdrifts_mhz = f_drifts/1000. #in MHz s-1
    for i in range(len(fdrifts_mhz)):
        col = "red" if fdrifts_mhz[i]<0 else "blue"
        yerr=freq_drifts["sigma_dfdt"][i]/1000. #in MHz s-1
        
        yerr = yerr if yerr!=np.inf else maxyerr/1000.
        if(limit_cases and yerr/np.abs(fdrifts_mhz[i])>=1):
            continue
        #print(yerr/np.abs(fdrifts_mhz[i]))
        
        
        ax.scatter(interval_centers[i],np.abs(fdrifts_mhz[i]),c=col,s=8)
        
        bar_alpha=0.3
        if(errorbars=="both"):
            markers, caps, bars = ax.errorbar(interval_centers[i],np.abs(fdrifts_mhz[i]),xerr=freq_drifts["sigma_f"][i],yerr=yerr,c=col,markersize=3,marker="o")
            [bar.set_alpha(bar_alpha) for bar in bars]
        elif(errorbars=="x"):
            markers, caps, bars = ax.errorbar(interval_centers[i],np.abs(fdrifts_mhz[i]),xerr=freq_drifts["sigma_f"][i],c=col,markersize=3,marker="o")
            [bar.set_alpha(bar_alpha) for bar in bars]
        elif(errorbars=="y"):
            markers, caps, bars = ax.errorbar(interval_centers[i],np.abs(fdrifts_mhz[i]),rerr=yerr,c=col,markersize=3,marker="o")
            [bar.set_alpha(bar_alpha) for bar in bars]
            
            
    
    neg_patch = mpatches.Patch(color="red", label='df/dt < 0')
    pos_patch = mpatches.Patch(color="blue", label='df/dt > 0')
    plt.legend(handles=[neg_patch,pos_patch],fontsize=8)
    
   # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    
    #for i in range(len(interval_centers)):
        #plt.text(interval_centers[i],fdrifts_mhz[i],str(int(interval_centers[i]))+" MHz",
        #         fontsize=10,horizontalalignment="center",verticalalignment="bottom")
    #plt.ylim(1,5)
    plt.yscale("log")
    plt.xscale("log",subs=[1,2,3,4,5,6,7,8,9])
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Frequency drift rate  [MHz sec$^{-1}$]")
    

def rpw_plot_fit_summary(rpw_psd,fit_results,freq_drifts,fit_limits=True,savepath=None,grid=True,errorbars=False):

    curve_fits = fit_results["curve_fits"]
    cf_meta = fit_results["metadata"]
    fit_interval = cf_meta["time_interval"]
    
    fig=plt.figure(figsize=(16,4),dpi=120)
    spec3 = gridspec.GridSpec(ncols=4, nrows=1)

    fig.add_subplot(spec3[0, 1:])
    rpw_plot_fit_results(curve_fits,rpw_psd,fit_limits=fit_limits)
    
    interv_times = curve_fits
    
    solar_event(event_type="interval",times={'start':fit_interval[0],'end':fit_interval[1]},hl_alpha=0.4).paint()
    fig.add_subplot(spec3[0,0])
    if(grid):
        plt.grid(which='both',color="#EEEEEE")
        plt.grid(color="#CCCCCC")
    rpw_plot_freq_drift(freq_drifts,errorbars)

    
    
    fig.tight_layout()
    if(savepath):
        plt.savefig(savepath,bbox_inches='tight')
        
    
# receivesfrequenciesin kHz
# returns plasma number density in cm-3
def ne_from_freq(freqs,coef=9.):
    return [(f/coef)**2 for f in freqs]

def freq_from_ne(ne,coef=9.):
    return [coef*np.sqrt(n) for n in ne]

def dfdn_from_ne(ne,coef=9.):
    return[(coef/2)*(1./np.sqrt(n)) for n in ne]
def ne_from_r_leblanc(radius):
    return [ 3.3e5*(r**(-2))+4.1e6*(r**(-4))+8.0e7*(r**(-6)) for r in radius]

def r_from_ne(nes,ne_model,r_interv=[1,400],n_iter=5,c=0.1,npoints=1000,error=False):
    
    r_mins = []
    r_err= []
    for  n in nes:
        bounds = r_interv.copy()
        

        for i in range(n_iter):
            
            r_span = np.linspace(bounds[0],bounds[1],npoints)
            r_span_l = bounds[1]-bounds[0]
            ne_span = ne_model(r_span)
            
            r_min = r_span[np.argmin(np.abs(np.array(ne_span)-n))]
            bounds =[max(r_min-c*r_span_l,r_interv[0]),min(r_min+c*r_span_l,r_interv[1])]
        r_err.append(bounds[1]-bounds[0])
        r_mins.append(r_min)
    if(error):
        return r_mins,r_err
    return r_mins
def r_from_freq(freqs,ne_model):
    return r_from_ne(ne_from_freq(freqs),ne_model)

def freq_from_r(r,ne_model):
    return freq_from_ne(ne_model(r))
def dndr_from_r(radius):
    return [ -6.6e5*(r**(-3))-16.4e6*(r**(-5))+48.0e7*(r**(-7)) for r in radius]

def convert_RoSec_to_c(vels):
    #c = 299792.458
    #conv_fact=695700  #695700km = 1 R0
    
    return [(v*km_per_Rs)/speed_c_kms for v in vels]

def convert_c_to_roSec(vels):
    #c = 299792.458
    #conv_fact= 695700  #695700km = 1 R0
    return [(v*speed_c_kms)/km_per_Rs for v in vels]


def rpw_estimate_beam_velocity(freq_drifts,density_model,r_interv=[0.1,300],n_iter=5,c=0.01,npoints=1000,only_neg_drifts=True):
    freqs = freq_drifts["conv_frequencies"]
    freqs_low_bound = freqs[:-1]
    freqs = (freqs[1:]+freqs[:-1])/2.
    

    delays = freq_drifts["delays"]
    delays = (delays[1:]+delays[:-1])/2.
    
    dfdt = freq_drifts["freq_drifts"]
    
    dt = freq_drifts["sigma_t"]

    if(only_neg_drifts):
        iidx = dfdt<0
        dfdt = dfdt[iidx]
        freqs = freqs[iidx]
        freqs_low_bound = freqs_low_bound[iidx]
        delays = delays[iidx]
    
    
    
    n_e = ne_from_freq(freqs)
    #print(len(n_e),len(freq_drifts["sigma_f"][:len(freqs)]),len(freqs))
    err_ne = n_e[:]*((2/9)*freq_drifts["sigma_f"][:len(freqs)]/freqs[:])#*((1/6)*(freq_drifts["sigma_f"][:len(freqs)]/freqs[:]))
    
    
    rads,err_r = r_from_ne(n_e,density_model,r_interv=r_interv,n_iter=n_iter,c=c,npoints=npoints,error=True)
    err_r = err_r[:] + np.array(rads[:])*(err_ne[:]/n_e[:])
    
    dfdn=dfdn_from_ne(n_e)
    err_dfdn = dfdn[:]*(freq_drifts["sigma_f"][:len(freqs)]/freqs[:])*np.sqrt(1+(1/36))
    
    dndr=dndr_from_r(rads)
    err_dndr = dndr[:]*(err_ne[:]/n_e[:])*np.sqrt(1+(1/(2*3.3e5+4*4.1e6+6*8.0e7)**2))
    
    drdt = []
    drdt_err=[]
    for i in range(len(dfdt)):
        if(dfdt[i]>0):
            continue
        v_trig = dfdt[i]*(dndr[i] **(-1) )*( dfdn[i]**(-1) )
        err_v = v_trig*np.sqrt((freq_drifts["sigma_dfdt"][i]/dfdt[i])**2 + (err_dfdn[i]/dfdn[i])**2 + (err_dndr[i]/dndr[i])**2)
        
        drdt.append( v_trig )
        drdt_err.append( err_v)
        
        
    return_dict = {
        "frequencies":freq_drifts["conv_frequencies"],
        "freq_average":freqs,
        "freq_low_bound":freqs_low_bound,
        "delays":delays,
        "n_e":n_e,
        "r":rads,
        "dfdt":dfdt,
        "drdt":drdt,
        "dndr":dndr,
        "dfdn":dfdn,
        "err_drdt":drdt_err,
        "err_dndr":err_dndr,
        "err_n_e":err_ne,
        "err_r":err_r,
        "dt":dt[:len(freqs)]
        
    }
    #print([(len(return_dict[x]),x) for x in list(return_dict.keys())])
    
    return return_dict
    
    
    
def rpw_plot_typeIII_diagnostics(rpw_psd,fit_results,freq_drifts,trigger_velocity,figsize=(14,15),errorbars="both",dfdt_errorbars="both",grid=True,fit_limits=False,cmap="jet"):
    
   # print(freq_drifts["conv_frequencies"])

    peak_fits=fit_results["freq_fits"]
    pf_meta = fit_results["metadata"]
    fit_interval = pf_meta["time_interval"]
    
    t0 = pf_meta["t0"]
    
    timeax = sec_t0_to_dt(trigger_velocity["delays"],t0)
    
    # TIME error
    delta_t = trigger_velocity["dt"]
    delta_t_dt = [dt.timedelta(seconds=t) for t in delta_t]
    
    terr = freq_drifts["sigma_tpeak"]
    terr_dt=[dt.timedelta(seconds=t) for t in terr]
    
    r_err = trigger_velocity["err_r"]
    f_err = freq_drifts["sigma_f"]
    

    cmap = mpl.cm.get_cmap(cmap)
    # create figure and grid
    fig=plt.figure(figsize=figsize,dpi=120)
    spec3 = gridspec.GridSpec(ncols=4, nrows=9,wspace=0.4,hspace=0.)
    
    # PLOT SPECTROGRAM
    ax=fig.add_subplot(spec3[:2, 1:])
    
    rpw_plot_fit_results(fit_results,rpw_psd,fit_limits=fit_limits)
    plt.gca().invert_yaxis()
    solar_event(event_type="interval",times={'start':fit_interval[0],'end':fit_interval[1]},hl_alpha=0.4,color="#AADDAA").paint()
    ax.xaxis.set_label_position('top')
    #ax.xaxis.tick_top()
    plt.xlabel("start time: {}".format(dt.datetime.strftime(rpw_psd["time"][0],"%d-%b-%Y %H:%M:%S")))
    #PLOT FREQ. DRIFTS
    ax=fig.add_subplot(spec3[:2,0])
    if(grid):
        plt.grid(which='both',color="#EEEEEE")
        plt.grid(color="#CCCCCC")
    rpw_plot_freq_drift(freq_drifts,errorbars)
    ax.xaxis.set_label_position('top')
    #ax.xaxis.tick_top()
    
    
    # VELOCITY DIAGNOSTICS
    vels = convert_RoSec_to_c(trigger_velocity["drdt"])
    err_vels = convert_RoSec_to_c(trigger_velocity["err_drdt"])
    
    
    
    # PLOT DIAGNOSTICS VS TIME
    ax=fig.add_subplot(spec3[3:4,:2])
    for f_i in range(len(trigger_velocity["freq_average"])):
        if(vels[f_i]<0):
            err_vels[f_i]=0
        rgba = cmap((f_i+1)/len(trigger_velocity["freq_average"]))
        ax.scatter(trigger_velocity["delays"][f_i],vels[f_i],color=rgba,marker="o",s=20)
        ax.errorbar(trigger_velocity["delays"][f_i],vels[f_i],xerr=terr[f_i]+delta_t[f_i],yerr=err_vels[f_i],c=rgba,markersize=2)
    
    mean_selected = np.mean([x for x in vels if (x>0 and x<=1) ])
    std_selected = np.std([x for x in vels if (x>0 and x<=1) ])
    ax.axhline(mean_selected,c="grey",linestyle="--",label="average = {:.2f} +- {:.2f}c".format(mean_selected,std_selected))
    ax.axhline(1,c="r",linestyle="--",label="speed of light")
   
    ax.set_xlabel("$t-t_0$ [sec] ",fontsize=13)
    ax.set_ylabel("v/c",fontsize=12)
    
    ax.xaxis.tick_top()
    plt.ylim(1e-3,10)
    ax.set_yscale("log")
    ax.xaxis.set_label_position('top')
    #ax.legend()
    
    
    ax2=fig.add_subplot(spec3[4:5,:2])
    ne_err = trigger_velocity["err_n_e"]
    for f_i in range(len(trigger_velocity["freq_average"])):
        rgba = cmap((f_i+1)/len(trigger_velocity["freq_average"]))
        ax2.scatter(timeax[f_i],trigger_velocity["n_e"][f_i],color=rgba,marker="o",s=20)
        ax2.errorbar(timeax[f_i],trigger_velocity["n_e"][f_i],yerr=ne_err[f_i],xerr=terr_dt[f_i]+delta_t_dt[f_i],c=rgba)
    ax2.set_yscale("log")

    ax2.set_ylabel("$n_e$ [cm$^{-3}$]",fontsize=12)
    ax2.set_xticks([])
    
    fig.add_subplot(spec3[5:6,:2])
    for f_i in range(len(trigger_velocity["freq_average"])):
        rgba = cmap((f_i+1)/len(trigger_velocity["freq_average"]))
        plt.scatter(timeax[f_i],trigger_velocity["r"][f_i],color=rgba,marker="o",s=20,label="{} kHz".format(int(trigger_velocity["frequencies"][f_i])))
        plt.errorbar(timeax[f_i],trigger_velocity["r"][f_i],xerr=terr_dt[f_i]+delta_t_dt[f_i],yerr=r_err[f_i],c=rgba,markersize=2)
    plt.xlabel("Time (UT)  $t_0$ = {}".format(dt.datetime.strftime(t0,"%d-%b-%Y %H:%M:%S")),fontsize=12)
    plt.ylabel("r $[R_o]$",fontsize=12)
    plt.xticks(rotation=45)
    #plt.legend(fontsize=7,ncol=4)
    
    
    
    # PLOT DIAGNOSTICS VS R
 
    ax=fig.add_subplot(spec3[3:4,2:])
    for f_i in range(len(trigger_velocity["freq_average"])):
        rgba = cmap((f_i+1)/len(trigger_velocity["freq_average"]))
        r_AU = trigger_velocity["r"][f_i]*(1/Rs_per_AU)
        r_err_AU = r_err[f_i]*(1/Rs_per_AU)
        if(vels[f_i]<0):
            err_vels[f_i]=0
        ax.errorbar(r_AU,vels[f_i],yerr=err_vels[f_i],xerr=r_err_AU,c=rgba,marker="o",markersize=2)
        
        ax.scatter(r_AU,vels[f_i],color=rgba,marker="o",s=20)
    plt.yscale("log")
    
    mean_selected = np.mean([x for x in vels if (x>0 and x<=1) ])
    std_selected = np.std([x for x in vels if (x>0 and x<=1) ])
    plt.axhline(mean_selected,c="grey",linestyle="--",label="average = {:.2f} c".format(mean_selected))
    plt.axhline(1,c="r",linestyle="--",label="speed of light")
    plt.xlabel("r $[AU]$",fontsize=13)
    plt.ylabel("v/c",fontsize=12)
    plt.ylim(1e-3,10)
    ax.xaxis.tick_top()
   # ax.yaxis.tick_right()
    ax.xaxis.set_label_position('top')
   # ax.yaxis.set_label_position('right')
    
    plt.legend(fontsize=9)
    
    ax2=fig.add_subplot(spec3[4:5,2:])
    for f_i in range(len(trigger_velocity["freq_average"])):
        rgba = cmap((f_i+1)/len(trigger_velocity["freq_average"]))
        ax2.errorbar(trigger_velocity["r"][f_i],trigger_velocity["n_e"][f_i],yerr=ne_err[f_i],xerr=r_err[f_i],c=rgba)
        ax2.scatter(trigger_velocity["r"][f_i],trigger_velocity["n_e"][f_i],color=rgba,marker="o",s=20)
    plt.yscale("log")

    plt.ylabel("$n_e$ [cm$^{-3}$]",fontsize=12)
    #ax2.yaxis.tick_right()
    #ax2.set_xticks([])
   # ax2.yaxis.set_label_position('right')
    

    #print(trigger_velocity["freq_average"])
    ax3=fig.add_subplot(spec3[5:6,2:],)
    for f_i in range(len(trigger_velocity["freq_low_bound"])):
        #print(freq_drifts["conv_frequencies"][f_i])
        rgba = cmap((f_i+1)/len(trigger_velocity["freq_low_bound"]))
        lbl = "{} MHz".format(round(trigger_velocity["freq_low_bound"][f_i]/1000.,2))
        if(len(trigger_velocity["freq_average"])>15 and f_i%2!=0 and f_i!=len(trigger_velocity["freq_average"])-1):
                lbl=None
        plt.scatter(trigger_velocity["r"][f_i],trigger_velocity["freq_average"][f_i],color=rgba,marker="o",s=20,label=lbl)
        ax3.errorbar(trigger_velocity["r"][f_i],trigger_velocity["freq_average"][f_i],xerr=r_err[f_i],yerr=f_err[f_i],c=rgba)
    plt.xlabel("r $[R_o]$",fontsize=12)
    plt.ylabel("Frequency [Hz]",fontsize=12)
    plt.yscale("log")
    plt.legend(fontsize=7,ncol=3)
    #plt.yscale("log")
    #plt.xscale("log")
   # ax3.yaxis.tick_right()
    #ax3.set_xticks([])
   # ax3.yaxis.set_label_position('right')
    


    




 