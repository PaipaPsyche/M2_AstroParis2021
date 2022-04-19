# -*- coding: utf-8 -*-

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





# RPW data read

def rpw_read_hfr_cdf(filepath, sensor=9, start_index=0, end_index=-99):
    import os
    os.environ["CDF_LIB"] = "/home/localuser/Documents/CDF/src/lib"
    from spacepy import pycdf
    import numpy as np
    import datetime

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
    if kwargs["which_freqs"]=="both":
        selected_freqs = [ j  for j in kwargs["proposed_indexes"] if j in freq_nozero]
        
    if(not kwargs["freq_range"]==None):
        #print(frequency[selected_freqs,fcol])
        selected_freqs = [selected_freqs[j] for j in range(len(selected_freqs)) if np.logical_and(frequency[selected_freqs[j],fcol] <= kwargs["freq_range"][1], frequency[selected_freqs[j], fcol]>=kwargs["freq_range"][0])]
    
    return selected_freqs,frequency[selected_freqs,fcol]

# create PSD from rpw data object
def rpw_create_PSD(data,freq_range=None,date_range=None,freq_col=0,proposed_indexes=None,which_freqs="both"):
    # return,x,y,z
    
    time_data = data["time"]
    date_idx = np.arange(len(time_data))
    start_date,end_date = time_data[0],time_data[-1]
    # when date range provided,select time indexes
    if(date_range):
        
        start_date = dt.datetime.strptime(date_range[0], '%d-%b-%Y %H:%M:%S')
        end_date = dt.datetime.strptime(date_range[1], '%d-%b-%Y %H:%M:%S')

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
    freq_idx,freq_axis = rpw_select_freq_indexes(freq_,freq_col=freq_col,freq_range=freq_range,
                                         proposed_indexes=proposed_indexes,which_freqs=which_freqs)
    freq_idx = np.array(freq_idx)
    print("  Selected frequencies: ",*freq_axis)
    
    # selecting Z axis (cropping)
    z_axis= data["voltage"][:,date_idx]
    z_axis = z_axis[freq_idx,:]
    
    
    return_dict = {
        "t_idx":date_idx,
        "freq_idx":freq_idx,
        "time":t_axis,
        "frequency":freq_axis,
        "v":z_axis
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
    ax.set_yticks([x  for x in [0,100,250,500,750,1000,2500,5000,7500,10000,12500,15000,20000] if np.logical_and(x<=f[-1],x>=f[0])])
    #ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    #plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    
    if(frequency_range):
        plt.ylim(max(frequency_range[0],np.min(f)),min(frequency_range[1],np.max(f)))
    
    
    if(xlabel):
        plt.xlabel("start time: "+t[0].strftime("%d-%b-%Y %H:%M:%S"),fontsize=axis_fontsize)
    plt.ylabel("Frequency [kHz]",fontsize=axis_fontsize)



 # STIX data read

def stix_create_counts(pathfile, is_bkg=False,time_arr=None):
    
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



def stix_remove_bkg_counts(pathfile,pathbkg):
    
    #import L1 data
    data_L1 = stix_create_counts(pathfile)
    #import BKG data
    data_BKG = stix_create_counts(pathbkg,is_bkg=True, time_arr=data_L1["time"])
    
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

def rpw_stix_combined_view(stx_cts,rpw_psd,date_range=None,dt_fmt="%d-%b-%Y %H:%M:%S",figsize=[15,9],
                          rpw_freq_range=None,stix_energy_range=None,invert_rpw_axis=True,markers={},markerwidth=1.5,cbars=True,
                          common_interval=True):

    
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
    rpw_plot_psd(rpw_psd,xlabel=False,colorbar=cbars,frequency_range=rpw_freq_range)
    if(invert_rpw_axis):
        plt.gca().invert_yaxis()
    plt.xlim(*time_interval)
    for mk in markers:
        ax.axvline(dt.datetime.strptime(mk,dt_fmt),c=markers[mk],lw=markerwidth)


    #STIX
    ax2 = plt.subplot(2,1,2)
    _=stix_plot_spectrogram(stx_cts,colorbar=cbars,energy_range=stix_energy_range)
    for mk in markers:
        _.axvline(dt.datetime.strptime(mk,dt_fmt),c=markers[mk],lw=markerwidth)
    plt.xlim(*time_interval)
