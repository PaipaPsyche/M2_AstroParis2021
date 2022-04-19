#package imports
# UTILS
from datetime import datetime,time,timedelta
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



## QL plots
def plot_counts_per_energy(counts_dict,ax,savename=None,
                      fmt=" %H:%M:%S\n %d-%m-%y",title=None,e_range=None,
                      date_range=None,n_ticks=10,legend=True,
                      integrate_bins=None,logscale=True):
    #get data   
    energies = counts_dict["energy_bins"]
    plot_time = counts_dict["time"].plot_date
    cts_per_sec = counts_dict["counts_per_sec"]
    # date range to Time object -> plot_date 
    if not date_range==None:
        date_range = [Time(x).plot_date for x in date_range]

    # date format
    myFmt = mdates.DateFormatter(fmt)
    
    if(date_range!=None):
        #max val for ylim
        maxval = 0
        # sum for energy bin counts integration
        cts_sum = []
        
        for i in range(len(energies)):
            # if energy in range 
            if e_range == None or (e_range != None and energies[i]["e_low"]>=e_range[0] and energies[i]["e_high"]<=e_range[1]):
                
                # if not integratin plot every bin
                if(not integrate_bins): 
                    ax.plot_date(plot_time, cts_per_sec[:,i],'-', label=f'{energies[i]["e_low"]}-{energies[i]["e_high"]} keV')
                    maxval = np.max(cts_per_sec[:,i]) if np.max(cts_per_sec[:,i])>maxval else maxval
                
                else:
                    # in integrating, sum counts to cts_sum array
                    cts_sum.append(cts_per_sec[:,i])
        # for integrate: sum over collected arrays and evaluate maxval
        if(integrate_bins):
            cts_sum_ = np.sum(np.array(cts_sum),axis=0)
            maxval = 2*np.max(cts_sum_)
            ax.plot_date(plot_time, cts_sum_,'-', label="Sumed energy bins")
        plt.xticks(np.linspace(date_range[0], date_range[1], n_ticks))
        plt.xlim(date_range[0],date_range[1])
        plt.ylim(1,2*maxval)

        if(savename):
            plt.savefig(savename,bbox_inch="tight")
    else:
        # maxval for ylim
        maxval = 0
        # plot energies  in range if provided
        for i in range(len(energies)):
            if e_range == None or (e_range != None and energies[i]["e_low"]>=e_range[0] and energies[i]["e_high"]<=e_range[1]):
                ax.plot_date(plot_time, cts_per_sec[:,i],'-', label=f'{energies[i]["e_low"]}-{energies[i]["e_high"]} keV')
                maxval = np.max(cts_per_sec[:,i]) if np.max(cts_per_sec[:,i])>maxval else maxval
        plt.xticks(np.linspace(min(plot_time), max(plot_time), n_ticks))
    # format date
    ax.xaxis.set_major_formatter(myFmt)
    #title
    if(title):
        plt.title(title)
    #labels
    plt.xlabel("Date",fontsize=14)
    plt.ylabel('Counts $s^{-1}$',fontsize=14)
        
    plt.ylim(1,2*maxval)
    # apply logscale if requested
    if(logscale and maxval>1):
        plt.yscale('log')
    plt.grid()
    if(legend):
        plt.legend(loc='lower center',bbox_to_anchor=(.5, -0.55),ncol=8,fontsize=13)
    if(savename):
        plt.savefig(savename,bbox_inch="tight")
        

        


def plot_energy_bins (counts_dict,savename=None,
                      fmt=" %H:%M:%S\n %d-%m-%y",title=None,e_range=None,
                      date_range=None,zoom=False,
                      integrate_bins=None,logscale=True):
    
    energies = counts_dict["energy_bins"]
    plot_time = counts_dict["time"].plot_date
    cts_per_sec = counts_dict["counts_per_sec"]
    if not date_range==None:
        date_range = [Time(x).plot_date for x in date_range]
    #print(date_range)
    
    
    # date format
    myFmt = mdates.DateFormatter(fmt)
    
    if(zoom and date_range!=None):
        fig, ax = plt.subplots(figsize=(20,5))
        
        # max value for plot limits
        maxval = 0
        
        for i in range(len(energies)):
            ax.plot_date(plot_time, cts_per_sec[:,i],'-', label=f'{energies[i]["e_low"]}-{energies[i]["e_high"]} keV')
            #update max val
            maxval = np.max(cts_per_sec[:,i]) if np.max(cts_per_sec[:,i])>maxval else maxval
        #date fomat x axis
        ax.xaxis.set_major_formatter(myFmt)
        # title if available
        if(title):
            plt.title(title)
        # if 
        if(not integrate_bins == None):
            plt.legend(loc='lower center',bbox_to_anchor=(.5, -0.55),ncol=8,fontsize=13)
        plt.axvline(date_range[0],linestyle="--",color="k")
        plt.axvline(date_range[1],linestyle="--",color="k")
        plt.xlabel("Date",fontsize=14)
        plt.ylabel('Counts $s^{-1}$',fontsize=14)
        plt.ylim(1,2*maxval)
        plt.xticks(np.linspace(min(plot_time), max(plot_time), 15))
        if(logscale and maxval>1):
            plt.yscale('log')
        plt.grid()
        
        
        plt.show()
        
        
        fig, ax = plt.subplots(figsize=(20,5))
        
        maxval = 0
        cts_sum = []
        for i in range(len(energies)):
           
            if e_range == None or (e_range != None and energies[i]["e_low"]>=e_range[0] and energies[i]["e_high"]<=e_range[1]):
                
                if(not integrate_bins):
                    
                    ax.plot_date(plot_time, cts_per_sec[:,i],'-', label=f'{energies[i]["e_low"]}-{energies[i]["e_high"]} keV')
                    maxval = np.max(cts_per_sec[:,i]) if np.max(cts_per_sec[:,i])>maxval else maxval
                else:
                    cts_sum.append(cts_per_sec[:,i])
        if(integrate_bins):
            cts_sum_ = np.sum(np.array(cts_sum),axis=0)
            maxval = 2*np.max(cts_sum_)
            ax.plot_date(plot_time, cts_sum_,'-', label="Sumed energy bins")
            
        ax.xaxis.set_major_formatter(myFmt)
        if(title):
            plt.title(title)
        plt.xlabel("Date",fontsize=14)
        plt.xlim(date_range[0],date_range[1])
        print(plot_time[0],plot_time[-1])
        plt.ylabel('Counts $s^{-1}$',fontsize=14)
        plt.ylim(1,2*maxval)
        if(logscale and maxval>1):
            plt.yscale('log')
        plt.grid()
        plt.xticks(np.linspace(date_range[0], date_range[1], 15))
        plt.legend(loc='lower center',bbox_to_anchor=(.5, -0.55),ncol=8,fontsize=13)

        plt.show()


        if(savename):
            plt.savefig(savename,bbox_inch="tight")

    else:
        fig, ax = plt.subplots(figsize=(20,5))
        maxval = 0
        for i in range(len(energies)):
            if e_range == None or (e_range != None and energies[i]["e_low"]>=e_range[0] and energies[i]["e_high"]<=e_range[1]):
                ax.plot_date(plot_time, cts_per_sec[:,i],'-', label=f'{energies[i]["e_low"]}-{energies[i]["e_high"]} keV')
                maxval = np.max(cts_per_sec[:,i]) if np.max(cts_per_sec[:,i])>maxval else maxval
        ax.xaxis.set_major_formatter(myFmt)
        if(title):
            plt.title(title)
        #labels
        plt.xlabel("Date",fontsize=14)
        plt.ylabel('Counts $s^{-1}$',fontsize=14)
        
        plt.ylim(1,2*maxval)
        # apply logscale if requested
        if(logscale and maxval>1):
            plt.yscale('log')
        plt.grid()
        
        plt.xticks(np.linspace(min(plot_time), max(plot_time), 15))
        plt.legend(loc='lower center',bbox_to_anchor=(.5, -0.55),ncol=8,fontsize=13)
        plt.show()


        if(savename):
            plt.savefig(savename,bbox_inch="tight")
        
    
        
def get_counts_data_per_energy(pathfile, is_bkg=False,time_arr=None):
    
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
    #normalise by time_bin duration ("timedel" keyword)
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

## SPECTRAL analysis (still cts space)

def plot_spectrogram(counts_dict,axes,savename=None,time_interval=None,
                      fmt=" %H:%M",title=None,x_axis=False,
                      date_ranges=None,energy_range=None,
                      logscale=True,**kwargs):
    # date_ranges param is used for visualizing delimiters for date range selection of the 
    # background and sample pieces (interactive plotting)
    # date_ranges = [[bkg_initial, bkg_final],[smpl_initial, smpl_final]]
    
    plot_time = counts_dict["time"].plot_date
    cts_per_sec = counts_dict["counts_per_sec"]
    energies = counts_dict ["energy_bins"]
    
    
    
    myFmt = mdates.DateFormatter(fmt)


    cts_data = np.log10(cts_per_sec, out=np.zeros_like(cts_per_sec), where=(cts_per_sec!=0)).T if logscale else cts_per_sec.T
    spec = axes.imshow(cts_data,aspect='auto',cmap="nipy_spectral",origin='lower',
               extent = [plot_time[0],
                        plot_time[-1],
                        0,32])
    yticks_idxs = [0,3,5,8,10,13,15,18,20,25,31]
    yticks_lbs = [f'{energies[idx]["e_low"]}-{energies[idx]["e_high"]} keV' for idx in yticks_idxs]
    axes.set_yticks(yticks_idxs)
    axes.set_yticklabels(yticks_lbs)
    if(date_ranges!=None):
        #if date ranges provided
        # bakground date range to plot_date
        date_ranges[0] = [Time(x).plot_date for x in date_ranges[0]]
        # sample date range to plot_date
        date_ranges[1] = [Time(x).plot_date for x in date_ranges[1]]
        
        #plot bkg delimiter
        plt.axvline(date_ranges[0][0],linestyle="--",color="white",label="Background selection")
        plt.axvline(date_ranges[0][1],linestyle="--",color="white")
        #plot sample delimiters
        plt.axvline(date_ranges[1][0],linestyle="--",color="cyan",label="Sample selection")
        plt.axvline(date_ranges[1][1],linestyle="--",color="cyan")
        plt.legend()

    
    cbar = plt.colorbar(spec)
    _ = cbar.set_label('$Log_{10}$ counts') if logscale else cbar.set_label('Counts')
    if(x_axis):
        axes.xaxis.set_major_formatter(myFmt)
        plt.xticks(np.linspace(min(plot_time), max(plot_time), 15))
        plt.xlabel("Date",fontsize=14)
    else:
        # for major ticks
        axes.set_xticks([])
        # for minor ticks
        axes.set_xticks([], minor=True)
    
    if(time_interval):
        plt.xlim(Time(time_interval[0]).plot_date,Time(time_interval[1]).plot_date)

    
    plt.ylabel('Energy bins',fontsize=14)
    if(energy_range):
        
        idx_enrg = [j for j in range(len(energies)) if np.logical_and(energies[j]["e_low"]>=energy_range[0],energies[j]["e_high"] <= energy_range[-1])]
        plt.ylim(idx_enrg[0],idx_enrg[-1])
    
    if(title):
            plt.title(title)
    #return fig, axes
    if(savename):
            plt.savefig(savename,bbox_inch="tight")
    
    #plt.show()




def create_spectrum(counts_dict,date_range):
    '''
    Function takes in as input the count arr for producing spectrogram
    and returns a spectrum with bins corresponding to the mean of the native energy binning.
    '''
    
    mean_energy = counts_dict["mean_energy"]
    cts_per_sec = counts_dict["counts_per_sec"]
    
    edges = [e_high for chn,e_low,e_high in counts_dict["energy_bins"]]
    edges = [0]+edges # edges for plotting
    
    
    
    time_dtm = counts_dict["time"]
    time_arr = time_dtm.plot_date
    
    #date_range to plot date to crop
    date_range = [Time(t).plot_date for t in date_range]

    time_idx = np.where((time_arr >=date_range[0]) &              #find the indexes corresponding to the time range requested
                          (time_arr <= date_range[1]))[0]
    #find the total time duration which we will use to divide the counts with
    time_dur = (time_dtm[time_idx[-1]]-time_dtm[time_idx[0]]).sec
    
    return_dict = {"spectrum":cts_per_sec[time_idx,:].sum(axis=0)/mean_energy/time_dur,
                   "duration":time_dur,
                   "mean_energy":mean_energy,
                   "energy_edges":edges,
                   "time_range":date_range}
    
    return return_dict

def remove_background_spectrum(spectr_dict,bkg_spectr_dict):
    if(len(spectr_dict["mean_energy"])==len(bkg_spectr_dict["mean_energy"])):
        return_dict = spectr_dict.copy()
        return_dict["spectrum"] = spectr_dict["spectrum"]-bkg_spectr_dict["spectrum"]
        return return_dict

def get_bkg_sample_spectrum(counts_dict,time_sample,time_bkg):
    
    specP=create_spectrum(counts_dict,time_sample)
    specB=create_spectrum(counts_dict,time_bkg)
    specPB = remove_background_spectrum(specP,specB)
    
    return specP,specB,specPB


def plot_spectrum(spectr_dict,**kwargs):
    
    #plt.plot(spectr_dict["mean_energy"],spectr_dict["spectrum"],**kwargs)
    plt.stairs(spectr_dict["spectrum"],
               spectr_dict["energy_edges"],orientation='vertical',linewidth=2,baseline=None,**kwargs)

#def 
def plot_counts_energy_spectra(counts_dict,time_sample,time_bkg,e_lim=None,**kwargs):
    # get spectra
    specP,specB,specPB = get_bkg_sample_spectrum(counts_dict,time_sample,time_bkg)
    # plot each spectra
    plot_spectrum(specPB,label="Spectrum w/o Background",color="red",linestyle="-")
    plot_spectrum(specB,label="Background spectrum",color="grey",linestyle="--")
    plot_spectrum(specP,label="Sample spectrum",color="orange",linestyle="--")  
    
    #fit
    if(not e_lim==None):
        popt,idx_mean,idx_edge = fit_power_law(specPB,e_lim)
        # use optim parameters to generate curve (considering that fit was made in log-log space)
        y_pred = np.exp(popt[1])*((np.array(specPB['mean_energy'])[idx_mean.astype(int)])**popt[0])
        
        plt_dict = {"spectrum":y_pred,
                    "energy_edges":np.array(specPB["energy_edges"])[idx_edge.astype(int)]}
        #plot
        plot_spectrum(plt_dict,color="blue",label=f"Fit ({round(popt[0],2)})")
        
    if("title" in kwargs):
        plt.title(kwargs["title"])
    
    plt.xlabel('Energy (keV)')
    plt.ylabel('$Counts \quad s^{-1} \quad keV^{-1}$')
    plt.legend(loc='best')
    plt.grid()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(1,2*np.max(counts_dict["energy_bins"]["e_low"]))
    #return specP,specB,specPB

def fit_power_law(spectr_dict,energy_range):
    #[low_energy, high_energy] (kev)
    
    def nonthermalindex(x, a, b):
        # in log-log-space powerlaw is lineal equation
        return a * x + b
    
    # indexes between selected energy limits for fit
    energy_idx = np.where(np.logical_and(np.array(spectr_dict['mean_energy'])>=energy_range[0],np.array(spectr_dict['mean_energy'])<=energy_range[1]))[0]
    # edges for stairs plot
    energy_idx_edge = np.insert(energy_idx,energy_idx.size,energy_idx[-1]+1)

    
    # log _ log space
    log_e = np.log(np.array(spectr_dict['mean_energy'])[energy_idx.astype(int)])
    log_spec = np.log(np.array(spectr_dict['spectrum'])[energy_idx.astype(int)])
    #fit powerlaw with observed spectra
    popt, _ = curve_fit(nonthermalindex,log_e ,log_spec, bounds=(-30, [20, 100]))
    print('Spectral index = ',popt[0])
    
    return popt,energy_idx,energy_idx_edge

## Interactive modes
    
def plot_interactive_energy_bins(counts_dict,**kwargs):
    
    # retrieve data
    plot_time = counts_dict["time"] # Time object
    cts_per_sec = counts_dict["counts_per_sec"]
    energies = counts_dict["energy_bins"]
    
    # options for energy range  
    e_options=np.unique(np.array(list(energies["e_low"])+list(energies["e_high"])))
    e_options = [("{} keV ".format(e),e) for e in e_options]
    

    # format
    #iso_frmt = '%Y-%m-%d %H:%M:%S' 
    format_display= "%m/%d %H:%M\n"

    # convert from string format to datetime format
    #date range options
    date_options=[(i.strftime(format_display),i) for i in plot_time]
    
    # SLIDERS
    energy_slider = widgets.SelectionRangeSlider(
    options=e_options,
    index=(0, len(e_options)-1),
    description='Energy range',
    disabled=False,
    layout=widgets.Layout(width='90%', padding='30px'),
    readout=True,
    readout_format='.2f',
    )
    
    
    time_slider = widgets.SelectionRangeSlider(
    options=date_options,
    index=(0, len(date_options)-1),
    description='Time range',
    disabled=False,
    layout=widgets.Layout(width='90%', padding='30px'),
    readout=True,
    )
    
    # BUTTONS
    
    butt_integrate=widgets.ToggleButton(
    value=False,
    description="sum bins",
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='sum over selected energy bins'
)
    butt_log=widgets.ToggleButton(
    value=True,
    description="log scale",
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Use log scale in Y axis'
)

    # use events to plot 
    def paint_plot(e_vals,t_interval,integ,logscale):
        
        plt.figure(figsize=(20,14))
        ax1 = plt.subplot(2,1,1)
        mod_t_interval = [Time(i, scale='tt', format='plot_date').iso.split(".")[0] for i in t_interval]
        plot_counts_per_energy(counts_dict,ax1,
                         e_range=None,date_range=None,
                         title="Counts per Energy bin",
                         integrate_bins=False,
                         logscale=logscale,
                         legend=False,
                         *kwargs)
        ax2 = plt.subplot(2,1,2)
        plot_counts_per_energy(counts_dict,ax2,
                         e_range=e_vals,date_range=t_interval,
                         title="Selected interval from {} to {}".format(mod_t_interval[0],mod_t_interval[1]),
                         integrate_bins=integ,
                         logscale=logscale,
                         *kwargs)
    # create widget group interactive session
    interact_manual(paint_plot,e_vals=energy_slider,t_interval=time_slider,integ=butt_integrate,
                    logscale=butt_log)
    
      
def plot_interactive_spectral_analysis(counts_dict,**kwargs):
    
    
    #get data   
    plot_time = counts_dict["time"] # Time object
    cts_per_sec = counts_dict["counts_per_sec"]
    energies = counts_dict["energy_bins"]
    
    # energy options for sliders
    e_options=np.unique(np.array(list(energies["e_low"])+list(energies["e_high"])))
    e_options = [("{} keV ".format(e),min(np.max(energies["e_low"]),e)) for e in e_options]
    

    # format
    #iso_frmt = '%Y-%m-%d %H:%M:%S' 
    format_display= "%m/%d %H:%M\n"

    # convert from string format to datetime format
    #date range options
    date_options=[(i.strftime(format_display),i) for i in plot_time]
    
    # SLIDERS
    fit_energy_slider = widgets.SelectionRangeSlider(
    options=e_options,
    index=(4, 17),
    description='fit E range',
    disabled=True,
    layout=widgets.Layout(width='90%', padding='30px'),
    readout=True,
    readout_format='.2f',
    )
    
    
    time_slider_bkg = widgets.SelectionRangeSlider(
    options=date_options,
    index=(0, len(date_options)-1),
    description='BKG Time',
    disabled=True,
    layout=widgets.Layout(width='90%', padding='30px'),
    readout=True,
    )
    
    time_slider_smp = widgets.SelectionRangeSlider(
    options=date_options,
    index=(0, len(date_options)-1),
    description='SMPL Time',
    disabled=True,
    layout=widgets.Layout(width='90%', padding='30px'),
    readout=True,
    )
    
    # BUTTONS
    
    butt_do_fit=widgets.ToggleButton(
    value=False,
    description="Do power-law fit",
    disabled=True,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='sum over selected energy bins'
    )
    
    butt_select_bits=widgets.ToggleButton(
    value=False,
    description="Correct background",
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Enable bkg and sample time range selection'
    )
    butt_log=widgets.ToggleButton(
    value=True,
    description="Log scale",
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Use log scale in spectrogram'
    )
    
    # update disabled attributes 
    def update_bits(select_bits):
        time_slider_bkg.disabled = not select_bits
        time_slider_smp.disabled = not select_bits
        butt_do_fit.disabled = not select_bits
    def update_fit(do_fit):
        fit_energy_slider.disabled = not do_fit
     
    # update widget each time these buttons have an event
    butt_select_bits.observe(update_bits,"value")
    butt_do_fit.observe(update_fit,"value")
    
    def paint_spectrum_plots(select_bits,time_bkg,time_smp,do_fit,e_range,logscale):
        
        # update widget elements
        update_bits(select_bits)
        update_fit(do_fit)
        
        
        if(not select_bits):
            # NO time bits NO do fit
            plt.figure(figsize=(20,5))
            ax=plt.subplot(1,1,1)
            plot_spectrogram(counts_dict,ax,logscale=logscale)
        elif (not do_fit):
            # YES time bits NO do fit
            plt.figure(figsize=(20,9))
            ax=plt.subplot(2,1,1)
            plot_spectrogram(counts_dict,ax,date_ranges=[time_bkg,time_smp],logscale=logscale)
            ax=plt.subplot(2,1,2)
            
            plot_counts_energy_spectra(counts_dict,time_smp,time_bkg)

        else:
            # YES time_bits YES do fit
            plt.figure(figsize=(20,11))
            ax=plt.subplot(2,1,1)
            plot_spectrogram(counts_dict,ax,date_ranges=[time_bkg,time_smp],logscale=logscale)
            ax=plt.subplot(2,1,2)
            plot_counts_energy_spectra(counts_dict,time_smp,time_bkg,e_lim=e_range)
            print(e_range)

    # interactive widgets       
    interact_manual(paint_spectrum_plots,select_bits=butt_select_bits,logscale=butt_log,time_bkg=time_slider_bkg,
                   time_smp=time_slider_smp,do_fit=butt_do_fit,e_range=fit_energy_slider)
                   