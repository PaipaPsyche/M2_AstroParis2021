import numpy as np
import reader_cdf as rcdf
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import os
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

def get_data(file,sensor=9):
    # data read
    print("Extracting info:")
    print("  File: ",os.path.basename(file))
    print("  Type: RPW L2")
    return rcdf.read_hfr_autoch_full(file,sensor=sensor)

    


def select_freq_indexes(frequency,**kwargs):#,freq_col=0,proposed_indexes=None):
    #indexes of frequencies different from 0 or -99 (column 0 in frequency matrix)
    
    fcol = kwargs["freq_col"]
    freq_nozero = np.where(frequency.T[fcol]>0)[0]
    
    selected_freqs = freq_nozero
    if kwargs["which_freqs"]=="both":
        selected_freqs = [ j  for j in kwargs["proposed_indexes"] if j in freq_nozero]
        
    if(not kwargs["freq_range"]==None):
        #print(frequency[selected_freqs,fcol])
        selected_freqs = [selected_freqs[j] for j in range(len(selected_freqs)) if np.logical_and(frequency[selected_freqs[j],fcol]<=kwargs["freq_range"][1],frequency[selected_freqs[j],fcol]>=kwargs["freq_range"][0]) ]
    
    return selected_freqs,frequency[selected_freqs,fcol]

def createPSD(data,freq_range=None,date_range=None,freq_col=0,proposed_indexes=None,which_freqs="both"):
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
    freq_idx,freq_axis = select_freq_indexes(freq_,freq_col=freq_col,freq_range=freq_range,
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

def plot_psd(psd,logscale=True,colorbar=True,cmap="jet",t_format="%H:%M:%S",
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
    ax.set_yticks([x  for x in [100,500,1000,5000] if np.logical_and(x<=f[-1],x>=f[0])])
    ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
    
    if(frequency_range):
        plt.ylim(max(frequency_range[0],np.min(f)),min(frequency_range[1],np.max(f)))
    plt.yscale("log")
    
    if(xlabel):
        plt.xlabel("start time: "+t[0].strftime("%d-%b-%Y %H:%M:%S"),fontsize=axis_fontsize)
    plt.ylabel("Frequency [kHz]",fontsize=axis_fontsize)



    
    