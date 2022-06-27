# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:24:41 2021

@author: David
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pref = "CHIMES_0.6/Out/"
interest_ab = ["h","h2","oh","h3+","h2o","c","co","electr","c-c3h","ch4","hco+","c+","c4h2","ch5o+"]
def reformat_grah_file(name_in,name_out):
    with open(name_out,"w") as k:
        with open(name_in,"r") as f:
            while True:
                line = f.readline()
                if(not line):
                    break
                line=line.strip()
                k.write(",".join(line.split()).strip()+"\n")


def get_graph_data(pref,name,ab_list=None,include_t = True):
    reformat_grah_file(pref+name+"/"+name+".graph",pref+name+"/"+name+"_rf.graph")
    data = pd.read_csv(pref+name+"/"+name+"_rf.graph",sep=",",engine="python")
    col_list = []
    if not ab_list:
        return data
    elif include_t:
        col_list.append("t(Myrs)")
    col_list = ["t(Myrs)"]+["t(yrs)"] + ab_list 
    data.drop(data.tail(1).index,inplace=True) # drop last n=1 rows (supressing final drop (???) )
    data.drop(data.head(1).index,inplace=True) # drop first n=1 rows (supressing initial peak (???) )
    data["t(yrs)"] = data["t(Myrs)"]*1e6
    return data[col_list]

def get_deriv_data(pref,name,ab_list=None,include_t = True):
    reformat_grah_file(pref+name+"/"+name+".deriv",pref+name+"/"+name+"_rf.deriv")
    data = pd.read_csv(pref+name+"/"+name+"_rf.deriv",sep=",",engine="python")
    col_list = []
    if not ab_list:
        return data
    elif include_t:
        col_list.append("t(Myrs)")
    col_list = ["t(Myrs)"]+["t(yrs)"] + ab_list 
    data.drop(data.tail(1).index,inplace=True) # drop last n=1 rows (supressing final drop (???) )
    data.drop(data.head(1).index,inplace=True) # drop first n=1 rows (supressing initial peak (???) )
    data["t(yrs)"] = data["t(Myrs)"]*1e6
    return data[col_list]


def get_log_values(logpath,to_text = False):
    
    ans = {}
    sc = None
    with open(logpath,"r") as f:
        while True:
            line = f.readline()
            if(not line):
                break
            if "---" in line:
                continue
            line = line.strip()
            if(line.startswith("*")):
                sc = line.split()[-1]
                ans[sc]={}
                if(to_text):
                    ans[sc]=""
            else:
                if(to_text):
                    ans[sc] = ans[sc] + "| " +line + " |   "
                else:
                    ln = line.split(":")
                    ans[sc][ln[0].strip()] = float(ln[1].strip())
    
    
    return ans


    
reff_vals = get_log_values("CHIMES_0.6/Data/log_testdepth.log",True)

spec_names = [
    ["h3+","$H_3$+ / $H_2$"],
    ["oh","OH / $H_2$"],
    ["electr","e- / $H_2$"],
    ["c+","C+ / $H_2$"],
    ["co","CO / $H_2$"],
    ["h","H / $H_2$"]
    
]
plt.figure(figsize=(17,10),dpi=200)
for  j in range(1,7):
    plt.subplot(2,3,j)
    for i in range(1,4):
        sc = "testdepth{}".format(i)
        ddbb = get_graph_data(pref,sc,ab_list=interest_ab)
        plt.title(spec_names[j-1][1],fontsize=12)
        newtitle =reff_vals[sc].replace("av","$A_v$").replace("|","").replace(":"," = ").strip()+" mag" 
        plt.plot(ddbb["t(yrs)"],ddbb[spec_names[j-1][0]]/ddbb["h2"],label=newtitle)
        plt.xlabel("Time (yrs)",fontsize=9)
        plt.ylabel("Fractional abundance ratio",fontsize=9)
        plt.yscale("log")
        plt.xscale("log")
        plt.grid()
        plt.ylim(1e-20,1e-1)
        #plt.legend(loc=2)
plt.savefig("testdepth.png")
plt.close()