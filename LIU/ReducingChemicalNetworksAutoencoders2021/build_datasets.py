# UTILS
import os
import time
import pandas as pd
import numpy as np
import scipy.stats as stats

# PLOT
import matplotlib.pyplot as plt

# LEARN
import torch
from sklearn.model_selection import train_test_split

# constants 
CLIPPING = [1e-35,1]


def create_dataset(dirpath,N_per_model,choice="equal",dataset_type="train_test", test_size=0.3, val_size=0.3, max_batch=None, savename = None, norm_abundances = None):
    """
    creates a dataset with the provided compressed files (.npz).
    when dataset_type  = "all_models"
        creates dict with the abundance data of different scenarios (scenario key ["sc_{i}"]) per timestep (same timestep in all scenarios)
            key ["sc_{i}"]["x"]       abundances data
            key ["sc_{i}"]["t"]       timestep data
            key ["sc_{i}"]["cols"]    name of the columns (species)
    when dataset_type  = "train_test"
        creates dict with the different samples for training the NN. X stores the abundances vectors (length= # of species N ), Y stores
        X concatenated with the derivatives vector (creating a vector fo size 2*N) which is not used in the conventional auntoencoder but will be
        used to train the specialized branch of the AE.
            key ["x_train"] / ["y_train"]   taining sample
            key ["x_test"] / ["y_test"]   test sample
            key ["x_val"] / ["y_val"]   validation sample

    PARAMS:
    (str)       dirname: root name of the directory containing files
    (str)       savename: path+name of saved compressed file (dict) containing the samples. Default None -> not saving the sample, just returning it
    (int)       N_per_model: number of lines imported per model (not used if dataset_type = "all_model")
    (str)       choice: time sampling method for each file ("random"[default] = random sampling,"equal" = uniform slpitting in time log space )
    (str)       dataset_type: type of dataset to be created ("train_test"[default] = returns a dict object with test,train,val sets by sampling scenarios, "all_model" 0 imports max_batch scenarios - all timesteps )
    (int)       max_batch: max number of scenario files used, Default "None" -> use all files
    (float)     test_size: perentage of whole sampling used to build test set (number from 0 to 1)
    (float)     val_size: percentage of training sample used for validation ( (1 - test_size)*val_size samples used for valdation)
    (float)     norm_abundances: normalie aundance values w.r.t. gas density if provided. Default -> None
    RETURN:

    """
    # list all compressed input files in the provided folder
    dirfiles = [f for f in os.listdir(dirpath)if f.endswith(".csv.gz")]
    # ake the first max_batch scenarios if defined
    if max_batch:
        dirfiles= dirfiles[:min(max_batch,len(dirfiles))]
        print(dirfiles)


    # TRAIN TEST DATASET
    if(dataset_type == "train_test"):
        # store X and Y vectors
        X = []
        Y = []

        total_s =  len(dirfiles)*N_per_model # total number of samples
        test_s = int(total_s*test_size) # test sample size
        val_s = int((total_s-test_s)*val_size) # validation sample chosen over the remaining sample
        train_s = int((total_s-test_s-val_s)) # training sample is the rest

        # Logs
        print("{} lines for each of the {} models = {} samples".format(N_per_model,len(dirfiles),total_s))
        print("  {} samples for training".format(train_s))
        print("  {} samples for validating".format(val_s))
        print("  {} samples for testing".format(test_s))

        # for each file
        for i in range(len(dirfiles)):
            # filename
            filename = dirfiles[i]

            # Logs
            s = "extracting {0} ... ".format(filename)
            if i == len(dirfiles)-1:
                s+='\n'
            if i>0:
                s = '\r'+s
            print(s, end='')
            # path
            filepath = os.path.join(dirpath, filename)
            #read file (HIGH MEM CONSUME: FIX)
            data = pd.read_csv(filepath,compression="gzip")
            #get column names, time step info, x vectors and  vectors using chosen sampling "choice"
            cols_,sel_t,x,y = get_vecs(data,N_per_model,chose=choice,norm=norm_abundances)
            # store the x,y vectors in order , the index is important to match x,y pairs for evaluating
            for i in range(len(x)):
                X.append(x[i])
                Y.append(y[i])

        print("Extraction completed. Splitting dataset...")
        # Split test sample from the total sample
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=test_size)
        # split validation sample from the other sample, and the train sample is the remaining
        X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=val_size)
        print ("Done.")
        # dictionary data structure
        pack = {"cols":cols_,"times":sel_t,"x_train":X_train,"x_val":X_val,"x_test":X_test,"y_train":y_train,"y_val":y_val,"y_test":y_test}
        # save if savename provided (.npz)
        if(savename):
            np.savez_compressed(savename,cols=cols_,times=sel_t,x_train=X_train,x_test=X_test,x_val=X_val,y_train=y_train,y_test=y_test,y_val=y_val)
        return pack

    # ALL MODEL DATASET (whole scenarios, no time sampling)
    elif(dataset_type=="all_model"):
        # dict structure to store models
        models = {}

        # total files used
        total_s =  len(dirfiles)
        print(" {} samples".format(total_s))

        # for each scenario (file)
        for i in range(len(dirfiles)):
            filename = dirfiles[i] # name
            # logs
            s = "extracting {0} ... ".format(filename)
            if i == len(dirfiles)-1:
                s+='\n'
            if i>0:
                s = '\r'+s
            print(s, end='')
            # filepath and read scenario data
            filepath = os.path.join(dirpath, filename)
            data = pd.read_csv(filepath,compression="gzip")
            # get column names, time step info, x vectors and  vectors using chosen sampling "choice"
            cols_,sel_t,x,y = get_vecs(data,N_per_model,use_all=True,norm=norm_abundances)
            # store all x and t vectors for each scenario
            models["sc_"+str(i+1)] = {"x":x,"t":sel_t}
        # columns are the same for every scenario
        models["cols"]=cols_


        print("Extraction completed.")
        # save dict structure if savename provided
        if(savename):
            np.save(savename,models)
        return models


def retrieve_dataset(filepath):
    """
    retrieve a saved samples compressed file

    PARAMS:
    (str)       filepath: path of the compressed dict structure (.npz)

    RETURN:
    Dict        contains the retrived samples

    """
    data = np.load(filepath,allow_pickle=True)
    return data

# SAMPLING METHODS

def get_vecs(df,N,use_all=False,chose="equal",norm=None):
    """
    retrieve the x,y vectors according to the chosen method of sampling

    PARAMS:
    Dataframe   df: dataframe containing the compiled graph and deriv data
    (int)       N: number of lines to sample
    (bool)      use_all: Use all timesteps, no time sampling
    (str)       chose: sampling in time [only used if use_all=False]
                ( "equal"=split the log time scale into N equal intervals, "random"= randomly choose N vectors (rows))
    (float)     norm: normalize values over the value of gas density if provided. Default -> None

    RETURN:
    list(str)   cols: names of the abundance columns (species)
    list(float) select_times: selected timesteps for sampling (time in Myrs)
    list(*arr)  x,y : x and y vectors extracted from the dataframe.

    Note: selec_times, x, y are in order (i.e. the first element in each list correponds to the same vector)

    """

    x = []
    y = []


    #print(df.columns)
    cols = df.columns[1:-2] #avoiding time column and two tag columns (deriv/graph and the scenario index)       #DEPRECATED[4:-21]
   # print(cols)

    nspec = len(cols)
    select_times = df["t(Myrs)"].unique()[5:-5]#appears to be a prblem with the last/first 5 lines (many ab 1e-16)

    if(not use_all):
        if chose=="equal":
            # create N points equally distanced in log time scale
            timelogspace = np.linspace(np.log10(select_times[0]),np.log10(select_times[-1]),N)
            # for each point in the log time scale select the timestep closest to its time value
            select_times = [min(select_times, key=lambda x:abs(x-10**t)) for t in timelogspace]

        elif chose=="random":
            # Randomly choose N timesteps
            select_times = np.random.choice(select_times,N)

    # for each selected time step
    for t in select_times:
        # x vector containing the chemical abundances of each of the nspec species
        xx = np.zeros(nspec)
        # y vector containing the abundances and the derivatives vector concatenated in a 2*nspec size vector
        yy = np.zeros(2*nspec)

        # select graph and deriv data of the selescted timestep
        grph =df[np.logical_and(df["t(Myrs)"]==t,df["datatype"]=="graph")][cols].iloc[0]
        drv =  df[np.logical_and(df["t(Myrs)"]==t,df["datatype"]=="deriv")][cols].iloc[0]

        # copy x values
        xx[:] = grph.values[:]
        if(norm):
            # normalize is provided
            xx = xx/norm
       # concatenate x and deriv vector to produce y vector
        yy[:nspec] = xx[:]
        yy[nspec:] = drv.values[:]

        # save x and y vector in ordererd lists. index is important for pairing
        x.append(xx)
        y.append(yy)


    return list(cols),select_times,x,y





#
# def loss_per_species(output,target):
#     loss = (output - target)**2
#     loss = loss.detach().apply_(lambda x: x/1)
#     return loss.numpy()
#
# def RMSE(output,target):
#     output = reconstruct_output(output,CLIPPING[0],CLIPPING[1])
#     target = reconstruct_output(target,CLIPPING[0],CLIPPING[1])
#
#     loss = (np.log10(output)-np.log10(target))**2
#
#     return np.sqrt(np.mean(loss))
