
# UTILS
import pandas as pd
import numpy as np
import pickle
import os

# STATS
import scipy.stats as stats
from sklearn.model_selection import train_test_split


#PLOT
import matplotlib.pyplot as plt

import os
import time
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

# scripts
from build_datasets import *

# constants
## ODE SOLVER
number_of_reactions = 12
S = np.array([
         [-2, 2, -1, 1, 0, 0, 0, 0, 1, -1, -1, 1],
         [1, -1, 0, 0, 0, 0, -1, 1, -1, 1, 0, 0],
         [0, 0, 1, -1, 0, 0, 2, -2, 1, -1, 1, -1],
         [0, 0, -1, 1, -2, 2, 0, 0, -1, 1, 1, -1],
         [0, 0, 0, 0, 1, -1, -1, 1, 0, 0, -1, 1]
        ], dtype="float64")
S_tensor = torch.from_numpy(S)

# METHODS

def normalize_input(vec,vmin,vmax):
    """
    takes an abundance vector (present in many orders of magnitude) and normalize it
    according to the clipping values defined  in build_datasets module


    PARAMS:
    (array)     vec: abundace vector to be normalized into the interval [-1,1]
    (float)     vmin: min value to clip the input vector (low consrain, mapped to -1)
    (float)     vmax: max value to clip the input vector (high consrain mapped to 1)

    RETURN
    (array)     constrained log vector in the range [-1,1]
    """


    # create torch element
    xvec = torch.from_numpy(vec)
    # log the contrained abunance vector
    xvec_log = torch.log10(torch.clamp(xvec,min=vmin,max=vmax))
    #log of the constrain values
    exp_vmin = np.log10(vmin)
    exp_vmax = np.log10(vmax)
    # interpolate to map the logscale in the [-1,1] interval
    xvec_log.apply_(lambda x: np.interp(x,[exp_vmin,exp_vmax],[-1,1]))

    # returns the normalized log vector
    return xvec_log

def reconstruct_output(vec,vmin,vmax):
    """
    takes anormalized log vector (present in interval [-1,1]) and re scale it
    according to the clipping values defined in build_datasets module to retrieve
    the values of chemical abundances


    PARAMS:
    (array)     vec: normalized log vector to be re-scaled into the interval [vmin,vmax]
    (float)     vmin: min value used to clip the input vector (low consrain, value correponding to re-sclaing -1)
    (float)     vmax: max value to clip the input vector (high consrain, value correponding to re-sclaing 1)

    RETURN
    (array)     recosntructed abundances vector
    """
    # log of the constrain values
    exp_vmin = np.log10(vmin)
    exp_vmax = np.log10(vmax)

    # map the interval [-1,1] to the log scale clipped. exponentiating base 10
    # result in values in the [vmin,vmax] interval across different orders of magnitudes

    vec.detach().apply_(lambda x: 10**np.interp(x,[-1,1],[exp_vmin,exp_vmax]))
    # retun numpy vector
    vec = vec.detach().numpy()

    return vec


# Metrics

def log_mse_loss(output, target):
    """
    Loss function L_0 evaluating the Mean suared error between the predicted
    and true vector

    PARAMS:
    (array)     output: output vector corresponding to y_predicted [-1,1]
    (array)     target: target vector corresponding to y_true [-1,1]

    RETURN
    (float)     MSE value
    """
    loss = torch.mean((output - target)**2)
    return loss


def RMSE(output,target,mean=True):
    """
    compares output and target to determine the total Root Mean Square Error
    (resulting in a scalar for output/target vectors) or Mean Absolute Error (residual) per specie
    (resulting in a vector of residuals of size N_species)

    PARAMS:
    (array)     output: output vector corresponding to y_predicted [-1,1]
    (array)     target: target vector corresponding to y_true [-1,1]
    (bool)      mean: average over all vector values before takin the square root

    RETURN:
    one of the following:
        (array)    residual vector if mean=False
        (float)    RMSE value between the two vectors if mean=True

    Note: The operations are done over the log value of the abunances value (not the value between -[1,1])
    """
    # retrieve abudnace values
    output = reconstruct_output(output,CLIPPING[0],CLIPPING[1])
    target = reconstruct_output(target,CLIPPING[0],CLIPPING[1])
    # square residual in the log space
    loss = (np.log10(output)-np.log10(target))**2

    # take the root. average over the vecto values if required
    ans = np.sqrt(np.mean(loss)) if mean else np.sqrt(loss)

    return ans


def generate_loss_report(model,columns,x_true,x_target):
    """
    generates a loss report for the provided model based on the residual measurements for each specie
    comparing the pairs of vactors in x_true and x_target

    PARAMS:
    (PyTorch nn) model: Trained model
    (list(str)) columns: list of the species names (orered names of the elements of the vector)
    (array)      x_true: predicted values
    (array)      x_target: true values

    RETURN:

    (Dataframe)  Loss report with the min, max, mean and std residual error per species
                 obtained by comparing the provided samples

    """
    losses_ps = []

    # for each index
    for i in range(len(x_true)):

        # get the log vectors of the true and predicted values as x and y
        xvec_log = normalize_input(x_true[i],CLIPPING[0],CLIPPING[1])
        yvec_log = normalize_input(x_target[i],CLIPPING[0],CLIPPING[1])
        # ===================forward=====================
        output = model(xvec_log.float())
        #estimate resudal loss per specie (mean=False)
        loss = RMSE(output, yvec_log.float(),mean=False)
        losses_ps.append(loss)

        #prit progress
        s = "Generating loss report: {0} % ".format(round(100*i/len(x_true),2))
        if i == len(x_true)-1:
            s+='\n'
        if i>0:
            s = '\r'+s
        print(s, end='')

    # compile statistical values per specie
    losses_report = [[columns[i],np.mean(losses_ps[i][:]),np.max(losses_ps[i][:]),np.min(losses_ps[i][:]),np.std(losses_ps[i][:])] for i in range(len(columns))]
    losses_report = pd.DataFrame(data=losses_report,columns=["species","mean_err","max_err","min_err","std_err"])
    print("Done.")
    return losses_report


def get_z_outliers(df,zlim,cols,sort=None):
    """
    generates a loss report for the provided model based on the residual measurements for each specie
    comparing the pairs of vactors in x_true and x_target

    PARAMS:
    (Dataframe)  df: Loss report
    (float)      zlim: threshold value of z-score to classidy ouliers (species is outlier if |z_score|>zlim)
    (array(str)) cols: columns to be used as classifiers (to detect outliers) [columns of loss report, not species names]
    (str)       sort: sort the resulting table by this column if provided (default None)
    RETURN:

    (Dataframe)  Loss report with the min, max, mean and std residual error per species
                 obtained by comparing the provided samples

    """
    new_df = df[(np.abs(stats.zscore(df[cols])) > zlim).all(axis=1)] #apply Zcore on the colvs columns

    #sorting
    if sort != None:
        new_df= new_df.sort_values(by=sort)

    return  new_df



##  AUTOENCODER AND CUSTOM LAYER
## CUSTOM LAYER (Based on the code provided by Grassi et al. 2020)
class RHS_layer(nn.Module):
    """ Custom Linear ODE layer but mimics a standard linear layer """
    def __init__(self,size_in):
        super().__init__()

        self.size_in = size_in
        weights = torch.Tensor(number_of_reactions)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        #bias = torch.Tensor(size_out)
        #self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.uniform_(self.weights) # weight init
        #fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        #bound = 1 / math.sqrt(fan_in)
        #nn.init.uniform_(self.bias, -bound, bound)  # bias init
    def ODE_forward(self,inputs):
        in_shape = list(inputs.size())
        self.batch_size = in_shape[0] if len(in_shape)>1 else 1
        if(self.batch_size ==1):
            try:
                z0 = inputs[0,0:1]  # A
                z1 = inputs[0,1:2]  # AA
                z2 = inputs[0,2:3]  # AB
                z3 = inputs[0,3:4]  # B
                z4 = inputs[0,4:5]  # BB
            except:
                z0 = inputs[0:1]  # A
                z1 = inputs[1:2]  # AA
                z2 = inputs[2:3]  # AB
                z3 = inputs[3:4]  # B
                z4 = inputs[4:5]  # BB

            # init species in fluxes (reactants)
            preflux = torch.cat([
             torch.mul(z0, z0),
             z1,
             torch.mul(z0, z3),
             z2,
             torch.mul(z3, z3),
             z4,
             torch.mul(z1, z4),
             torch.mul(z2, z2),
             torch.mul(z1, z3),
             torch.mul(z2, z0),
             torch.mul(z4, z0),
             torch.mul(z2, z3)
            ],axis=0)

            p = 1e1**self.weights
            flux = torch.mul(p, preflux)
            xdot = torch.matmul(S_tensor.double(),flux.double())

            return xdot


        else:
            z0 = inputs[:, 0:1]  # A
            z1 = inputs[:, 1:2]  # AA
            z2 = inputs[:, 2:3]  # AB
            z3 = inputs[:, 3:4]  # B
            z4 = inputs[:, 4:5]  # BB
                    # init species in fluxes (reactants)
            preflux = torch.cat([
             torch.mul(z0, z0),
             z1,
             torch.mul(z0, z3),
             z2,
             torch.mul(z3, z3),
             z4,
             torch.mul(z1, z4),
             torch.mul(z2, z2),
             torch.mul(z1, z3),
             torch.mul(z2, z0),
             torch.mul(z4, z0),
             torch.mul(z2, z3)
            ],axis=1)


            p = 1e1**self.weights
            flux = torch.mul(p, preflux)
            xdot = torch.zeros(self.batch_size,self.size_in)
            for i, x in enumerate(flux):
                k = torch.matmul(S_tensor.double(),x.double())
                #print(np.shape(S_tensor),np.shape(x),np.shape(k))
                xdot[i:i+1,:] = k
            return xdot

    def forward(self, x):

        return self.ODE_forward(x)

class autoencoder(nn.Module):
    def __init__(self,latent_dim,num_specs):
        super(autoencoder, self).__init__()
        # architecture according to the original publciation
        # Grassi et al. 2020

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(num_specs, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.Linear(8, latent_dim),
            nn.Tanh())
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(True),
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, num_specs),
            nn.Tanh())

        self.RHS = RHS_layer(latent_dim)

    def forward(self, x):
        # WITHOUT EXTRA RHS_layer
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

        # WITH RHS_layer
        encoded = self.encoder(x)
        xdot = self.RHS(encoded)
        decoded = self.decoder(encoded)

        # TODO


        return decoded



def train_AE(model,criterion,num_epochs,training_x,validation_x=None, batch_size= 32 ,epochs_validation=5,epochs_print=10,learning_rate=1e-3):
    """
    Trains a conventional autoencoder with the provided training sample and paramters. Evaluates the
    model over a valdition set if such set is provided.

    PARAMS:
    (PyTorch nn)  model: PyTorch autoencoder model. Can be already trained or not.
    (function)    criterion: loss function evaluated over two tensors, y_true and y_predicted
    (int)         num_epochs: number of epochs to train the model
    (array)       training_x: traing sample. only X vectors are required since its an autoencoder (input == target==y_true)
    (array)       validation_x: validation sample. only X vectors are required. Is splitted into batches as well(same number as training set). default None.
    (int)         batch_size: number of vectors per batch. Default 32
    (int)         epochs_validtion: interval of epochs between each evaluation of the validation set loss. Default 5
    (int)         epochs_print: numer of eapoch between progress updates (print loss) . Default 10
    (float)       learning_rate: learning rate used by the optimizer (Adam)
    RETURN:

    (PyTorch nn)  PyTorch autoencoder model after training.
    (array)       loss per epoch
    (array)       validation loss [[epochs][mean loss][std loss]] . empty if validation set is not provided.
                 obtained by comparing the provided samples

    """
    print("Training Autoencoder \n Epochs {}  \n Batch Size {} \n Learning Rate {:.2e}".format(num_epochs, batch_size , learning_rate))

    #efine optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training X
    X=Variable(torch.stack(([normalize_input(np.array(xvec),CLIPPING[0],CLIPPING[1]) for xvec in training_x])))

    # store loss value per epoch
    loss_epoch = np.zeros([num_epochs])
    # validation performance if validation done
    val_perform = []

    # timing
    start_time = time.time()


    for epoch in range(num_epochs):
        # randomly permute the trianing set ang generate index
        permutation = torch.randperm(X.size()[0])

        # for each batch
        for i in range(0,X.size()[0], batch_size):
            optimizer.zero_grad()
            # get the batch
            indices = permutation[i:i+batch_size]
            batch_x= X[indices]
            # forward batch
            outputs = model.forward(batch_x.float())
            #estimate loss
            loss = criterion(outputs,batch_x)
            #back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # print loss every epochs_print epochs
        if(epoch%epochs_print == 0 ):
            print('epoch [{}/{}], Log(loss):{:.3f}'.format(epoch, num_epochs, np.log10(loss.data.item())))
        loss_epoch[epoch] = loss.data.item()

        # evaluate with the validation set every epochs_validation epochs
        if(  epoch%epochs_validation == 0 and not validation_x is None):

            # X validation
            Xeval=Variable(torch.stack(([normalize_input(np.array(xvec),CLIPPING[0],CLIPPING[1]) for xvec in validation_x])))
            # store loses per batch
            losses_eval = []
            #permutations of sample
            permutation_v = torch.randperm(Xeval.size()[0])

            for g in range(0,Xeval.size()[0], batch_size):
                #get batch
                indices_v = permutation_v[g:g+batch_size]
                batch_x_v= Xeval[indices_v]
                # forward batch
                outputs_v = model.forward(batch_x_v.float())
                # evalaute validation loss
                loss_v= criterion(outputs_v,batch_x_v)
                losses_eval.append(loss.data.item())
                # mean and std validati0n loss
                mean_loss_eval = np.mean(np.log10(losses_eval))
                std_loss_eval = np.std(np.log10(losses_eval))

            # store epoch, mean(loss), std(loss)
            val_perform.append([epoch,mean_loss_eval,std_loss_eval])
            print("-->    [{}] Validation Log(loss) = {:.2f} +- {:.2e}".format(epoch,mean_loss_eval,std_loss_eval))

    # timing
    print("--- %s seconds ---" % (time.time() - start_time))

    # return validation performance if validation set was provided
    if(not validation_x is None):
        performance = [[],[],[]]
        for i in range(len(val_perform)):
            for j in range(3):
                performance[j].append(val_perform[i][j])
        return model,loss_epoch,performance
    else:
        return model,loss_epoch, []
