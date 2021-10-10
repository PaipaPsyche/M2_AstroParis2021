import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg

# genenrqtes q matrix with the chebyshev polynomial coefficients
def generate_chebyshev(n,func_type=1):
    # a new array for polynomials up to n degree to store coefficients
    ans = np.zeros([n+1,n+1]) 
    
    ans[0,0] = 1
    
    ans[1,1] = func_type
    
    for i in range(2,n+1):
        ans[i,1:] = 2 * ans[i-1,:-1] 
        ans[i,:] = ans[i,:] - ans[i-2,:]
    return ans

#given a coefficient ,atrix, returns the coefficients of the derivative of each polynomial
def derivate_coefs(coefs):
    new_c = np.zeros([len(coefs),len(coefs)])
    for p in range(len(coefs)):
        for i in range(len(coefs)-1):
            new_c[p][i] = coefs[p][i+1]*(i+1)
    return new_c


# generate a polynomial given the coefficients
def generate_poly(coefs):
    def func(x):
        total = 0
        for j in range(len(coefs)):
            total = total + (coefs[j])*(x**j)
        return total
    return func

# using generated coefficients, generate a polynomial python function; can be evluated over any x
def func_chebyshev(coefs,func_type=1):
    funcs = []
    for i in range(len(coefs)):  
        
        # get coefficients of i-th polynomial
        cc = coefs[i,:]
        
        #print(cc)
        func_n = generate_poly(cc)
        funcs.append(func_n)
    return funcs
    
    
    
    
def project_function(f,pols,eps=1e-6):
    # storing the coefficients for projection
    coefs = []
    
    
    for i in range(len(pols)):
        Ti = pols_list[i] # for eqch base polynomial
        
        # calculate the inner product with the arbitrary function
        def new_fun(x):
            return Ti(x)*r_func(x)/np.sqrt(1-x*x)
        # calculate the norm of the polynomial 
        def norm_fun(x):
            return Ti(x)*Ti(x)/np.sqrt(1-x*x)

        # integrate to calculate the inner products
        res,error = intg.quad(new_fun,-1,1,epsabs=eps)
        res_norm,error_norm = intg.quad(norm_fun,-1,1,epsabs=eps)
        
        # append the coefficient of projection for the i-th polynomial
        coefs.append(res/res_norm)

    # create the projected function as a linear combination of the m polynomials 
    def new_fun(x):
        total = 0
        for i in range(len(coefs)):
            total = total + coefs[i]*pols[i](x)
        return total
    
    return new_fun



# generate weights and colloc for a given index 
def get_weight_colloc(idx,m):
    wght = 0
    cllc = 0
    m=m-1
    
    if idx == 0 or idx == m:
        wght, cllc = [np.pi/(2*m), -np.cos(idx*np.pi/m)]
    else:
        wght, cllc = [np.pi/m, -np.cos(idx*np.pi/m)]
        
    return [wght,cllc]
        
# generate gamma for given index
def get_gamma(idx,pols):

    total = 0
    for i in range(len(pols)):
        wght,cllc = get_weight_colloc(i,len(pols))
        total = total + ( (pols[idx](cllc))**2 ) * wght
    return total
    
# generate fn for any function and index
def get_fn(idx,f,pols):
    gamma = get_gamma(idx,pols)
    total = 0
    
    for i in range(len(pols)):
        wght,cllc = get_weight_colloc(i,len(pols))
        total = total + f(cllc)*pols[idx](cllc) * wght
    
    return total/gamma


def get_f_coefs(f,pols):
    fs = []
    for i in range(len(pols)):
        fn = get_fn(i,f,pols)
        fs.append(fn)
    return fs
# generate the interpolant function of any function f

def get_interpolant(f_coefs,pols):
    if(len(f_coefs)!=len(pols)):
        print("Different number of polynomials and coefficients : ",len(pols)," - ",len(f_coefs))
    
    def interpolant(x):
        total = 0
        for i in range(len(pols)):
            fn = f_coefs[i]
            total = total + fn * pols[i](x)
        return total
    return interpolant
    
    
# ERRORS====================

def error_MSE(y,y_pred): # Mean Squared Error Function
    return np.mean((y-y_pred)**2)

def error_max(y,y_pred):
    return np.max(np.abs(y-y_pred))