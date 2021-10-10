import numpy as np


def get_params(filename):
    params = {}
    with open(filename,"r") as f:

        while True:
            line = f.readline()

            if not line:
                break

            line = line.split()

            if(len(line)==2):
                try:
                    params[line[0]] = float(line[1])
                except:
                    params[line[0]] = line[1]
    return params


def make_grid(params):

    dx =  give_deltas(params)[0]
    cell_center = np.linspace(params["x_init"],params["x_end"],params["N_points"])
    cell_center = cell_center + dx/2.

    return cell_center

def initial_condition(x,params):
    density = np.array([params["density_left"]  if x[i]<=params["x_split"] else params["density_right"] for i in range(len(x)) ])
    return density

def give_deltas(params,coef=0.5):
    dx = (params["x_end"]-params["x_init"])/params["N_points"]
    dt = coef*dx/params["velocity"]
    return dx,dt

def advance_step(density,x,params):

    dx,dt = give_deltas(params)
    grid_dens = np.zeros(len(x)+2)

    vel = params["velocity"]

    grid_dens[1:-1] = density[:]
    grid_dens[0] = grid_dens[1]
    grid_dens[-1] = grid_dens[-2]

    new_dens = np.array([ grid_dens[i]+(dt/dx)*(grid_dens[i-1]*vel - grid_dens[i+1]*vel) for i in range(1,len(grid_dens)-1) ])


    return new_dens




# save/load

def save_obj(obj,name):
    np.save("../saves/"+name+".npy",obj)

def retrieve_save(name):
    return np.load("../saves/"+name+".npy")


def main():
    p = get_params("../params/params.txt")
    dx,dt = give_deltas(p)
    k = make_grid(p)
    dens = initial_condition(k,p)
    #save_obj(k,"centers")
    #print("Retrieved values: ")
    #print(retrieve_save("centers"))
    #print(dens)
    for i in range(10):

        dens = advance_step(dens,k,p)
        print(dens)







if __name__ == '__main__':
    main()
