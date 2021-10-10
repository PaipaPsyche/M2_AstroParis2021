import numpy as np


def read_config(fname):
    """ read the configuration file and return the parameters """
    # Match parameter names
    fp = open(fname, "r")
    lines = fp.readlines()
    config_params = ["x_start", "x_end", "N_points"]
    params = {}

    for line in lines:
        elements = line.split()
        if elements[0] in config_params:
            params[elements[0]] = float(elements[1])
        else:
            print("Warning, unknown parameter in config file:")
            print(elements[0], elements[1])
    return params


def create_grid(x_start, x_end, N_points):
    """ WE ARE USING THE MID POINTS of the CELL"""
    off_Set = (x_end - x_start) / (N_points)
    xgrid = np.linspace(x_start + 0.5*off_Set,
                        x_end - 0.5 * off_Set, int(N_points))
    return xgrid


def main():
    """ Create a grid using the config values """
    # we want to use the mid points, not the end points
    fname = "params.txt"
    params = read_config(fname)
    xgrid = create_grid(params["x_start"], params["x_end"], params["N_points"])
    print(xgrid)


if __name__ == "__main__":
    main()
