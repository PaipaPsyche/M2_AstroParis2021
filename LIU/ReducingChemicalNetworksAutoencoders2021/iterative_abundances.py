### IMPORTS
# utils
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import pandas as pd
# time
import time
from datetime import datetime,timedelta
#sys
import os
from shutil import copyfile
import subprocess

# relative path to Chimes (these files work better in the parent dir of chimes)
CHIMESPATH ="CHIMES_0.6/"

# excluded tags from .graph file (to create dataset)
excluded = ['zeta','nh','T(K)','bthb','ynn','bnn','bee','bchim','gamh2','gamgr','gampeg','zlambd','xlamoh','xlamco','xlah2o','xlamc','xlamcp','xlamo','xlambn','wthb','xlambe','gamrc']




def plot_info(path_info,path_img):
    """
    Once built the .info file, this function can plot the abundances for C and C+
    chosen randomly: Ideally its a uniform log distribution.

    PARAMS:
    (str) path_info: Path to info file
    (str) path_img: Path for generating the plot

    RETURN:
    (array) x: ordered list of C abundances
    (array) y: ordered list of C+ abundances
    """

    # Reading and interpreting lines
    lines =[]
    with open(path_info,"r") as f:
        lines = f.readlines()
    # omit commentary lines
    lines = [l for  l in lines if not l.startswith("#")]
    # x,y values = C,C+ abundances
    lines = [l.split()[1:] for l in lines]
    x,y = np.array([[float(l[0].split(":")[1]),float(l[1].split(":")[1])] for l in lines]).T


    #PLOTTING
    plt.figure(figsize=(8,8))
    plt.scatter(x,y,s=5)
    plt.xlabel("$\chi_C$",fontsize=15)
    plt.ylabel("$\chi_{C+}$",fontsize=15)
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(path_img,bbox_inches="tight")

    return x,y


def eformat(f, prec, exp_digits):
    """
    Format any number in UpperCase scientific notation
    PARAMS:
    (float) f: number to format
    (int)   prec: decimal precision
    (int)   exp_digits: digits of exponent

    RETURN:
    (str) str_num: formatted number
    """
    s = "%.*E"%(prec, f)
    mantissa, exp = s.split('E')
    # add 1 to digits as 1 is taken by sign +/-
    str_num = "%sE%+0*d"%(mantissa, exp_digits+1, int(exp))
    return str_num




def change_speci(path,newvals,new_path):
    """
    Given the path to a template _Speci File, a new file is created (in a
    provided path) with the changes defined in the newvals directory
    PARAMS:
    (str)   path: path to template _Speci file
    (dir)   newvals: directory containing {species name: new abundace}
    (str)   new_path: path to save resulting _Speci file

    RETURN:
    """
    lines = []
    with open(path,"r") as f:

        #read file line by line
        while(True):
            line_ = f.readline()
            if not line_:
                break
            #append unedited comments to new file
            if(line_.startswith("#")):
                lines.append(line_)
                continue

            #split line in whitespaces
            line = line_.split()

            # append file markers like 888 (for continuity, i dont really know what they do)
            if(len(line)==1):
                lines.append(line_)
                continue

            # if line its not a file marker, and the line has a species name
            # that appears in the new vals dir ...
            if line[1] in newvals.keys():
                val = newvals[line[1]]
                #print(line[1],val)
                newline =""
                newline = newline+line_[:44] # first 44 char are the same
                newline = newline+'%.3E' % Decimal(val) # append the new value
                newline = newline+line_[53:] # append the rest of the line
                lines.append(newline)
            else:
                lines.append(line_)

    # write new file
    with open(new_path,"w") as f:
        for l in lines:
            f.write(l)

def change_deple(path,newvals,new_path):
    """
    Given the path to a template _Deple File, a new file is created (in a
    provided path) with the changes defined in the newvals directory
    PARAMS:
    (str)   path: path to template _Deple file
    (dir)   newvals: directory containing {elemnt name: new depletion value}
    (str)   new_path: path to save resulting _Deple file

    RETURN:
    """
    lines = []

    # Reading the file line by line
    with open(path,"r") as f:
        while(True):
            line_ = f.readline()
            if not line_:
                break
            # not changing comments
            if(line_.startswith("#")):
                lines.append(line_)
                continue
            # split in white spaces
            line = line_.split()
            # keep file markers
            if(len(line)==1):
                lines.append(line_)
                continue

            # if the line Element ID matches ne of hte newvals directory...
            if line[2] in newvals.keys():
                val = newvals[line[2]]  # get the new value
                newline =""
                newline = newline+line_[:4] # first 4 char are the same
                newline = newline+eformat(Decimal(val), 4, 3) # append formatted decimal
                newline = newline+line_[15:] # append the rest of the line
                lines.append(newline)
            else: # if not in dir, just append the line
                lines.append(line_)
    # write file
    with open(new_path,"w") as f:
        for l in lines:
            f.write(l)



def generate_cco_abundances(lims=[-6,-3]):
    """
    Generates a pair of numbers randmly chosen from a log unifrorm
    distribution. C,C+ abundances values.
    PARAMS:
    (list(float))   lims: lower and upper limmit in log space (1e-6 to 1e-3 default)

    RETURN:
    (list(float))   proposed abundances for C and C+

    """
    num1 = np.random.uniform(lims[0],lims[1])
    num2 = np.random.uniform(lims[0],lims[1])

    return [10**num1,10**num2]


def generate_scenario(paramspath,outdir,name):
    """
    Given the path to a template _Params File (and assuming the other
    _SPeci, _Deple and _Chemi files are in the same folder), A new
    random scenario is generated (with random abundaces for C and C+)
    resulting in the creation of 4 new input files (copy of all 4 mentioned)
    with the given name replacing the original file name (format name_Deple.dat)

    PARAMS:
    (str)   paramspath: path to template _Params file
    (str)   outdir: path to folder where new files should be stored
    (str)   name: new name for all files

    RETURN:
    (list(float)) abundances of C,C+ for the scenario

    """
    exn = paramspath.split(".")[-1] # file extension (.dat)
    root = os.path.basename(paramspath).split("_")[0] #(original name)
    vals = generate_cco_abundances() # new random abundances
    indir = os.path.dirname(paramspath) # directory of params file


    # assumed path of original depletion file
    file_speci = indir+"/"+root+"_Speci."+exn
    # directory to edit speci file
    editspeci = {"c":vals[0],"c+":vals[1],"o":vals[0]+vals[1]}
    # new path of speci file
    new_file_speci = outdir+"/"+name+"_Speci."+exn
    # change speci file
    change_speci(file_speci, editspeci, new_file_speci)


     # assumed path to original depletion file
    file_deple = indir+"/"+root+"_Deple."+exn
    # directory to edit deple file
    editdeple = {"C":vals[0]+vals[1],"O":vals[0]+vals[1]}
    # new path to deple file
    new_file_deple = outdir+"/"+name+"_Deple."+exn
    # change deple file
    change_deple(file_deple, editdeple, new_file_deple)


    # assumed path to Chemi file
    file_chemi = indir+"/"+root+"_Chemi."+exn
    # new path of Chemi file
    new_file_chemi = outdir+"/"+name+"_Chemi."+exn
    try:
        # try copy the file
        copyfile(file_chemi,new_file_chemi)
    except:
        print("Seems like there was a problem trying to copy ",file_chemi)

    # new path of params file
    new_file_params = outdir+"/"+name+"_Param."+exn
    try:
        # try copy the file in new location
        copyfile(paramspath,new_file_params)
    except:
        print("Seems like there was a problem trying to copy ",paramspath)

    return vals




def gen_execute(paramspath,outdir,name):
    """
    Given the path to a template _Params File (and assuming the other
    _SPeci, _Deple and _Chemi files are in the same folder), A new
    random scenario is generated qnd Executed in the CHIMES script.

    PARAMS:
    (str)   paramspath: path to template _Params file
    (str)   outdir: path to folder where new files should be stored
    (str)   name: new name for all files

    RETURN:
    (list(float)) abundances of C,C+ for the scenario

    """
    # eenrate new scenario
    vals = generate_scenario(paramspath,outdir,name)

    # open a file to store the cosnole verbose of CHIMES
    with open('{}{}_exec.txt'.format(outdir,name), 'w') as f:

        #execute the scenario given the name of the input files
        subprocess.call(['./Chem_Evol {}'.format(name)],cwd=CHIMESPATH,shell=True,stdout=f)
        # cut the last 5 rows of the exec file (steady state information) qnd save it in name_depldata.text file
        subprocess.call(['echo "$(tail -5 {}_exec.txt)" > ../Out/{}/{}_depldata.txt'.format(name,name,name)],cwd=outdir,shell=True)
        #rm exec file
        subprocess.call(['rm {}_exec.txt'.format(name)],cwd=outdir,shell=True)

    return vals



def generate_train_sample(N,paramspath,paramdir,outdir,name,date_flag=False):
    """
    Generate a training sample raw data set. Genrates N scenarios and executes them,
    Storing all used input files in a single directory with the same name in the Data folder
    of chimes. all scenarios are named name_i where i is the number of this scenario (1,N).
    Creates a .info file with the abndances for each scenario, elapsed time of execution and
    steady state information.

    PARAMS:
    (int)   N: number of scenairios to generate
    (str)   paramspath: path to template _Params file
    (str)   paramdir: path to folder with param files (realtive to CHIMES folder , in this case /Data)
    (str)   outdir: path to CHIMES oututs (/Out since its also relative to chimes folder)
    (str)   name: new name for all files

    RETURN:
    (dir) directory with folder name and all scenarios name

    """
    #measuring time
    timestart = time.time()

    # appending chimes path
    paramdir = CHIMESPATH+paramdir
    outdir = CHIMESPATH+outdir

    print("Creating {} samples with name {} :".format(N,name))


    # date flag for batch name
    if(date_flag):
        startdate = datetime.today().strftime('%m%d')
        name = name+"_"+startdate


    names = [] # scenario names
    description=[] # info file lines

    for i in range(1,N+1):

        print("generating sample ",i,"...")

        newname = name+"_"+str(i)
        names.append(outdir+name+"/"+newname+"/"+newname)
        # execute new scenario
        vals = gen_execute(paramspath,paramdir,newname)

        # entry for info file
        new_desc = "{}  c:{:.3e}  c+:{:.3e}  o:{:.3e}".format(newname,vals[0],vals[1],vals[0]+vals[1])
        description.append(new_desc)

    print("cleaning...")
    # create destiny directory for input files
    subprocess.call(['mkdir {}'.format(name)],cwd=paramdir,shell=True)
    #  move all related files to destiny directory
    subprocess.call(['mv {}_* {}'.format(name,name)],cwd=paramdir,shell=True)

    #make directory for output files
    subprocess.call(['mkdir {}'.format(name)],cwd=outdir,shell=True)
    subprocess.call(['mv {}_* {}'.format(name,name)],cwd=outdir,shell=True)

    # Generating info file
    print("generating log file...")
    elapsed = time.time() - timestart

    with open("{}{}/{}.info".format(outdir,name,name),"w") as ff:
        # count all depldata.txt files that report reaching a steady state
        subprocess.call(['var="$(grep "Steady state has been computed" {}/*/*_depldata.txt | wc -l)"; echo "# $var models reached a stready state" >> {}/{}.info'.format(name,name,name)],cwd=outdir,shell=True)
        # elapsed time of executin
        ff.write("# Elapsed_time  hh:mm:ss  {}\n".format(timedelta(seconds=elapsed)))
        # abundances for scenario
        for l in description:
            ff.write(l+"\n")
    # plot info file
    plot_info("{}{}/{}.info".format(outdir,name,name), "{}{}/{}.png".format(outdir,name,name))

    print("Done.")
    return {"filenames":names,"dirname":name}



# build the database
def reformat_graph_file(name_in,name_out):
    """
    graph files are not easy to read in pandas. this method reads the file and
    reformat it into a destiny file

    PARAMS:
    (str)   name_in: path to original graph file
    (str)   name_out: path to new formatted file


    RETURN:

    """
    with open(name_out,"w") as k:
        with open(name_in,"r") as f:
            while True:
                line = f.readline()
                if(not line):
                    break
                line=line.strip()
                k.write(",".join(line.split()).strip()+"\n")


def get_graph_data(name,ab_list=None,include_t = True,exclude=None):
    """
    with the path to a graph file, it i reformatted and read as a pandas dataframe

    PARAMS:
    (str)       name: root name of the file ("{DIRPATH}/name_i")
    (list(str)) ab_list: list of species column keys to include in dataframe. all included if None
    (bool)      inclde_t: include time column
    (list(str)) exclude: list of column keys to exclude


    RETURN:
    Dataframe   Dataframe with the selected columns of the graph file

    """
    reformat_graph_file(name+".graph",name+"_rf.graph")

    if exclude:
        headers = [*pd.read_csv(name+"_rf.graph",sep=",", nrows=1)]
        data = pd.read_csv(name+"_rf.graph",sep=",", usecols=[c for c in headers if c not in exclude])
    else:
        data = pd.read_csv(name+"_rf.graph",sep=",",engine="python")

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


#convfactor from Myrs to seconds (s/Myrs)
def get_deriv_data(name,ab_list=None,exclude = None,include_t = True,convfactor =31556925216000):
    """
    with the path to a deriv file, it i reformatted and read as a pandas dataframe

    PARAMS:
    (str)       name: root name of the file ("{DIRPATH}/name_i")
    (list(str)) ab_list: list of species column keys to include in dataframe. all included if None
    (bool)      inclde_t: include time column
    (list(str)) exclude: list of column keys to exclude
    (int)       convfactor: comvertion factor (multiplicative) applied to time derivative values. Defult - the number of second in a 1e6 years


    RETURN:
    Dataframe   Dataframe with the selected columns of the deriv file

    """

    # file is reformatted and saved
    reformat_graph_file(name+".deriv",name+"_rf.deriv")

    # exclude defined columns otherwise import all columns
    if exclude:
        headers = [*pd.read_csv(name+"_rf.deriv",sep=",", nrows=1)]
        data = pd.read_csv(name+"_rf.deriv",sep=",", usecols=[c for c in headers if c not in exclude])
    else:
        data = pd.read_csv(name+"_rf.deriv",sep=",",engine="python")

    col_list = []
    if not ab_list:
        return data
    elif include_t:
        col_list.append("t(Myrs)")
    col_list = ["t(Myrs)"]+["t(yrs)"] + ab_list
    data.drop(data.tail(1).index,inplace=True) # drop last n=1 rows (supressing final drop in data (header?) )
    data.drop(data.head(1).index,inplace=True) # drop first n=1 rows (supressing initial peak (???) )
    data["t(yrs)"] = data["t(Myrs)"]*1e6

    # convert abundances derivate in time from seconds to Myears
    for n in ab_list:
        data[n] = data[n].apply(lambda x : x*convfactor)

    return data[col_list]



def compile_database(dirname,pref=None,ab_list=None,exclude = None,include_t = True):
    """
    compile the graph and deriv data in big dataframes that are then compressed and saved

    PARAMS:
    (str)       dirname: root name of the directories (execution root name for in/out dirs)
    (str)       pref: preffix path to the CHIMESPATH directory. Defaut - None
    (list(str)) ab_list: list of species column keys to include in dataframe. all included if None
    (bool)      inclde_t: include time column
    (list(str)) exclude: list of column keys to exclude

    RETURN:

    """
    # name of output directory
    outname = CHIMESPATH + "Out/" +dirname +"/"+dirname
    # create out directory for compressed csv files
    subprocess.call(['mkdir {}_csv'.format(outname)],shell=True)
    # list all dirnames for different scenarios, each dir constains graph and deriv files
    names = [CHIMESPATH+"Out/"+dirname+"/"+x+"/"+x for x in os.listdir(CHIMESPATH+"/Out/"+dirname)  if not x.endswith("info") and not x.endswith("csv") and not  x.endswith("png")]


    ("compiling database of ",len(names)," samples")
    # for each dirname (scenario) reformat and get the graph and deriv data
    for n in names:
        iter = n.split("_")[-1]
        print("compiling sample "+iter+"...")
        name = n if not pref else pref+n
        derivdata = get_deriv_data(name,ab_list=ab_list,exclude=exclude,include_t=include_t)
        graphdata = get_graph_data(name,ab_list=ab_list,exclude=exclude,include_t=include_t)
        # to concatenate, add a fag to indicate it de data row is from the graph or deriv file
        derivdata["datatype"]="deriv"
        graphdata["datatype"]="graph"

        # concatenate graph and deriv data in one table
        # delete variables to save memory
        newdat = pd.concat([graphdata,derivdata])
        del derivdata
        del graphdata

        # flag to inidicate the number tag of the scenario
        newdat["name"]=iter
        # name of the outfile in formate  OUTDIR/name_i.csv.gz
        fileout = '{}_csv'.format(outname)+"/"+n.split("/")[-1]+".csv.gz"
        # save to compressed csv (gzip)
        newdat.to_csv(fileout, index=False,compression="gzip")


    print("done")
