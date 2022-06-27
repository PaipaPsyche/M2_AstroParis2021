import numpy as np
import itertools as itr
from decimal import Decimal
import os
from shutil import copyfile

def read_input(path):
    param_values = {}
    param_out = {}
    
    read_vals = False
    with open(path,"r") as f:
        while(True):
            line = f.readline()
            if not line:
                break
            if(line.startswith("---")):
                read_vals = True
                continue
            line = line.split(":")
            name = line[0]
            if(read_vals):
                values = [float(x) for x in line[1].split()]
                param_values[name] = values
            else:
                values = [x for x in line[1].split()]
                param_out[name] = values
    return param_out,param_values


def edit_param_file(path,params,paramout,number=None,suff="Param"):
    lines = []
    with open(path,"r") as f:
        while(True):
            line_ = f.readline()
            if not line_:
                break
            if(line_.startswith("!")):
                lines.append(line_)
                continue
            line = line_.split("!")
            if(len(line)==1):
                lines.append(line_)
                continue
            #print(line)
            name = line[1].strip().split(" ")[0].replace(":","")
            val = line[0]
            if(name in list(params.keys())):
                new_val = '%.2E' % Decimal(params[name])
                sp_ = (len(val)-len(new_val))*" "
            
                new_line = new_val+sp_+"!"+line[1]
                lines.append(new_line)
            else:
                lines.append(line_)
                
    
    extn = path.split(".")[-1]
    new_path = paramout["outdir"][0]+paramout["outname"][0]
    if(number):
        new_path = new_path+str(number)
    new_path = new_path+"_"+suff+"."+extn
    
    with open(new_path,"w") as f:
        for l in lines:
            f.write(l)
        #f.write("! -------- params file produced in batch:")
            
            
def replicate_file(path_template,path_input,copy_files=["Speci","Deple","Chemi"]):
    param_out,param_values = read_input(path_input)
    scenarios = list( itr.product( *list( param_values.values() ) ) )
    names = list( param_values.keys() )          
    logg = []
    
    exn = path_template.split(".")[-1]
    root = os.path.basename(path_template).split("_")[0]
      

    
    
    for i in range(len(scenarios)):
        sc  = {names[j]:scenarios[i][j] for j in range(len(names))}
        edit_param_file(path_template,sc,param_out,number=i+1)
        logg.append("* {}{}".format(param_out["outname"][0],i+1))
        for elem in sc:
            logg.append("  "+elem+" : "+'%.2E' % Decimal(sc[elem]))
        logg.append("--"*10)
        
        for cf in copy_files:
            file_add = os.path.dirname(path_template)+"/"+root+"_"+cf+"."+exn 
            new_file_add = param_out["outdir"][0]+param_out["outname"][0]+str(i+1)+"_"+cf+"."+exn
            try:
                #print(file_add,new_file_add)
                copyfile(file_add,new_file_add)
            except:
                print("Seems like there was a problem trying to copy ",file_add)
                
        
    with open(param_out["outdir"][0]+"log_"+param_out["outname"][0]+".log","w") as lg:
        for l in logg:
            lg.write(l+"\n")
            
    