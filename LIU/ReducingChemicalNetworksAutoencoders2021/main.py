import iterative_abundances as it_ab
import iterative_params as it_p


if __name__ == '__main__':

    name_ = "final"
    N = 150

    n_names =it_ab.generate_train_sample(N,it_ab.CHIMESPATH+"Data/iterCorr1_Param.dat","Data/","Out/",name_)
    # the _1117 is for the 17 of Nov . the date is used to tag the samples generated
    it_ab.compile_database(name_+"_1207",10,exclude=it_ab.excluded)



# in case sample generated but not compiled
# print("Destination file: CHIMES_0.6/Out/sample_20211107")
# xnames = []
# namedate ="nsample_20211107"
# xdirname = "CHIMES_0.6/Out/"+namedate
# with open("names.txt","r") as f:
#     xnames = f.readlines()
# xnames = [xdirname+"/"+x.replace("\n","")+"/"+x.replace("\n","") for x in xnames if "info" not in x]
#
# compile_database(xnames,xdirname+"/"+namedate)
