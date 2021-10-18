To compile every C file use "make"
To delete all executable use "make clean"

To run Tokenring (Blocking) use "make run_tr_b"
To run Tokenring (non Blocking) use "make run_tr_nb"
To run ParallelMax (point-to-point) use "make run_pm_ptp"
To run ParallelMax (global) use "make run_pm_gc"


comments:
The example on global communications does not compare the values in pairs, but instead 
reduces the returns of all the processes with MPI_Gather and computes the max over the 
return list. Also this way  may not be comaptible with the case in which the size of the
initial array is not divisible by the number of precesses, in such case, some elements are 
left out from the scatter (i dont really know). I do not know enough(or find enough examples) 
about gatherv to use it properly qnd gave me trouble. Also i would like to know how to do this
with REDUCE (if possible).

