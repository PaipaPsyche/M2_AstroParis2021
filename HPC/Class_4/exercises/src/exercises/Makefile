All: cmpl 

cmpl: 	
	@echo 'Compiling C files ...'
	@mpicc tokenring_b.c -o tokenring_blocking
	@mpicc tokenring_nb.c -o tokenring_nonblocking
	@mpicc parallelmax_ppc.c -o parallelmax_ptp
	@mpicc parallelmax_gc.c -o parallelmax_gc
	@echo 'Finished'
	
run_tr_b:
	@echo ' ====== Executing Tokenring (blocking) ====== '
	@mpirun -np 8 ./tokenring_blocking 

run_tr_nb:
	@echo ' ====== Executing Tokenring (non-blocking) ====== '
	@mpirun -np 8 ./tokenring_nonblocking 

run_pm_ptp:
	@echo ' ====== Executing Parallel max (point to point) ====== '
	@mpirun -np 8 ./parallelmax_ptp

run_pm_gc:
	@echo ' ====== Executing Parallel max (global) ====== '
	@mpirun -np 8 ./parallelmax_gc 

clean: 
	rm parallelmax_ptp
	rm parallelmax_gc
	rm tokenring_blocking
	rm tokenring_nonblocking
