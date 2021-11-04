All: cmpl 

cmpl: 	
	@echo 'Compiling C files ...'
	@mpicc MatMulOpenMP.c -o MatMulOpenMP

	@echo 'Finished'
	
run_matmul:
	@echo ' ====== Executing Matrix multiplication (openMP) ====== '
	@mpirun -np 4 ./MatMulOpenMP
