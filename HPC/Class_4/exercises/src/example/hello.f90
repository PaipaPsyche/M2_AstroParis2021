program hello

    use iso_fortran_env, only : ERROR_UNIT, OUTPUT_UNIT
    use mpi
  
    implicit none
  
    integer :: ierr 
    integer :: process_rank
    integer :: communicator_size
    
    call mpi_init(ierr)
    call mpi_comm_rank(MPI_COMM_WORLD, process_rank, ierr)
    call mpi_comm_size(MPI_COMM_WORLD, communicator_size, ierr)
    
    write(OUTPUT_UNIT, '(a,i0)') 'Hello from process ', process_rank
    if(process_rank == 0) write(OUTPUT_UNIT, '(a,i0,a)') 'There are ', communicator_size, ' processes.'
    
    call mpi_finalize(ierr)
    
end program hello