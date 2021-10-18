program pingpong

    use iso_fortran_env, only : ERROR_UNIT, OUTPUT_UNIT
    use mpi
  
    implicit none
  
    integer :: alloc_stat
    integer :: communicator_size
    integer :: destination
    character(100) :: error_message
    integer :: ierr
    integer :: i_loop
    integer, allocatable, dimension(:) :: message
    integer :: message_size
    integer :: n_loop
    integer :: process_rank
    integer, allocatable, dimension(:) :: reception
    integer :: source
    integer, dimension(MPI_STATUS_SIZE) :: status
    integer :: tag
  
    call mpi_init(ierr)  
    call mpi_comm_rank(MPI_COMM_WORLD, process_rank, ierr)
    call mpi_comm_size(MPI_COMM_WORLD, communicator_size, ierr)
  
    if(communicator_size /= 2) then
      if(process_rank == 0) write(ERROR_UNIT, '(a)') 'There should be 2 processes. Aborting.'
      call mpi_abort(MPI_COMM_WORLD, 1, ierr)
    end if
  
    destination = mod(process_rank+1,communicator_size)
    source = destination
    
    message_size = 10
    allocate(message(message_size), stat=alloc_stat, errmsg=error_message)
    if(alloc_stat /= 0) then
      write(ERROR_UNIT,'(a)') 'allocation error for array message'
      call mpi_abort(MPI_COMM_WORLD, 1, ierr)
    end if
    allocate(reception(message_size), stat=alloc_stat, errmsg=error_message)
    if(alloc_stat /= 0) then
      write(ERROR_UNIT,'(a)') 'allocation error for array reception'
      call mpi_abort(MPI_COMM_WORLD, 1, ierr)
    end if
  
    message = process_rank
    tag = 0
    n_loop = 10
  
    do i_loop = 1, n_loop
      if(process_rank == 0) write(OUTPUT_UNIT,'(a,i0)') 'loop ', i_loop
      call mpi_send(message, message_size, MPI_INTEGER, destination, tag, MPI_COMM_WORLD, ierr)
      call mpi_recv(reception, message_size, MPI_INTEGER, source, tag, MPI_COMM_WORLD, status, ierr)
      write(OUTPUT_UNIT,'(a,i0,a,i0)') 'First integer received by ', process_rank, ': ', reception(1)
    end do
  
    deallocate(message)
    deallocate(reception)
  
    call mpi_finalize(ierr)
  
  end program