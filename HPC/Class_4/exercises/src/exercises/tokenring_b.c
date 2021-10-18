#include <mpi.h>
#include <stdio.h>
   int main(int argc, char **argv)
   {
       // MPI params
      int my_id,  ierr, num_procs,rcv,snd,tag,obt;
      // MPI elements
      MPI_Status status;
      
      ierr = MPI_Init(&argc, &argv);
      
      ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
      ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
      
      tag = 0;

      // Receive information from process rcv
      rcv =  (my_id-1) % num_procs;
      // Send infromation to process snd
      snd =  (my_id+1) % num_procs;
      
      MPI_Send(&my_id,1,MPI_INT,snd,tag,MPI_COMM_WORLD);
      MPI_Recv(&obt,1,MPI_INT,rcv,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      
      printf("Process %d received the message %d\n",my_id,obt);
      
      ierr = MPI_Finalize();
      return 0;
   }
