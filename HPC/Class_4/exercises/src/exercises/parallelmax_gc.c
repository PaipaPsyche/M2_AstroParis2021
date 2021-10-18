#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>





/* Returns a rqndom array of n size */
int * generate_randint_array(int n){
    /* Intializes random number generator*/
    time_t t;
    
   srand((unsigned) time(&t));
   int* arr_rand = malloc(sizeof(int) * n);
   for(int c = 0; c<n ; c++){
        arr_rand[c]= rand();
    }
    
    return arr_rand;
   
}
/* Returns the max value of an array */
int give_max(int* numarr, int n){
   int maxval = 0;
   for(int g = 0; g <n ; g++){
     if(numarr[g]>maxval){
	maxval = numarr[g];
  }
 }
return maxval;
}


   int main(int argc, char **argv)
   {
//       MPI params
      int my_id, ierr, num_procs,tag,obt;
      
//       array params
      int arr_size, nums_left,nums_per_proc, nums_recv;
      
      // iters
      int i,j,k,l,m,idx;
      
      //ARRAY
      arr_size = 1000000;
      
// 	MPI elements
      MPI_Status status;
      MPI_Request request = MPI_REQUEST_NULL;

      // GLOBAL DATA
int *globaldata=NULL;
      tag = 0;
      
      
      ierr = MPI_Init(&argc, &argv);
     
      ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
      ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
      
 nums_per_proc = arr_size/num_procs;  
      
//  for process 0
      if (my_id == 0){
          
        //generate array
        globaldata = generate_randint_array(arr_size);  
        nums_per_proc = arr_size/num_procs;  // numbers per processs
        printf("numbers per process :  %d \n",nums_per_proc);
        
        

    } 
        // array to receive piece of global array
        int* localdata = malloc(sizeof(int) * nums_per_proc);
        
        // scatter the global array into all processes (each one with a slice of size nums_per_proc)
        MPI_Scatter(globaldata, nums_per_proc, MPI_INT, localdata, nums_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

        
        
        // COMPUTE LOCAL MAX
        int sub_max = give_max(localdata,nums_per_proc);
        printf("Max found in process %d : %d\n",my_id,sub_max);
        
        // array to contain the local max of each process (num_procs size)
        int *sub_maxs;
        if (my_id == 0) { // other process different from root ignore buffer
        sub_maxs = malloc(sizeof(int) * num_procs);
        }
        
        // send local maximum and build the receive array sub_maxs 
        MPI_Gather(&sub_max, 1, MPI_INT, sub_maxs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    
    // the root prcess computes the max value among the local maxima received from all processes
    if(my_id==0){
        
        int maxtot = give_max(sub_maxs,num_procs);
        
        printf("MAX VALUE FOUND :%d\n",maxtot);
        printf("RAND_MAX - value = %d\n",RAND_MAX-maxtot);

    }
 
      ierr = MPI_Finalize();
      return 0;
   }
