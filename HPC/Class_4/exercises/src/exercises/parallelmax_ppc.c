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
   for(int i = 0; i<n ; i++){
        arr_rand[i]= (int)rand();
    }
    
    return arr_rand;
   
}
/* Returns the max value of an array */
int give_max(int* numarr, int n){
   int maxval = 0;
   for(int i = 0; i <n ; i++){
     if(numarr[i]>maxval){
	maxval = numarr[i];
  }
 }
return maxval;
}


   int main(int argc, char **argv)
   {
//       MPI params
      int my_id, ierr, num_procs,rcv,snd,tag,obt;
      
//       array params
      int arr_size, nums_left,nums_per_proc, nums_recv;
      
      // iters
      int i,j,k,idx;
      
// 	MPI elements
      MPI_Status status;
      MPI_Request request = MPI_REQUEST_NULL;


// PARAMS

      tag = 0;
      arr_size = 1000000;
      

      ierr = MPI_Init(&argc, &argv);
     
      ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
      ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
      
  
      
//  for process 0
      if (my_id == 0){
          
        //generate array
        int* arr_to_split = generate_randint_array(arr_size);  
        nums_per_proc = arr_size/num_procs;  // numbers per processs
        
        printf("numbers per process :  %d \n",nums_per_proc);
        
        
        for ( j = 1; j<num_procs-1;j++){
            // distribute the array among the different processes except last one
            
	    idx = j * nums_per_proc;
            int* drop = malloc(sizeof(int) * nums_per_proc);
            drop = &(arr_to_split[idx]);
            
         // send the count of numbers sent and the array of numbers itself
            MPI_Send(&nums_per_proc,1,MPI_INT,j,tag,MPI_COMM_WORLD);
            MPI_Send(drop,nums_per_proc,MPI_INT,j,tag,MPI_COMM_WORLD);
            
            printf("%d nums sent to process %d \n",nums_per_proc, j);

 //            for(int k = 0;k<nums_per_proc ;k++){
  //               printf("S %d %d \n",my_id,drop[k]);
   //          }
        }
		
	   // for the last process assign all values that are left (in case the number of processors do not divide exactly the size of the array)
        
            idx = j * nums_per_proc;
            nums_left = arr_size - idx; 
            int* drop = malloc(sizeof(int) * nums_left);
            drop = &(arr_to_split[idx]);
            
            
            MPI_Send(&nums_left,1,MPI_INT,j,tag,MPI_COMM_WORLD);
            MPI_Send(drop,nums_left,MPI_INT,j,tag,MPI_COMM_WORLD);
//         
            printf("%d nums sent to process %d \n",nums_left, j);
//             for(int k = 0;k<nums_left ;k++){
//                 printf("S %d %d \n",my_id, drop[k]);
//             }


//         MAX OF PROCESS 0 
	
	// array 0 also has a set of number to find max

	    int* dropp = malloc(sizeof(int) * nums_per_proc);
            dropp = &(arr_to_split[0]);

	    int max_num = give_max(dropp,nums_per_proc);

		printf("max number 0 : %d\n",max_num);

      


	

           // RECEIVE MAX FROM OTHER PROCESS
	    int max_recv,maxtot;
	    MPI_Recv(&max_recv,1,MPI_INT,MPI_ANY_SOURCE,tag,MPI_COMM_WORLD,&status);

	      //compare the max found in process 0 with the max received
	    maxtot = max_num;
	    if(max_recv > max_num){
		maxtot = max_recv;			
            }
	
	// print the max value found 
	printf("MAXIMUM VALUE FOUND : %d\n",maxtot);
	printf("RAND_MAX - value = %d\n",RAND_MAX-maxtot);

	    


	    
    // SLAVE PROCESSES
    } else{
	
	    // first receive the number of elements sent
            MPI_Recv(&nums_recv,1,MPI_INT,0,tag,MPI_COMM_WORLD,&status);
            
            int* catch = malloc(sizeof(int) * nums_recv);
            
            // receive the array ofnumber of size nums_recv
            MPI_Recv(catch,nums_recv,MPI_INT,0,tag,MPI_COMM_WORLD,&status);
            
	
            
            printf("%d nums received in process %d \n",nums_recv, my_id);
//             for(int k = 0;k<nums_recv ;k++){
//                 printf(" R %d %d \n",my_id,catch[k]);
//             }


	   // calculate the max of the array received
	    int max_num = give_max(catch,nums_recv);
		printf("max number %d : %d\n",my_id,max_num);


	
          if(my_id < num_procs-1){
// if the process is any but the last, it will receive a value to compare from other process with id = my_id+1
	
	      int nrecv;
              MPI_Recv(&nrecv,1,MPI_INT,my_id+1,tag,MPI_COMM_WORLD,&status);
	
	// compare the value received with the value calculated 
	      int maxtot = max_num;
	    if(nrecv > max_num){		
		maxtot = nrecv;			
            }
		printf("P%d -> P%d: max number found : %d\n",my_id+1,my_id,maxtot);

		// send the max value of these two to the proceess with id = my_id-1
		MPI_Send(&maxtot,1,MPI_INT,my_id-1,tag,MPI_COMM_WORLD);

	}else{
	// for the last prcess, just send the max value calculated to the process with id = my_id-1
	// this process do not receive q value to compare
            MPI_Send(&max_num,1,MPI_INT,my_id-1,tag,MPI_COMM_WORLD);
	}	    
        
    }
 

      ierr = MPI_Finalize();
      return 0;
   }
