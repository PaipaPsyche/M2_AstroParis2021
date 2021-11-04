#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


#define N1 ((int)4)
#define N2 ((int)3)


int A1[N1][N2];
int A2[N2][N1];
int R[N1][N1];
int i,j,k;


void printMatrix(int m, int n, int arr[m][n]){
    
    for(i =0;i<m;i++){
        for(j =0;j<n;j++){
             printf("%d  " ,arr[i][j]);
        
        }
        printf("\n");
        
    }
}

int main() 
{
    
//     omp_set_num_threads(omp_get_num_procs());
    for (i= 0; i< N1; i++){
        for (j= 0; j< N2; j++)
        {
            A1[i][j] = 2*i-j;
        }
	}
	
	for (i= 0; i< N2; i++){
        for (j= 0; j< N1; j++)
        {
            A2[i][j] = j+i-1;
        }
	}

    #pragma omp parallel for private(i,j,k) shared(A,B,C)
    for (i = 0; i < N1; i++) {
        for (j = 0; j < N1; j++) {
            R[i][j] = 0;
            for (k = 0; k < N2; k++) {
                R[i][j] += A1[i][k] * A2[k][j];
            }
        }
    }
    
    
    #pragma omp critical
    printf("Input matrix 1\n");
    printMatrix(N1,N2,A1);
    printf("Input matrix 2\n");
    printMatrix(N2,N1,A2);
    printf("Output matrix\n");
    printMatrix(N1,N1,R);



}
 
