#include <iostream>
#include <math.h>

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
     y[i] = x[i] + y[i];
}

int main(void)
{
  int RUNS = 1000;
  int N = 1<<20; // 1M elements
  float *x = new float[N];
  float *y = new float[N];

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
     x[i] = 1.0f;
     y[i] = 2.0f;
  }

  // Run RUNS times kernel on 1M elements on the CPU
  for (int i=0; i < RUNS; i++)
     add(N, x, y);

  // Check for errors (all values should be RUNS+2.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
      maxError = fmax(maxError, fabs(y[i]-2.0f-RUNS));
  std::cout << "Max error: " << maxError << std::endl;
     
  // Free memory
  delete [] x;
  delete [] y;
  return 0;
}  
