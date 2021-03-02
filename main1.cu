#include <iostream>
using namespace std;
__global__ void add(int a, int b, int *c)
{
*c = a * b;
}  
 
int main(void) {
int a, b, c; // host copies of a, b, c
int *d_c; // device copies of a, b, c
int size = sizeof(int);
 
// Allocate space for device copies of a, b, c
//cudaMalloc((void **)&d_a, size);
//cudaMalloc((void **)&d_b, size);
cudaMalloc((void **)&d_c, size);
 
// Setup input values
a = 2;
b = 17;
 
 
//cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
//cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
 
// Launch add() kernel on GPU
add<<<1,1>>>(a, b, d_c);
 
// Copy result back to host
cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
cout << "answer is " << c <<endl;
// Cleanup
cudaFree(d_c);
 
 
//return 0;
}