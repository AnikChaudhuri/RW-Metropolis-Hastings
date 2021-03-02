#include <iostream>
#include <ctime>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
using namespace std;
__global__ void AddVectors(float* a, float* b,int count)
{
int idx = threadIdx.x + blockIdx.x * blockDim.x;
if(idx < count)
{
a[idx] += b[idx];
}
}
int main() {
// Number of items in the arrays
const int count = 100;

thrust::device_vector<float> d_a(count);
thrust::device_vector<float> d_b(count);
for(int j = 0; j<2; j++){
thrust::sequence(d_a.begin(), d_a.end(),j+0.2);
thrust::sequence(d_b.begin(), d_b.end(),j+.5);

float * dv_ptra = thrust::raw_pointer_cast(d_a.data());
float * dv_ptrb = thrust::raw_pointer_cast(d_b.data());

dim3 gridSize((count / 512) + 1);
dim3 blockSize(512);
//AddVectors<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(&d_a[0]),thrust::raw_pointer_cast(&d_b[0]),count );
AddVectors<<<gridSize, blockSize>>>(dv_ptra, dv_ptrb,count );

// Print out the results
for(int i = 0; i < count; i++){
cout<<"Result["<<i<<"]="<<d_a[i]<<endl;
}
}
return 0;
}