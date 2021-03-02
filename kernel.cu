#include <iostream>
#include <ctime>
#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
using namespace std;
__global__ void AddVectors(float* a, float* b,int count)
{
    //int j = 0;
    
int idx = threadIdx.x+ blockIdx.x * 10;

if(idx < count)
{
    //do{
        //j++;
        float d = a[idx]-1.0/3;
        float c = 1.0/sqrtf(9*d);
        //float u1 = curand_uniform(&state); //one uniform for proposal
        //float u2 = curand_normal(&state); //one uniform for accept/reject step
        //float v = powf((1 + c*u1), 3);
        
        //a[idx] += b[idx]+j;
        a[idx] = v;
        __syncthreads();
    //}while(a[idx] < 15.0);

}
}



int main() {
// Number of items in the arrays
const int count = 10;

thrust::device_vector<float> d_a(count);
thrust::device_vector<float> d_b(count);

thrust::sequence(d_a.begin(), d_a.end(),1);
thrust::sequence(d_b.begin(), d_b.end(),.5);

float * dv_ptra = thrust::raw_pointer_cast(d_a.data());
float * dv_ptrb = thrust::raw_pointer_cast(d_b.data());


    //setup<<<1, 100>>>(d_states);

//curandStatePhilox4_32_10_t *philox;
//cudaMalloc((void **)&philox, 64 * 64 * sizeof(curandStatePhilox4_32_10_t));

dim3 gridSize(1);
dim3 blockSize(10);


//for(int j = 0; j<100; j++){


AddVectors<<<gridSize, blockSize>>>(dv_ptra, dv_ptrb, count);


for(int i = 0; i < count; i++){

    cout<<"Result["<<i<<"]="<<d_a[i]<<endl;
    
    }

//}
cudaFree(d_states);
/*
for(int i = 0; i < count; i++){

    cout<<"Result["<<i<<"]="<<d_a[i]<<endl;
    
    }*/


}