#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

#include <iostream>


int main(void){
const int N = 1000;
thrust::device_vector<float> V1(N);
thrust::device_vector<float> V2(N);
thrust::device_vector<float> V3(N);
thrust::sequence(V1.begin(), V1.end(), 1);
thrust::fill(V2.begin(), V2.end(), 75);
thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
                  thrust::multiplies<float>());
for(int i = 0; i< N; i++){

    std::cout << V3[i] << std::endl;
}

}