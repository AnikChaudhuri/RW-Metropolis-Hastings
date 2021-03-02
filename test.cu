#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>

#include <thrust/detail/config.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

std::default_random_engine rng;
thrust::host_vector<float> random_vector(thrust::device_vector<float>& u, const size_t N)
{
    
//thrust::default_random_engine rng(seed);
//thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
thrust::host_vector<float> temp(N);
for(size_t i = 0; i < N; i++) {
    std::gamma_distribution<double> u01 (u[i]  ,1.0);
temp[i] = u01(rng);
}
return temp;
}


struct cgamma
{
    template <typename Tuple>
    __device__
    void operator()(Tuple ft)
    {
         
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
            thrust::get<2>(ft) = 1;

        /*
        if(thrust::get<2>(ft) < 3){
            
            thrust::get<2>(ft) = thrust::get<1>(ft);

        }*/
        
    
        
    }
};

struct cgamma1
{
    template <typename Tuple>
    __device__
    void operator()(Tuple ft)
    {
        do{
            
            thrust::get<1>(ft) = thrust::get<1>(ft)+thrust::get<0>(ft);

        }
        while(thrust::get<1>(ft) < 4);
    
        
    }
};

int main(void)
{
    // number of vectors
    const size_t N = 10;
    thrust::device_vector<float> u(N);
    thrust::device_vector<float> v(N);
    thrust::device_vector<float> z(N);
    thrust::device_vector<float> md(N);
    thrust::device_vector<float> md1(N);

    thrust::sequence(u.begin(), u.end(), 1);
    thrust::sequence(v.begin(), v.end(), 1,2);
    thrust::fill(z.begin(), z.end(), 0);
    

    thrust::device_vector<float> A0 = random_vector(u,N);

    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(u.begin(),z.begin(), md.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(u.end(), z.end(), md.end())),cgamma());

    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(u.begin(), md1.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(u.end(), md1.end())),cgamma1());

    thrust::sort(md.begin(),md.end(), thrust::less<float>());
    if(md[0] <= 0 ){
        std::cout << "zero" <<std::endl;
    }
    

        for(int i = 0; i<10; i++){
            std::cout << md[i] << std::endl;
        }
    

    
    
}

