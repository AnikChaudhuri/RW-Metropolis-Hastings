#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/detail/config.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>

#include <iostream>

struct cgamma
{
    template <typename Tuple>
    __device__
    void operator()(Tuple ft)
    {
         
        
            thrust::get<3>(ft) = thrust::get<0>(ft)/(thrust::get<0>(ft)+thrust::get<1>(ft)+thrust::get<2>(ft));
            thrust::get<4>(ft) = thrust::get<1>(ft)/(thrust::get<0>(ft)+thrust::get<1>(ft)+thrust::get<2>(ft));
            thrust::get<5>(ft) = thrust::get<2>(ft)/(thrust::get<0>(ft)+thrust::get<1>(ft)+thrust::get<2>(ft));
            

        }
};

int main(){
    thrust::device_vector<float> a1(10);
    thrust::device_vector<float> b1(10);
    thrust::device_vector<float> c1(10);

    thrust::device_vector<float> a2(10);
    thrust::device_vector<float> b2(10);
    thrust::device_vector<float> c2(10);

    thrust::device_vector<float> alpha1(10);
    thrust::device_vector<float> alpha2(10);
    thrust::device_vector<float> alpha3(10);

    thrust::device_vector<float> alpha(30);

    thrust::sequence(a1.begin(), a1.end(), 1);
    thrust::sequence(b1.begin(), b1.end(), 11);
    thrust::sequence(c1.begin(), c1.end(), 21);

    thrust::copy(a1.begin(), a1.end(), alpha.begin());
    thrust::copy(b1.begin(), b1.end(), alpha.begin()+10);
    thrust::copy(c1.begin(), c1.end(), alpha.begin()+20);

    thrust::copy_n(alpha.begin(), 10, a2.begin() );
    thrust::copy_n(alpha.begin()+10, 20, b2.begin() );
    thrust::copy_n(alpha.begin()+20, 30, c2.begin() );

    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(a2.begin(), b2.begin(), c2.begin(), alpha1.begin(),alpha2.begin(),alpha3.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(a2.end(), b2.end(), c2.end(),alpha1.end(),alpha2.end(),alpha3.end())),cgamma());

    for(int i =0; i < 10; i++){
        std::cout << alpha1[i] <<std::endl;
    }


}