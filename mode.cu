#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/random.h>
#include <fstream>
#include <iostream>
#include <iterator>
using namespace std;
// This example compute the mode [1] of a set of numbers.  If there
// are multiple modes, one with the smallest value it returned.
//
// [1] http://en.wikipedia.org/wiki/Mode_(statistics)

int main(void)
{
    

    ifstream ifs("alpha1.txt", ifstream::in);
			
			std::vector<double> a;
 
			while((!ifs.eof()) && ifs)
			{
			double iNumber = 0;
 
			ifs >> iNumber;
			a.push_back(iNumber);
			}
			ifs.close();
	/*		
    // generate random data on the host
    thrust::host_vector<float> h_data(N);
    for(size_t i = 0; i < N; i++)
        h_data[i] = dist(rng);*/

    // transfer data to device
    thrust::device_vector<float> d_data(500);
    d_data = a;
    
    // print the initial data
    std::cout << "initial data" << std::endl;
    thrust::copy(d_data.begin(), d_data.end(), std::ostream_iterator<double>(std::cout, " "));
    std::cout << std::endl;

    // sort data to bring equal elements together
    thrust::sort(d_data.begin(), d_data.end());
    
    // print the sorted data
    std::cout << "sorted data" << std::endl;
    thrust::copy(d_data.begin(), d_data.end(), std::ostream_iterator<double>(std::cout, " "));
    std::cout << std::endl;

    // count number of unique keys
    size_t num_unique = thrust::inner_product(d_data.begin(), d_data.end() - 1,
                                              d_data.begin() + 1,
                                              0,
                                              thrust::plus<double>(),
                                              thrust::not_equal_to<double>()) + 1;

    // count multiplicity of each key
    thrust::device_vector<double> d_output_keys(num_unique);
    thrust::device_vector<double> d_output_counts(num_unique);
    thrust::reduce_by_key(d_data.begin(), d_data.end(),
                          thrust::constant_iterator<double>(1),
                          d_output_keys.begin(),
                          d_output_counts.begin());
    
    // print the counts
    //std::cout << "values" << std::endl;
    thrust::copy(d_output_keys.begin(), d_output_keys.end(), std::ostream_iterator<double>(std::cout, " "));
   // std::cout << std::endl;

    // print the counts
    //std::cout << "counts" << std::endl;
    thrust::copy(d_output_counts.begin(), d_output_counts.end(), std::ostream_iterator<double>(std::cout, " "));
    //std::cout << std::endl;

    // find the index of the maximum count
    thrust::device_vector<double>::iterator mode_iter;
    mode_iter = thrust::max_element(d_output_counts.begin(), d_output_counts.end());

    double mode = d_output_keys[mode_iter - d_output_counts.begin()];
    double occurances = *mode_iter;
    
    std::cout << "Modal value " << mode << " occurs " << occurances << " times " << std::endl;
    
    return 0;
}