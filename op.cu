#include <thrust/iterator/counting_iterator.h>
#include <iostream>
#include <thrust/reduce.h>
// create iterators
int main(void){
thrust::counting_iterator<int> first(0);
thrust::counting_iterator<int> last(101);

//first[0]   // returns 10
//first[1]   // returns 11
//first[100] // returns 110

// sum of [first, last)
float e = thrust::reduce(thrust::counting_iterator<int> (0), thrust::counting_iterator<int> (101));   
float f = thrust::reduce(first,last);   

//std::cout << last[0] <<std::endl;
std::cout << e <<std::endl;
std::cout << f <<std::endl;




}