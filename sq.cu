#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <iostream>

//   This example computes the norm [1] of a vector.  The norm is 
// computed by squaring all numbers in the vector, summing the 
// squares, and taking the square root of the sum of squares.  In
// Thrust this operation is efficiently implemented with the 
// transform_reduce() algorith.  Specifically, we first transform
// x -> x^2 and the compute a standard plus reduction.  Since there
// is no built-in functor for squaring numbers, we define our own
// square functor.
//
// [1] http://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm


// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x, const T&y ) const { 
            float f = x + y ;
            return f;
        }
};

int main(void)
{
    // initialize host array
    float x[4] = {1.0, 2.0, 3.0, 4.0};
    float y[4] = {2.0, 2.0, 5.0, 6.0};
    //float z[4] = {3.0, 6.0, 15.0, 8.0};


    // transfer to device
    thrust::device_vector<float> d_x(x, x + 4);
    thrust::device_vector<float> d_y(y, y + 4);
   // thrust::device_vector<float> d_z(z, z + 4);

    // setup arguments
    square<float>        unary_op;
    //thrust::plus<float> binary_op;
    //float init = 0;

    // compute norm
    thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_y.begin(),unary_op) ;

    for(int i = 0; i<4; i++)
        std::cout << d_y[i] <<std::endl;

    //std::cout << "norm is " << norm << std::endl;

    return 0;
}