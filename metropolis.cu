#include <iostream>
#include <chrono>
#include <random>
#include <vector>

#include <thrust/device_vector.h>


void dsample(double al1, double al2, double al3, double *alpha1, double *alpha2, double *alpha3 ){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    
    std::default_random_engine generator(seed);
    
        std::gamma_distribution<double> distribution (al1,1.0);
        std::gamma_distribution<double> distribution1 (al2,1.0);
        std::gamma_distribution<double> distribution2 (al3,1.0);
        

        double a = distribution(generator);
        double b = distribution1(generator);
        double c = distribution2(generator);
        

    
    *alpha1 = a /(a + b +c );
    *alpha2 = b /(a + b +c );
    *alpha3 = c /(a + b +c );


}

int main()
{
    thrust::device_vector<double> ac1(1);
    thrust::device_vector<double> ac2(1);
    thrust::device_vector<double> ac3(1);
    ac1[0] = .5; ac2[0] = .3; ac3[0] = .2;
    //double alpha1; double alpha2; double alpha3;

    //for(int i =0; i< 3 ; i++){
      //dsample(ac1[0], ac2[0], ac3[0],  &alpha1,  &alpha2,  &alpha3 );
    //std::cout << alpha1 << std::endl;
    //std::cout << alpha2 << std::endl;
    //std::cout << alpha3 << std::endl;
 

    //}

    double x2[4] = {.081899588,.072795849,.334481889,.435275282};//DCN
        thrust::device_vector<double> r2(x2, x2 + 4);
        for( int j = 0; j < 4; j++){
          std::cout << r2[j] << std::endl;

        }
        

    
 
    
    
    

  
}