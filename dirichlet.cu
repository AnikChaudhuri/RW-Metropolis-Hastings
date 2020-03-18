#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <iostream>
#include <fstream>

#include <thrust/detail/config.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/normal_distribution.h>

struct arbitrary_functor2
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple f)
    {
        thrust::get<2>(f) = powf(thrust::get<0>(f), (thrust::get<1>(f) - 1)) * expf(-(thrust::get<0>(f)))/tgammaf(thrust::get<1>(f));
        
    }
};

int main(void)
{
    // allocate storage
    int num = 500000;

    thrust::device_vector<float> x(1);

    thrust::device_vector<float> xp(1);//proposal
    thrust::device_vector<float> xc(1);

    thrust::device_vector<float> resp(1);//proposal result
    thrust::device_vector<float> resc(1);

    thrust::device_vector<float> al(1);// alpha

    //vectors to hold the samples
    thrust::device_vector<float> d_v(num);
    thrust::device_vector<float> d_v1(num);
    thrust::device_vector<float> d_v2(num);

    float R;

    float al1 = .6; float al2 = .3; float al3 = .1;

    thrust::minstd_rand rng;
    thrust::random::normal_distribution<float> dist(0.0f, 1.0f);
    
    thrust::uniform_int_distribution<float> dist1(0.0f,1.0f);

    thrust::uniform_int_distribution<int> distu(0,num);

    for(int k =0; k<= 2; k++){

    
    

    // initialize input vectors
    if(k == 0){
        al[0] = al1;
        std::cout << "loop 1 and al is: " << al[0]<< std::endl;

    }
    else if(k == 1){
        al[0] = al2;
        std::cout << "loop 2 and al is: " << al[0]<< std::endl;
    }
    else{
        al[0] = al3;
        std::cout << "loop 3 and al is: " << al[0]<< std::endl;
    }
    
    x[0] = .3; //B[0] = .6;  C[0] = .2; 
    
    
  for(int i = 0; i < num;i++){
      //std::cout << x <<std::endl;
      //curr = x;
      //std::cout << "curr is :" << curr <<std::endl;
      xp[0] = x[0] + dist(rng);
      xc[0] = x[0];
      thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(xp.begin(), al.begin(), resp.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(xp.end(),al.begin(), resp.end())),
                     arbitrary_functor2());

      thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(xc.begin(),al.begin(), resc.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(xc.end(),al.begin(), resc.end())),
                     arbitrary_functor2());

      R = resp[0]/ resc[0];

      if(dist1(rng) < R){
          x[0] = xp[0];
          
      }
      else{
        x[0] = xc[0];
    }// Metropolis hastings ends here

    if(k == 0){
        d_v[i] = x[0];

    }
    else if(k == 1){
        d_v1[i] = x[0];
    }
    else{
        d_v2[i] = x[0];
    }

     
    } //for loop ends here.
}// k loop ends here

    thrust::minstd_rand rng1;
    thrust::minstd_rand rng2;
    thrust::minstd_rand rng3;

    int u1 = distu(rng1);
    int u2 = distu(rng1);
    int u3 = distu(rng1);

    std::cout << u1 <<std::endl;
    std::cout << u2 <<std::endl;
    std::cout << u3 <<std::endl;

    float d1 = d_v[u1];
    float d2 = d_v1[u2];
    float d3 = d_v2[u3];

    float alpha1 = d1/(d1 + d2 + d3);
    float alpha2 = d2/(d1 + d2 + d3);
    float alpha3 = d3/(d1 + d2 + d3);

    std::cout << "alpha 1 is: " << alpha1 <<std::endl;
    std::cout << "alpha 2 is: " <<alpha2 <<std::endl;
    std::cout << "alpha 3 is: " <<alpha3 <<std::endl;

    std::cout << "sum is: " << alpha3 + alpha1 + alpha2 <<std::endl;
    


    //std::ofstream myfile;
    //myfile.open ("example.txt");
    
    //for(int j = 0; j < num; j++){
        //myfile << d_v[j] << std::endl;

    //}
    //myfile.close();

    //std::ofstream myfile1;
    //myfile1.open ("example1.txt");
    
    //for(int l = 0; l < num; l++){
        //myfile1 << d_v1[l] << std::endl;

    //}
    //myfile1.close();

    //std::ofstream myfile2;
    //myfile2.open ("example2.txt");
    
    //for(int m = 0; m < num; m++){
       // myfile2 << d_v2[m] << std::endl;

   // }
    //myfile2.close();
        
        

}