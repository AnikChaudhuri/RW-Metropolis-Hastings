#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <iostream>
#include <fstream>

#include <thrust/detail/config.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <vector>
#include <cuda.h>
#include <curand_kernel.h>

using namespace std;

struct alpres
{
    template <typename Tuple>
    __device__
    void operator()(Tuple tb)
    {
        thrust::get<7>(tb) = log(pow(thrust::get<0>(tb),thrust::get<3>(tb)) * pow(thrust::get<1>(tb),thrust::get<4>(tb)) * pow(thrust::get<2>(tb),thrust::get<5>(tb)) / thrust::get<6>(tb));
        
    }
};



double dch(double k1, double k2, double k3, thrust::device_vector<double>& A1, thrust::device_vector<double>& A2, thrust::device_vector<double>& A3){

    double S;
    thrust::device_vector<double> res_D(500);
    thrust::device_vector<double> k_1(500);
    thrust::device_vector<double> k_2(500);
    thrust::device_vector<double> k_3(500);
    thrust::device_vector<double> beta_k(500);


    thrust::fill(k_1.begin(), k_1.end(), k1-1);
    thrust::fill(k_2.begin(), k_2.end(), k2-1);
    thrust::fill(k_3.begin(), k_3.end(), k3-1);

    double gam = tgamma(k1) * tgamma(k2) * tgamma(k3) / (tgamma(k1 + k2 + k3));
    thrust::fill(beta_k.begin(), beta_k.end(), gam);

    

    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(A1.begin(), A2.begin(), A3.begin(),k_1.begin(),k_2.begin(),k_3.begin(),beta_k.begin(), res_D.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(A1.end(),A2.begin(), A3.end(), k_1.end(), k_2.end(), k_3.end(),beta_k.end(), res_D.end())),
                     alpres());

	S = thrust::reduce(res_D.begin(), res_D.end());
	//std::cout << A1.size() <<std::endl;

    return S;
    
}
 
__global__ void setup(curandState_t *d_states, int j)
{
    int id = threadIdx.x;
    curand_init(j, id, 0, &d_states[id]);
}

__global__ void uni(double* a, int count, curandState_t* d_states)
{
    //int j = 0;
    
int idx = threadIdx.x+ blockIdx.x * 3;
curandState_t state = d_states[idx]; //copy random number generator state to local memory
    //skipahead((unsigned long long) (6*threadIdx.x), &state); //give each thread its own pseudorandom subsequence
    
if(idx < count)
{
   
        
        float u1 = curand_uniform(&state); //one uniform for proposal
        //float u2 = curand_normal(&state); //one uniform for accept/reject step
        
        a[idx] = u1;
        //a[idx] += b[idx]+j;
        
        
       
}


}

__global__ void dch(double p_k1, double p_k2, double p_k3,double k1,double k2,double k3,double* dv_A1,double* dv_A2,
	double* dv_A3,double* dv_intrp, double* dv_intrc){

		double gam = tgamma(p_k1)*tgamma(p_k2)*tgamma(p_k3)/(tgamma(p_k1+p_k2+p_k3));
		double gam1 = tgamma(k1)*tgamma(k2)*tgamma(k3)/(tgamma(k1+k2+k3));

		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if(idx<500){
			dv_intrp[idx] = log(pow(dv_A1[idx],p_k1-1)*pow(dv_A2[idx],p_k2-1)*pow(dv_A3[idx],p_k3-1)/gam);
			dv_intrc[idx] = log((pow(dv_A1[idx],(k1-1))*pow(dv_A2[idx],(k2-1))*pow(dv_A3[idx],(k3-1)))/gam1);
		}

}

int main()
{ 
	
	
	
			ifstream ifs("alpha1.txt", ifstream::in);
			
			vector<double> a;
 
			while((!ifs.eof()) && ifs)
			{
			double iNumber = 0;
 
			ifs >> iNumber;
			a.push_back(iNumber);
			}
			ifs.close();
			//std::cout << "al1" << std::endl;
  			//for(int i = 0; i< 500; i++){
    		//std::cout << a[i] << std::endl;
			  //}
			  
			ifstream ifu("alpha2.txt", ifstream::in);
			
			vector<double> b;
 
			while((!ifu.eof()) && ifu)
			{
			double iNumber2 = 0;
 
			ifu >> iNumber2;
			b.push_back(iNumber2);
			}
			ifs.close();
			//std::cout << "###############al2#############" << std::endl;
  			//for(int j = 0; j< 500; j++){
    		//std::cout << b[j] << std::endl;
			  //}
			  

			ifstream ifv("alpha3.txt", ifstream::in);
			
			vector<double> c;
   
			while((!ifv.eof()) && ifv)
			{
			double iNumber3 = 0;
   
			ifv >> iNumber3;
			c.push_back(iNumber3);
			}
			ifv.close();
			//std::cout << "###############al3#############" << std::endl;
			//for(int k = 0; k< 500; k++){
			//std::cout << c[k] << std::endl;
			//}
			

			int num = 10000;
			const int count = 3;

			thrust::device_vector<double> d_a(count);
			thrust::device_vector<double> d_b(count);
			thrust::device_vector<double> A1(500);
			thrust::device_vector<double> A2(500);
			thrust::device_vector<double> A3(500);
			thrust::device_vector<double> intrp(500);
			thrust::device_vector<double> intrc(500);
			A1 = a;
			A2 = b; 
			A3 = c;
			double * dv_ptra = thrust::raw_pointer_cast(d_a.data());
			double * dv_ptrb = thrust::raw_pointer_cast(d_b.data());

			double * dv_A1 = thrust::raw_pointer_cast(A1.data());
			double * dv_A2 = thrust::raw_pointer_cast(A2.data());
			double * dv_A3 = thrust::raw_pointer_cast(A3.data());

			double * dv_intrp = thrust::raw_pointer_cast(intrp.data());
			double * dv_intrc = thrust::raw_pointer_cast(intrc.data());





			//std::cout << a.size() <<std::endl;

			
			
			thrust::device_vector<double> K1(num);
			thrust::device_vector<double> K2(num);
			thrust::device_vector<double> K3(num);

			
			//current K
			double k1;
			double k2;
			double k3;
			//proposal K
			double p_k1;
			double p_k2;
			double p_k3;

			curandState_t* d_states;
    		cudaMalloc(&d_states, sizeof(curandState_t)*100);
			

		//thrust::minstd_rand rng2;
		
		//thrust::random::uniform_real_distribution<double> distal(0.0f, 1.0f);
		k1 = 2; k2 = 6; k3 = 4;
		double acq = 0;
	for(int priork = 0; priork < num; priork++){

				//std::cout << "loop no: " << priork <<std::endl;
				
		
				setup<<<1, 10>>>(d_states, priork);

				uni<<<1, 3>>>(dv_ptra, count, d_states );
				double Uk1 = .2;
				
				//thrust::random::uniform_real_distribution<double> dist2(k1[0] - Uk1, k1[0] + Uk1);
				double t = ((k1 + Uk1)-(k1 - Uk1)+.55)*d_a[0] + (k1 - Uk1);
				
				if(t < 0){
					p_k1 = -t;
				}
				else{
					p_k1 = t;
				}
		
				double Uk2 = .2;
				//thrust::random::uniform_real_distribution<double> dist3(k2[0] - Uk2, k2[0] + Uk2);
				double t2 = ((k2 + Uk2)-(k2 - Uk2)+.55)*d_a[1] + (k2 - Uk2);
				
				if(t2 < 0){
					p_k2 = -t2;
				}
				else{
					p_k2 = t2;
				}
		
				double Uk3 = .2;
				//thrust::random::uniform_real_distribution<double> dist4(k3[0] - Uk3, k3[0] + Uk3);
				double t3 = ((k3 + Uk3)-(k3 - Uk3)+.55)*d_a[2] + (k3 - Uk3);
				
				if(t3 < 0){
					p_k3 = -t3;
				}
				else{
					p_k3 = t3;
				}
				
				dch<<<1,500>>>(p_k1, p_k2,p_k3,k1,k2,k3,dv_A1,dv_A2,dv_A3,dv_intrp, dv_intrc);
				double rp = thrust::reduce(intrp.begin(), intrp.end());
				double rc = thrust::reduce(intrc.begin(), intrc.end());

			   /*double rp = dch(p_k1[0], p_k2[0], p_k3[0], A1,A2,A3);
			   double rc = dch(k1[0], k2[0], k3[0], A1,A2,A3);*/
		
			   double kp1 = ((-1*p_k1)) + ((-2*p_k2)) +((-3*p_k3));
			   double kc1 = ((-1*k1)) + ((-2*k2)) + ((-3*k3));
				
			
			   double a1 = ( rp + kp1 )-( rc + kc1 );
				
			   //AddVectors<<<1, 3>>>(dv_ptrb, count, d_states );

			   if(log(d_a[0]) < a1){
					k1 = p_k1;
					k2 = p_k2;
					k3 = p_k3;
					acq = acq+1;
				}
		
		
				else{
					k1 = k1;
					k2 = k2;
					k3 = k3;
					
		
				}
		
				K1[priork] = k1;
				K2[priork] = k2;
				K3[priork] = k3;
				//cudaFree(d_states);
				/*for(int i = 0; i<3; i++){
					std::cout << d_a[i] <<std::endl;
				}*/
			}
			cudaFree(d_states);
			std::cout<<"acceptance " << acq/num <<std::endl;
	std::ofstream myfile3;
    myfile3.open ("K1.txt");
    
    for(int m = 0; m < num; m++){
        myfile3 << K1[m] << std::endl;

    }
    myfile3.close();


    std::ofstream myfile4;
    myfile4.open ("K2.txt");
    
    for(int n = 0; n < num; n++){
        myfile4 << K2[n] << std::endl;

    }
    myfile4.close();


    std::ofstream myfile5;
    myfile5.open ("K3.txt");
    
    for(int p = 0; p < num; p++){
        myfile5 << K3[p] << std::endl;

    }
    myfile5.close();

		
	
}