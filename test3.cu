#include <iostream>
#include <fstream>
#include <ctime>
#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

using namespace std;

__global__ void gen_gamma(double* a, double* b, double* dc, int count1, curandState_t *d_states, 
                            double U, double* dv_Uni)
{
    
int idx = threadIdx.x+ blockIdx.x * blockDim.x;
curandState_t state = d_states[idx]; //copy random number generator state to local memory
    //skipahead((unsigned long long) (6*threadIdx.x), &state); //give each thread its own pseudorandom subsequence
    
if(idx < count1)
{
    if((a[idx])/U < 1.){
    double d = ((a[idx]/U) + 1.0) -1.0/3.;
    double c = 1.0/sqrt(9*d);
    do{
        
        
        double u1 = curand_uniform(&state); //one uniform for proposal
        double u2 = curand_normal(&state); //one uniform for accept/reject step
        double v = pow((1. + c*u2), 3);
        int j3 = (v > 0) && (log(u1) < 0.5*pow(u2, 2)+d - d*v+d*log(v));
        dc[idx] = j3;
        b[idx] = d*v*pow(u1,1/(a[idx]/U));//samples
        dv_Uni[idx] = u1;
        
        
    }while(dc[idx] == 0||b[idx]==0);
}
else{
    double ds1 = ((a[idx])/U ) -1.0/3.;
    double c1 = 1.0/sqrt(9*ds1);
    do{
        
        
        double u11 = curand_uniform(&state); //one uniform for proposal
        double u21 = curand_normal(&state); //one uniform for accept/reject step
        double v1 = pow((1. + c1*u21), 3);
        int j1 = (v1 > 0) && (log(u11) < 0.5*pow(u21, 2)+ds1 - ds1*v1+ds1*log(v1));
        dc[idx] = j1;
        b[idx] = ds1*v1;//samples
        dv_Uni[idx] = u11;
        
        
    }while(dc[idx] == 0);

}/*if(b[idx]==0){
    b[idx] = .001;
}*/

}
}

__global__ void c_gamma(double c2a, double* c2b, double* dv_Cg, int count2, curandState_t *d_states)
{
    
int idx = threadIdx.x+ blockIdx.x * blockDim.x;
curandState_t state = d_states[idx]; //copy random number generator state to local memory
    //skipahead((unsigned long long) (6*threadIdx.x), &state); //give each thread its own pseudorandom subsequence
    

    
    double d22 = (c2a) -1.0/3.;
    double c2 = 1.0/sqrt(9*d22);
    do{
        
        
        double u13 = curand_uniform(&state); //one uniform for proposal
        double u23 = curand_normal(&state); //one uniform for accept/reject step
        double v2 = pow((1. + c2*u23), 3);
        int j2 = v2 > 0 && log(u13) < 0.5*pow(u23, 2)+d22 - d22*v2+d22*log(v2);
        count2 = j2;
        dv_Cg[idx] = d22*v2/c2b[idx];//samples
        
        
    }while(count2 == 0);


}

__global__ void calacp(double* dv_ptra, double* dv_ptrb, double* dv_ptrc, double* dv_a1, double*dv_a2, double*dv_a3, int count, 
                        double* dv_resp, double* dv_resc, double* dv_Rkp, double* dv_Rkc, curandState_t *d_states, 
                        double U, double* dv_Acc){
    int idx = threadIdx.x+ blockIdx.x * blockDim.x;
    curandState_t state = d_states[idx]; //copy random number generator state to local memory
    //skipahead((unsigned long long) (6*threadIdx.x), &state); 
    if(idx < count){
        
       int jacc;  
       double g= tgamma(dv_ptra[idx]/U)*tgamma(dv_ptrb[idx]/U)*tgamma(dv_ptrc[idx]/U)/(tgamma(dv_ptra[idx]/U+dv_ptrb[idx]/U+dv_ptrc[idx]/U));
        __syncthreads();
        double g1 = (pow(dv_a1[idx],(dv_ptra[idx]/U)-1)*pow(dv_a2[idx],(dv_ptrb[idx]/U)-1)*
                    pow(dv_a3[idx],(dv_ptrc[idx]/U)-1))/(g);
        __syncthreads();

        double g2 = tgamma(dv_a1[idx]/U)*tgamma(dv_a2[idx]/U)*tgamma(dv_a3[idx]/U)/(tgamma(dv_a1[idx]/U+dv_a2[idx]/U+dv_a3[idx]/U));
        __syncthreads();
        double g3 = (pow(dv_ptra[idx],(dv_a1[idx]/U)-1)*pow(dv_ptrb[idx],(dv_a2[idx]/U)-1)*
                    pow(dv_ptrc[idx],(dv_a3[idx]/U)-1))/(g2);
        __syncthreads();

        dv_Rkc[idx] = log(g3) + log(dv_resp[idx]) - log(g1) - log(dv_resc[idx]) ;// Ralpha
        //dv_Rkc[idx] = (g3)  ;// Ralpha
        dv_Rkp[idx] = curand_uniform(&state);

        if(log(dv_Rkp[idx]) < (dv_Rkc[idx])){
            jacc = (dv_Rkp[idx]) < (dv_Rkc[idx]);
            dv_ptra[idx] = dv_a1[idx];
            dv_ptrb[idx] = dv_a2[idx];
            dv_ptrc[idx] = dv_a3[idx];
            dv_Acc[idx] = jacc;
            
        }
        else{
            dv_ptra[idx] = dv_ptra[idx];
            dv_ptrb[idx] = dv_ptrb[idx];
            dv_ptrc[idx] = dv_ptrc[idx];
        }
       
    }

}

__global__ void calalpha(double* a1, double* a2, double* a3, double* m, double* r, double* rp, double* ex,
                        double k1, double k2, double k3, double c_1, int count, double* dv_ptra, double* dv_ptrb, double* dv_ptrc,
                        double* d_mc, double* dv_resc, double* dv_exc){

    int idx = threadIdx.x+ blockIdx.x * blockDim.x;
    if(idx < count){
        double eqnp = pow(m[idx], 2) + pow(r[idx],2);
        double first = (m[idx] * (m[idx] + r[idx])/(pow(eqnp,1.5)))*
        exp(-.5*c_1*ex[idx]) * pow(a1[idx], (k1 - 1))*pow(a2[idx], (k2 - 1))*pow(a3[idx], (k3 - 1));

       
        rp[idx] = first;//proposed result

        double eqnc = pow(d_mc[idx], 2) + pow(r[idx],2);
        double second = (d_mc[idx] * (d_mc[idx] + r[idx])/(pow(eqnc,1.5)))*
        exp(-.5*c_1*dv_exc[idx]) * pow(dv_ptra[idx], (k1 - 1))*pow(dv_ptrb[idx], (k2 - 1))*pow(dv_ptrc[idx], (k3 - 1));

       
        dv_resc[idx] = second;

}
}

__global__ void calm(double* a11,double* a12,double* a13,double* a21,double* a22,double* a23,double* vr,double* ex,double* exc,
                        double*d_1, double* d_2, double* d_3, double* m_d, double* m_c, int count){
    int idx = threadIdx.x+ blockIdx.x * blockDim.x;
    if(idx < count){
        
        double firstc = a11[idx] * d_1[idx] + a12[idx] * d_2[idx] + a13[idx] * d_3[idx];
        

        double firstp = a21[idx] * d_1[idx] + a22[idx] * d_2[idx] + a23[idx] * d_3[idx];
        

        m_c[idx] = firstc;
        m_d[idx] = firstp;

        
        double secondc = pow((vr[idx] - m_c[idx]), 2)/(pow(vr[idx], 2) + pow(m_c[idx], 2));

        
        double secondp = pow((vr[idx] - m_d[idx]), 2)/(pow(vr[idx], 2) + pow(m_d[idx], 2));
        

        exc[idx] = secondc;
        ex[idx] = secondp;
}
}

__global__ void setup(curandState_t *d_states, int j)
{
    int id = threadIdx.x+ blockIdx.x * blockDim.x;
    
    curand_init(j, id, 0, &d_states[id]);
}

struct dirichlet
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple dc1)
    {
         
        
            thrust::get<3>(dc1) = thrust::get<0>(dc1)/(thrust::get<0>(dc1)+thrust::get<1>(dc1)+thrust::get<2>(dc1));
            thrust::get<4>(dc1) = thrust::get<1>(dc1)/(thrust::get<0>(dc1)+thrust::get<1>(dc1)+thrust::get<2>(dc1));
            thrust::get<5>(dc1) = thrust::get<2>(dc1)/(thrust::get<0>(dc1)+thrust::get<1>(dc1)+thrust::get<2>(dc1));
            

        }
};

struct alpres
{
    template <typename Tuple>
    __device__
    void operator()(Tuple tb)
    {
        thrust::get<7>(tb) = log(pow(thrust::get<0>(tb),thrust::get<3>(tb)) * pow(thrust::get<1>(tb),thrust::get<4>(tb)) * 
                                pow(thrust::get<2>(tb),thrust::get<5>(tb)) / (thrust::get<6>(tb)));
        
    }
};



double dch(double k11, double k12, double k13, thrust::device_vector<double>& A1, 
            thrust::device_vector<double>& A2, thrust::device_vector<double>& A3){

    double S;
    thrust::device_vector<double> res_D(28);
    thrust::device_vector<double> k_1(28);
    thrust::device_vector<double> k_2(28);
    thrust::device_vector<double> k_3(28);
    thrust::device_vector<double> beta_k(28);


    thrust::fill(k_1.begin(), k_1.end(), k11-1);
    thrust::fill(k_2.begin(), k_2.end(), k12-1);
    thrust::fill(k_3.begin(), k_3.end(), k13-1);

    double gam = tgamma(k11) * tgamma(k12) * tgamma(k13) / (tgamma(k11 + k12 + k13));
    thrust::fill(beta_k.begin(), beta_k.end(), gam);

    

    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(A1.begin(), A2.begin(), A3.begin(),k_1.begin(),k_2.begin(),k_3.begin(),beta_k.begin(), res_D.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(A1.end(),A2.begin(), A3.end(), k_1.end(), k_2.end(), k_3.end(),beta_k.end(), res_D.end())),
                     alpres());

    //S = res_D[0]*res_D[1]*res_D[2]*res_D[3]*res_D[4]*res_D[5]*res_D[6]*res_D[7]*res_D[8]*res_D[9];
    S = thrust::reduce(res_D.begin(), res_D.end());
	//std::cout << A1.size() <<std::endl;

    return S;
    
}

__global__ void ccopy(double *dv_Af1, double *dv_Af2, double *dv_Af3, double *dv_Ath1, double *dv_Ath2, double *dv_Ath3,
                        int count3, int num1){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx <= count3){
        
        dv_Ath1[idx] = dv_Af1[num1+ (idx * 100)];
        dv_Ath2[idx] = dv_Af2[num1 + (idx * 100)];
        dv_Ath3[idx] = dv_Af3[num1 + (idx * 100)];
    }


}

__global__ void g_samp(double *a, double *b, double *dc, int counter, curandState_t *d_states){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t state = d_states[idx];
    if(idx <= counter){
        if((a[idx]/.1) < 1.){
            double d = ((a[idx]/.1) + 1.0) -1.0/3.;
            double c = 1.0/sqrt(9*d);
            do{
                
                
                double u1 = curand_uniform(&state); //one uniform for proposal
                double u2 = curand_normal(&state); //one uniform for accept/reject step
                double v = pow((1. + c*u2), 3);
                int j3 = (v > 0) && (log(u1) < 0.5*pow(u2, 2)+d - d*v+d*log(v));
                dc[idx] = j3;
                b[idx] = d*v*pow(u1,1/(a[idx]/.1));//samples
                
                
                
            }while(dc[idx] == 0);
        }
        else{
            double ds1 = ((a[idx]/.1) ) -1.0/3.;
            double c1 = 1.0/sqrt(9*ds1);
            do{
                
                
                double u11 = curand_uniform(&state); //one uniform for proposal
                double u21 = curand_normal(&state); //one uniform for accept/reject step
                double v1 = pow((1. + c1*u21), 3);
                int j1 = (v1 > 0) && (log(u11) < 0.5*pow(u21, 2)+ds1 - ds1*v1+ds1*log(v1));
                dc[idx] = j1;
                b[idx] = ds1*v1;//samples
                
                
                
            }while(dc[idx] == 0);
        
        }/*if(b[idx]==0){
            b[idx] = .001;
        }*/
        
        }
    


}

int main() {
// Number of items in the arrays
const int count = 28;
int num = 800000;
int counter = ((num)-(.4*num))/(100)+1;

thrust::device_vector<double> d_a(28);//current
thrust::device_vector<double> d_b(28);
thrust::device_vector<double> d_c(28);

thrust::device_vector<double> Acc(28);

thrust::device_vector<double> Cg(1);

thrust::device_vector<double> ad1(28);//dirichlet
thrust::device_vector<double> ad2(28);
thrust::device_vector<double> ad3(28);

thrust::device_vector<double> aj1(28*3);//join all alphas
thrust::device_vector<double> aj2(28*3);//samples
thrust::device_vector<double> aj3(28*3);
/*
thrust::device_vector<double> Af1(10*num);
thrust::device_vector<double> Af2(10*num);
thrust::device_vector<double> Af3(10*num);*/

thrust::device_vector<double> alpha1(28);//proposed
thrust::device_vector<double> alpha2(28);
thrust::device_vector<double> alpha3(28);

thrust::device_vector<double> Rk_p(28);//D(alphai |... )
thrust::device_vector<double> Rk_c(28);//D(alphai |... )

thrust::device_vector<double> Uni(28*3);

thrust::device_vector<double> K1(num);
thrust::device_vector<double> K2(num);
thrust::device_vector<double> K3(num);

thrust::device_vector<double> Kth1(counter);
thrust::device_vector<double> Kth2(counter);
thrust::device_vector<double> Kth3(counter);

//current K
thrust::device_vector<double> c_k1(1);
thrust::device_vector<double> c_k2(1);
thrust::device_vector<double> c_k3(1);

thrust::device_vector<double> c2b(1);
//proposal K
thrust::device_vector<double> p_k1(1);
thrust::device_vector<double> p_k2(1);
thrust::device_vector<double> p_k3(1);

thrust::fill(d_a.begin(), d_a.end(),.1);
thrust::fill(d_b.begin(), d_b.end(),.2);
thrust::fill(d_c.begin(), d_c.end(),.1);
/*
thrust::fill(d_a.begin(), d_a.end(),.6);
thrust::fill(d_b.begin(), d_b.end(),.3);
thrust::fill(d_c.begin(), d_c.end(),.1);*/

thrust::device_vector<double> Ath1(num);
thrust::device_vector<double> Ath2(num);
thrust::device_vector<double> Ath3(num);

thrust::device_vector<double> d1(28);
thrust::device_vector<double> d2(28);
thrust::device_vector<double> d3(28);

thrust::device_vector<double> md(28);//proposed d*proposed alpha
thrust::device_vector<double> mc(28);//current d*current alpha

thrust::device_vector<double> ex_alpha(28);//proposed (r-m)^2/(r^2 + m^2)
thrust::device_vector<double> ex_c(28);//current (r-m)^2/(r^2 + m^2)
//thrust::device_vector<double> c_alpha(10);
thrust::device_vector<double> resp(28);//proposed eqn 16 result
thrust::device_vector<double> resc(28);//current eqn 16 result

thrust::device_vector<double> re(num);
   
    d1[0] = 0; d2[0] = 1; d3[0] = 1; //d for each alpha
    d1[1] = 0; d2[1] = 1; d3[1] = 1;
    d1[2] = 0; d2[2] = 1; d3[2] = 1;
    d1[3] = 0; d2[3] = 1; d3[3] = 1;

    d1[4] = 0; d2[4] = 1; d3[4] = 0;
    d1[5] = 0; d2[5] = 1; d3[5] = 0;
    d1[6] = 0; d2[6] = 1; d3[6] = 0;
    d1[7] = 0; d2[7] = 1; d3[7] = 0;

    d1[8] = 0; d2[8] = 1; d3[8] = 0;

    d1[9] = 0; d2[9] = 1; d3[9] = 0;
    d1[10] = 0; d2[10] = 1; d3[10] = 0; //d for each alpha
    d1[11] = 0; d2[11] = 1; d3[11] = 0;
    d1[12] = 0; d2[12] = 1; d3[12] = 0;

    d1[13] = 0; d2[13] = 1; d3[13] = 0;
    d1[14] = 0; d2[14] = 1; d3[14] = 0;
    d1[15] = 0; d2[15] = 1; d3[15] = 0;
    d1[16] = 0; d2[16] = 1; d3[16] = 0;
    d1[17] = 0; d2[17] = 1; d3[17] = 0;
    d1[18] = 0; d2[18] = 1; d3[18] = 0;
    d1[19] = 0; d2[19] = 1; d3[19] = 0;

    d1[20] = 0; d2[20] = 1; d3[20] = 0;
    d1[21] = 0; d2[21] = 1; d3[21] = 0;
    d1[22] = 0; d2[22] = 1; d3[22] = 0;
    d1[23] = 0; d2[23] = 1; d3[23] = 0;

    d1[24] = 0; d2[24] = 1; d3[24] = 0;
    d1[25] = 0; d2[25] = 1; d3[25] = 0;
    d1[26] = 0; d2[26] = 1; d3[26] = 0;
    d1[27] = 0; d2[27] = 1; d3[27] = 0;


double * dv_ptra = thrust::raw_pointer_cast(d_a.data());//current
double * dv_ptrb = thrust::raw_pointer_cast(d_b.data());
double * dv_ptrc = thrust::raw_pointer_cast(d_c.data());

double * dv_Uni = thrust::raw_pointer_cast(Uni.data());

double * dv_Acc = thrust::raw_pointer_cast(Acc.data());

double * dv_a1 = thrust::raw_pointer_cast(alpha1.data());//proposed
double * dv_a2 = thrust::raw_pointer_cast(alpha2.data());
double * dv_a3 = thrust::raw_pointer_cast(alpha3.data());

double * dv_md = thrust::raw_pointer_cast(md.data());//proposed d*proposed alpha
double * dv_mc = thrust::raw_pointer_cast(mc.data());//current d*current alpha

double * dv_ex = thrust::raw_pointer_cast(ex_alpha.data());//proposed (r-m)^2/(r^2 + m^2)
double * dv_exc = thrust::raw_pointer_cast(ex_c.data());//current (r-m)^2/(r^2 + m^2)

double * dv_resp = thrust::raw_pointer_cast(resp.data());//proposed eqn 16 result
double * dv_resc = thrust::raw_pointer_cast(resc.data());//current eqn 16 result

double * dv_Rkp = thrust::raw_pointer_cast(Rk_p.data());// Rk eqn 28
double * dv_Rkc = thrust::raw_pointer_cast(Rk_c.data());

double * dv_Cg = thrust::raw_pointer_cast(Cg.data());

double * dv_d1 = thrust::raw_pointer_cast(d1.data());
double * dv_d2 = thrust::raw_pointer_cast(d2.data());
double * dv_d3 = thrust::raw_pointer_cast(d3.data());

double * dv_aj1 = thrust::raw_pointer_cast(aj1.data());//joining all alphas
double * dv_aj2 = thrust::raw_pointer_cast(aj2.data());//samples
double * dv_aj3 = thrust::raw_pointer_cast(aj3.data());

double * dv_Ath1 = thrust::raw_pointer_cast(Ath1.data());//thinning
double * dv_Ath2 = thrust::raw_pointer_cast(Ath2.data());
double * dv_Ath3 = thrust::raw_pointer_cast(Ath3.data());

double * dv_re = thrust::raw_pointer_cast(re.data());
/*
double * dv_Af1 = thrust::raw_pointer_cast(Af1.data());//thinning
double * dv_Af2 = thrust::raw_pointer_cast(Af2.data());
double * dv_Af3 = thrust::raw_pointer_cast(Af3.data());*/

double * dv_K1 = thrust::raw_pointer_cast(K1.data());//thinning
double * dv_K2 = thrust::raw_pointer_cast(K2.data());
double * dv_K3 = thrust::raw_pointer_cast(K3.data());

double * dv_Kth1 = thrust::raw_pointer_cast(Kth1.data());//thinning
double * dv_Kth2 = thrust::raw_pointer_cast(Kth2.data());
double * dv_Kth3 = thrust::raw_pointer_cast(Kth3.data());

double * dv_c2b = thrust::raw_pointer_cast(c2b.data());


//curandStatePhilox4_32_10_t *d_states;
//cudaMalloc((void **)&d_states, 64 * 64 * sizeof(curandStatePhilox4_32_10_t));

curandState_t* d_states;
cudaMalloc(&d_states, sizeof(curandState_t)*num);
   

dim3 gridSize(1);
dim3 blockSize(28);

//double x2[10] = {.598739352,.493116352,.579867973,.320856474,.081899588,.072795849,.334481889,.435275282,.517632462,.44421341};
double x2[28] = {0.4796,
    0.2892,
    0.2535,
    0.2698,
    0.1550,
    0.2793,
    0.3015,
    0.3415,
    0.3842,
    0.2570,
    0.2059,
    0.2717,
    0.2679,
    0.0087,
    0.0242,
    0.1661,
    0.2793,
    0.3789,
    0.3737,
    0.3536,
    0.2624,
    0.2553,
    0.2253,
    0.2031,
    0.3164,
    0.4623,
    0.4633,
    0.3660};

thrust::device_vector<double> r(x2, x2 + 28);
double * dv_r = thrust::raw_pointer_cast(r.data());

int count1 = 28*3;
int count2;
//double U = 9,28;
double U = .6;

//double nu0 = .1; double c20 = 1000; 
double nu0 = 1; double c20 = 0; 
double c2a = (nu0 + 28.)/2.; double c_1;
double acp = 0;//acceptance fro alpha
double ack = 0;//accceptance for K
//double acc = 0;//increment
c_k1[0] = 2; c_k2[0] =6; c_k3[0] = 4;
//c_k1[0] = .3; c_k2[0] =.6; c_k3[0] = .4;


double k1; double k2 ; double k3 ; 
double a1k;
for(int j = 0; j<num; j++){
    std::cout << "loop " << j <<std::endl;
setup<<<1, 28*3>>>(d_states,j);

thrust::copy(d_a.begin(), d_a.end(), aj1.begin());  //joining all the alphas
thrust::copy(d_b.begin(), d_b.end(), aj1.begin()+28);
thrust::copy(d_c.begin(), d_c.end(), aj1.begin()+56);

gen_gamma<<<gridSize, 28*3>>>(dv_aj1, dv_aj2, dv_aj3, count1, d_states, U, dv_Uni);//generating gamma samples (proposals)

thrust::copy_n(aj2.begin(), 28, alpha1.begin() );   //separating the alphas
thrust::copy_n(aj2.begin()+28, 56, alpha2.begin() );
thrust::copy_n(aj2.begin()+56, 84, alpha3.begin() );

thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(alpha1.begin(),alpha2.begin(),alpha3.begin(), 
                                            ad1.begin(),ad2.begin(),ad3.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(alpha1.end(),alpha2.end(),alpha3.end(),
                                        ad1.end(),ad2.end(),ad3.end())),dirichlet());
alpha1 = ad1;
alpha2 = ad2;
alpha3 = ad3;


k1 = c_k1[0]; k2 = c_k2[0]; k3 = c_k3[0];

calm<<<gridSize,blockSize>>>(dv_ptra, dv_ptrb, dv_ptrc, dv_a1, dv_a2, dv_a3, dv_r, dv_ex, dv_exc,
                            dv_d1, dv_d2, dv_d3, dv_md, dv_mc, count);

double cb = thrust::reduce(ex_c.begin(), ex_c.end());
c2b[0] = (nu0*c20 + cb)/2.0;

c_gamma<<<1, 1>>>(c2a, dv_c2b, dv_Cg, count2, d_states );
c_1 = Cg[0];

calalpha<<<gridSize,blockSize>>>(dv_a1, dv_a2, dv_a3, dv_md, dv_r, dv_resp, dv_ex, k1, k2, 
                                k3, c_1, count, dv_ptra, dv_ptrb, dv_ptrc, dv_mc, dv_resc, dv_exc);

calacp<<<gridSize, blockSize>>>(dv_ptra, dv_ptrb, dv_ptrc, dv_a1, dv_a2, dv_a3, count, dv_resp, dv_resc, 
                                dv_Rkp, dv_Rkc, d_states, U, dv_Acc);

acp = thrust::reduce(Acc.begin(), Acc.end()) + acp;


//double Uk1 = .7;
double Uk1 =.6;
double adj = 0.83;
double t = ((c_k1[0] + Uk1)-(c_k1[0] - Uk1)+adj)*Uni[0] + (c_k1[0] - Uk1);

if(t < 0){
    p_k1[0] = -t;
}
else{
    p_k1[0] = t;
}

//double Uk2 = 1;
double t2 = ((c_k2[0] + Uk1)-(c_k2[0] - Uk1)+adj)*Uni[1] + (c_k2[0] - Uk1);

if(t2 < 0){
    p_k2[0] = -t2;
}
else{
    p_k2[0] = t2;
}

//double Uk3 = 1;
double t3 = ((c_k3[0] + Uk1)-(c_k3[0] - Uk1)+adj)*Uni[2] + (c_k3[0] - Uk1);

if(t3 < 0){
    p_k3[0] = -t3;
}
else{
    p_k3[0] = t3;
}

double rpk = dch(p_k1[0], p_k2[0], p_k3[0], d_a, d_b, d_c);
double rc = dch(c_k1[0], c_k2[0], c_k3[0], d_a, d_b, d_c);

double kp1 = ((-.5*p_k1[0]) + (-2.*p_k2[0]) + (-6.5*p_k3[0]));
double kc1 = ((-.5*c_k1[0]) + (-2.*c_k2[0]) + (-6.5*c_k3[0]));

a1k = (( rpk + kp1 )-( rc + kc1 ));

if(log(Uni[0]) < a1k){
    c_k1[0] = p_k1[0];
    c_k2[0] = p_k2[0];
    c_k3[0] = p_k3[0];
    ack++;
}


else{
    c_k1[0] = c_k1[0];
    c_k2[0] = c_k2[0];
    c_k3[0] = c_k3[0];

}

K1[j] = c_k1[0];
K2[j] = c_k2[0];
K3[j] = c_k3[0];


       
}

thrust::device_vector<double> di1(num);
thrust::device_vector<double> di2(num);
thrust::device_vector<double> di3(num);

double * dv_di1 = thrust::raw_pointer_cast(di1.data());
double * dv_di2 = thrust::raw_pointer_cast(di2.data());
double * dv_di3 = thrust::raw_pointer_cast(di3.data());

int count3 = counter - 1;
int num1 = (.4*num) - 1;
ccopy<<<400, 1000>>>(dv_K1, dv_K2, dv_K3, dv_Kth1, dv_Kth2, dv_Kth3, count3, num1);
g_samp<<<400, 1000>>>(dv_Kth1, dv_Ath1, dv_re, counter, d_states);
g_samp<<<400, 1000>>>(dv_Kth2, dv_Ath2, dv_re, counter, d_states);
g_samp<<<400, 1000>>>(dv_Kth3, dv_Ath3, dv_re, counter, d_states);

thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(Ath1.begin(),Ath2.begin(),Ath3.begin(), 
                                            di1.begin(),di2.begin(),di3.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(Ath1.end(),Ath2.end(),Ath3.end(),
                                        di1.end(),di2.end(),di3.end())),dirichlet());
cudaFree(d_states);

//std::cout << "acceptance: "<< acp/(10*num) <<std::endl;
std::cout << "acceptance of K: "<< ack/(1.0*num) <<std::endl;

//
std::cout<<"K: "<<a1k<<std::endl;



    std::ofstream myfile3;
    myfile3.open ("K1.txt");
    
    for(int p1 = 0; p1 < counter; p1++){
        myfile3 << di1[p1] << std::endl;

    }
    myfile3.close();

    std::ofstream myfile4;
    myfile4.open ("K2.txt");
    
    for(int p1 = 0; p1 < counter; p1++){
        myfile4 << di2[p1] << std::endl;

    }
    myfile4.close();

    std::ofstream myfile5;
    myfile5.open ("K3.txt");
    
    for(int p1 = 0; p1 < counter; p1++){
        myfile5 << di3[p1] << std::endl;

    }
    myfile5.close();

std::cout << counter <<std::endl;
}