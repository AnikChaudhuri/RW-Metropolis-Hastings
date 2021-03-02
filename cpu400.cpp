//400 synthetic data CPU code.
//500 alphas were generated from Dir(10, 6, 3) in python and they were stored in files named alpha1.txt, alpha2.txt, alpha3.txt.
//This code will read the alphas from alpha1.txt, alpha2.txt and alpha3.txt to generate synthetic gene expression ratio r.
//The results i.e. K1, K2, K3 will be saved on your hard disk.
//K1 will be saved in K1.txt, K2 will be saved in K2.txt and K3 will be saved in K3.txt

#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <random>

using namespace std;
std::default_random_engine rng;
std::uniform_real_distribution<double> uni(0.0,1.0);
std::normal_distribution<double> norm(0.0,1.0);
std::uniform_int_distribution<int> uni1(0,1);

//this function samples gamma variables using Marsaglia and Tsang algorithm, these samples are for the alpha proposals.
//a1 is the alpha and U is the tunning parameter.
double gen_gamma(double a1, double U){
    
    int dc;double b;//dc checks the condition for iteration, samples are stored in b
	//checking if alpha is less than 1, this loop works for alpha < 1
    if((a1)/U < 1.){
    double d = ((a1/U) + 1.0) -1.0/3.;
    double c = 1.0/sqrt(9*d);
    do{
        
        
        double u1 = uni(rng); //one uniform for proposal
        double u2 = norm(rng); //one uniform for accept/reject step
        double v = pow((1. + c*u2), 3);
        int j3 = (v > 0) && (log(u1) < 0.5*pow(u2, 2)+d - d*v+d*log(v));
        dc = j3;
        b = d*v*pow(u1,1/(a1/U));//samples
        
        
        
    }while(dc == 0);//condition check
}

//this loop works for alpha > 1
else{
    double ds1 = ((a1)/U ) -1.0/3.;
    double c1 = 1.0/sqrt(9*ds1);
    do{
        
        
        double u11 = uni(rng); //one uniform for proposal
        double u21 = norm(rng); //one uniform for accept/reject step
        double v1 = pow((1. + c1*u21), 3);
        int j1 = (v1 > 0) && (log(u11) < 0.5*pow(u21, 2)+ds1 - ds1*v1+ds1*log(v1));
        dc = j1;
        b = ds1*v1;//samples
        
        
        
    }while(dc == 0);//condition check.
    


}return b;//returning the samples.
}

//this function samples a gamma variable for 1/c^2, (Marsaglia and Tsang algorithm).
//c2a and c2b are the parameters of the gamma distribution 1/c^2
double c_gamma(double c2a, double c2b)
{
    double dv_Cg; int count2;//dv_Cg stores the sample, count2 is used for condition checking.

    double d22 = (c2a) -1.0/3.;
    double c2 = 1.0/sqrt(9*d22);
    do{
        
        
        double u13 = uni(rng); //one uniform for proposal
        double u23 = norm(rng); //one uniform for accept/reject step
        double v2 = pow((1. + c2*u23), 3);
        int j2 = v2 > 0 && log(u13) < 0.5*pow(u23, 2)+d22 - d22*v2+d22*log(v2);
        count2 = j2;
        dv_Cg = d22*v2/c2b;//samples
        
        
    }while(count2 == 0);//condition check
	return dv_Cg;//returning the sample.


}

//this function is calucating the full conditional of alpha given everything else, thus function will be called twice
//once for current value and once for proposed values.
//a1, b1,c1 are the alphas, mc is d*alpha, r is the data, k1,k2,k3 are the Ks, c is a sample from 1/c^2.
//a1 + b1 + c1 =1.
double calalpha(double a1,double b1, double c1, double mc,double r, double k1, double k2, double k3, double c){

	double result;//conditional prob. is stored here.

	//calculating (m*(r+m)/(r^2+m^2)^1.5) * exp(-0.5*(r-m)^2/c^2*(r^2+m^2)) *a1^(k1-1)*a2^(k2-1)*a3^(k3-1)

	result = (mc*(r + mc)/pow((pow(mc,2)+pow(r,2)),1.5)) * exp(-0.5*c * pow((r - mc),2)/(r*r + mc*mc))*
			pow(a1,(k1-1))*pow(b1,(k2-1))*pow(c1,(k3-1));

	return result;//returning the cond. prob.
		
	}

//this function the acceptance ratio for alpha, this function will be called twice.
//res is the full condtional of alpha calculated in calalpha.
//a,b,c are the support of Dirichlet distribution and ap,bp,cp are the concentration parameters of Dirichlet distriburion.
//a + b + c = 1
double ralpha(double res, double a, double b, double c, double ap, double bp, double cp){
	//calculating Ralpha
	double beta = tgamma(ap)*tgamma(bp)*tgamma(cp)/(tgamma(ap + bp + cp));
	double dirich = pow(a,ap-1)*pow(b,bp-1)*pow(c,cp-1)/beta;
	double r_result = res * dirich;

	return r_result;


}

//this function is calculating Dir(a1,b1,c1|p_k1,p_k2,p_k3). a1,b1,c1 are the alphas and p_k1, p_k2, p_k3 are the Ks.
//this function will be called twice. Used for calculating Rk.
double dch(double p_k1, double p_k2, double p_k3, double a1, double b1, double c1){
	double de = tgamma(p_k1)*tgamma(p_k2)*tgamma(p_k3)/(tgamma(p_k1 + p_k2 + p_k3));
	
	 double mult = log(pow(a1, p_k1-1)* pow(b1, p_k2-1)* pow(c1, p_k3-1)/de);
	return mult;

}
int main(){
	//loading synthetic data. alpha1, alpha2, alpha3 generated from Dir(10,6,3).
	//this data is stored in a separate file. alpha1.txt, alpha2.txt, alpha3.txt are the file name.
	//loading starts here
    ifstream ifs("alpha1.txt", ifstream::in);
			
			std::vector<double> a;//storing the data from alpha1.txt in a.
 
			while((!ifs.eof()) && ifs)
			{
			double iNumber = 0;
 
			ifs >> iNumber;
			a.push_back(iNumber);
			}
			ifs.close();
			
			  
			ifstream ifu("alpha2.txt", ifstream::in);
			
			std::vector<double> b;//storing the data from alpha2.txt in b.
 
			while((!ifu.eof()) && ifu)
			{
			double iNumber2 = 0;
 
			ifu >> iNumber2;
			b.push_back(iNumber2);
			}
			ifs.close();
			
			  

			ifstream ifv("alpha3.txt", ifstream::in);
			
			std::vector<double> c;//storing the data from alpha3.txt in c.
   
			while((!ifv.eof()) && ifv)
			{
			double iNumber3 = 0;
   
			ifv >> iNumber3;
			c.push_back(iNumber3);
			}
			ifv.close();// loading ends here.
    
	int num = 10000;// number of iterations
	std::vector<double> ac(400);// current value of alpha1
	std::vector<double> bc(400);// current value of alpha2
	std::vector<double> cc(400);// current value of alpha3

	std::vector<double> r(400);// synthetic r
	std::vector<double> mu(400);// mean, used in generating r.

	//expression profile d1, d2, d3
	std::vector<int> d1(400);
	std::vector<int> d2(400);
	std::vector<int> d3(400);
    
    double U = .02;//tunning parameter
	
	//generating synthetic data.
    for(int i = 0; i<400; i++){
        
		do{
			//randomly setting d1, d2, d3
			d1[i] = uni1(rng);
			d2[i] = uni1(rng);
			d3[i] = uni1(rng);

		}while(d1[i]==d2[i]==d3[i]==0);// because I don't want all the d's to be 0.
		
		mu[i] = d1[i]*a[i] + d2[i]*b[i] + d3[i]*c[i];// calculating d*alpha
		do{
			//sampling r from a normal distribution.
			std::normal_distribution<double> norm1(mu[i],0.1*mu[i]);
			r[i] = norm1(rng);

		}while(r[i] >=1);
		
        

    }// end of data generation
	double mpr, mc;//mpr = d*proposed alpha; mc = d*current alpha.

	std::vector<double> m_b(400);// used for calculating the parameters for 1/c^2.
	//gap, gbp, gcp are used for holding the generated gamma variables for alpha1, alpha2, alpha3.
	std::vector<double> gap(400);
	std::vector<double> gbp(400);
	std::vector<double> gcp(400);

	//apr, bpr, cpr are used for storing the proposals of alpha1, alpha2, alpha3
	std::vector<double> apr(400);
	std::vector<double> bpr(400);
	std::vector<double> cpr(400);

	std::vector<double> resp(400);//this holds the conditional prob. of proposed alpha 
	std::vector<double> resc(400);//this holds the conditional prob. of current alpha 

	//K1, K2, K3 holds the sampled Ks. 
	std::vector<double> K1(num);
	std::vector<double> K2(num);
	std::vector<double> K3(num);

	std::vector<double> rpk(400);//holds the result from function dch.
	std::vector<double> rc(400);//holds the result from function dch.
	
	//initiating random alphas
	std::fill(ac.begin(), ac.end(), .6);
	std::fill(bc.begin(), bc.end(), .3);
	std::fill(cc.begin(), cc.end(), .1);

	double nu0 = .01; double c20 = 1000; double c2_b;//c2_b is the b parameter for 1/c^2.

	double c2a = (nu0 + 400.)/2.; double c_1; //c2a is the a parameter for 1/c^2

	double c_k1, c_k2, c_k3, p_k1, p_k2, p_k3;// c_k1,c_k2,c_k3 holds the current samples of K. 
	//p_k1,p_k2,p_k3 holds the proposed K

	double ac_b;// used for summing the b parameter in 1/c^2

	c_k1 = 2., c_k2 = 6. , c_k3 = 4.;// initializing random values to K.

	double a1k; long fast;//a1k holds Rk. fast is used for calculatin time.

	long start = clock();//clock

	//Big MCMC loop starts here.
	for(int j = 0; j< num; j++){
		
		//std::cout << "loop no: " << j<<std::endl;
		ac_b = 0;
		//summing the b parameter for 1/c^2
		for(int l= 0; l<400; l++ ){
				m_b[l] = ac[l] * d1[l] + bc[l] * d2[l] + cc[l] * d3[l];
				ac_b = (pow((m_b[l]-r[l]),2)/(pow(m_b[l],2)+pow(r[l],2))) + ac_b;
			}
		
		//MH for alpha starts here
		 for(int k = 0; k< 400; k++){
			
			//generating gamma variables; generating proposals for alpha.
			gap[k] = gen_gamma(ac[k], U);
			gbp[k] = gen_gamma(bc[k], U);
			gcp[k] = gen_gamma(cc[k], U);

			//normalizing the gamma variables to obtain dirichlet distributed samples.
			apr[k] = gap[k]/(gap[k] + gbp[k] + gcp[k]);
			bpr[k] = gbp[k]/(gap[k] + gbp[k] + gcp[k]);
			cpr[k] = gcp[k]/(gap[k] + gbp[k] + gcp[k]);

			mpr = apr[k] * d1[k] + bpr[k] * d2[k] + cpr[k] * d3[k];
			mc = ac[k] * d1[k] + bc[k] * d2[k] + cc[k] * d3[k];

			c2_b = nu0*c20 + ac_b;//b parameter for 1/c^2
			c_1 = c_gamma(c2a, c2_b );//sampling 1/c^2

			double k1 =c_k1; double k2 = c_k2; double k3 = c_k3;
			resc[k] = calalpha(ac[k], bc[k], cc[k], mc,r[k], k1, k2, k3, c_1);

			resp[k] = calalpha(apr[k], bpr[k], cpr[k], mpr,r[k], k1, k2, k3, c_1);

			double Ralp = ralpha(resp[k], ac[k], bc[k], cc[k], apr[k]/U, bpr[k]/U, cpr[k]/U);
			double Ralc = ralpha(resc[k], apr[k], bpr[k], cpr[k], ac[k]/U, bc[k]/U, cc[k]/U);
			double RA = Ralp/Ralc;//calculating Ralpha

			//MH accept starts here.
			if(uni(rng) < RA ){
				ac[k] = apr[k];
				bc[k] = bpr[k];
				cc[k] = cpr[k];
 			}
			 else{
				ac[k] = ac[k];
				bc[k] = bc[k];
				cc[k] = cc[k];
			 }//MH accept ends here
		}// MH for alpha ends here.

		double Uk = 1;//tunning parameter for K.

		//Generating proposals for K starts here.
        double adj = .99;
		double t = ((c_k1 + Uk)-(c_k1 - Uk)+adj)*uni(rng) + (c_k1 - Uk);
		if(t < 0){
    		p_k1 = -t;
			}
		else{
    		p_k1 = t;
		}

		double t1 = ((c_k2 + Uk)-(c_k2 - Uk)+adj)*uni(rng) + (c_k2 - Uk);
		if(t1 < 0){
    		p_k2 = -t1;
			}
		else{
    		p_k2 = t1;
		}

		double t2 = ((c_k3 + Uk)-(c_k3 - Uk)+adj)*uni(rng) + (c_k3 - Uk);
		if(t2 < 0){
    		p_k3 = -t2;
			}
		else{
    		p_k3 = t2;
		}//proposal generation for K ends here.
		
		double multp = 0;double multc = 0;
		//loop to calculate P(alpha|K)
		for(int l=0; l<400; l++){
			rpk[l] = dch(p_k1, p_k2, p_k3, ac[l], bc[l], cc[l]);
			rc[l] = dch(c_k1, c_k2, c_k3, ac[l], bc[l], cc[l]);
			 multp = multp + rpk[l];
			 multc = multc + rc[l];
			
		}
		
		double kp1 = (-1.*p_k1) + (-2.*p_k2) + (-3.*p_k3);//prior over K proposed
		double kc1 = (-1.*c_k1) + (-2.*c_k2) + (-3.*c_k3);//prior over K current

		a1k = (( multp + kp1 )-( multc + kc1 ));//calculating RK

		//MH acceptance for K
		if(log(uni(rng)) < a1k){
    		c_k1 = p_k1;
    		c_k2 = p_k2;
    		c_k3 = p_k3;
    		
		}
		else{
   		 c_k1 = c_k1;
    	 c_k2 = c_k2;
    	 c_k3 = c_k3;

		}

		//storing the sampled Ks in K1, K2,K3.
		K1[j] = c_k1;
		K2[j] = c_k2;
		K3[j] = c_k3;


		
		

	}//MCMC loop ends here.
	long stop = clock();
		fast = stop - start;//time taken
	
	std::cout << "time "<< fast<<std::endl;

	//saving the sampled K1 in K1.txt
	std::ofstream myfile3;
    myfile3.open ("K1.txt");
    
    for(int p1 = 0; p1 < num; p1++){
        myfile3 << K1[p1] << std::endl;

    }
    myfile3.close();

	//saving K2 in K2.txt
    std::ofstream myfile4;
    myfile4.open ("K2.txt");
    
    for(int p1 = 0; p1 < num; p1++){
        myfile4 << K2[p1] << std::endl;

    }
    myfile4.close();

	//saving K3 in K3.txt
    std::ofstream myfile5;
    myfile5.open ("K3.txt");
    
    for(int p1 = 0; p1 < num; p1++){
        myfile5 << K3[p1] << std::endl;

    }
    myfile5.close();
    
    
}

