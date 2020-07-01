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

double gen_gamma(double a1, double U){
    
    int dc;double b;

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
        
        
        
    }while(dc == 0);
}
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
        
        
        
    }while(dc == 0);
    


}return b;
}

double c_gamma(double c2a, double c2b)
{
    double dv_Cg; int count2;

    double d22 = (c2a) -1.0/3.;
    double c2 = 1.0/sqrt(9*d22);
    do{
        
        
        double u13 = uni(rng); //one uniform for proposal
        double u23 = norm(rng); //one uniform for accept/reject step
        double v2 = pow((1. + c2*u23), 3);
        int j2 = v2 > 0 && log(u13) < 0.5*pow(u23, 2)+d22 - d22*v2+d22*log(v2);
        count2 = j2;
        dv_Cg = d22*v2/c2b;//samples
        
        
    }while(count2 == 0);
	return dv_Cg;


}

double calalpha(double a1,double b1, double c1, double mc,double r, double k1, double k2, double k3, double c){
	double result;
	result = (mc*(r + mc)/pow((pow(mc,2)+pow(r,2)),1.5)) * exp(-0.5*c * pow((r - mc),2)/(r*r + mc*mc))*
			pow(a1,(k1-1))*pow(b1,(k2-1))*pow(c1,(k3-1));

	return result;
		
	}

double ralpha(double res, double a, double b, double c, double ap, double bp, double cp){
	double beta = tgamma(ap)*tgamma(bp)*tgamma(cp)/(tgamma(ap + bp + cp));
	double dirich = pow(a,ap-1)*pow(b,bp-1)*pow(c,cp-1)/beta;
	double r_result = res * dirich;

	return r_result;


}

double dch(double p_k1, double p_k2, double p_k3, double a1, double b1, double c1){
	double de = tgamma(p_k1)*tgamma(p_k2)*tgamma(p_k3)/(tgamma(p_k1 + p_k2 + p_k3));
	
	 double mult = log(pow(a1, p_k1-1)* pow(b1, p_k2-1)* pow(c1, p_k3-1)/de);
	return mult;

}
int main(){
    ifstream ifs("alpha1.txt", ifstream::in);
			
			std::vector<double> a;
 
			while((!ifs.eof()) && ifs)
			{
			double iNumber = 0;
 
			ifs >> iNumber;
			a.push_back(iNumber);
			}
			ifs.close();
			
			  
			ifstream ifu("alpha2.txt", ifstream::in);
			
			std::vector<double> b;
 
			while((!ifu.eof()) && ifu)
			{
			double iNumber2 = 0;
 
			ifu >> iNumber2;
			b.push_back(iNumber2);
			}
			ifs.close();
			
			  

			ifstream ifv("alpha3.txt", ifstream::in);
			
			std::vector<double> c;
   
			while((!ifv.eof()) && ifv)
			{
			double iNumber3 = 0;
   
			ifv >> iNumber3;
			c.push_back(iNumber3);
			}
			ifv.close();
    
	int num = 10000;
	std::vector<double> ac(400);
	std::vector<double> bc(400);
	std::vector<double> cc(400);

	std::vector<double> r(400);
	std::vector<double> mu(400);
	std::vector<int> d1(400);
	std::vector<int> d2(400);
	std::vector<int> d3(400);
    //std::fill(x.begin(), x.end(), 1);
    double U = .02;
	

    for(int i = 0; i<400; i++){
        
		do{
			d1[i] = uni1(rng);
			d2[i] = uni1(rng);
			d3[i] = uni1(rng);

		}while(d1[i]==d2[i]==d3[i]==0);
		
		mu[i] = d1[i]*a[i] + d2[i]*b[i] + d3[i]*c[i];
		do{
			std::normal_distribution<double> norm1(mu[i],0.1*mu[i]);
			r[i] = norm1(rng);

		}while(r[i] >=1);
		
        //std::cout << r[i] << std::endl;

    }
	double mpr, mc;

	std::vector<double> m_b(400);

	std::vector<double> gap(400);
	std::vector<double> gbp(400);
	std::vector<double> gcp(400);

	std::vector<double> apr(400);
	std::vector<double> bpr(400);
	std::vector<double> cpr(400);

	std::vector<double> resp(400);
	std::vector<double> resc(400);

	std::vector<double> K1(num);
	std::vector<double> K2(num);
	std::vector<double> K3(num);

	std::vector<double> rpk(400);
	std::vector<double> rc(400);

	std::fill(ac.begin(), ac.end(), .6);
	std::fill(bc.begin(), bc.end(), .3);
	std::fill(cc.begin(), cc.end(), .1);

	double nu0 = .01; double c20 = 1000; double c2_b;
	double c2a = (nu0 + 400.)/2.; double c_1;
	double c_k1, c_k2, c_k3, p_k1, p_k2, p_k3;
	double ac_b;
	c_k1 = 2., c_k2 = 6. , c_k3 = 4.;
	double a1k; long fast;
	long start = clock();

	for(int j = 0; j< num; j++){
		
		std::cout << "loop no: " << j<<std::endl;
		ac_b = 0;

		for(int l= 0; l<400; l++ ){
				m_b[l] = ac[l] * d1[l] + bc[l] * d2[l] + cc[l] * d3[l];
				ac_b = (pow((m_b[l]-r[l]),2)/(pow(m_b[l],2)+pow(r[l],2))) + ac_b;
			}
		
		for(int k = 0; k< 400; k++){
			
			gap[k] = gen_gamma(ac[k], U);
			gbp[k] = gen_gamma(bc[k], U);
			gcp[k] = gen_gamma(cc[k], U);

			apr[k] = gap[k]/(gap[k] + gbp[k] + gcp[k]);
			bpr[k] = gbp[k]/(gap[k] + gbp[k] + gcp[k]);
			cpr[k] = gcp[k]/(gap[k] + gbp[k] + gcp[k]);

			mpr = apr[k] * d1[k] + bpr[k] * d2[k] + cpr[k] * d3[k];
			mc = ac[k] * d1[k] + bc[k] * d2[k] + cc[k] * d3[k];

			c2_b = nu0*c20 + ac_b;
			c_1 = c_gamma(c2a, c2_b );

			double k1 =c_k1; double k2 = c_k2; double k3 = c_k3;
			resc[k] = calalpha(ac[k], bc[k], cc[k], mc,r[k], k1, k2, k3, c_1);

			resp[k] = calalpha(apr[k], bpr[k], cpr[k], mpr,r[k], k1, k2, k3, c_1);

			double Ralp = ralpha(resp[k], ac[k], bc[k], cc[k], apr[k]/U, bpr[k]/U, cpr[k]/U);
			double Ralc = ralpha(resc[k], apr[k], bpr[k], cpr[k], ac[k]/U, bc[k]/U, cc[k]/U);
			double RA = Ralp/Ralc;

			if(uni(rng) < RA ){
				ac[k] = apr[k];
				bc[k] = bpr[k];
				cc[k] = cpr[k];
 			}
			 else{
				ac[k] = ac[k];
				bc[k] = bc[k];
				cc[k] = cc[k];
			 }
		}
		double Uk = 1;
		double t = ((c_k1 + Uk)-(c_k1 - Uk))*uni(rng) + (c_k1 - Uk);
		if(t < 0){
    		p_k1 = -t;
			}
		else{
    		p_k1 = t;
		}

		double t1 = ((c_k2 + Uk)-(c_k2 - Uk))*uni(rng) + (c_k2 - Uk);
		if(t1 < 0){
    		p_k2 = -t1;
			}
		else{
    		p_k2 = t1;
		}

		double t2 = ((c_k3 + Uk)-(c_k3 - Uk))*uni(rng) + (c_k3 - Uk);
		if(t2 < 0){
    		p_k3 = -t2;
			}
		else{
    		p_k3 = t2;
		}
		
		double multp = 1;double multc = 1;
		for(int l=0; l<400; l++){
			rpk[l] = dch(p_k1, p_k2, p_k3, ac[l], bc[l], cc[l]);
			rc[l] = dch(c_k1, c_k2, c_k3, ac[l], bc[l], cc[l]);
			 multp = multp + rpk[l];
			 multc = multc + rc[l];
			
		}//std::cout<< multp<<std::endl;
		
		double kp1 = (-1.*p_k1) + (-2.*p_k2) + (-3.*p_k3);
		double kc1 = (-1.*c_k1) + (-2.*c_k2) + (-3.*c_k3);

		a1k = (( multp + kp1 )-( multc + kc1 ));

		if(log(uni(rng)) < a1k){
    		c_k1 = p_k1;
    		c_k2 = p_k2;
    		c_k3 = p_k3;
    		//ack++;
		}
		else{
   		 c_k1 = c_k1;
    	 c_k2 = c_k2;
    	 c_k3 = c_k3;

		}

		K1[j] = c_k1;
		K2[j] = c_k2;
		K3[j] = c_k3;


		
		

	}
	long stop = clock();
		fast = stop - start;
	std::cout << a1k<<std::endl;
	std::cout << "time "<< fast<<std::endl;
/*
	for(int m = 0; m<400; m++){
		std::cout << ac[m] <<std::endl;
	}*/
	std::ofstream myfile3;
    myfile3.open ("K1.txt");
    
    for(int p1 = 0; p1 < num; p1++){
        myfile3 << K1[p1] << std::endl;

    }
    myfile3.close();

    std::ofstream myfile4;
    myfile4.open ("K2.txt");
    
    for(int p1 = 0; p1 < num; p1++){
        myfile4 << K2[p1] << std::endl;

    }
    myfile4.close();

    std::ofstream myfile5;
    myfile5.open ("K3.txt");
    
    for(int p1 = 0; p1 < num; p1++){
        myfile5 << K3[p1] << std::endl;

    }
    myfile5.close();
    
    
}

