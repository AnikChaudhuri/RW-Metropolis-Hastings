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
    
	int num = 400000;
	std::vector<double> ac(10);
	std::vector<double> bc(10);
	std::vector<double> cc(10);
	std::vector<double> mu(10);
	
    //std::fill(x.begin(), x.end(), 1); .02,2,14,15
    double U = 14;
	
    //double r[10] = {.47963206,.154963462,.384218795,.257028457,.008668512,.024180703,.166085727,.279321785,.262429171,.316439148};
    double r[10] = {.598739352,.493116352,.579867973,.320856474,.081899588,.072795849,.334481889,.435275282,.517632462,.44421341};

    std::vector<double> d1(10);
    std::vector<double> d2(10);
    std::vector<double> d3(10);
/*
    d1[0] = 0; d2[0] = 1; d3[0] = 1; //d for each alpha
    d1[1] = 0; d2[1] = 1; d3[1] = 0;
    d1[2] = 0; d2[2] = 1; d3[2] = 0;
    d1[3] = 0; d2[3] = 1; d3[3] = 0;
    d1[4] = 0; d2[4] = 1; d3[4] = 0;
    d1[5] = 0; d2[5] = 1; d3[5] = 0;
    d1[6] = 0; d2[6] = 1; d3[6] = 0;
    d1[7] = 0; d2[7] = 1; d3[7] = 0;
    d1[8] = 0; d2[8] = 1; d3[8] = 0;
    d1[9] = 0; d2[9] = 1; d3[9] = 0;*/

    
    d1[0] = 1; d2[0] = 1; d3[0] = 1; //d for each alpha
    d1[1] = 1; d2[1] = 1; d3[1] = 1;
    d1[2] = 1; d2[2] = 1; d3[2] = 1;
    d1[3] = 1; d2[3] = 1; d3[3] = 1;
    d1[4] = 1; d2[4] = 1; d3[4] = 1;
    d1[5] = 1; d2[5] = 1; d3[5] = 1;
    d1[6] = 1; d2[6] = 1; d3[6] = 1;
    d1[7] = 1; d2[7] = 1; d3[7] = 1;
    d1[8] = 1; d2[8] = 1; d3[8] = 1;
    d1[9] = 1; d2[9] = 1; d3[9] = 1;
    
    
	double mpr, mc;

	std::vector<double> m_b(10);

	std::vector<double> gap(10);
	std::vector<double> gbp(10);
	std::vector<double> gcp(10);

	std::vector<double> apr(10);
	std::vector<double> bpr(10);
	std::vector<double> cpr(10);

	std::vector<double> resp(10);
	std::vector<double> resc(10);

	std::vector<double> K1(num);
	std::vector<double> K2(num);
	std::vector<double> K3(num);

	std::vector<double> rpk(10);
	std::vector<double> rc(10);

	std::fill(ac.begin(), ac.end(), .1);
	std::fill(bc.begin(), bc.end(), .3);
	std::fill(cc.begin(), cc.end(), .5);

	double nu0 = 1; double c20 = 0; double c2_b;
	double c2a = (nu0 + 10.)/2.; double c_1;
	double c_k1, c_k2, c_k3, p_k1, p_k2, p_k3;
	double ac_b;
	c_k1 = .3, c_k2 = .6 , c_k3 = .4;
	double a1k;
	for(int j = 0; j< num; j++){
		std::cout << "loop no: " << j<<std::endl;
		ac_b = 0;

		for(int l= 0; l<10; l++ ){
				m_b[l] = ac[l] * d1[l] + bc[l] * d2[l] + cc[l] * d3[l];
				ac_b = (pow((m_b[l]-r[l]),2)/(pow(m_b[l],2)+pow(r[l],2))) + ac_b;
			}
		
		for(int k = 0; k< 10; k++){
			
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
			double RA = log(Ralp/Ralc);

			if(log(uni(rng)) < RA ){
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
		double t = ((c_k1 + Uk)-(c_k1 - Uk)+.45)*uni(rng) + (c_k1 - Uk);
		if(t < 0){
    		p_k1 = -t;
			}
		else{
    		p_k1 = t;
		}

		double t1 = ((c_k2 + Uk)-(c_k2 - Uk)+.45)*uni(rng) + (c_k2 - Uk);
		if(t1 < 0){
    		p_k2 = -t1;
			}
		else{
    		p_k2 = t1;
		}

		double t2 = ((c_k3 + Uk)-(c_k3 - Uk)+.45)*uni(rng) + (c_k3 - Uk);
		if(t2 < 0){
    		p_k3 = -t2;
			}
		else{
    		p_k3 = t2;
		}
		
		double multp = 1;double multc = 1;
		for(int l=0; l<10; l++){
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


		
		

	}std::cout << "ac:"<<std::endl;

	for(int m = 0; m<10; m++){
		std::cout << ac[m] <<std::endl;
	}
    std::cout << "bc:"<<std::endl;
    for(int m = 0; m<10; m++){
		std::cout << bc[m] <<std::endl;
	}
    std::cout << "cc:"<<std::endl;
    for(int m = 0; m<10; m++){
		std::cout << cc[m] <<std::endl;
	}
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

