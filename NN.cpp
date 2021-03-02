#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <cmath>
#include <random>
#include <fstream>

int main(){
    std::vector<double> n_mig(16);
    std::vector<double> n_sym(16);
    std::vector<double> n_pos(16);

    double w14, w24, w15, w25, w34, w35, w46, w56, b4, b5,b6;
    w14 = .2; w24 = -.3; w15 = .5; w25 = .6; w34 = .4; w35 = -.8; b4 = .14; b5 = .023; b6 = -.4;w46 = .1; w56 = .3;

    double mig[16] = {500,1500,1400,2500,3500,4500,5525,6666,8000,12000,25000,40000,47000,58000,75000,85000};
    double sym[16] = {100,400,675,900,1250,1600,1800,2300,2800,3000,3300,3700,4000,4300,5000,4900};
    double pos[16] = {10,35,45,53,67,62,71,81,91,101,120,122,142,162,172,240};

    for(int k = 0; k< 16; k++){
        n_mig[k] = (mig[k] - 500)/(85000-500);
        n_sym[k] = (sym[k] - 100)/(4900-100);
        n_pos[k] = (pos[k] - 10)/(240-10);
    }
    int num = 10000;
for(int j = 0; j<num; j++){
    double Y;

    for(int i =0; i<16; i++){
        

        double I1 = w14*n_mig[i] + w24*n_sym[i] + w34*n_pos[i] + b4;
        double I2 = w15*n_mig[i] + w25*n_sym[i] + w35*n_pos[i] + b5;

        double act1 = exp(I1) - exp(-I1)/(exp(I1) + exp(-I1));
        double act2 = exp(I2) - exp(-I2)/(exp(I2) + exp(-I2));

        double J = act1 * w46 + act2 * w56 + b6;

         Y = 1/(1 + exp(-J));
         double del6 = (n_pos[i] - Y)*Y*(1-Y);
         double del5 = del6 * w56 *act2*(1-act2);
         double del4 = del6 * w46 *act1*(1-act1);
         w14 = w14 - 0.9*(n_mig[i])*del4;
         w15= w15 - 0.9*(n_mig[i])*del5;
         w24 = w24 - 0.9*(n_sym[i])*del4;
         w25 = w25 - 0.9*n_sym[i]*del5;
         w34 = w34 -0.9*n_sym[i]*del4;
         w35 = w35 - 0.9*n_sym[i]*del5;
         w46 = w46 - 0.9*act1*del6;
         w56 = w56 - 0.9*act2*del6;
         b4 = b4 - .9*del4;
         b5 = b5 - .9*del5;
         b6 = b6 -.9*del6;


        //std::cout << n_pos[i] << std::endl;
    }

    
    
}
    std::cout << w14 <<std::endl;
    std::cout << w24 <<std::endl;
    std::cout << w34 <<std::endl;
    std::cout << w15 <<std::endl;
    std::cout << w25 <<std::endl;
    std::cout << w35 <<std::endl;
    std::cout << w46 <<std::endl;
    std::cout << w56 <<std::endl;
    std::cout << b4 <<std::endl;
    std::cout << b5 <<std::endl;
    std::cout << b6 <<std::endl;

}