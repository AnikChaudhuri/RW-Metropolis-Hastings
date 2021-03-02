#include<iostream>
#include<string>
int main(){
    std::cout << "enter your name: ";
    std::string name;
    std::cin >> name;
    std::cout << "hello " << name << std::endl;
    std::cout << "length = " << name.size() << std::endl;

}