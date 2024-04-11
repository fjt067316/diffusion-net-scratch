#include <string>
#include <iostream>
#include "Tensor.cpp"

using namespace std;

class Layer{
    public:
        const string tag;

        Layer(string t) : tag(t)
        {}

        virtual ~Layer(){
            cout << "Layer destructor called" << endl;
        }

    Tensor<double, 4> forward();
};