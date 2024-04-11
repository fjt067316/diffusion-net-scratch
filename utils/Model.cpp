#include "Layer.cpp"
#include <vector>

class Model{
    public:
        vector<Layer*> layers;
        int size = 0;

    Model(){
        return;
    }

    vector<vector<vector<float>>> forward(){
        return;
    }

    vector<vector<vector<float>>> backward(){
        return;
    }
}