#ifndef MLP_H
#define MLP_H

#include "dataitem.h"
#include <vector>

class MLP
{
private:
    class Weights
    {
    public:
        Weights(unsigned int inputdim, unsigned int outputdim);
        float get(const unsigned int outidx, const unsigned int inidx) const;
        void add(const unsigned int outidx, const unsigned int inidx,
                 float value);
    private:
        const unsigned int inputdim;
        const unsigned int outputdim;
        std::vector<float> mat;
    };

public:
    MLP(std::initializer_list<unsigned int> dims);
    std::vector<float> operator()(const std::vector<float> &input);
    void train(const std::vector<DataItem> &dataset, const float eta);

private:
    const std::vector<unsigned int> dimlist;
    std::vector<Weights> weightlist;
    std::vector<std::vector<float> > slist;
    std::vector<std::vector<float> > ylist;
    static float activation(float x);
    static float derivactiv(float x);
    static std::vector<float> softmax(const std::vector<float> &input);
};


#endif  // MLP_H
