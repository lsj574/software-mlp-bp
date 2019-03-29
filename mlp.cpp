#include "mlp.h"
#include <random>
#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <cmath>

MLP::Weights::Weights(unsigned int inputdim, unsigned int outputdim)
    : inputdim(inputdim != 0 ? inputdim : throw std::invalid_argument("")),
      outputdim(outputdim != 0 ? outputdim : throw std::invalid_argument("")),
      mat(inputdim * outputdim)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::generate_n(mat.begin(), inputdim * outputdim,
                    [&dist, &rng](){return dist(rng);});
}

float MLP::Weights::get(const unsigned int outidx,
                        const unsigned int inidx) const
{
    return mat[outidx * inputdim + inidx];
}

void MLP::Weights::add(const unsigned int outidx,
                       const unsigned int inidx, float value)
{
    mat[outidx * inputdim + inidx] += value;
}

MLP::MLP(std::initializer_list<unsigned int> dims) :
    dimlist((dims.size() >= 2 ? dims :
             throw std::invalid_argument("At least 2 layers are required.")))
{
    for (unsigned int i = 0; i < dimlist.size()-1; ++i)
        weightlist.push_back(Weights(dimlist[i], dimlist[i+1]));
    for (unsigned int i = 0; i < dimlist.size(); ++i) {
        slist.push_back(std::vector<float>(dimlist[i]));
        ylist.push_back(std::vector<float>(dimlist[i]));
    }
}

std::vector<float> MLP::operator()(const std::vector<float> &input)
{
    if (input.size() != dimlist[0])
        throw std::invalid_argument("Input dimension mismatch.");

    // ylist[l][j] <=> y^l_j
    // slist[l][j] <=> s^l_j
    // weightlist[l-1] : weights between layer l-1 and l <=> w^l

    //for (unsigned int j = 0; j < dimlist.front(); ++j)
    //ylist[0][j] = input[j];
    ylist[0] = input;

    for (unsigned int l = 1; l < dimlist.size(); ++l) {
        // s^l_j = \sum_{i=0}^{N_{l-1}} w^l_{ji} y^{l-1}_i
        for (unsigned int j = 0; j < dimlist[l]; ++j) {
            slist[l][j] = 0;
            for (unsigned int i = 0; i < dimlist[l-1]; ++i)
                slist[l][j] += ylist[l-1][i] * weightlist[l-1].get(j, i);
        }

        // y^l_j = f(s^l_j)
        std::transform(slist[l].begin(), slist[l].end(), ylist[l].begin(),
                       activation);
    }

    return softmax(ylist[dimlist.size()-1]);
}

float MLP::activation(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

float MLP::derivactiv(float x)
{
    return activation(x) * (1.0f - activation(x));
}

std::vector<float> MLP::softmax(const std::vector<float> &input)
{
    std::vector<float> ret(input);
    const float max_elem(*std::max_element(input.begin(), input.end()));

    float expsum = 0;
    std::transform(ret.begin(), ret.end(), ret.begin(),
                   [&](float x){
                       float e = std::exp(x - max_elem);
                       expsum += e;
                       return e;});
    std::transform(ret.begin(), ret.end(), ret.begin(),
                   std::bind2nd(std::divides<float>(), expsum));

    return ret;
}

void MLP::train(const std::vector<DataItem> &dataset, const float eta)
{
    for (unsigned int idx = 0; idx < dataset.size(); ++idx) {
        if (dataset[idx].in.size() != dimlist.front() ||
            dataset[idx].out.size() != dimlist.back())
            throw std::invalid_argument("dataset dimension mismatch.");

        std::vector<float> result(this->operator()(dataset[idx].in));

        // e^l = expected - result (output layer)
        std::vector<float> error(dataset[idx].out);
        std::transform(error.begin(), error.end(), result.begin(),
                       error.begin(), std::minus<float>());

        for (unsigned int l = dimlist.size()-1; l >= 1; --l) {
            // \delta^l_j(t) = f'(s^l_j(t))\times e^l_j(t)
            std::vector<float> delta(slist[l]);
            std::transform(delta.begin(), delta.end(), delta.begin(),
                           derivactiv);
            std::transform(delta.begin(), delta.end(), error.begin(),
                           delta.begin(), std::multiplies<float>());

            // e^l_j = \sum_{k=0}^{N_{l+1}} \delta_k^{l+1}(t) w_{kj}(t)
            // calculate e^{l-1}_j for next loop
            // this is actually not the error
            error.clear();
            error.resize(dimlist[l-1], 0.0f);
            for (unsigned int j = 0; j < dimlist[l-1]; ++j)
                for (unsigned int k = 0; k < dimlist[l]; ++k)
                    error[j] += delta[k] * weightlist[l-1].get(k, j);

            // w^l_{ji}(t+1) = w^l_{ji}(t) + \eta \delta^l_j(t) y^{l-1}_i(t)
            for (unsigned int j = 0; j < dimlist[l]; ++j)
                for (unsigned int i = 0; i < dimlist[l-1]; ++i)
                    weightlist[l-1].add(j, i, eta * delta[j] * ylist[l-1][i]);
        }
    }
}
