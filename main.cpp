#include "mlp.h"
#include "mnist.h"
#include <iostream>
#include <algorithm>

int main(int argc, char *argv[])
{
    MNIST trainset("../MNIST/train-images.idx3-ubyte", "../MNIST/train-labels.idx1-ubyte");
    MNIST testset("../MNIST/t10k-images.idx3-ubyte", "../MNIST/t10k-labels.idx1-ubyte");

    const unsigned int inputdim = trainset.getrows() * trainset.getcols();
    const unsigned int outputdim = MNIST::LABEL_MAX;

    MLP mlp{inputdim, 64, 36, outputdim};
    std::vector<DataItem> dataset(trainset.getdataset());

    for (int epoch = 0; epoch < 50; ++epoch) {
        std::cout << "Epoch " << epoch << std::endl;
        mlp.train(dataset, 0.3f);
    }

    int correct = 0;
    for (int i = 0; i < testset.getsize(); ++i) {
        std::vector<float> out(mlp(testset.getimage(i)));
        unsigned int maxidx = std::distance(out.begin(), std::max_element(out.begin(), out.end()));

        if (maxidx == testset.getlabel(i))
            correct++;
    }

    std::cout << correct << " per " << testset.getsize() << std::endl;

    return 0;
}
