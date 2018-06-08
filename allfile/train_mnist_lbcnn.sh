#!/usr/bin/env bash

#mnist
#th main.lua -netType resnet-binary-felix -dataset mnist -data './datasets/data' -save './datasets/output' -numChannels 16 -batchSize 10 -depth 75 -full 128 -sparsity 0.5 -nEpochs 1 -resume './datasets/output/mnist_resnet-binary-felix_75_16_512_128_0.5_3_10'

#th main.lua -netType resnet-binary-felix -dataset mnist -data '/media/Freya/juefeix/LBCNN' -save '/media/Freya/juefeix/LBCNN-Weights' -numChannels 16 -batchSize 10 -depth 75 -full 128 -sparsity 0.5

#SVHN
th main.lua -netType resnet-binary-felix -dataset svhn -data './datasets/data' -save './datasets/output2' -numChannels 16 -batchSize 10 -depth 40 -full 512 -sparsity 0.9

#cifar10
#th main.lua -netType resnet-binary-felix -dataset cifar10 -data './datasets/data' -save './datasets/lbcnn-output' -numChannels 384 -numWeights 704 -batchSize 5 -depth 50 -full 512 -sparsity 0.001 -datper 0.1
