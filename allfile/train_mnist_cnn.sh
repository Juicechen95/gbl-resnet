#!/usr/bin/env bash
th main.lua -netType resnet-dense-felix -dataset mnist -data './datasets/data' -save './datasets/output' -numChannels 16 -batchSize 10 -depth 75 -full 128

#svhn
#th main.lua -netType resnet-dense-felix -dataset svhn -data './datasets/data' -save './datasets/output' -numChannels 16 -batchSize 10 -depth 40 -full 512

#cifar10
#th main.lua -netType resnet-dense-felix -dataset cifar10 -data './datasets/data' -save './datasets/output' -numChannels 384 -numWeights 704 -batchSize 3 -depth 50 -full 512
