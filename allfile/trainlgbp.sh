#!/usr/bin/env bash
th main.lua -netType lgbp-net -dataset cifar10 -data './datasets/data' -save './datasets/output' -numChannels 16 -batchSize 10 -depth 75 -full 128 -sparsity 0.5 -nEpochs 1

#svhn
CUDA_VISIBLE_DEVICES=2 th main.lua -netType lgbp-net -dataset svhn -data './datasets/data' -save './datasets/output2/compare' -numChannels 16 -batchSize 10 -depth 40 -full 512 -sparsity 0.9 -gbsz 9 -gbst 2 -ngbout 80 -mode t2 


#cifar10
CUDA_VISIBLE_DEVICES=0 th main.lua -netType lgbp-net -dataset cifar10 -data './datasets/data' -save './datasets/output3/gbsparse' -numChannels 384 -numWeights 704 -batchSize 5 -depth 50 -full 512 -sparsity 0.001 -datper 0.25 -mode tig -gbsz 3 -gbst 1 -ngbout 180 -gbsparse 0.01 -nEpochs 100

CUDA_VISIBLE_DEVICES=1 th main.lua -netType lgbp-net -dataset cifar10 -data './datasets/data' -save './datasets/output3/gbsparse' -numChannels 384 -numWeights 704 -batchSize 5 -depth 50 -full 512 -sparsity 0.001 -datper 0.25 -mode tig -gbsz 5 -gbst 1 -ngbout 60 -gbsparse 0.01

 CUDA_VISIBLE_DEVICES=2 th main.lua -netType resnet-binary-felix -dataset cifar10 -data './datasets/data' -save './datasets/lbcnn-output' -numChannels 384 -numWeights 704 -batchSize 5 -depth 50 -full 512 -sparsity 0.001 -datper 0.1

 CUDA_VISIBLE_DEVICES=2 th main.lua -netType lgbp-net -dataset cifar10 -data './datasets/data' -save './datasets/output3/gbsparse' -numChannels 384 -numWeights 704 -batchSize 5 -depth 50 -full 512 -sparsity 0.001 -datper 0.25 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2

 CUDA_VISIBLE_DEVICES=3 th main.lua -netType resnet-binary-felix -dataset cifar10 -data './datasets/data' -save './datasets/lbcnn-output' -numChannels 384 -numWeights 704 -batchSize 5 -depth 50 -full 512 -sparsity 0.001 -datper 0.05

1.8

--CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output2/cifar100' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 100 -LR 0.1


CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output2/gbsparse' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.4 -nEpochs 140 -LR 0.1

CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output2/gbsparse' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.1 -nEpochs 140 -LR 0.1

CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output2/gbsparse' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.05 -nEpochs 140 -LR 0.1

CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output2/szst' -batchSize 32 -depth 56 -mode t1 -gbsz 7 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 140 -LR 0.1

CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output2/szst' -batchSize 32 -depth 56 -mode t1 -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 140 -LR 0.1
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output2' -batchSize 32 -depth 110 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 140 -LR 0.1


CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output2/cifar100' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 140 -LR 0.1


CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output2/mode' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 140 -LR 0.1

CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output2/mode' -batchSize 32 -depth 56 -mode tg1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 140 -LR 0.1

CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output2/ngbout' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 40 -gbsparse 0.2 -nEpochs 140 -LR 0.1

CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output2/ngbout' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 120 -gbsparse 0.2 -nEpochs 140 -LR 0.1

CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output2/ngbout' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 180 -gbsparse 0.2 -nEpochs 140 -LR 0.1

CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output2/mode' -batchSize 32 -depth 56 -mode tg1 -gbsz 5 -gbst 1 -ngbout 120 -gbsparse 0.2 -nEpochs 140 -LR 0.1

CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output2/mode' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 120 -gbsparse 0.2 -nEpochs 140 -LR 0.1

1.9--------------------------------------------------------------------------------
terminal3
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output3/LR' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.1  ==
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/LR' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.06


CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output3/LR' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.01==
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/gbsparse' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.9 -nEpochs 164 -LR 0.06


CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/gbsparse' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.9 -nEpochs 164 -LR 0.1


CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output3/LR' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.001 ==
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/gbsparse' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 60 -gbsparse 0.2 -nEpochs 164 -LR 0.06


terminal_4
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output3/LR' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.2  ==
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output3/LR' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.06 ==
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output3/gbsparse' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.5 -nEpochs 164 -LR 0.04


CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output3/gbsparse' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.03 ==
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output3/gbsparse' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.9 -nEpochs 164 -LR 0.04


CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output3/gbsparse' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.005 ==
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output3/gbsparse' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.02 -nEpochs 164 -LR 0.04

cd gbl
terminal3
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/gbsparse' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.9 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/gbsparse' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/gbsparse' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.1 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/gbsparse' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.01 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/gbsparse' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.001 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/gbsparse' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.7 -nEpochs 164 -LR 0.05

CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/mode' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/mode' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 120 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/mode' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 60 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/mode' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 180 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/ngbout' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 120 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/ngbout' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 180 -gbsparse 0.2 -nEpochs 164 -LR 0.05


cd gbl

terminal4
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/ngbout' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 120 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/ngbout' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 120 -gbsparse 0.9 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/szst' -batchSize 32 -depth 56 -mode t1 -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/szst' -batchSize 32 -depth 56 -mode t1 -gbsz 7 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/szst' -batchSize 32 -depth 56 -mode t1 -gbsz 9 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/szst' -batchSize 32 -depth 56 -mode t1 -gbsz 9 -gbst 2 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05

CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/ngbout' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 60 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/ngbout' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 60 -gbsparse 0.9 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/ngbout' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 120 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/ngbout' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 120 -gbsparse 0.9 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/szst' -batchSize 32 -depth 56 -mode t1 -gbsz 7 -gbst 2 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/szst' -batchSize 32 -depth 56 -mode t1 -gbsz 7 -gbst 2 -ngbout 80 -gbsparse 0.9 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100' -batchSize 32 -depth 56 -mode t1 -gbsz 9 -gbst 1 -ngbout 80 -gbsparse 0.9 -nEpochs 164 -LR 0.05

tig 可达93，提升gbsparse有效果，改变ngbout效果不显著，sz3反而更好一些。cifar100效果近似

1.10
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/tig' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/tig' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/tig' -batchSize 32 -depth 56 -mode tig -gbsz 7 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05

cd gbl

CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/tig' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.4 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/tig' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.9 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/tig' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.1 -nEpochs 164 -LR 0.05


CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/tig' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.06
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/tig' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.07
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/tig' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.08



CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100' -batchSize 32 -depth 110 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05



1.11 cd gbl
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/smitrain/epoch70' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05 
-nEpochs 70
-resume './datasets/gbl-resnet-output4/smitrain/epoch70/cifar10_gbl-resnet_d56_sz5_st1_1_tig_gblr0_gbsp0.2_up_0LR0.06_gbl-resnet80_conv3_56_128_512_512_0.9_3_32'  
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/smitrain/epoch70' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05 -nEpochs 70
-resume './datasets/gbl-resnet-output4/smitrain/epoch70/cifar10_gbl-resnet_d56_sz3_st1_1_tig_gblr0_gbsp0.2_up_0LR0.05_gbl-resnet80_conv3_56_128_512_512_0.9_3_32'



CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/smitrain' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05 -nEpochs 50 
-resume './datasets/gbl-resnet-output4/smitrain/cifar10_gbl-resnet_d56_sz5_st1_1_tig_gblr0_gbsp0.2_up_0LR0.05_gbl-resnet80_conv3_56_128_512_512_0.9_3_32'
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/smitrain' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05 -nEpochs 50 
-resume './datasets/gbl-resnet-output4/smitrain/cifar10_gbl-resnet_d56_sz3_st1_1_tig_gblr0_gbsp0.2_up_0LR0.05_gbl-resnet80_conv3_56_128_512_512_0.9_3_32'



CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/smitrain/resume' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.06 -resume './datasets/gbl-resnet-output4/smitrain/cifar10_gbl-resnet_d56_sz5_st1_1_tig_gblr0_gbsp0.2_up_0LR0.06_gbl-resnet80_conv3_56_128_512_512_0.9_3_32'
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/smitrain/resume' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.07 -resume './datasets/gbl-resnet-output4/smitrain/cifar10_gbl-resnet_d56_sz5_st1_1_tig_gblr0_gbsp0.2_up_0LR0.07_gbl-resnet80_conv3_56_128_512_512_0.9_3_32'



CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/smitrain/resume' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.06 -resume './datasets/gbl-resnet-output4/smitrain/cifar10_gbl-resnet_d56_sz3_st1_1_tig_gblr0_gbsp0.2_up_0LR0.06_gbl-resnet80_conv3_56_128_512_512_0.9_3_32'
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/smitrain/resume' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.07 -resume './datasets/gbl-resnet-output4/smitrain/cifar10_gbl-resnet_d56_sz3_st1_1_tig_gblr0_gbsp0.2_up_0LR0.07_gbl-resnet80_conv3_56_128_512_512_0.9_3_32'

CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/d110/szst' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 140 -LR 0.1
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/d110/LR' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 140 -LR 0.1
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/d110/LR' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 140 -LR 0.07
CUDA_VISIBLE_DEVICES=1 th main.lua -n etType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/d110/LR' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 140 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/d110/LR' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 140 -LR 0.04


CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/smitrain' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 50 -LR 0.8
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/smitrain/resume/epoch70' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05 -resume './datasets/gbl-resnet-output4/smitrain/epoch70/cifar10_gbl-resnet_d56_sz5_st1_1_tig_gblr0_gbsp0.2_up_0LR0.05_gbl-resnet80_conv3_56_128_512_512_0.9_3_32'  
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/smitrain/resume/epoch70' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05 -resume './datasets/gbl-resnet-output4/smitrain/epoch70/cifar10_gbl-resnet_d56_sz3_st1_1_tig_gblr0_gbsp0.2_up_0LR0.05_gbl-resnet80_conv3_56_128_512_512_0.9_3_32'
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100/LR' -batchSize 32 -depth 110 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100/LR' -batchSize 32 -depth 110 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100/LR' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100/LR' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100/ngbout' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 180 -gbsparse 0.2 -nEpochs 164 -LR 0.1

CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100/LR' -batchSize 32 -depth 110 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.06
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100/LR' -batchSize 32 -depth 110 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.07
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100/ngbout' -batchSize 32 -depth 110 -mode tig -gbsz 5 -gbst 1 -ngbout 180 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100/ngbout' -batchSize 32 -depth 110 -mode tig -gbsz 5 -gbst 1 -ngbout 120 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100/ngbout' -batchSize 32 -depth 110 -mode tig -gbsz 5 -gbst 1 -ngbout 40 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100/szst' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100/szst' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 180 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100/sp' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 180 -gbsparse 0.3 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100/sp' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 180 -gbsparse 0.1 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100/sp' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 180 -gbsparse 0.9 -nEpochs 164 -LR 0.05

CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100/fulltrain' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100/fulltrain' -batchSize 32 -depth 110 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/fulltrain' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/fulltrain' -batchSize 32 -depth 110 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05

good :sz3-st1 sp0.3 ngbout 180 lr 0.05 no-train depth 110 cifar100

sz3 better
ngbout 180 better sp0.3 better
fulltrain better 而且不会发散可以调大lr

！！fulltrain cifar10 depth 110 sz3st1 tig sp0.2 ngbout 180 lr0.05 is better than baseline
baselinen run again
1.12
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/fulltrain/lr' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/fulltrain/lr' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.08
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100/fulltrain/lr' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/cifar100/fulltrain/lr' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.08
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/gbconv4' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05 -gbconv 4
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/gbconv4' -batchSize 32 -depth 110 -mode t1 -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05 -gbconv 4
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/smitrain2' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 70 -LR 0.05 -gbconv 4
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/smitrain2' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 70 -LR 0.05 -gbconv 3
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/smitrain2' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 180 -gbsparse 0.3 -nEpochs 70 -LR 0.05 -gbconv 3

CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/fulltrain/lr' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.3
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/fulltrain/lr' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.2
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/smitrain2/resume/70' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05 -gbconv 4 -resume './datasets/gbl-resnet-output4/smitrain2/cifar100_gbl-resnet_d110_sz3_st1_1_tig_gblr0_gbsp0.2_up_0LR0.05_gbl-resnet80_conv4_110_128_512_512_0.9_3_32'
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/smitrain2/resume/70' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05 -gbconv 3 -resume './datasets/gbl-resnet-output4/smitrain2/cifar100_gbl-resnet_d110_sz3_st1_1_tig_gblr0_gbsp0.2_up_0LR0.05_gbl-resnet80_conv3_110_128_512_512_0.9_3_32'
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/smitrain2/resume/70' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 180 -gbsparse 0.3 -nEpochs 164 -LR 0.05 -gbconv 3 -resume './datasets/gbl-resnet-output4/smitrain2/cifar100_gbl-resnet_d110_sz3_st1_1_tig_gblr0_gbsp0.3_up_0LR0.05_gbl-resnet180_conv3_110_128_512_512_0.9_3_32'
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/fulltrain' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.1 -gbconv 4
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/fulltrain' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 180 -gbsparse 0.3 -nEpochs 164 -LR 0.1 -gbconv 4
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/fulltrain' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 180 -gbsparse 0.3 -nEpochs 164 -LR 0.1 -gbconv 4


1.13
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/fulltrain/ngbout' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/fulltrain/ngbout' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/d56/fulltrain' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/d56/fulltrain' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.08
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/d56/fulltrain' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.07
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/d110/notrain' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/d110/notrain' -batchSize 32 -depth 110 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/d110/notrain' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/d110/notrain' -batchSize 32 -depth 110 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/d56/notrain' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/d56/notrain' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05


1.16
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/d56/fulltrain' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/d56/fulltrain' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.08
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/d56/fulltrain' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/d56/fulltrain/ngbout' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.2 -nEpochs 164 -LR 0.05


CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/fulltrain/ngbout' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.7 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/fulltrain/ngbout' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/fulltrain/ngbout' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/fulltrain/ngbout' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05


CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/baseline1/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05 -gbconv 5
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/baseline1/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.1 -gbconv 5
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/baseline1/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.5 -nEpochs 164 -LR 0.05 -gbconv 5


CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/baseline2/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05 -gbconv 6
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/baseline2/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.1 -gbconv 6
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/baseline2/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.5 -nEpochs 164 -LR 0.05 -gbconv 6





CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/fulltrain/ngbout' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/fulltrain/ngbout' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/fulltrain/ngbout' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/d56/fulltrain' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/d56/fulltrain/ngbout' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/d56/fulltrain/ngbout' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/d56/fulltrain/ngbout' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.2 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/d110/fulltrain/ngbout' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.2 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/d110/fulltrain/ngbout' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/d110/fulltrain/ngbout' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/d110/fulltrain' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/baseline2' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.1 -gbconv 6
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/baseline2' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.1 -gbconv 6
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/baseline1' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.1 -gbconv 5
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/baseline1' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.1 -gbconv 5
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/baseline2' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.1 -gbconv 6
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/baseline1' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.1 -gbconv 5
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/baseline2' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.1 -gbconv 6
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output4/baseline1' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.1 -gbconv 5
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/baseline2/cifar100' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.1 -gbconv 6
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/baseline1/cifar100' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 80 -gbsparse 0.2 -nEpochs 164 -LR 0.1 -gbconv 5
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/baseline2/cifar100' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.1 -gbconv 6
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output4/baseline1/cifar100' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.1 -gbconv 5




1.18
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.06
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.07
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.08
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.06
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.07
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.08


CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/d56' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/d56' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/d110' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/mode' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/mode' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.08



CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/mode' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/mode' -batchSize 32 -depth 56 -mode t1 -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/szst' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/szst' -batchSize 32 -depth 56 -mode tig2 -gbsz 5 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/szst' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/szst' -batchSize 32 -depth 56 -mode t1 -gbsz 7 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05


CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/szst' -batchSize 32 -depth 56 -mode t1 -gbsz 5 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/szst' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/szst' -batchSize 32 -depth 56 -mode tig2 -gbsz 5 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/szst' -batchSize 32 -depth 56 -mode tig -gbsz 7 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05




CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/d56' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/d56' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.03
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/d56' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.07
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/d110' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.07
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/szst' -batchSize 32 -depth 56 -mode tig -gbsz 9 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/mode' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.5 -nEpochs 164 -LR 0.1



CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d110/lr' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d110/lr' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d110/lr' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d110/lr' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.5 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d110/szst' -batchSize 32 -depth 110 -mode tig2 -gbsz 5 -gbst 1 -ngbout 8 -gbsparse 0.5 -nEpochs 164 -LR 0.1


CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d110/szst' -batchSize 32 -depth 110 -mode tig2 -gbsz 5 -gbst 1 -ngbout 8 -gbsparse 0.5 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d110/lr' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d110/lr' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.5 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d110/lr' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d110/lr' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.05

1.19
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.08
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.09
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/LR' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.1

CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/LR/d110' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.08
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/LR/d110' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/LR/d110' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d110/lr' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d110/lr' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.03
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d110/lr' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.06


CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d110/szst' -batchSize 32 -depth 110 -mode tig2 -gbsz 5 -gbst 1 -ngbout 8 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d110/szst' -batchSize 32 -depth 110 -mode tig2 -gbsz 5 -gbst 1 -ngbout 8 -gbsparse 0.5 -nEpochs 164 -LR 0.08
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d110/szst' -batchSize 32 -depth 110 -mode tig2 -gbsz 5 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d110/szst' -batchSize 32 -depth 110 -mode tig2 -gbsz 7 -gbst 1 -ngbout 8 -gbsparse 0.5 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d110/mode' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.07
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d56/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.05

CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d56/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d56/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.5 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d56/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.07
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d56/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.03
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d56/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d56/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.8 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/d56/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.5 -nEpochs 164 -LR 0.05






CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.02
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.08
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.2 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.8 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.005


CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8fix/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8fix/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.5 -nEpochs 164 -LR 0.08
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8fix/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.5 -nEpochs 164 -LR 0.02
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8fix/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.5 -nEpochs 164 -LR 0.005
 


CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8fix/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.3 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8fix/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 1 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8fix/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.7 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8fix/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.1 -nEpochs 164 -LR 0.05



CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8fix/lr' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8fix/lr' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.05

goog result:
http://localhost:8889/edit/gbl/datasets/gbl-resnet-output5/16fix/lr/cifar10_gbl-resnet_d110_sz3_st1_1_tig_gblr0_gbsp0.9_up_0LR0.05_gbl-resnet16_conv3_110_128_512_512_0.9_3_32/accuracy.log
http://localhost:8889/edit/gbl/datasets/gbl-resnet-output5/16fix/lr/cifar10_gbl-resnet_d56_sz3_st1_1_tig_gblr0_gbsp0.5_up_0LR0.04_gbl-resnet16_conv3_56_128_512_512_0.9_3_32/accuracy.log


http://localhost:8889/edit/gbl/datasets/gbl-resnet-output5/8fix/lr/cifar10_gbl-resnet_d110_sz3_st1_1_tig2_gblr0_gbsp0.5_up_0LR0.05_gbl-resnet8_conv3_110_128_512_512_0.9_3_32/accuracy.log
http://localhost:8889/edit/gbl/datasets/gbl-resnet-output5/8fix/lr/cifar10_gbl-resnet_d56_sz3_st1_1_tig2_gblr0_gbsp0.7_up_0LR0.05_gbl-resnet8_conv3_56_128_512_512_0.9_3_32/accuracy.log

CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.06

CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8fix/lr' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8fix/lr' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.06
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8fix/lr' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.7 -nEpochs 164 -LR 0.04

CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8fix/lr' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.5 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8fix/lr' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.04

CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.06



1.21
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.03
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.02
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.03
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.02
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.03
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.03



CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr/100' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.03
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr/100' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.02
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr/100' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.03
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr/100' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.02
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr/100' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.03
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr/100' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.03



CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr/100' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr/100' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.04




CUDA_VISIBLE_DEVICES=1 th main.lua -netType lgbp-net -dataset svhn -data './datasets/data' -save './datasets/gbl-lbcnn-output1/16fix/lr' -numChannels 384 -numWeights 704 -batchSize 5 -depth 50 -full 512 -sparsity 0.001 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 80 -LR 0.01
CUDA_VISIBLE_DEVICES=2 th main.lua -netType lgbp-net -dataset svhn -data './datasets/data' -save './datasets/gbl-lbcnn-output1/16fix/lr' -numChannels 384 -numWeights 704 -batchSize 5 -depth 50 -full 512 -sparsity 0.001 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 80 -LR 0.001
CUDA_VISIBLE_DEVICES=3 th main.lua -netType lgbp-net -dataset svhn -data './datasets/data' -save './datasets/gbl-lbcnn-output1/16fix/lr' -numChannels 384 -numWeights 704 -batchSize 5 -depth 50 -full 512 -sparsity 0.001 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 80 -LR 0.0001


CUDA_VISIBLE_DEVICES=1 th main.lua -netType lgbp-net -dataset svhn -data './datasets/data' -save './datasets/gbl-lbcnn-output1/16fix/lr' -numChannels 384 -numWeights 704 -batchSize 5 -depth 50 -full 512 -sparsity 0.001 -mode tig -gbsz 9 -gbst 2 -ngbout 16 -gbsparse 0.9 -nEpochs 80 -LR 0.0001


1.23
cd gbl
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr/100' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr/100' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/lr/100' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.05
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/mode' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/mode' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/mode' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/mode' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/mode' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar100 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/mode' -batchSize 32 -depth 110 -mode tig2 -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.04





CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 24 -save './datasets/imagnet/16fix/lr' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 64 -gbsparse 0.9 -nEpochs 164 -LR 0.04  
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 24 -save './datasets/imagnet/16fix/lr' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 64 -gbsparse 0.9 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=3 th main.lua -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -save './imagnet' -depth 50 -batchSize 32 -nThreads 24
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 24 -save './datasets/imagnet/16/lr' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 64 -gbsparse 0.9 -nEpochs 164 -LR 0.08



1.23
th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -save './datasets/imagnet/16fix/lr' -depth 50 -batchSize 256 -mode tig -gbsz 7 -gbst 1 -ngbout 64 -gbsparse 0.9 -nEpochs 80 -LR 0.04 -nGPU 4 -nThreads 8 -shareGradInput true
CUDA_VISIBLE_DEVICES=3 th main.lua -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -save './imagnet' -depth 50 -batchSize 32 -nThreads 24
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 24 -save './datasets/imagnet/16fix/lr' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 64 -gbsparse 0.9 -nEpochs 80 -LR 0.04 -resume './datasets/imagnet/16fix/lr/imagenet_gbl-resnet_d50_sz7_st1_1_tig_gblr0_gbsp0.9_up_0LR0.04_gbl-resnet64_conv3_50_128_512_512_0.9_3_64'
CUDA_VISIBLE_DEVICES=0 th main.lua -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -save './imagnet' -depth 50 -batchSize 64 -nThreads 24
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 24 -save './datasets/imagnet/16fix/lr/resume' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 64 -gbsparse 0.9 -nEpochs 80 -LR 0.1 -resume './datasets/imagnet/16fix/lr/imagenet_gbl-resnet_d50_sz7_st1_1_tig_gblr0_gbsp0.9_up_0LR0.1_gbl-resnet64_conv3_50_128_512_512_0.9_3_64'
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 24 -save './datasets/imagnet/16/lr/resume' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 64 -gbsparse 0.9 -nEpochs 80 -LR 0.08 -resume './datasets/imagnet/16/lr/imagenet_gbl-resnet_d50_sz7_st1_1_tig_gblr0_gbsp0.9_up_0LR0.08_gbl-resnet64_conv3_50_128_512_512_0.9_3_64'

CUDA_VISIBLE_DEVICES=0 th main.lua -netType resnet-binary-felix -dataset svhn -data './datasets/data' -save './datasets/gbl-lbcnn-output1/16fix/lr' -numChannels 384 -numWeights 704 -batchSize 5 -depth 50 -full 512 -sparsity 0.001 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 80 -LR 0.0001

1.24
CUDA_VISIBLE_DEVICES=2 th main.lua -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -save './imagnet' -depth 50 -LR 0.1 -nThreads 24 -batchSize 64
CUDA_VISIBLE_DEVICES=0 th main.lua -netType lgbp-net -dataset svhn -data './datasets/data' -save './datasets/gbl-lbcnn-output1/16fix/lr' -numChannels 16 -batchSize 10 -depth 40 -full 512 -sparsity 0.9 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 80 -LR 0.0001
CUDA_VISIBLE_DEVICES=0 th main.lua -netType lgbp-net -dataset svhn -data './datasets/data' -save './datasets/gbl-lbcnn-output1/16fix/lr' -numChannels 16 -batchSize 10 -depth 40 -full 512 -sparsity 0.9 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 80 -LR 0.00005

1.25
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 24 -save './datasets/imagnet/16/ngbout' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 80 -LR 0.08
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 24 -save './datasets/imagnet/16fix/ngbout' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 80 -LR 0.04
CUDA_VISIBLE_DEVICES=0 th main.lua -netType lgbp-net -dataset svhn -data './datasets/data' -save './datasets/gbl-lbcnn-output1/16fix/lr' -numChannels 16 -batchSize 10 -depth 40 -full 512 -sparsity 0.9 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 80 -LR 0.0005
CUDA_VISIBLE_DEVICES=0 th main.lua -netType lgbp-net -dataset svhn -data './datasets/data' -save './datasets/gbl-lbcnn-output1/16/lr' -numChannels 16 -batchSize 10 -depth 40 -full 512 -sparsity 0.9 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 80 -LR 0.0001


1.28
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 24 -save './datasets/imagnet/16fix/ngbout/lr' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 80 -LR 0.01
=====CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 12 -save './datasets/imagnet/16fix/ngbout/lr' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 80 -LR 0.005

CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/pre' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.7 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/pre' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.3 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/pre' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 1 -nEpochs 164 -LR 0.04


1.30
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/pre' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.1 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/pre' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 1 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/pre' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.7 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/pre' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/pre' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.3 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/pre' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.1 -nEpochs 164 -LR 0.04


2.1

CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 16 -save './datasets/imagnet/16fix/ngbout/lr' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 80 -LR 0.02

CUDA_VISIBLE_DEVICES=0 th main.lua -dataset cifar100 -data './datasets/data' -save './datasets/output2/try4' -depth 56 -nEpochs 164 -LR 0.1
CUDA_VISIBLE_DEVICES=0 th main.lua -dataset cifar100 -data './datasets/data' -save './datasets/output2/try4' -depth 110 -nEpochs 164 -LR 0.1

CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/4fix/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 4 -gbsparse 0.9 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/32fix/lr' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 32 -gbsparse 0.9 -nEpochs 164 -LR 0.04


2.2\
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/4/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 4 -gbsparse 0.9 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/8/lr' -batchSize 32 -depth 56 -mode tig2 -gbsz 3 -gbst 1 -ngbout 8 -gbsparse 0.9 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/32/lr' -batchSize 32 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 32 -gbsparse 0.9 -nEpochs 164 -LR 0.04


CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/pre/gbsz/' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 32 -gbsparse 0.9 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/pre/gbsz/' -batchSize 32 -depth 56 -mode tig -gbsz 7 -gbst 1 -ngbout 32 -gbsparse 0.9 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/pre/gbsz/' -batchSize 32 -depth 56 -mode tig -gbsz 9 -gbst 1 -ngbout 32 -gbsparse 0.9 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16fix/pre/gbsz/' -batchSize 32 -depth 56 -mode tig -gbsz 11 -gbst 1 -ngbout 32 -gbsparse 0.9 -nEpochs 164 -LR 0.04

2.3
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/pre/gbsz/' -batchSize 32 -depth 56 -mode tig -gbsz 5 -gbst 1 -ngbout 32 -gbsparse 0.9 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/pre/gbsz/' -batchSize 32 -depth 56 -mode tig -gbsz 7 -gbst 1 -ngbout 32 -gbsparse 0.9 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/pre/gbsz/' -batchSize 32 -depth 56 -mode tig -gbsz 9 -gbst 1 -ngbout 32 -gbsparse 0.9 -nEpochs 164 -LR 0.04
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/pre/gbsz/' -batchSize 32 -depth 56 -mode tig -gbsz 11 -gbst 1 -ngbout 32 -gbsparse 0.9 -nEpochs 164 -LR 0.04

2.5
cd gbl2
CUDA_VISIBLE_DEVICES=0 th main.lua -netType lgbp-net -dataset svhn -data './datasets/data' -save './datasets/svhn/80fix/sz' -numChannels 16 -batchSize 10 -depth 40 -full 512 -sparsity 0.9 -gbsz 3 -gbst 1 -ngbout 80 -mode tig -gbsparse 0.9
CUDA_VISIBLE_DEVICES=3 th main.lua -netType lgbp-net -dataset svhn -data './datasets/data' -save './datasets/svhn/80/' -numChannels 16 -batchSize 10 -depth 40 -full 512 -sparsity 0.9 -gbsz 9 -gbst 2 -ngbout 80 -mode tig -gbsparse 0.9


2.6
CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/16/d110' -batchSize 32 -depth 110 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 164 -LR 0.08

2.7
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 16 -save './datasets/imagnet/16fix/ngbout/lr' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 32 -gbsparse 0.9 -nEpochs 80 -LR 0.01

2.8
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 24 -save './datasets/imagnet/16fix/ngbout/lr/resume' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 80 -LR 0.01 -resume './datasets/imagnet/16fix/ngbout/lr/imagenet_gbl-resnet_d50_sz7_st1_1_tig_gblr0_gbsp0.9_up_0LR0.01_gbl-resnet16_conv3_50_128_512_512_0.9_3_64'

2.9
CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 24 -save './datasets/imagnet/16fix/ngbout/lr/resume2' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 100 -LR 0.001 -resume './datasets/imagnet/16fix/ngbout/lr/imagenet_gbl-resnet_d50_sz7_st1_1_tig_gblr0_gbsp0.9_up_0LR0.01_gbl-resnet16_conv3_50_128_512_512_0.9_3_64'

CUDA_VISIBLE_DEVICES=0 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 16 -save './datasets/imagnet/16fix/ngbout/lr' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 32 -gbsparse 0.9 -nEpochs 80 -LR 0.01

3.7
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 8 -save './datasets/imagnet/32fix/lr/' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 32 -gbsparse 0.9 -nEpochs 100 -LR 0.01

CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 8 -save './datasets/imagnet/16fix/mode/lr' -depth 50 -batchSize 64 -mode t1 -gbsz 7 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 100 -LR 0.01

3.9
CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 8 -save './datasets/imagnet/32fix/lr/resume0' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 32 -gbsparse 0.9 -nEpochs 100 -LR 0.01 -resume './datasets/imagnet/32fix/lr/imagenet_gbl-resnet_d50_sz7_st1_1_tig_gblr0_gbsp0.9_up_0LR0.01_gbl-resnet32_conv3_50_128_512_512_0.9_3_64'

CUDA_VISIBLE_DEVICES=3 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 8 -save './datasets/imagnet/16fix/mode/lr' -depth 50 -batchSize 64 -mode t1 -gbsz 7 -gbst 1 -ngbout 16 -gbsparse 0.9 -nEpochs 100 -LR 0.01 -resume './datasets/imagnet/16fix/mode/lr/imagenet_gbl-resnet_d50_sz7_st1_1_t1_gblr0_gbsp0.9_up_0LR0.01_gbl-resnet16_conv3_50_128_512_512_0.9_3_64'

4.2
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 8 -save './datasets/imagnet/32fix/lr/resume1' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 32 -gbsparse 0.9 -nEpochs 100 -LR 0.001 -resume './datasets/imagnet/32fix/lr//home/jcz/gbl/datasets/imagnet/32fix/lr/resume0/imagenet_gbl-resnet_d50_sz7_st1_1_tig_gblr0_gbsp0.9_up_0LR0.01_gbl-resnet32_conv3_50_128_512_512_0.9_3_64' -nEpochs 60

4.7
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 8 -save './datasets/imagnet/32fix/lr/resume2' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 32 -gbsparse 0.9 -nEpochs 100 -LR 0.001 -resume '/home/jcz/gbl/datasets/imagnet/32fix/lr/resume1/imagenet_gbl-resnet_d50_sz7_st1_1_tig_gblr0_gbsp0.9_up_0LR0.001_gbl-resnet32_conv3_50_128_512_512_0.9_3_64' -nEpochs 60


4.13
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 2 -save './datasets/imagnet/64fix/lr' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 64 -gbsparse 0.9 -LR 0.01 -nEpochs 40

4.14
CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 8 -save './datasets/imagnet/32fix/lr/resume3' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 32 -gbsparse 0.9 -nEpochs 100 -LR 0.001 -resume '/home/jcz/Desktop/Link to gbl/datasets/imagnet/32fix/lr/resume2/imagenet_gbl-resnet_d50_sz7_st1_1_tig_gblr0_gbsp0.9_up_0LR0.001_gbl-resnet32_conv3_50_128_512_512_0.9_3_64' -nEpochs 80

4.17

CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -dataset cifar10 -data './datasets/data' -save './datasets/gbl-resnet-output5/test' -batchSize 1000 -depth 56 -mode tig -gbsz 3 -gbst 1 -ngbout 16 -gbsparse 0.5 -nEpochs 164 -LR 0.08

CUDA_VISIBLE_DEVICES=1 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 2 -save './datasets/imagnet/64fix/lr' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 64 -gbsparse 0.9 -LR 0.02 -nEpochs 40

CUDA_VISIBLE_DEVICES=2 th main.lua -netType gbl-resnet -data '/data1/datasets/imageNet/ILSVRC2012/' -dataset imagenet -nThreads 2 -save './datasets/imagnet/64fix/lr' -depth 50 -batchSize 64 -mode tig -gbsz 7 -gbst 1 -ngbout 64 -gbsparse 0.9 -LR 0.01 -nEpochs 40

