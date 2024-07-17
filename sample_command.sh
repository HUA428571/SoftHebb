python multi_layer.py --gpu-id 1 --preset 5SoftHebbCnnCIFAR --dataset-unsup CIFAR10_1 --dataset-sup CIFAR10_50
python multi_layer.py --gpu-id 2 --preset 5SoftHebbCnnCIFAR_noClassifier --dataset-unsup CIFAR10_1

python multi_layer.py --gpu-id 2 --preset 4SoftHebbCnnCIFAR_ft --dataset-sup CIFAR10_50 --resume without_classifier --model-name 4SoftHebbCnnCIFAR_ft