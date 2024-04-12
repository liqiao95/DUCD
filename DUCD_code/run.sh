cd /root/project/QUCD/distillation-based-surrogate-model-generation;

python train.py --batch_size 256 --dataset cifar101 --model resnet34_8x --ckpt /root/project/base_model/cifar/resnet34_cifar10.pt --device 1 --grad_m 1 --query_budget 30 --log_dir /root/project/datafree-model-extraction/dfme/save_results/cifar10 --student_model resnet18_8x --loss l1
