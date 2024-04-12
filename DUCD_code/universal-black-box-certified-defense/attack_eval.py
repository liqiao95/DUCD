"""
对抗样本测试
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from art.attacks.evasion import  AutoProjectedGradientDescent, HopSkipJump, CarliniL0Method, CarliniL2Method, CarliniLInfMethod, AutoAttack, SquareAttack,BoundaryAttack,GeoDA,AutoConjugateGradient,SignOPTAttack
from art.estimators.classification import PyTorchClassifier, BlackBoxClassifierNeuralNetwork
from art.estimators import BaseEstimator
from art.utils import preprocess
import os 
import torch
from torchvision import datasets, transforms
import resnet_8x
import sys
import six
import argparse
from ae_get_data import load_cifar10, load_mnist, load_svhn
from architectures import ARCHITECTURES, get_architecture, get_architecture_adv

parser = argparse.ArgumentParser(description='TRAIN BASE MODEL')
parser.add_argument('--device', type=str, default='1', choices=['0', '1','2'], help='device name (default: 1)')
parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18','resnet34','resnet50'], help='model type (default: resnet18)')
parser.add_argument('--path', type=str, default='/root/project/base_model/cifar/resnet18_cifar10.pt',  help='model path')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist','cifar10','svhn'], help='dataset type (default: cifar10)')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--attack', type=str, default='autopgd', choices=['autopgd', 'hsja','cw','geoda','sa','zoo','autoattack','acg','sign-opt','hsja'], help='dataset name (default: cifar10)')
parser.add_argument('--norm', type=str, default='l2', choices=['l1', 'l2','linf'], help='attack norm')
parser.add_argument('--eps', type=float, default= 0.25,  help='perturbation')
parser.add_argument('--target', type=str, default= True,choices=['True', 'False'],  help='targeted /untargeted attack')
parser.add_argument('--batch_size', type=int, default= 256,  help='batchsize')
parser.add_argument('--type', type=str, default= 'certified', choices=['certified','base'], help='certified model or classifier')
args = parser.parse_args()
#python /root/project/UniCR-main/attack_eval.py --device 0 --model resnet18 --path /root/project/UniCR-main/results2.75/substitute/cifar101/resnet34_substitute_cifar10_L1.pt --attack autopgd --target False --eps 2.75 --type certified --norm l1
#python /root/project/UniCR-main/attack_eval.py --device 2 --model resnet18 --path /root/project/UniCR-main/results0.25/substitute/cifar10/resnet34_substitute_cifar10_L-1.pt --attack hjsa --target False --eps 0.25 --type certified --norm linf
#python /root/project/UniCR-main/attack_eval.py --device 2 --model resnet18 --path /root/project/UniCR-main/results2.75/substitute/cifar101/resnet34_substitute_cifar10_L-1.pt --attack autopgd --target False --eps 2.75 --type certified --norm linf

#python /root/project/UniCR-main/attack_eval.py --device 1 --model resnet34 --path /root/project/UniCR-main/results0.5/base/resnet34_base_cifar10_L1.pt --attack simba --norm l1 --target False --eps 0.5 --type certified
#设备选定
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#载入数据集
if args.dataset == 'mnist':
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
elif args.dataset == 'cifar10':
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
if args.dataset == 'svhn':
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_svhn()

#Step 1a: Swap axes to PyTorch's NCHW format

x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

# Step 2: Create the model
if args.type == 'certified':
    model = get_architecture_adv(args.model, args.dataset)
    model = torch.load(args.path, map_location=device)
else:
    if args.model == 'resnet18':
        model = resnet_8x.ResNet18_8x(num_classes=10) 
    elif args.model == 'resnet34':
        model = resnet_8x.ResNet34_8x(num_classes=10) 
    elif args.model == 'resnet50':
        model = resnet_8x.ResNet50_8x(num_classes=10)   
    model.load_state_dict(torch.load(args.path, map_location=device)) 


# Step 3: Create the model/创建模型 但不训练
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=10,)

    

# Step 4: Train the ART classifier
#classifier.fit(x_train, y_train, batch_size=256, nb_epochs=0)
# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

#范数
if args.norm == 'l1':
    norm = 1
elif args.norm =='l2':
    norm = 2
elif args.norm == 'linf':
    norm = np.inf
# Step 6: Generate adversarial test examples
if args.attack == 'autopgd':
    attack = AutoProjectedGradientDescent(classifier, norm=norm, eps=args.eps)
elif args.attack =='acg':
    attack = AutoConjugateGradient(classifier, norm=norm,eps=args.eps,targeted=False,batch_size =args.batch_size)
elif args.attack == 'sa':
    attack = SquareAttack(classifier,norm=norm, batch_size=args.batch_size, eps=args.eps)
elif args.attack == 'sign-opt':
    attack = SignOPTAttack(classifier, batch_size=args.batch_size, epsilon=args.eps,targeted=False,)
elif args.attack == 'hsja':
    attack = HopSkipJump(classifier, batch_size=args.batch_size,targeted=False,norm=norm,max_eval=5000)
    
if args.target == 'True':
    y_target = np.zeros(len(x_test), dtype=int) 
    x_test_adv = attack.generate(x=x_test, y=y_target)
else:
    x_test_adv = attack.generate(x=x_test)   


# Step 7: Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
### 存储数据和原始标签
if args.target == 'True':
    save_data_path = '/root/project/cifar10_data/adv_data/' + 'targeted_' + str(args.eps) + args.attack + args.norm + '_' + 'data' + '_' + args.model + '_' +args.dataset + '.npy'
    save_label_path = '/root/project/cifar10_data/adv_data/' + 'targeted_' + str(args.eps) +args.attack + args.norm + '_' + 'label' + '_' + args.model + '_' +args.dataset +  '.npy'
elif args.target == 'False':
    save_data_path = '/root/project/cifar10_data/adv_data/' + 'untargeted_' + str(args.eps) + args.attack + args.norm + '_' + 'data' + '_' + args.model + '_' +args.dataset + '.npy'
    save_label_path = '/root/project/cifar10_data/adv_data/' + 'untargeted_' + str(args.eps) +args.attack + args.norm + '_' + 'label' + '_' + args.model + '_' +args.dataset +  '.npy'
np.save(save_data_path, x_test_adv)
np.save(save_label_path, y_test)
