###############################
######### 训练分类器 ###########
###############################

# import packages
import torch
import torchvision
import torch.nn.functional as F
import os
import dill
import network
import cifar10_models
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import argparse

'''
运行指令
python /root/project/datafree-model-extraction-test/dfme/classifier_train.py  --dataset cifar10 --model resnet34 --device 2 --num_epochs 30
python /root/project/datafree-model-extraction-test/dfme/classifier_train.py  --dataset mnist --model inception_v3 --device 2 --batch_size 128
python /root/project/datafree-model-extraction-test/dfme/classifier_train.py  --dataset cifar10-split --model resnet18 --device 2 --num_epochs 80 --batch_size 256

'''
# Device configuration.

parser = argparse.ArgumentParser(description='TRAIN BASE MODEL')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'svhn','cifar10','cifar10-split', 'cifar10.1'], help='dataset name (default: cifar10)')
parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18','resnet34','resnet50','resnet101','inception_v3'], help='model name (default: resnet18)')
parser.add_argument('--device', type=str, default='0', choices=['0','1','2'], help='device number (default: 0)')
# Hyper-parameters
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--learning_rate', type=int, default=0.001)

args = parser.parse_args()

# Device configuration.
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(args.num_classes)

# Transform configuration and data augmentation.
transform_train = torchvision.transforms.Compose([torchvision.transforms.Pad(4),
                                                torchvision.transforms.RandomHorizontalFlip(),
                                                 torchvision.transforms.RandomCrop(32),
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
mnist_transform = transforms.Compose([
    transforms.Ｇrayscale(num_output_channels=3),
	transforms.ToTensor(), 
	transforms.Normalize([0.5], [0.5],),
    
])
'''数据集设置'''
if args.dataset == 'cifar10':
    train_set = torchvision.datasets.CIFAR10("/root/project/cifar10_data/", download=True, train=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10("root='/root/project/cifar10_data/'", download=True, train=False, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size = args.batch_size, shuffle=True )
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size)
elif args.dataset == 'cifar10-split':
    with open('/root/project/cifar10_data/cifar10_split/train_teacher.pkl','rb') as f:
        train_loader = dill.load(f)
    with open('/root/project/cifar10_data/cifar10_split/test.pkl','rb') as f:
        test_loader = dill.load(f)
elif args.dataset == 'cifar10.1':
    with open('/root/project/cifar10_data/cifar10.1/train_cifar10_1_enhance.pkl','rb') as f:
        train_loader = dill.load(f)
    with open('/root/project/cifar10_data/cifar10.1/test_cifar10_1_enhance.pkl','rb') as f:
        test_loader = dill.load(f)    
elif args.dataset == 'mnist':
    train_set = torchvision.datasets.MNIST(root='/root/project/cifar10_data/mnist',train=True, transform=mnist_transform, download=False)
    test_set = torchvision.datasets.MNIST(root='/root/project/cifar10_data/mnist',train=False, transform=mnist_transform, download=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size = args.batch_size, shuffle=True )
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                               batch_size = args.batch_size,
                                               shuffle = True)
elif args.dataset == 'svhn':
    train_set = torchvision.datasets.SVHN(root='/root/project/cifar10_data/svhn',
                                              split='train',
                                              download=True,
                                              transform=torchvision.transforms.ToTensor())
    test_set = torchvision.datasets.SVHN(
        root='/root/project/cifar10_data/svhn',
        split='test',
        download=True,
        transform=torchvision.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size = args.batch_size, shuffle=True )
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                               batch_size = args.batch_size,
                                               shuffle = True)

'''模型设置'''
if args.model == 'resnet18':
    model = network.resnet_8x.ResNet18_8x(num_classes=args.num_classes)
elif args.model == 'resnet34':
    model = network.resnet_8x.ResNet34_8x(num_classes=args.num_classes)  
elif args.model == 'resnet50':
    model = network.resnet_8x.ResNet50_8x(num_classes=args.num_classes)    
elif args.model == 'resnet101':
    model = network.resnet_8x.ResNet101_8x(num_classes=args.num_classes)  
elif args.model == 'inception_v3':
    model = cifar10_models.inception.inception_v3(num_classes=args.num_classes)  

     

model.cuda()

# Loss ans optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# For updating learning rate.
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def val(dataloader, model, loss_fn):
    # Test the model.
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Train the model.
total_step = len(train_loader)
curr_lr = args.learning_rate
for epoch in range(args.num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass.
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, args.num_epochs, i+1, total_step, loss.item()))
    # Decay learning rate.
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

save_path = '/root/project/base_model/' + args.model + '_' + args.dataset + '.pt'
torch.save(model.state_dict(), save_path)
print("Done!")
 





