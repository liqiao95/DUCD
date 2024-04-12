from torchvision import datasets, transforms
import torchvision
import torch
import dill

def get_dataloader(args):
    if args.dataset.lower()=='mnist':
        print("Loading mnist data")
        mnist_transform = transforms.Compose([
            transforms.Ｇrayscale(num_output_channels=3),
            ##transforms.Resize((32, 32)),####不知道行不行得通
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5],),])
        train_set = torchvision.datasets.MNIST(root='/root/project/cifar10_data/mnist',train=True, transform=mnist_transform, download=False)
        test_set = torchvision.datasets.MNIST(root='/root/project/cifar10_data/mnist',train=False, transform=mnist_transform, download=False)
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size = args.batch_size, shuffle=True )
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                               batch_size = args.batch_size,
                                               shuffle = True)                  
    elif args.dataset.lower()=='svhn':
        print("Loading SVHN data")
        train_loader = torch.utils.data.DataLoader( 
            datasets.SVHN(root='/root/project/cifar10_data/svhn', split='train', download=True,
                       transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader( 
            datasets.SVHN(root='/root/project/cifar10_data/svhn', split='test', download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif args.dataset.lower()=='cifar10':
        train_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR10(root='/root/project/cifar10_data/cifar-10-batches-py', train=True, download=True,
                       transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR10(root='/root/project/cifar10_data/cifar-10-batches-py', train=False, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif args.dataset.lower()=='cifar10_split':
        #固定batchsize，要更改batchsize需要重新去分割，见文件/root/project/split_dataset.py
        with open('/root/project/cifar10_data/cifar10_split/train_student.pkl','rb') as f:
            train_loader = dill.load(f)
        with open('/root/project/cifar10_data/cifar10_split/test.pkl', 'rb') as f:
            test_loader = dill.load(f)
    elif args.dataset.lower()=='cifar101':  
        transform_train = transforms.Compose([
                            transforms.Resize((32, 32)),                            
                            transforms.RandomCrop(32, padding=4),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])
        transform_test = transforms.Compose([
                            transforms.Resize((32, 232)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])
        data_dir = "/root/project/A_self_dataset/cifar10.1/enhance"
        data_set = datasets.ImageFolder(data_dir, transform = transform_train)
        train_size = int(0.7 * len(data_set))
        test_size = int(0.3 * len(data_set)) + 1
        train_dataset, test_dataset = torch.utils.data.random_split(data_set,[train_size, test_size])
 
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=False  )
        with open('/root/project/cifar10_data/cifar10_split/test.pkl', 'rb') as f:
            test_loader = dill.load(f)       

    return train_loader, test_loader