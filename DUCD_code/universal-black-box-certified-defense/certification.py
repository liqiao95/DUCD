import argparse
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from tqdm import tqdm
from pdf_functions import *
from universal_certified_robustness import Universal_CR
os.environ["NUMEXPR_MAX_THREADS"] = "64"
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--norm', type=int, default=2, help='l_p norm for radius computation')
parser.add_argument('--model_path', type=str, default='./model_saved/ResNet101_CIFAR10_our_2,0.707_best.pth')
parser.add_argument('--MonteNum', type=int, default=2000,help='Monte Carlo samping number')
parser.add_argument('--input_size', type=int, default=32 * 32 * 3)
parser.add_argument('--iid', type=bool, default=1, help='bool type value indicates if noise is i.i.d.')
parser.add_argument('--save_name', type=str, default='Gaussian', help='name for saving results')
parser.add_argument('--pdf_args', action='append', type=float,help='pdf hyper-parameters, set the first parameters as -1 for indicating i.i.d.')
parser.add_argument('--samples_begin', type=int, default=0, help='begin index of the test samples')
parser.add_argument('--samples_end', type=int, default=500, help='end index of the test samples')

#python /root/project/UniCR-main/certification.py cifar10 resnet18 /root/project/UniCR-main/model_saved/ --norm=2 --model_path="/root/project/UniCR-main/results1.0/substitute/cifar-s/resnet34_substitute_cifar10_L2.pt" --MonteNum=500 --input_size=3072 --batch=500 --iid=1 --gpu=0 --save_name=Gaussian_L2 --pdf_args=-1 --pdf_args=1.41 --pdf_args=2 --samples_begin=0 --samples_end=1000 --pdf_type gaussian
#python /root/project/UniCR-main/certification.py cifar10 resnet18 /root/project/UniCR-main/model_saved/ --norm=-1 --model_path="/root/project/UniCR-main/results1.0/substitute/cifar-s/resnet34_substitute_cifar10_L-1.pt" --MonteNum=500 --input_size=3072 --batch=500 --iid=1 --gpu=1 --save_name=Laplace_L-1 --pdf_args=-1 --pdf_args=1 --pdf_args=0.707 --samples_begin=0 --samples_end=1000 --pdf_type laplace
#python /root/project/UniCR-main/certification.py cifar10 resnet18 /root/project/UniCR-main/model_saved/ --norm=2 --model_path="/root/project/UniCR-main/results1.0/substitute/cifar-s/resnet34_substitute_cifar10_L2.pt" --MonteNum=500 --input_size=3072 --batch=500 --iid=1 --gpu=2 --save_name=Cauthy_L2 --pdf_args=-1 --pdf_args=-1 --pdf_args=0.3345 --samples_begin=0 --samples_end=1000 --pdf_type cauthy
#python /root/project/UniCR-main/certification.py cifar10 resnet18 /root/project/UniCR-main/model_saved/ --norm=2 --model_path="/root/project/UniCR-main/results1.0/substitute/cifar-s/resnet34_substitute_cifar10_L2.pt" --MonteNum=500 --input_size=3072 --batch=500 --iid=1 --gpu=0 --save_name=Pareto_L2 --pdf_args=-1 --pdf_args=1 --pdf_args=1 --samples_begin=0 --samples_end=1000 --pdf_type pareto
########添加参数##########
parser.add_argument('--para_dir', type=str, default=500, help='parameters save path')
parser.add_argument('--pdf_type', type=str, default=500, help='pdf type')

########添加结束##########

args = parser.parse_args()

if __name__ == '__main__':
    from multiprocessing import set_start_method

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset)

    #model = model.cuda()
    #checkpoint = torch.load(args.model_path)
    #model.load_state_dict(checkpoint['state_dict'])
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.model_path, map_location=device)
    model = model.cuda()

    model.eval()

    criterion = CrossEntropyLoss().cuda()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    #Universal_CR = Universal_CR(Gen_normal, args.iid, args.norm, args.dataset, args.MonteNum, args.batch, args.pdf_args)
    if args.pdf_type == 'laplace':
        Universal_CR = Universal_CR(Laplace, args.iid, args.norm, args.dataset, args.MonteNum, args.batch, args.pdf_args)
    elif args.pdf_type == 'gaussian':
        Universal_CR = Universal_CR(Gaussian, args.iid, args.norm, args.dataset, args.MonteNum, args.batch, args.pdf_args)        
    elif args.pdf_type == 'pareto':
        Universal_CR = Universal_CR(Pareto, args.iid, args.norm, args.dataset, args.MonteNum, args.batch, args.pdf_args) 
    elif args.pdf_type == 'cauthy':
        Universal_CR = Universal_CR(cauthy_iid, args.iid, args.norm, args.dataset, args.MonteNum, args.batch, args.pdf_args) 
    
    
    
    print('start')
    PA_number = []
    Results_R = []
    Results_R_other = []
    for j in tqdm(range(args.samples_end - args.samples_begin)):
        i = j + args.samples_begin
        print('fig {} certifying'.format(i))
        (x, y) = test_dataset[i]
        # lq添加
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        x = x.cuda().unsqueeze(0)
        y = y.cuda().unsqueeze(0)
        #lq添加结束
        #x = x.cuda().unsqueeze(0)
        #y = y.cuda().unsqueeze(0)

        PA, CA = Universal_CR.compute_PA_PB_binary(x, args.pdf_args, model)

        if PA == 'abstain':
            R = -1
            R_other = -1
            PA_save = -1
        else:
            PA_save = float(PA)
            R = Universal_CR.direction_optimization_binary(PA, args.pdf_args)
            if args.norm == 1:
                R_other = Laplace_R(PA, sigma=1.0)
                #cauthy_iid/Laplace_R/Pareto_R_L1
                #R_other = Gaussian_R_binary(PA, sigma=1.0)
            elif args.norm == 2:
                R_other = Gaussian_R_binary(PA, sigma=1.0)
                #R_other = Gaussian_R_binary(PA, sigma=1.0)
            else:
                R_other = Gaussian_R_infnorm(PA, 3072, sigma=1.0)
                #R_other = Gaussian_R_binary(PA, sigma=1.0)

        Results_R.append(R)
        Results_R_other.append(R_other)
        PA_number.append(PA_save)

        #####lq添加       
        print('PA: {}, our radius: {} , theoretical radius: {}'.format(PA_save, R, R_other))
        #####lq添加
        np.save('PA_{}'.format(args.save_name), PA_number)
        #####lq添加结束
        np.save('R_{}'.format(args.save_name), Results_R)
        np.save('R_other_{}'.format(args.save_name), Results_R_other)
