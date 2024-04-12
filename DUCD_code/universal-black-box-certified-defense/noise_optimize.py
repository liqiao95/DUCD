from universal_certified_robustness import Universal_CR
import argparse
import os
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR

from pdf_functions import *
os.environ['NUMEXPR_MAX_THREADS'] = '8'
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
parser.add_argument('--MonteNum', type=int, default=2000, help='Monte Carlo samping number')
parser.add_argument('--input_size', type=int, default=32 * 32 * 3, help='input dimension, 32*32*3 for Cifar10 and 224*224*3 for ImageNet')
parser.add_argument('--iid', type=int,default=0, help='bool value indicates if noise is i.i.d.')
parser.add_argument('--save_name', type=str, default='Gaussian', help='name for saving results')
parser.add_argument('--pdf_args', action='append', type=float, help='hyper-parameters of pdf function, set the first parameters as -1 for indicating i.i.d.')
parser.add_argument('--opt_args_low',action='append', type=float, help='lower bound for opitmizing pdf parameters')
parser.add_argument('--opt_args_high',action='append', type=float, help='upper bound for optimizing pdf parameters')
parser.add_argument('--opt_args_step',action='append', type=float, help='steps for optimizing pdf parameters')
parser.add_argument('--samples_begin', type=int, default=0, help='begin index of test samples')
parser.add_argument('--samples_end', type=int, default=500, help='end index of test samples')
parser.add_argument('--save_path', type=str, help='results_save_path')
args = parser.parse_args()

if __name__ == '__main__':

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    os.environ['IMAGENET_LOC_ENV']="/home/cc/data/"
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    # train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
    #                           num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset)
    print('load model: {}'.format(args.model_path))
    if args.dataset=='cifar10':
        #model.load_state_dict(torch.load(args.model_path))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(args.model_path, map_location=device)
        #model.load_state_dict(torch.load(args.model_path, map_location=device), False)
        
    else:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])
    criterion = CrossEntropyLoss().cuda()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
    torch.multiprocessing.set_start_method('spawn')
    Universal_CR=Universal_CR(Gen_normal,args.iid,args.norm,args.dataset,args.MonteNum,args.batch,args.pdf_args)
    print('start')

    if os.path.exists('R_L{}_ours_{}_{}_cifar{}-{}.npy'.format(args.norm,args.save_name,args.pdf_args, args.samples_begin, args.samples_end)):
        print("continue certifying")
        Results_index=np.load('index_L{}_ours_{}_{}_cifar{}-{}.npy'.format(args.norm,args.save_name,args.pdf_args, args.samples_begin, args.samples_end),allow_pickle=True).tolist()
        Results_R=np.load('R_L{}_ours_{}_{}_cifar{}-{}.npy'.format(args.norm,args.save_name,args.pdf_args, args.samples_begin, args.samples_end),allow_pickle=True).tolist()
        Results_RM_Rs=np.load('other_Rs_L{}_{}_{}_cifar{}-{}.npy'.format(args.norm,args.save_name,args.pdf_args, args.samples_begin, args.samples_end),allow_pickle=True).tolist()
        Results_args=np.load('Args_L{}_ours_{}_{}_cifar{}-{}.npy'.format(args.norm,args.save_name,args.pdf_args, args.samples_begin, args.samples_end),allow_pickle=True).tolist()
        Results_CA=np.load('CA_L{}_ours_{}_{}_cifar{}-{}.npy'.format(args.norm,args.save_name,args.pdf_args, args.samples_begin, args.samples_end),allow_pickle=True).tolist()
        Results_CA_other=np.load('CA_L{}_other_{}_{}_cifar{}-{}.npy'.format(args.norm,args.save_name,args.pdf_args, args.samples_begin, args.samples_end),allow_pickle=True).tolist()
        skip=len(Results_index)
    else:
        Results_index = []
        Results_R = []
        Results_args = []
        Results_RM_Rs = []
        Results_CA=[]
        Results_CA_other=[]
        skip=0

    #样本总数计数
    sample_num = 0
    abstain_num = 0
    for j in range(args.samples_end - args.samples_begin-skip):
        i = j + args.samples_begin+skip
        print('fig {} certifying'.format(i))
        Results_index.append(i)
        (x, y) = test_dataset[i]
        best_Rs, best_args, RM_Rs,CA,CA_other,abstain_value = Universal_CR.noise_optimization_binary(x,y,args.opt_args_low,args.opt_args_high,args.opt_args_step,model,max_iter=20)
        '''
        best_Rs:
        best_args
        RM_Rs
        CA
        CA_other
        abstain_value:添加的对弃权样本的计数
        '''
        #认证样本计数
        sample_num = sample_num + 1 
        #弃权总数计数
        if abstain_value == 1:
            abstain_num = abstain_num + 1
        print('abstain percent of the model on the test images: {} %'.format(100 * abstain_num / sample_num))    


        if best_Rs:
            Results_R.append(best_Rs)
            Results_args.append(best_args)
            Results_RM_Rs.append(RM_Rs)
            Results_CA.append(CA)
            Results_CA_other.append(CA_other)
        else:
            Results_R.append(-1)
            Results_args.append(-1)
            Results_RM_Rs.append(-1)
            Results_CA.append(-1)
            Results_CA_other.append(-1)
        np.save('index_L{}_ours_{}_{}_cifar{}-{}'.format(args.norm,args.save_name,args.pdf_args, args.samples_begin, args.samples_end), Results_index)
        np.save('R_L{}_ours_{}_{}_cifar{}-{}'.format(args.norm,args.save_name,args.pdf_args, args.samples_begin, args.samples_end), Results_R)
        np.save('other_Rs_L{}_{}_{}_cifar{}-{}'.format(args.norm,args.save_name,args.pdf_args, args.samples_begin, args.samples_end), Results_RM_Rs)
        np.save('Args_L{}_ours_{}_{}_cifar{}-{}'.format(args.norm,args.save_name,args.pdf_args, args.samples_begin, args.samples_end), Results_args)
        np.save('CA_L{}_ours_{}_{}_cifar{}-{}'.format(args.norm,args.save_name,args.pdf_args, args.samples_begin, args.samples_end), Results_CA)
        np.save('CA_L{}_other_{}_{}_cifar{}-{}'.format(args.norm,args.save_name,args.pdf_args, args.samples_begin, args.samples_end), Results_CA_other)
        torch.cuda.empty_cache()
  ########添加统计计算到底认证通过率有多少的功能。