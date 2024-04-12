
from universal_certified_robustness import Universal_CR
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


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, choices=DATASETS)
parser.add_argument('--arch', type=str, choices=ARCHITECTURES)
parser.add_argument('--outdir', type=str, help='folder to save model and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=128, type=int, metavar='N',
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
parser.add_argument('--model_path', type=str, default='/root/project/datafree-model-extraction-test/dfme/save_results/cifar10/checkpoint/student.pt')
parser.add_argument('--MonteNum', type=int, default=2000,help='Monte Carlo sampling number')
parser.add_argument('--input_size', type=int, default=32 * 32 * 3, help='input dimension')
parser.add_argument('--iid', type=bool, help='bool value indicating if noise is i.i.d.')
parser.add_argument('--pdf_args', action='append', type=float, help='pdf hyper-parameters, set the first parameters as -1 for indicating i.i.d.')
parser.add_argument('--model_structure', type=str, default='resnet34', help='model structure type')
parser.add_argument('--model_type', type=str, default='substitude',help='model type, base model or substitude model')
parser.add_argument('--model_data_type', type=str, default='cifar10', help='model structure type')




args = parser.parse_args()

def train(model,optimizer,trainloader,epoch,Universal_CR,args):
    model.train()
    pbar=tqdm(enumerate(trainloader))
    for batch_idx,(x,y) in pbar:
        X = x.cuda()
        label = y.cuda()
        optimizer.zero_grad()
        print(X.shape)
        epsilons=Universal_CR.noise_sampling(X.shape[0],args)
        noise = epsilons.float()
        clean_outputs = model(X + noise.cuda().reshape(X.shape))
        pred_clean = torch.max(clean_outputs, 1)[1]
        loss = F.cross_entropy(clean_outputs, label)
        loss.backward()
        optimizer.step()
        step_score = accuracy_score(label.cpu().data.squeeze().numpy(), pred_clean.cpu().data.squeeze().numpy())
        pbar.set_description('E|{}|Ls:{:.4f}|C:{:.2f}|'.format(epoch+1,loss.item(),step_score*100))

def test(model,optimizer,testloader,Universal_CR,args):
    model.eval()
    all_label = []
    all_pred = []

    with torch.no_grad():
        pbar=tqdm(enumerate(testloader))
        for batch_idx,(x,y) in pbar:
            X = x.cuda()
            label = y.cuda()
            optimizer.zero_grad()
            epsilons = Universal_CR.noise_sampling(X.shape[0], args)
            noise = epsilons.float()
            clean_outputs = model(X + noise.cuda().reshape(X.shape))
            pred_clean = torch.max(clean_outputs, 1)[1]
            all_label.extend(label)
            all_pred.extend(pred_clean)

    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)


    test_score = accuracy_score(all_label.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())

    return test_score*100

if __name__ == '__main__':
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(args.model_path, map_location=device), False)
    criterion = CrossEntropyLoss().cuda()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    Universal_CR = Universal_CR(Gaussian, args.iid, args.norm, args.dataset, args.MonteNum, args.batch,
                                args.pdf_args)
    save_path = args.outdir + '/' + args.model_structure + '_'+ args.model_type +'_'+ args.model_data_type+'_L'+ str(args.norm) + '.pt'
    num_epoch = args.epochs
    best_acc = 0
    best_epoch = 0
    for epoch in range(num_epoch):

        train(model, optimizer, train_loader, epoch, Universal_CR, args.pdf_args)
        test_score = test(model, optimizer, test_loader, Universal_CR, args.pdf_args)
        if test_score > best_acc:
            best_epoch = epoch + 1
            best_acc = test_score
        print('Epoch:{},Test Acc:{:.4f},Best Acc:{:.4f} at epoch {}'.format(epoch + 1, test_score, best_acc, best_epoch))
        torch.save(model, save_path)

       

