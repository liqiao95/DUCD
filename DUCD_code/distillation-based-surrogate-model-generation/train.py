from __future__ import print_function
import argparse, ipdb, json
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import network
from dataloader import get_dataloader
import os, random
import numpy as np
import torchvision
from pprint import pprint
from time import time
from approximate_gradients import *
import torchvision.models as models
from my_utils import *

print("torch version", torch.__version__)
def myprint(a):
    """Log the print statements"""
    global file
    print(a); file.write(a); file.write("\n"); file.flush()

#学生模型loss
def student_loss(args, s_logit, t_logit, return_t_logits=False):
    """Kl/ L1 Loss for student"""
    print_logits =  False
    if args.loss == "l1":
        loss_fn = F.l1_loss
        loss = loss_fn(s_logit, t_logit.detach())
    elif args.loss == "kl":
        loss_fn = F.kl_div
        s_logit = F.log_softmax(s_logit, dim=1)
        t_logit = F.softmax(t_logit, dim=1)
        loss = loss_fn(s_logit, t_logit.detach(), reduction="batchmean")
    else:
        raise ValueError(args.loss)

    if return_t_logits:
        return loss, t_logit.detach()
    else:
        return loss

# 数据生成器 (一次只取一个batch)
def get_infinite_batches(data_loader):
    while True:
        for i, (images, label) in enumerate(data_loader):
            yield images
            
def train(args, teacher, student, device, optimizer, epoch,num):
    """Main Loop for one epoch of Training Student"""
    global file
    teacher.eval()
    student.train()
    optimizer_S = optimizer
    gradients = []
    train_loader, _ = get_dataloader(args)
    data = get_infinite_batches(train_loader)
    for i in range(args.epoch_itrs):
        img_batch = next(data)
        fake =img_batch.to(device) 
        optimizer_S.zero_grad()
        with torch.no_grad(): 
            t_logit = teacher(fake)
        if args.loss == "l1" and args.no_logits:
            t_logit = F.log_softmax(t_logit, dim=1).detach()
            if args.logit_correction == 'min':
                t_logit -= t_logit.min(dim=1).values.view(-1, 1).detach()
            elif args.logit_correction == 'mean':
                t_logit -= t_logit.mean(dim=1).view(-1, 1).detach()
        s_logit = student(fake)
        loss_S = student_loss(args, s_logit, t_logit)
        loss_S.backward()
        optimizer_S.step() 
        # Log Results
        if i % args.log_interval == 0:
            myprint(f'Train Epoch: {epoch} [{i}/{args.epoch_itrs} ({100*float(i)/float(args.epoch_itrs):.0f}%)]\tS_loss: {loss_S.item():.6f}') 
            log_root_dir = '/root/project/QUCD_results/log/' + args.model + '_' + args.student_model + '_' + args.dataset  + '_' + num
            if os.path.exists(log_root_dir) == False:
                os.mkdir(log_root_dir)
            if i == 0:
                with open(log_root_dir + "/loss.csv", "a") as f:
                    f.write("%d,%f\n"%(epoch, loss_S))
            if args.rec_grad_norm and i == 0:
                S_grad_norm = compute_grad_norms(student)
                
                
        # update query budget
        args.query_budget -= args.cost_per_iteration
        if args.query_budget < args.cost_per_iteration:
            return 

def test(args,querynum, student = None, device = "cuda", test_loader = None, epoch=0):    
    global file
    student.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = student(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    myprint('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    log_root_dir = '/root/project/QUCD_results/log/' + args.model + '_' + args.student_model + '_' + args.dataset  + '_' + querynum
    if os.path.exists(log_root_dir) == False:
        os.mkdir(log_root_dir)
    with open(log_root_dir + "/accuracy.csv", "a") as f:
        f.write("%d,%f\n"%(epoch, accuracy))
    acc = correct/len(test_loader.dataset)
    return acc
def compute_grad_norms(student):
    S_grad = []
    for n, p in student.named_parameters():
        if "weight" in n:
            S_grad.append(p.grad.norm().to("cpu"))
    return   np.mean(S_grad)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='DFAD CIFAR')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',help='input batch size for training (default: 256)')
    parser.add_argument('--query_budget', type=float, default=20, metavar='N', help='Query budget for the extraction attack in millions (default: 20M)')
    parser.add_argument('--epoch_itrs', type=int, default=50)  
    parser.add_argument('--g_iter', type=int, default=1, help = "Number of generator iterations per epoch_iter")
    parser.add_argument('--d_iter', type=int, default=5, help = "Number of discriminator iterations per epoch_iter")

    parser.add_argument('--lr_S', type=float, default=0.1, metavar='LR', help='Student learning rate (default: 0.1)')
    parser.add_argument('--nz', type=int, default=256, help = "Size of random noise input to generator")

    parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'kl'],)
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep', 'cosine', "none"],)
    parser.add_argument('--steps', nargs='+', default = [0.1, 0.3, 0.5], type=float, help = "Percentage epochs at which to take next step")
    parser.add_argument('--scale', type=float, default=3e-1, help = "Fractional decrease in lr")
    #dataset cifar10_split：C10-S cifar101: C10.1
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist','svhn','cifar10','cifar10_split','cifar101'], help='dataset name (default: cifar10)')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--model', type=str, default='resnet34_8x', choices=['resnet34_8x','resnet18','resnet50'], help='Target model name (default: resnet34_8x)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=random.randint(0, 100000), metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--ckpt', type=str, default='checkpoint/teacher/cifar10-resnet34_8x.pt')
    

    parser.add_argument('--student_load_path', type=str, default=None)
    parser.add_argument('--model_id', type=str, default="debug")

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default="results")

    # Gradient approximation parameters
    parser.add_argument('--approx_grad', type=int, default=1, help = 'Always set to 1')
    parser.add_argument('--grad_m', type=int, default=1, help='Number of steps to approximate the gradients')
    parser.add_argument('--grad_epsilon', type=float, default=1e-3) 
    

    parser.add_argument('--forward_differences', type=int, default=1, help='Always set to 1')
    

    # Eigenvalues computation parameters
    parser.add_argument('--no_logits', type=int, default=1)
    parser.add_argument('--logit_correction', type=str, default='mean', choices=['none', 'mean'])

    parser.add_argument('--rec_grad_norm', type=int, default=1)

    parser.add_argument('--MAZE', type=int, default=0) 

    parser.add_argument('--store_checkpoints', type=int, default=1)

    parser.add_argument('--student_model', type=str, default='resnet18_8x',
                        help='Student model architecture (default: resnet18_8x)')


    args = parser.parse_args()

    save_budget_num = str(args.query_budget)
    args.query_budget *=  10**6
    args.query_budget = int(args.query_budget)
    if args.MAZE:

        print("\n"*2)
        print("#### /!\ OVERWRITING ALL PARAMETERS FOR MAZE REPLCIATION ####")
        print("\n"*2)
        args.scheduer = "cosine"
        args.loss = "kl"
        args.batch_size = 128
        args.g_iter = 1
        args.d_iter = 5
        args.grad_m = 10
        args.lr_S = 1e-1


    if args.student_model not in classifiers:
        if "wrn" not in args.student_model:
            raise ValueError("Unknown model")


    pprint(args, width= 80)
    print(args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.store_checkpoints:
        os.makedirs(args.log_dir + "/checkpoint", exist_ok=True)

    log_root_dir = '/root/project/QUCD_results/' + args.model + '_' + args.student_model + '_' + args.dataset  + '_' + save_budget_num 
    if os.path.exists(log_root_dir) == False:
        os.mkdir(log_root_dir)  
          
    # Save JSON with parameters
    with open(log_root_dir + "/parameters.json", "w") as f:
        json.dump(vars(args), f)

    with open(log_root_dir + "/loss.csv", "w") as f:
        f.write("epoch,loss_S\n")

    with open(log_root_dir + "/accuracy.csv", "w") as f:
        f.write("epoch,accuracy\n")

    if args.rec_grad_norm:
        with open(log_root_dir + "/norm_grad.csv", "w") as f:
            f.write("epoch,S_grad_norm,grad_wrt_X\n")

    with open("latest_experiments.txt", "a") as f:
        f.write(log_root_dir + "\n")
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda:%d"%args.device if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    global file
    model_root_dir = '/root/project/QUCD_results/modelinfo/' + args.model + '_' + args.student_model + '_' + args.dataset  + '_' + save_budget_num 
    model_dir = model_root_dir
    if(not os.path.exists(model_dir)):
        os.makedirs(model_dir)
    with open(f"{model_dir}/model_info.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)  
    file = open(f"{model_dir}/logs.txt", "w") 
    print(args)

    args.device = device
    _, test_loader = get_dataloader(args)


    args.normalization_coefs = None
    args.G_activation = torch.tanh

    num_classes = 10 if args.dataset in ['cifar10', 'svhn', 'mnist','cifar101','cifar10_split'] else 100
    args.num_classes = num_classes

    if args.model == 'resnet34_8x':
        teacher = network.resnet_8x.ResNet34_8x(num_classes=num_classes)
        if args.dataset == 'svhn':
            print("Loading SVHN TEACHER")
        teacher.load_state_dict( torch.load( args.ckpt, map_location=device), strict=False)
    elif args.model == 'resnet18':
        teacher = network.resnet_8x.ResNet18_8x(num_classes=num_classes)
        if args.dataset == 'svhn':
            print("Loading SVHN TEACHER resnet18")
        elif args.dataset == 'mnist':
            print("Loading MNIST TEACHER resnet18")
        else:
            print("Loading CIFAR TEACHERresnet18")
        teacher.load_state_dict( torch.load( args.ckpt, map_location=device), strict=False)
    elif args.model == 'resnet50':
        if args.dataset == 'svhn':
            print("Loading SVHN TEACHER resnet50")
        elif args.dataset == 'mnist':
            print("Loading MNIST TEACHER resnet50")
        else:
            print("Loading CIFAR TEACHER resnet50")
        teacher = network.resnet_8x.ResNet50_8x(num_classes=num_classes)
        teacher.load_state_dict(torch.load( args.ckpt, map_location=device), strict=False)
    else: 
        teacher = get_classifier(args.model, pretrained=True, num_classes=args.num_classes)
    print('loading success!')  
    
    ################evaluate teacher model
    teacher.eval()
    teacher = teacher.to(device)
    myprint("Teacher restored from %s"%(args.ckpt)) 
    print(f"\n\t\tTraining with {args.model} as a Target\n") 
    correct = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = teacher(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTeacher - Test set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, len(test_loader.dataset),accuracy))
    




    
    student = get_classifier(args.student_model, pretrained=False, num_classes=args.num_classes)
    student = student.to(device)

    args.student = student
    args.teacher = teacher 
    #student model architecture loading
    if args.student_load_path :
        student.load_state_dict( torch.load( args.student_load_path ) )
        myprint("Student initialized from %s"%(args.student_load_path))
        acc = test(args,querynum=save_budget_num, student=student, device = device, test_loader = test_loader)
    args.cost_per_iteration = args.batch_size * (args.g_iter * (args.grad_m+1) + args.d_iter)

    number_epochs = args.query_budget // (args.cost_per_iteration * args.epoch_itrs) + 1

    print (f"\nTotal budget: {args.query_budget//1000}k")
    print ("Cost per iterations: ", args.cost_per_iteration)
    print ("Total number of epochs: ", number_epochs)

    optimizer_S = optim.SGD( student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9 )



    steps = sorted([int(step * number_epochs) for step in args.steps])
    print("Learning rate scheduling at steps: ", steps)
    print()

    if args.scheduler == "multistep":
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, steps, args.scale)
    elif args.scheduler == "cosine":
        scheduler_S = optim.lr_scheduler.CosineAnnealingLR(optimizer_S, number_epochs)


    best_acc = 0
    acc_list = []

    for epoch in range(1, number_epochs + 1):
        # Train
        if args.scheduler != "none":
            scheduler_S.step()
        train(args, teacher=teacher, student=student, device=device, optimizer=optimizer_S, epoch=epoch,num = save_budget_num)
        # Test
        acc = test(args,querynum=save_budget_num, student=student, device = device, test_loader = test_loader, epoch=epoch)
        acc_list.append(acc)
        if acc>best_acc:
            best_acc = acc
            name = 'resnet34_8x'
            save_model_dir_student = model_dir + args.dataset + '-' + name + '.pt'
            torch.save(student.state_dict(),save_model_dir_student)
        if args.store_checkpoints:
            save_student_model_path = '/root/project/QUCD_results/model_pt/' + args.model + '_' + args.student_model + '_' + args.dataset +'_' + save_budget_num + 'student.pt'
            torch.save(student.state_dict(), save_student_model_path)

    myprint("Best Acc=%.6f"%best_acc)

    with open(args.log_dir + "/Max_accuracy = %f"%best_acc, "w") as f:
        f.write(" ")

     

    import csv
    os.makedirs('/root/project/QUCD_results/log', exist_ok=True)
    with open('log/QUCD-%s.csv'%(args.dataset), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(acc_list)


if __name__ == '__main__':
    main()


