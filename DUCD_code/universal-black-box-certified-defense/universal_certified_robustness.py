import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torchvision
import pyswarms as ps
from scipy.stats import norm as Norm
import numpy as np
from statsmodels.stats.proportion import proportion_confint
import time
import torch
import argparse
import os
from pdf_functions import *


class Universal_CR:
    def __init__(self,pdf_function,iid,norm,dataset,MC_Num,batch_size,pdf_args,input_size=None):
        """ pdf_function: pre-defined function for noise PDF
            iid: bool type indicate if the noise is i.i.d.
            norm: \ell_p norm for radius computation, e.g., 1, 2, 3, ..., and -1 for inf norm
            dataset: dataset name
            MC_Num: Monte Carlo sampling number
            batch_size: batch size
            pdf_args: the hyper-parameters for pdf function, set first parameter as -1 as default.
            input_size: input dimension
        """
        self.pdf=pdf_function
        self.dataset=dataset
        self.iid=iid
        self.args=pdf_args
        self.MC_Num=MC_Num
        self.batch_size=batch_size
        if self.dataset=='cifar10':
            self.input_size=3*32*32
            self.num_class=10
        #######添加数据集
        if self.dataset=='cifar101':
            self.input_size=3*32*32
            self.num_class=10
        if self.dataset=='svhn':
            self.input_size = 32*32
            self.num_class = 10
        if self.dataset=='cifars':
            self.input_size = 32*32
            self.num_class = 10
        ########添加结束
        if self.dataset=='imagenet':
            self.input_size=3*224*224
            self.num_class=1000
        if self.dataset=='mnist':
            self.input_size = 28*28
            self.num_class = 10
        if input_size:
            self.input_size=input_size
        self.norm=norm
        if not self.iid:
            if self.args[0] == -1:
                self.pdf_norm = np.inf
            else:
                self.pdf_norm = self.args[0]

    def initial_args(self, x,y,args,opt_args_low,opt_args_high,opt_args_step,model):
        scaler_arg=args[1]
        step=opt_args_step[1]
        if scaler_arg < opt_args_low[1] or scaler_arg>opt_args_high[1]:
            return False, scaler_arg, False, False
        PA, PB = self.compute_PA_PB(x, y,args,model)

        if PA == 'abstain':
            scaler_arg = scaler_arg - step
            args[1]=scaler_arg
            R, scaler_arg, PA, PB = self.initial_args(x,y,args,opt_args_low,opt_args_high,opt_args_step,model)
            return R, scaler_arg, PA, PB
        elif PA <= PB:
            return False, False, False, False
        else:
            R = self.direction_optimization(PA, PB,args)
            return R, scaler_arg, PA, PB

    def initial_args_binary(self, x,y,args,opt_args_low,opt_args_high,opt_args_step,model):
        scaler_arg=args[1]
        step=opt_args_step[1]
        if scaler_arg < opt_args_low[1] or scaler_arg>opt_args_high[1]:
            return False, scaler_arg, False
        PA, CA= self.compute_PA_PB_binary(x,args,model)
        if PA == 'abstain':
            scaler_arg = scaler_arg - step
            args[1]=scaler_arg
            R, scaler_arg, PA,CA = self.initial_args_binary(x,y,args,opt_args_low,opt_args_high,opt_args_step,model)
            return R, scaler_arg, PA, CA
        else:
            R = self.direction_optimization_binary(PA,args)
            return R, scaler_arg, PA, CA

    def compute_R(self,x,y,args,model):
        PA, PB = self.compute_PA_PB(x, y,args,model)
        if PA == 'abstain' or PA < PB:
            R = -1
        else:
            R = self.direction_optimization(PA,PB,args)
        return R

    def compute_R_binary(self,x,args,model):
        PA,CA= self.compute_PA_PB_binary(x,args,model)
        if PA == 'abstain':
            R = -1
        else:
            R = self.direction_optimization_binary(PA,args)
        return R, CA

    def noise_optimization(self,x,y,opt_args_low,opt_args_high,opt_args_step,model,max_iter=20):
        args=self.args.copy()
        best_Rs = []
        best_args=[]
        R, scaler_arg, PA, PB = self.initial_args(x,y,args,opt_args_low,opt_args_high,opt_args_step,model)
        args[1]=scaler_arg
        R_old=R
        print('scaler_arg: {}'.format(scaler_arg))
        if not R:
            return False, False, False
        best_Rs.append(R)
        best_args.append(args)
        print(R)

        PA,PB= self.compute_PA_PB(x, y,args,model)
        if PA=='abstain':
            radius=-1
        else:
            radius = Gaussian_R(PA,PB,self.args.copy())
        print('radius by RM: {}'.format(radius))
        RM_Rs=radius

        iter=max_iter
        init_time = time.time()
        while iter>=0:
            iter -= 1
            for i in range(len(opt_args_step)):
                if iter<0:
                    break
                if opt_args_step[i]>0:
                    args[i]=args[i]+opt_args_step[i]
                    print(args)
                    if args[i]>opt_args_high[i]:
                        iter=-1
                    else:
                        R_next=self.compute_R(x,y,args,model)
                        if R_next>R:
                            R=R_next
                        else:
                            args[i]=args[i]-2*opt_args_step[i]
                            print(args)
                            if args[i]<opt_args_low[i]:
                                iter=-1
                            else:
                                R_next = self.compute_R(x, y, args,model)
                                if R_next>R:
                                    R=R_next
                                else:
                                    args[i]=args[i]+opt_args_step[i]
            print('iter:{}, best R:{}, best args:{}, time_used:{} min'.format(max_iter-iter, R, args, (time.time() - init_time) / 60))
            if R==R_old:
                iter=-1
            else:
                R_old=R
            best_Rs.append(R)
            best_args.append(args)
        return best_Rs,best_args,RM_Rs

    def noise_optimization_binary(self,x,y,opt_args_low,opt_args_high,opt_args_step,model,max_iter=20):
        args=self.args.copy()
        best_Rs = []
        best_args=[]
        #对弃权样本本计数
        abstain_num = 0

        PA,CA= self.compute_PA_PB_binary(x,self.args.copy(),model)
        if PA=='abstain':
            while PA=='abstain' and args[1] > opt_args_low[1] and args[1] < opt_args_high[1]:
                args[1]=args[1]-opt_args_step[1]
                PA,CA=self.compute_PA_PB_binary(x,args,model)
        if PA=='abstain':
            print('abstain')
            #对弃权样本本计数
            abstain_num = 1
            #return False, False, False,False,False
            return False, False, False,False,False, abstain_num
        else:
            R = self.direction_optimization_binary(PA,args) 
            if R<0 or R>10: # rule out extreme values
                return False, False, False,False,False, abstain_num


        R_old=R
        CA_HAT=CA
        print('scaler_arg: {}'.format(args[1]))
        print('PA={}'.format(PA))
        if not R:
            print('not valid R')
            return False, False, False,False,False
        best_Rs.append(R)
        best_args.append(args)
        print(R)

        PA,CA_other= self.compute_PA_PB_binary(x,self.args.copy(),model)
        if PA=='abstain':
            radius=-1
        else:
            # radius = Gaussian_R_infnorm(PA,3072,self.args.copy()[1]/np.sqrt(2))
            radius = Gaussian_R_binary(PA, self.args.copy()[1]/np.sqrt(2))
            # radius = Laplace_R(PA, self.args.copy()[1]*np.sqrt(2))
            # radius=1
        print('radius by RM: {}'.format(radius))
        RM_Rs=radius
        # RM_Rs=1


        iter=max_iter
        init_time = time.time()
        while iter>=0:
            iter -= 1
            for i in range(len(opt_args_step)):
                if iter<0:
                    break
                if opt_args_step[i]>0:
                    args[i]=args[i]+opt_args_step[i]
                    if args[i]>opt_args_high[i]:
                        iter=-1
                    else:
                        R_next,CA=self.compute_R_binary(x,args,model)
                        if R_next>R:
                            R=R_next
                            CA_HAT = CA
                        else:
                            args[i]=args[i]-2*opt_args_step[i]
                            if args[i]<opt_args_low[i]:
                                iter=-1
                            else:
                                R_next,CA = self.compute_R_binary(x, args,model)
                                if R_next>R:
                                    R=R_next
                                    CA_HAT = CA
                                else:
                                    args[i]=args[i]+opt_args_step[i]
            print('iter:{}, best R:{}, best args:{}, time_used:{} min'.format(max_iter-iter, R, args, (time.time() - init_time) / 60))
            if R==R_old:
                iter=-1
            else:
                R_old=R
            best_Rs.append(R)
            best_args.append(args)
        return best_Rs,best_args,RM_Rs,CA_HAT,CA_other,abstain_num

    def direction_optimization(self,PA,PB,args,n_particals=10,iters=3,n_processes=2):
        epsilons = self.noise_sampling(self.MC_Num,args)
        dimension = self.input_size
        bound = (-np.ones(dimension), np.ones(dimension))

        if self.norm==1:
            init_pos = np.zeros((n_particals, dimension))
            for i in range(n_particals):
                index=int(np.random.rand(1)*dimension)
                init_pos[i,index]=1
        elif self.norm==-1:
            init_pos = np.ones((n_particals,dimension))
        else:
            init_pos=None
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=n_particals, dimensions=dimension, bounds=bound,init_pos=init_pos,
                                            options=options)
        R, best_delta = optimizer.optimize(self.parel_scale_optimization, iters=iters, n_processes=n_processes,
                                                 verbose=False, PA=PA, PB=PB,epsilons=epsilons,args=args)
        return R

    def direction_optimization_binary(self,PA,args,n_particals=4,iters=3,n_processes=2):
        epsilons = self.noise_sampling(self.MC_Num,args)
        dimension = self.input_size
        bound = (-np.ones(dimension), np.ones(dimension))

        if self.norm>0:
            init_pos = np.zeros((n_particals, dimension))
            for i in range(n_particals):
                index=int(np.random.rand(1)*dimension)
                init_pos[i,index]=1
        elif self.norm==-1:
            init_pos = np.ones((n_particals,dimension))
        else:
            init_pos=None
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=n_particals, dimensions=dimension, bounds=bound,init_pos=init_pos,
                                            options=options)
        R, best_delta = optimizer.optimize(self.parel_scale_optimization_binary, iters=iters, n_processes=n_processes,
                                                 verbose=False, PA=PA, epsilons=epsilons,args=args)
        if R==1000:
            R=-1
        return R

    def parel_scale_optimization(self,deltas, PA, PB,epsilons,args):
        Rs = np.zeros(deltas.shape[0])
        for n in range(deltas.shape[0]):
            R = self.scale_optimization(deltas[n], PA, PB,epsilons,args)
            Rs[n] = R
        return Rs

    def parel_scale_optimization_binary(self,deltas, PA,epsilons,args):
        Rs = np.zeros(deltas.shape[0])
        for n in range(deltas.shape[0]):
            R = self.scale_optimization_binary(deltas[n], PA,epsilons,args)
            Rs[n] = R
        return Rs

    def scale_optimization(self,delta,PA,PB,epsilons,args):
        error=0.01
        max_iteration=20
        Lambda_init = torch.mean(torch.abs(epsilons))/2
        Lambda = Lambda_init
        delta_scaler = delta * (Lambda_init)
        diff_initial = self.compute_K(PA, PB, epsilons, delta_scaler,args)
        n = max_iteration
        if diff_initial >= 0:
            diff = diff_initial
            while diff >= 0 and n > 0:
                Lambda = Lambda * 2
                delta_scaler = delta * (Lambda)
                diff = self.compute_K(PA, PB, epsilons,delta_scaler,args)
                # print(K,diff)
                n -= 1
        else:
            diff = diff_initial
            while diff < 0 and n > 0:
                Lambda = Lambda * 1 / 2
                delta_scaler = delta * (Lambda)
                diff = self.compute_K(PA, PB, epsilons, delta_scaler,args)
                n -= 1
        if n == 0:
            return 1000

        if Lambda > Lambda_init:
            Lambdaa = Lambda_init
            Lambdab = Lambda
        else:
            Lambdaa = Lambda
            Lambdab = Lambda_init

        n = max_iteration
        while np.abs(diff) > error and n > 0:
            Lambda = (Lambdaa + Lambdab) / 2
            delta_scaler = delta * (Lambda)
            diff = self.compute_K(PA, PB, epsilons,delta_scaler,args)
            if diff >= 0:
                Lambdaa = Lambda
            else:
                Lambdab = Lambda
            n -= 1
        if n == 0:
            return 1000
        else:
            if self.norm == -1:
                R = torch.linalg.norm(delta_scaler,ord=np.inf)
            else:
                R = torch.linalg.norm(delta_scaler,ord=self.norm)
            return R.cpu()

    def scale_optimization_binary(self,delta,PA,epsilons,args):
        delta=torch.from_numpy(delta).cuda()
        error=0.01
        max_iteration=20
        Lambda_init = torch.mean(torch.abs(epsilons))/2
        Lambda = Lambda_init
        delta_scaler = delta * (Lambda_init)
        diff_initial = self.compute_K_binary(PA, epsilons, delta_scaler,args)
        n = max_iteration
        if diff_initial >= 0:
            diff = diff_initial
            while diff >= 0 and n > 0:
                Lambda = Lambda * 2
                delta_scaler = delta * (Lambda)
                diff = self.compute_K_binary(PA,epsilons,delta_scaler,args)
                n -= 1
        else:
            diff = diff_initial
            while diff < 0 and n > 0:
                Lambda = Lambda * 1 / 2
                delta_scaler = delta * (Lambda)
                diff = self.compute_K_binary(PA,epsilons, delta_scaler,args)
                n -= 1
        if n == 0:
            return 1000

        if Lambda > Lambda_init:
            Lambdaa = Lambda_init
            Lambdab = Lambda
        else:
            Lambdaa = Lambda
            Lambdab = Lambda_init

        n = max_iteration
        while np.abs(diff) > error and n > 0:
            Lambda = (Lambdaa + Lambdab) / 2
            delta_scaler = delta * (Lambda)
            diff = self.compute_K_binary(PA,epsilons,delta_scaler,args)
            if diff >= 0:
                Lambdaa = Lambda
            else:
                Lambdab = Lambda
            n -= 1
        if n == 0:
            return 1000
        else:
            if self.norm == -1:
                R = torch.linalg.norm(delta_scaler,ord=np.inf)
                # print(Lambda, R, np.max(delta_scaler))
            else:
                R = torch.linalg.norm(delta_scaler,ord=self.norm)
            return R.cpu()

    def compute_PA_PB(self,x,y,args,model):
        """using Monte Carlo method to compute the upper bound and lower bound of
        the probability
        x: the input image
        y: the label"""
        model.eval()
        N = self.MC_Num
        num_classes = self.num_class
        alpha = 0.001
        with torch.no_grad():

            X = x.cuda()
            # label = y.cuda()
            epsilons = self.noise_sampling(N,args)
            predictions_g = torch.tensor([]).cuda().long()
            num = N
            for i in range(np.ceil(num / self.batch_size).astype(int)):
                this_batch_size = min(self.batch_size, num)
                num -= this_batch_size

                batch = X.repeat((this_batch_size, 1, 1, 1))
                noise = torch.from_numpy(epsilons[i * self.batch_size:i * self.batch_size + this_batch_size]).float()
                predictions = model(batch + noise.cuda().reshape(batch.shape)).argmax(1)
                predictions_g = torch.cat([predictions_g, predictions], 0)
            pred = predictions_g.cpu().numpy()
            counts_g = np.zeros(num_classes)
            for i in range(num_classes):
                counts_g[i] = (pred == i).sum()

            if counts_g.argmax(0) == y:
                NA = np.sort(counts_g)[-2:][1].astype(int)
                NB = np.sort(counts_g)[-2:][0].astype(int)
                pABar = proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
                pBBar = proportion_confint(NB, N, alpha=2 * alpha, method="beta")[1]
                return pABar, pBBar
            else:
                return 'abstain', 'abstain'

    def compute_PA_PB_binary(self,x,args,model):
        """using Monte Carlo method to compute the upper bound and lower bound of
        the probability
        x: the input image
        y: the label"""

        model.eval()
        N = self.MC_Num
        N0=100
        num_classes = self.num_class
        alpha = 0.001
        with torch.no_grad():

            X = x.cuda()
            epsilons = self.noise_sampling(N,args)

            predictions_g = torch.tensor([]).cuda().long()
            num = N
            for i in range(np.ceil(num / self.batch_size).astype(int)):
                this_batch_size = min(self.batch_size, num)
                num -= this_batch_size
                batch = X.repeat((this_batch_size, 1, 1, 1))
                noise = epsilons[i * self.batch_size:i * self.batch_size + this_batch_size].float()
                predictions = model(batch + noise.cuda().reshape(batch.shape)).argmax(1)
                predictions_g = torch.cat([predictions_g, predictions], 0)
            pred = predictions_g.cpu().numpy()
            counts_g = np.zeros(num_classes)
            for i in range(num_classes):
                counts_g[i] = (pred == i).sum()


            batch_select = X.repeat((N0, 1, 1, 1))
            epsilons_select = self.noise_sampling(N0,self.args)
            noise_select = epsilons_select.float()
            predictions = model(batch_select + noise_select.cuda().reshape(batch_select.shape)).argmax(1)
            pred_select = predictions.cpu().numpy()
            counts_select = np.zeros(num_classes)
            for i in range(num_classes):
                counts_select[i] = (pred_select == i).sum()

            if counts_g.argmax(0) == counts_select.argmax(0):
                NA = np.sort(counts_g)[-1].astype(int)
                pABar = proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
                if pABar>0.5:
                    return pABar,counts_g.argmax(0)
                else:
                    return 'abstain','abstain'
            else:
                return 'abstain','abstain'

    def compute_PA_PB_binary_standard(self,x,args,model):
        """using Monte Carlo method to compute the upper bound and lower bound of
        the probability
        x: the input image
        y: the label"""

        model.eval()
        N = self.MC_Num
        num_classes = self.num_class
        alpha = 0.001
        with torch.no_grad():

            X = x.cuda()
            epsilons = self.noise_sampling(N,args)

            predictions_g = torch.tensor([]).cuda().long()
            num = N
            for i in range(np.ceil(num / self.batch_size).astype(int)):
                this_batch_size = min(self.batch_size, num)
                num -= this_batch_size
                batch = X.repeat((this_batch_size, 1, 1, 1))
                noise = epsilons[i * self.batch_size:i * self.batch_size + this_batch_size].float()
                # print(batch.shape, noise.shape)
                predictions = model(batch + noise.cuda().reshape(batch.shape)).argmax(1)
                predictions_g = torch.cat([predictions_g, predictions], 0)
            pred = predictions_g.cpu().numpy()
            # print(pred)
            counts_g = np.zeros(num_classes)
            for i in range(num_classes):
                counts_g[i] = (pred == i).sum()

            predictions_f = model(X.unsqueeze(0)).argmax(1)

            if counts_g.argmax(0) == predictions_f[0]:
                NA = np.sort(counts_g)[-1].astype(int)
                #     # print(NA,N)
                pABar = proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
                if pABar>0.5:
                    return pABar
                else:
                    return 'abstain'
            else:
                return 'abstain'


    def compute_ta_tb(self,PA,PB,epsilons,delta,args):
        epsilons_delta = epsilons - delta
        if self.iid:
            gammas=torch.prod(self.pdf(epsilons_delta,args) / self.pdf(epsilons, args),dim=1)
        sorted_gammas = torch.sort(gammas,dim=0)[0]
        ta = sorted_gammas[np.ceil(len(gammas) * PA).astype(int)]
        tb = sorted_gammas[np.floor(len(gammas) * (1 - PB)).astype(int)]
        return ta, tb

    def compute_ta_tb_binary(self,PA,epsilons,delta,args):
        epsilons_delta = epsilons - delta
        if self.iid:
            gammas=torch.prod(self.pdf(epsilons_delta,args) / self.pdf(epsilons, args),dim=1)

        sorted_gammas = torch.sort(gammas,dim=0)[0]
        ta = sorted_gammas[np.ceil(len(gammas) * PA).astype(int)]
        return ta

    def compute_K(self,PA, PB, epsilons, delta,args):
        """"""
        ta, tb = self.compute_ta_tb(PA, PB, epsilons, delta,args)
        epsilons_delta = epsilons + delta
        if self.iid:
            gammas=torch.prod(self.pdf(epsilons, args) / self.pdf(epsilons_delta, args),dim=1)

        P_ta = (gammas <= ta).sum() / len(gammas)
        P_tb = (gammas >= tb).sum() / len(gammas)
        K=P_ta-P_tb
        return K.cpu()

    def compute_K_binary(self,PA,epsilons, delta,args):
        """"""
        ta= self.compute_ta_tb_binary(PA,epsilons, delta,args)
        epsilons_delta = epsilons + delta
        if self.iid:
            gammas=torch.prod(self.pdf(epsilons, args) / self.pdf(epsilons_delta, args),dim=1)

        P_ta = (gammas <= ta).sum() / len(gammas)
        K=P_ta-1/2
        return K.cpu()

    def noise_sampling(self,Sample_Num,args):
        """samping Sample_Num*input_size noises from the discrete pdf"""
        pdf, discrete_scaler = self.discrete_pdf_function(args)
        s_pdf = []
        s = 0
        for p in pdf:
            s = s + p
            s_pdf.append(s)

        epsilons = torch.rand(Sample_Num, self.input_size,device='cuda')
        left_bound = 0
        for i, v in enumerate(s_pdf):
            epsilons[torch.logical_and(epsilons < v, epsilons >= left_bound)] = i
            left_bound = v
        return (epsilons - pdf.shape[0] / 2) * discrete_scaler


    def discrete_pdf_function(self,args,discrete_range_factor=5):
        """prepare the discrete version of pdf function"""
        #estimate the sigma
        s=torch.linspace(-5,5,500)
        t=self.pdf(s,args)
        t=t/torch.sum(t*10/500)
        sigma=torch.sqrt(torch.sum(s**2*t*10/500))
        #prepare the discrete pdf
        s_=torch.linspace(-sigma*discrete_range_factor,sigma*discrete_range_factor,1000)
        discrete_pdf=self.pdf(s_,args)
        discrete_pdf=discrete_pdf/torch.sum(discrete_pdf*1)

        discrete_scaler=1/1000*2* sigma * discrete_range_factor

        return discrete_pdf.data, discrete_scaler
