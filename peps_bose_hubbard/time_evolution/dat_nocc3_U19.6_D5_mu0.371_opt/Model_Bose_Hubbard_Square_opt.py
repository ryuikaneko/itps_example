# coding:utf-8
import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import time
import argparse

## import basic routines
from PEPS_Basics import *
from PEPS_Parameters import *
from Square_lattice_CTM import *

def parse_args():
    parser = argparse.ArgumentParser(description='iPEPS simulation')
    parser.add_argument('-s',metavar='seed',dest='seed',type=int,default=11,help='set random seed')
    parser.add_argument('-part',metavar='part',dest='part',type=float,default=0.05,help='set part')
    parser.add_argument('-parU',metavar='parU',dest='parU',type=float,default=1.0,help='set parU')
    parser.add_argument('-parmu',metavar='parmu',dest='parmu',type=float,default=0.3,help='set parmu')
    parser.add_argument('-ts',metavar='tau',dest='tau',type=float,default=0.01,help='set tau for simple update')
    parser.add_argument('-tf',metavar='tau_full',dest='tau_full',type=float,default=0.01,help='set tau for full update')
    parser.add_argument('-ss',metavar='tau_step',dest='tau_step',type=int,default=100,help='set step for simple update')
    parser.add_argument('-sf',metavar='tau_full_step',dest='tau_full_step',type=int,default=0,help='set step for full update')
    parser.add_argument('-i',metavar='initial_type',dest='initial_type',type=int,default=0,help='set initial state type \
                        (0:random, 1:FM, 2:AF, -1 or negative value:from file (default input_Tn.npz))')
    parser.add_argument('--second_ST',action='store_const',const=True,default=False,help='use second order Suzuki Trotter decomposition for ITE')
    parser.add_argument('-o',metavar='output_file',dest='output_file',default="output",help='set output filename prefix for optimized tensor')
    parser.add_argument('-in',metavar='input_file',dest='input_file',default="input_Tn.npz",help='set input filename for initial tensors')
    return parser.parse_args()

def Set_Hamiltonian(parz,parnocc,part,parU,parmu):
    z = parz
    nocc = parnocc
    t = part
    U = parU * 1.0/z
    mu = parmu * 1.0/z
#
    Ham = np.zeros((nocc**2,nocc**2),dtype=np.float64)
    for n2 in range(nocc):
        for n1 in range(nocc):
# diag
            sa = n2*nocc + n1
            Ham[sa,sa] += 0.5*U*(n1*(n1-1) + n2*(n2-1)) - mu*(n1+n2)
# offdiag   
            n1p = n1+1
            n2m = n2-1
            if n1p<nocc and n2m>=0:
                sb = n2m*nocc + n1p
                Ham[sa,sb] += -t*np.sqrt(n1p)*np.sqrt(n2)
            n2p = n2+1
            n1m = n1-1
            if n2p<nocc and n1m>=0:
                sb = n2p*nocc + n1m
                Ham[sa,sb] += -t*np.sqrt(n2p)*np.sqrt(n1)
#    print(Ham)
    return Ham

def Initialize_Tensors(Tn,seed,initial_type=0):
    np.random.seed(seed)
    ## Random tensors back_ground
#    back_ground_amp = 0.001
    back_ground_amp = 0.0
    Tn_temp = back_ground_amp * (np.random.rand(D,D,D,D,DPHYS)-0.5)

    for i in range(0,N_UNIT):
        Tn[i][:] = Tn_temp.copy()

    if initial_type == 1:
        ## MI
        for i in range(0,N_UNIT):
            Tn[i][0,0,0,0,1]=1.0
    elif initial_type == 2:
        ## CDW, nocc >= 3
        for i in range(0,N_UNIT):
            ix = i%LX_ori
            iy = i/LX_ori
            if (ix + iy)%2 == 0:
                Tn[i][0,0,0,0,0]=1.0
            else:
                Tn[i][0,0,0,0,2]=1.0
    elif initial_type == 3:
        ## x FM when nocc = 2
        for i in range(0,N_UNIT):
            for j in range(DPHYS):
                Tn[i][0,0,0,0,j]=1.0

def main():
    ## timers
    time_simple_update=0.0
    time_full_update=0.0
    time_env=0.0
    time_obs=0.0

    ## Parameters
    args = parse_args()

    seed = args.seed
    part = args.part
    parU = args.parU
    parmu = args.parmu
    parz = 4
    parnocc = DPHYS
#
    tau = args.tau
    tau_step = args.tau_step
    tau_full = args.tau_full
    tau_full_step = args.tau_full_step
    second_ST = args.second_ST
    initial_type = args.initial_type
    output_prefix=args.output_file
    input_file = args.input_file

    print "## Logs: seed =",seed
    print "## Logs: t =",part
    print "## Logs: U =",parU
    print "## Logs: mu =",parmu
    print "## Logs: z =",parz
    print "## Logs: nocc =",parnocc
    print "## Logs: tau =",tau
    print "## Logs: tau_step =",tau_step
    print "## Logs: tau_full =",tau_full
    print "## Logs: tau_full_step =",tau_full_step
    print "## Logs: second_ST =",second_ST
    print "## Logs: initial_type =",initial_type
    if initial_type < 0:
        print "## Logs: input_file =",input_file
    print "## Logs: output_file =",output_prefix

    ## Tensors
    Tn=[np.zeros((D,D,D,D,DPHYS),dtype=TENSOR_DTYPE)]
    eTt=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]
    eTr=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]
    eTb=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]
    eTl=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]

    C1=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)
    C2=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)
    C3=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)
    C4=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)

    for i in range(1,N_UNIT):
        Tn.append(np.zeros((D,D,D,D,DPHYS),dtype=TENSOR_DTYPE))
        eTt.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
        eTr.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
        eTb.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
        eTl.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))

    ## Initialize tensor every step
    if initial_type >= 0:
        Initialize_Tensors(Tn,seed,initial_type)
    else:
        #input from file
        input_tensors = np.load(input_file)
        for i in range(N_UNIT):
            key='arr_'+repr(i)
            Tn_in = input_tensors[key]
            Tn[i][0:min(Tn[i].shape[0],Tn_in.shape[0]),0:min(Tn[i].shape[1],Tn_in.shape[1]),0:min(Tn[i].shape[2],Tn_in.shape[2]),0:min(Tn[i].shape[3],Tn_in.shape[3]),0:min(Tn[i].shape[4],Tn_in.shape[4])] \
                = Tn_in[0:min(Tn[i].shape[0],Tn_in.shape[0]),0:min(Tn[i].shape[1],Tn_in.shape[1]),0:min(Tn[i].shape[2],Tn_in.shape[2]),0:min(Tn[i].shape[3],Tn_in.shape[3]),0:min(Tn[i].shape[4],Tn_in.shape[4])]

    lambda_tensor = np.ones((N_UNIT,4,D),dtype=float) ## (N_UNIT,4,D) --> "4" corresponds to left, top, right, bottom
    Ham = Set_Hamiltonian(parz,parnocc,part,parU,parmu)
    s,U = linalg.eigh(Ham)

#    op12 =  np.dot(np.dot(U,np.diag(np.exp(-tau * s))),U.conj().T).reshape(DPHYS,DPHYS,DPHYS,DPHYS).transpose(2,3,0,1)
#    op12_2 =  np.dot(np.dot(U,np.diag(np.exp(-tau*0.5 * s))),U.conj().T).reshape(DPHYS,DPHYS,DPHYS,DPHYS).transpose(2,3,0,1)
    op12 =  np.dot(np.dot(U,np.diag(np.exp(-1j*tau * s))),U.conj().T).reshape(DPHYS,DPHYS,DPHYS,DPHYS).transpose(2,3,0,1)
    op12_2 =  np.dot(np.dot(U,np.diag(np.exp(-1j*tau*0.5 * s))),U.conj().T).reshape(DPHYS,DPHYS,DPHYS,DPHYS).transpose(2,3,0,1)
    start_simple=time.time()
    for int_tau in range(0,tau_step):

        if second_ST:
            ## simple update

            ## x-bond A sub-lattice
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_2,2)
                lambda_tensor[num,2] = lambda_c
                lambda_tensor[num_j,0] = lambda_c

            ## x-bond B sub-lattice
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_2,2)
                lambda_tensor[num,2] = lambda_c
                lambda_tensor[num_j,0] = lambda_c

            ## y-bond A sub-lattice
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_2,1)
                lambda_tensor[num,1] = lambda_c
                lambda_tensor[num_j,3] = lambda_c

            ## y-bond B sub-lattice
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12,1)
                lambda_tensor[num,1] = lambda_c
                lambda_tensor[num_j,3] = lambda_c

            ## y-bond A sub-lattice
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_2,1)
                lambda_tensor[num,1] = lambda_c
                lambda_tensor[num_j,3] = lambda_c

            ## x-bond B sub-lattice
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_2,2)
                lambda_tensor[num,2] = lambda_c
                lambda_tensor[num_j,0] = lambda_c

            ## x-bond A sub-lattice
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_2,2)
                lambda_tensor[num,2] = lambda_c
                lambda_tensor[num_j,0] = lambda_c

        else:
            ## simple update

            ## x-bond A sub-lattice
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12,2)
                lambda_tensor[num,2] = lambda_c
                lambda_tensor[num_j,0] = lambda_c

            ## x-bond B sub-lattice
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12,2)
                lambda_tensor[num,2] = lambda_c
                lambda_tensor[num_j,0] = lambda_c

            ## y-bond A sub-lattice
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12,1)
                lambda_tensor[num,1] = lambda_c
                lambda_tensor[num_j,3] = lambda_c

            ## y-bond B sub-lattice
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12,1)
                lambda_tensor[num,1] = lambda_c
                lambda_tensor[num_j,3] = lambda_c

        ## done simple update
    time_simple_update += time.time() - start_simple

    ## Start full update

    ## !!! no second_ST yet !!!

    if tau_full_step > 0:
        Ham = Set_Hamiltonian(parz,parnocc,part,parU,parmu)
        s,U = linalg.eigh(Ham)

        op12 =  np.dot(np.dot(U,np.diag(np.exp(-tau_full * s))),U.conj().T).reshape(DPHYS,DPHYS,DPHYS,DPHYS).transpose(2,3,0,1)

        ## Environment 
        start_env = time.time()
        Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)
        time_env += time.time() - start_env

    start_full = time.time()
    for int_tau in range(0,tau_full_step):

        ## x-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12,2)

            if Full_Use_FFU:
                ix = num%LX
                ix_j = num_j%LX
                Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                

            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

        ## x-bond B sub-lattice
        for i in range(0,N_UNIT/2):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12,2)

            if Full_Use_FFU:
                ix = num%LX
                ix_j = num_j%LX
                Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                
            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

        ## y-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12,1)

            if Full_Use_FFU:
                iy = num/LX
                iy_j = num_j/LX
                Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

        ## y-bond B sub-lattice
        for i in range(0,N_UNIT/2):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12,1)

            if Full_Use_FFU:
                iy = num/LX
                iy_j = num_j/LX
                Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

    ## done full update
    time_full_update += time.time()-start_full

    ## output optimized tensor
    output_file = output_prefix+'_Tn.npz'
    np.savez(output_file, *Tn)

#    ## Calc physical quantities
#    start_env = time.time()
#    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)
#    time_env += time.time() - start_env
#
#
#
##    for i in range(0,N_UNIT):
##        print "## i eTl[i] ",i,eTl[i]
##        print "## i eTt[i] ",i,eTt[i]
##        print "## i eTr[i] ",i,eTr[i]
##        print "## i eTb[i] ",i,eTb[i]
#
###    |    |
### --[2]  [3]--
###    |    |
###    |    |
###   [0]--[1]
###    |    |
###
###   2,3
###    |
### 1-[0]-0
###   
###   2,3
###    |
### 1-[1]-0
###   
### 0-[2]-1
###    |
###   2,3
###   
### 0-[3]-1
###    |
###   2,3
###   
#
#    T0b = eTb[0]
#    T1b = eTb[1]
##    T2t = eTt[2]
##    T3t = eTt[3]
#    T2t = eTt[1]
#    T3t = eTt[0]
#
#    T02 = np.tensordot(T0b,T2t,axes=((2,3),(2,3)))
#    T13 = np.tensordot(T1b,T3t,axes=((2,3),(2,3)))
##    T0123h = np.tensordot(T02,T13,axes=((0,3),(1,2))).transpose(0,3,1,2).reshape(CHI*CHI,CHI*CHI)
#    T0123h = np.tensordot(T02,T13,axes=((0,3),(1,2))).reshape(CHI*CHI,CHI*CHI)
#
#    print "## np.allclose(T0123h,T0123h.T,rtol=1e-6,atol=1e-6)", np.allclose(T0123h,T0123h.T,rtol=1e-6,atol=1e-6)
##    wT0123h = linalg.eigvalsh(T0123h)
#    wT0123h = linalg.eigvals(T0123h)
#    idx = np.abs(wT0123h).argsort()[::-1]
#    wT0123h = wT0123h[idx]
#    print "## eigenvalues of T0123h abs",' '.join([str(elem) for elem in np.abs(wT0123h)])
#    idx = np.abs(wT0123h.real).argsort()[::-1]
#    wT0123h = wT0123h[idx]
##    print "## eigenvalues of T0123h",' '.join([str(elem) for elem in wT0123h])
#    print "## eigenvalues of T0123h real",' '.join([str(elem) for elem in wT0123h.real])
#    print "## eigenvalues of T0123h imag",' '.join([str(elem) for elem in wT0123h.imag])
#    print "## eigenvalues of T0123h imag max",np.max(np.abs(wT0123h.imag))
#
#    T0l = eTl[0]
#    T1r = eTr[1]
##    T2l = eTl[2]
##    T3r = eTr[3]
#    T2l = eTl[1]
#    T3r = eTr[0]
#
#    T23 = np.tensordot(T2l,T3r,axes=((2,3),(2,3)))
#    T01 = np.tensordot(T0l,T1r,axes=((2,3),(2,3)))
##    T0123v = np.tensordot(T23,T01,axes=((0,3),(1,2))).transpose(0,3,1,2).reshape(CHI*CHI,CHI*CHI)
#    T0123v = np.tensordot(T23,T01,axes=((0,3),(1,2))).reshape(CHI*CHI,CHI*CHI)
#
#    print "## np.allclose(T0123v,T0123v.T,rtol=1e-6,atol=1e-6)", np.allclose(T0123v,T0123v.T,rtol=1e-6,atol=1e-6)
##    wT0123v = linalg.eigvalsh(T0123v)
#    wT0123v = linalg.eigvals(T0123v)
#    idx = np.abs(wT0123v).argsort()[::-1]
#    wT0123v = wT0123v[idx]
#    print "## eigenvalues of T0123v abs",' '.join([str(elem) for elem in np.abs(wT0123v)])
#    idx = np.abs(wT0123v.real).argsort()[::-1]
#    wT0123v = wT0123v[idx]
##    print "## eigenvalues of T0123v",' '.join([str(elem) for elem in wT0123v])
#    print "## eigenvalues of T0123v real",' '.join([str(elem) for elem in wT0123v.real])
#    print "## eigenvalues of T0123v imag",' '.join([str(elem) for elem in wT0123v.imag])
#    print "## eigenvalues of T0123v imag max",np.max(np.abs(wT0123v.imag))
#
#
#
#    op_identity = np.identity(parnocc)
#    op_n = np.zeros((parnocc,parnocc))
#    op_n2 = np.zeros((parnocc,parnocc))
#    op_a = np.zeros((parnocc,parnocc))
#    op_adag = np.zeros((parnocc,parnocc))
#    for i in range(parnocc):
#        op_n[i,i] = i
#        op_n2[i,i] = i**2
#    for i in range(parnocc-1):
#        op_a[i,i+1] = np.sqrt(i+1)
#        op_adag[i+1,i] = np.sqrt(i+1)
##    op_a[0,1] = 1.0
##    op_a[1,2] = np.sqrt(2.0)
##    op_adag[1,0] = 1.0
##    op_adag[2,1] = np.sqrt(2.0)
##    print(op_a)
##    print(op_adag)
#    val_n = np.zeros(N_UNIT)
#    val_n2 = np.zeros(N_UNIT)
#    val_a = np.zeros(N_UNIT)
#    val_adag = np.zeros(N_UNIT)
#    val_aadag = np.zeros((N_UNIT,2))
#    val_adaga = np.zeros((N_UNIT,2))
##    val_aadag11 = np.zeros((N_UNIT,4))
##    val_adaga11 = np.zeros((N_UNIT,4))
#    val_aadag_x_2 = np.zeros(N_UNIT)
#    val_aadag_x_3 = np.zeros(N_UNIT)
#    val_aadag_y_2 = np.zeros(N_UNIT)
#    val_aadag_y_3 = np.zeros(N_UNIT)
#    val_adaga_x_2 = np.zeros(N_UNIT)
#    val_adaga_x_3 = np.zeros(N_UNIT)
#    val_adaga_y_2 = np.zeros(N_UNIT)
#    val_adaga_y_3 = np.zeros(N_UNIT)
#
#    start_obs = time.time()
#
#    for num in range(0,N_UNIT):
#        norm = Contract_one_site(C1[num],C2[num],C3[num],C4[num],eTt[num],eTr[num],eTb[num],eTl[num],Tn[num],op_identity)
#        val_n[num] = np.real(Contract_one_site(C1[num],C2[num],C3[num],C4[num],eTt[num],eTr[num],eTb[num],eTl[num],Tn[num],op_n)/norm)
#        val_n2[num] = np.real(Contract_one_site(C1[num],C2[num],C3[num],C4[num],eTt[num],eTr[num],eTb[num],eTl[num],Tn[num],op_n2)/norm)
#        val_a[num] = np.real(Contract_one_site(C1[num],C2[num],C3[num],C4[num],eTt[num],eTr[num],eTb[num],eTl[num],Tn[num],op_a)/norm)
#        val_adag[num] = np.real(Contract_one_site(C1[num],C2[num],C3[num],C4[num],eTt[num],eTr[num],eTb[num],eTl[num],Tn[num],op_adag)/norm)
#        print "## t,U,mu,z,nocc,num,norm,val_n[num],val_n2[num],val_a[num],val_adag[num]",part,parU,parmu,parz,parnocc,num,norm,val_n[num],val_n2[num],val_a[num],val_adag[num]
#
#    for num in range(0,N_UNIT):
#        ## x direction
#        num_j = NN_Tensor[num,2]
#        norm_x = Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_identity,op_identity)
#        val_aadag[num,0] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_a,op_adag)/norm_x)
#        val_adaga[num,0] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_adag,op_a)/norm_x)
#
#        ## y direction
#        num_j = NN_Tensor[num,3]
#        norm_y = Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_identity,op_identity)
#        val_aadag[num,1] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_a,op_adag)/norm_y)
#        val_adaga[num,1] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_adag,op_a)/norm_y)
#        print "## t,U,mu,z,nocc,num,norm_x,norm_y,val_aadag[num],val_adaga[num]",part,parU,parmu,parz,parnocc,num,norm_x,norm_y,val_aadag[num],val_adaga[num]
#
#    for i in range(0,N_UNIT):
#        ##
#        ## direction 0
#        ##
#        ## k--l
#        ## |  |
#        ## j--i
#        ##
#        ## n.n.n = i-->k 
#        ##
#        j = NN_Tensor[i,0]
#        k = NNN_Tensor[i,0]
#        l = NN_Tensor[i,1]
#        norm = Contract_four_sites(C1[k],C2[l],C3[i],C4[j],eTt[k],eTt[l],eTr[l],eTr[i],eTb[i],eTb[j],eTl[j],eTl[k],Tn[k],Tn[l],Tn[i],Tn[j],op_identity,op_identity,op_identity,op_identity)
#        val_aadag11[i,0] = np.real(Contract_four_sites(C1[k],C2[l],C3[i],C4[j],eTt[k],eTt[l],eTr[l],eTr[i],eTb[i],eTb[j],eTl[j],eTl[k],Tn[k],Tn[l],Tn[i],Tn[j],op_adag,op_identity,op_a,op_identity)/norm)
#        val_adaga11[i,0] = np.real(Contract_four_sites(C1[k],C2[l],C3[i],C4[j],eTt[k],eTt[l],eTr[l],eTr[i],eTb[i],eTb[j],eTl[j],eTl[k],Tn[k],Tn[l],Tn[i],Tn[j],op_a,op_identity,op_adag,op_identity)/norm)
#        ##
#        ## direction 1
#        ##
#        ## j--k
#        ## |  |
#        ## i--l
#        ##
#        ## n.n.n = i-->k
#        ##
#        j = NN_Tensor[i,1]
#        k = NNN_Tensor[i,1]
#        l = NN_Tensor[i,2]
#        norm = Contract_four_sites(C1[j],C2[k],C3[l],C4[i],eTt[j],eTt[k],eTr[k],eTr[l],eTb[l],eTb[i],eTl[i],eTl[j],Tn[j],Tn[k],Tn[l],Tn[i],op_identity,op_identity,op_identity,op_identity)
#        val_aadag11[i,1] = np.real(Contract_four_sites(C1[j],C2[k],C3[l],C4[i],eTt[j],eTt[k],eTr[k],eTr[l],eTb[l],eTb[i],eTl[i],eTl[j],Tn[j],Tn[k],Tn[l],Tn[i],op_identity,op_adag,op_identity,op_a)/norm)
#        val_adaga11[i,1] = np.real(Contract_four_sites(C1[j],C2[k],C3[l],C4[i],eTt[j],eTt[k],eTr[k],eTr[l],eTb[l],eTb[i],eTl[i],eTl[j],Tn[j],Tn[k],Tn[l],Tn[i],op_identity,op_a,op_identity,op_adag)/norm)
#        ##
#        ## direction 2
#        ##
#        ## i--j
#        ## |  |
#        ## l--k
#        ##
#        ## n.n.n = i-->k
#        ##
#        j = NN_Tensor[i,2]
#        k = NNN_Tensor[i,2]
#        l = NN_Tensor[i,3]
#        norm = Contract_four_sites(C1[i],C2[j],C3[k],C4[l],eTt[i],eTt[j],eTr[j],eTr[k],eTb[k],eTb[l],eTl[l],eTl[i],Tn[i],Tn[j],Tn[k],Tn[l],op_identity,op_identity,op_identity,op_identity)
#        val_aadag11[i,2] = np.real(Contract_four_sites(C1[i],C2[j],C3[k],C4[l],eTt[i],eTt[j],eTr[j],eTr[k],eTb[k],eTb[l],eTl[l],eTl[i],Tn[i],Tn[j],Tn[k],Tn[l],op_a,op_identity,op_adag,op_identity)/norm)
#        val_adaga11[i,2] = np.real(Contract_four_sites(C1[i],C2[j],C3[k],C4[l],eTt[i],eTt[j],eTr[j],eTr[k],eTb[k],eTb[l],eTl[l],eTl[i],Tn[i],Tn[j],Tn[k],Tn[l],op_adag,op_identity,op_a,op_identity)/norm)
#        ##
#        ## direction 3
#        ##
#        ## l--i
#        ## |  |
#        ## k--j
#        ##
#        ## n.n.n = i-->k
#        ##
#        j = NN_Tensor[i,3]
#        k = NNN_Tensor[i,3]
#        l = NN_Tensor[i,0]
#        norm = Contract_four_sites(C1[l],C2[i],C3[j],C4[k],eTt[l],eTt[i],eTr[i],eTr[j],eTb[j],eTb[k],eTl[k],eTl[l],Tn[l],Tn[i],Tn[j],Tn[k],op_identity,op_identity,op_identity,op_identity)
#        val_aadag11[i,3] = np.real(Contract_four_sites(C1[l],C2[i],C3[j],C4[k],eTt[l],eTt[i],eTr[i],eTr[j],eTb[j],eTb[k],eTl[k],eTl[l],Tn[l],Tn[i],Tn[j],Tn[k],op_identity,op_a,op_identity,op_adag)/norm)
#        val_adaga11[i,3] = np.real(Contract_four_sites(C1[l],C2[i],C3[j],C4[k],eTt[l],eTt[i],eTr[i],eTr[j],eTb[j],eTb[k],eTl[k],eTl[l],Tn[l],Tn[i],Tn[j],Tn[k],op_identity,op_adag,op_identity,op_a)/norm)
#    for num in range(0,N_UNIT):
#        for nnn in range(4):
#            print "## t,U,mu,z,nocc,num,nnn,norm,val_aadag11[num,nnn],val_adaga11[num,nnn]",part,parU,parmu,parz,parnocc,num,nnn,norm,val_aadag11[num,nnn],val_adaga11[num,nnn]
#
#    for i in range(0,N_UNIT):
#        #
#        # x=2
#        # i-j-k
#        #
#        j = FAR_X_Tensor[i,1]
#        k = FAR_X_Tensor[i,2]
#        norm = \
#        Contract_scalar_3x1(\
#         C1[i],eTt[i],eTt[j],eTt[k], C2[k],\
#        eTl[i], Tn[i], Tn[j], Tn[k],eTr[k],\
#         C4[i],eTb[i],eTb[j],eTb[k], C3[k],\
#        op_identity,op_identity,op_identity\
#        )
#        val_aadag_x_2 = np.real(\
#        Contract_scalar_3x1(\
#         C1[i],eTt[i],eTt[j],eTt[k], C2[k],\
#        eTl[i], Tn[i], Tn[j], Tn[k],eTr[k],\
#         C4[i],eTb[i],eTb[j],eTb[k], C3[k],\
#        op_a,op_identity,op_adag\
#        )\
#        /norm)
#        val_adaga_x_2 = np.real(\
#        Contract_scalar_3x1(\
#         C1[i],eTt[i],eTt[j],eTt[k], C2[k],\
#        eTl[i], Tn[i], Tn[j], Tn[k],eTr[k],\
#         C4[i],eTb[i],eTb[j],eTb[k], C3[k],\
#        op_adag,op_identity,op_a\
#        )\
#        /norm)
#        print "## t,U,mu,z,nocc,num,distx,disty,norm,val_aadag,val_adaga",\
#        part,parU,parmu,parz,parnocc,i,2,0,norm,\
#        val_aadag_x_2,val_adaga_x_2
#        #
#        # x=3
#        # i-j-k-l
#        #
#        j = FAR_X_Tensor[i,1]
#        k = FAR_X_Tensor[i,2]
#        l = FAR_X_Tensor[i,3]
#        norm = \
#        Contract_scalar_4x1(\
#         C1[i],eTt[i],eTt[j],eTt[k],eTt[l], C2[l],\
#        eTl[i], Tn[i], Tn[j], Tn[k], Tn[l],eTr[l],\
#         C4[i],eTb[i],eTb[j],eTb[k],eTb[l], C3[l],\
#        op_identity,op_identity,op_identity,op_identity\
#        )
#        val_aadag_x_3 = np.real(\
#        Contract_scalar_4x1(\
#         C1[i],eTt[i],eTt[j],eTt[k],eTt[l], C2[l],\
#        eTl[i], Tn[i], Tn[j], Tn[k], Tn[l],eTr[l],\
#         C4[i],eTb[i],eTb[j],eTb[k],eTb[l], C3[l],\
#        op_a,op_identity,op_identity,op_adag\
#        )\
#        /norm)
#        val_adaga_x_3 = np.real(\
#        Contract_scalar_4x1(\
#         C1[i],eTt[i],eTt[j],eTt[k],eTt[l], C2[l],\
#        eTl[i], Tn[i], Tn[j], Tn[k], Tn[l],eTr[l],\
#         C4[i],eTb[i],eTb[j],eTb[k],eTb[l], C3[l],\
#        op_adag,op_identity,op_identity,op_a\
#        )\
#        /norm)
#        print "## t,U,mu,z,nocc,num,distx,disty,norm,val_aadag,val_adaga",\
#        part,parU,parmu,parz,parnocc,i,3,0,norm,\
#        val_aadag_x_3,val_adaga_x_3
#        #
#        # y=2
#        # k
#        # |
#        # j
#        # |
#        # i
#        #
#        j = FAR_Y_Tensor[i,1]
#        k = FAR_Y_Tensor[i,2]
#        norm = \
#        Contract_scalar_1x3(\
#         C1[k],eTt[k], C2[k],\
#        eTl[k], Tn[k],eTr[k],\
#        eTl[j], Tn[j],eTr[j],\
#        eTl[i], Tn[i],eTr[i],\
#         C4[i],eTb[i], C3[i],\
#        op_identity,\
#        op_identity,\
#        op_identity\
#        )
#        val_aadag_y_2 = np.real(\
#        Contract_scalar_1x3(\
#         C1[k],eTt[k], C2[k],\
#        eTl[k], Tn[k],eTr[k],\
#        eTl[j], Tn[j],eTr[j],\
#        eTl[i], Tn[i],eTr[i],\
#         C4[i],eTb[i], C3[i],\
#        op_adag,\
#        op_identity,\
#        op_a\
#        )\
#        /norm)
#        val_adaga_y_2 = np.real(\
#        Contract_scalar_1x3(\
#         C1[k],eTt[k], C2[k],\
#        eTl[k], Tn[k],eTr[k],\
#        eTl[j], Tn[j],eTr[j],\
#        eTl[i], Tn[i],eTr[i],\
#         C4[i],eTb[i], C3[i],\
#        op_a,\
#        op_identity,\
#        op_adag\
#        )\
#        /norm)
#        print "## t,U,mu,z,nocc,num,distx,disty,norm,val_aadag,val_adaga",\
#        part,parU,parmu,parz,parnocc,i,0,2,norm,\
#        val_aadag_y_2,val_adaga_y_2
#        #
#        # y=3
#        # l
#        # |
#        # k
#        # |
#        # j
#        # |
#        # i
#        #
#        j = FAR_Y_Tensor[i,1]
#        k = FAR_Y_Tensor[i,2]
#        l = FAR_Y_Tensor[i,3]
#        norm = \
#        Contract_scalar_1x4(\
#         C1[l],eTt[l], C2[l],\
#        eTl[l], Tn[l],eTr[l],\
#        eTl[k], Tn[k],eTr[k],\
#        eTl[j], Tn[j],eTr[j],\
#        eTl[i], Tn[i],eTr[i],\
#         C4[i],eTb[i], C3[i],\
#        op_identity,\
#        op_identity,\
#        op_identity,\
#        op_identity\
#        )
#        val_aadag_y_3 = np.real(\
#        Contract_scalar_1x4(\
#         C1[l],eTt[l], C2[l],\
#        eTl[l], Tn[l],eTr[l],\
#        eTl[k], Tn[k],eTr[k],\
#        eTl[j], Tn[j],eTr[j],\
#        eTl[i], Tn[i],eTr[i],\
#         C4[i],eTb[i], C3[i],\
#        op_adag,\
#        op_identity,\
#        op_identity,\
#        op_a\
#        )\
#        /norm)
#        val_adaga_y_3 = np.real(\
#        Contract_scalar_1x4(\
#         C1[l],eTt[l], C2[l],\
#        eTl[l], Tn[l],eTr[l],\
#        eTl[k], Tn[k],eTr[k],\
#        eTl[j], Tn[j],eTr[j],\
#        eTl[i], Tn[i],eTr[i],\
#         C4[i],eTb[i], C3[i],\
#        op_a,\
#        op_identity,\
#        op_identity,\
#        op_adag\
#        )\
#        /norm)
#        print "## t,U,mu,z,nocc,num,distx,disty,norm,val_aadag,val_adaga",\
#        part,parU,parmu,parz,parnocc,i,0,3,norm,\
#        val_aadag_y_3,val_adaga_y_3
#
#
#    ene_t = - part * (np.sum(val_aadag) + np.sum(val_adaga))/N_UNIT
#    ene_U = 0.5*parU * (np.sum(val_n2) - np.sum(val_n))/N_UNIT
#    ene_mu = - parmu * np.sum(val_n)/N_UNIT
#    ene = ene_t + ene_U + ene_mu
#    print "## t,U,mu,z,nocc,ene,ene_t,ene_U,ene_mu",part,parU,parmu,parz,parnocc,ene,ene_t,ene_U,ene_mu
#
#    time_obs += time.time() - start_obs

    print "## time simple update=",time_simple_update
    print "## time full update=",time_full_update
#    print "## time environment=",time_env
#    print "## time observable=",time_obs

if __name__ == "__main__":
    main()
