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
    parser = argparse.ArgumentParser(description='iPEPS simulation for TFI model')
#    parser = argparse.ArgumentParser(description='iPEPS simulation for Heisenberg model')
    parser.add_argument('-s', metavar='seed',dest='seed', type=int, default=11,
                        help='set random seed')
#    parser.add_argument('-J', metavar='J',dest='J', type=float, default=1.0,
#                        help='set Heisenberg coupling J')
#    parser.add_argument('-Jxy', metavar='Jxy',dest='Jxy', type=float, default=1.0,
#                        help='set XY coupling Jxy')
    parser.add_argument('-Jxy', metavar='Jxy',dest='Jxy', type=float, default=0.0,
                        help='set XY coupling Jxy')
#    parser.add_argument('-Jz', metavar='Jz',dest='Jz', type=float, default=1.0,
#                        help='set Ising coupling Jz')
    parser.add_argument('-Jz', metavar='Jz',dest='Jz', type=float, default=-1.0,
                        help='set Ising coupling Jz')
#    parser.add_argument('-hx', metavar='hx',dest='hx', type=float, default=0.0,
#                        help='set magnetic field hx')
    parser.add_argument('-hx', metavar='hx',dest='hx', type=float, default=1.5,
                        help='set magnetic field hx')
    parser.add_argument('-ts', metavar='tau',dest='tau', type=float, default=0.01,
                        help='set tau for simple update')
    parser.add_argument('-tf', metavar='tau_full',dest='tau_full', type=float, default=0.01,
                        help='set tau for full update')
    parser.add_argument('-ss', metavar='tau_step',dest='tau_step', type=int, default=100,
                        help='set step for simple update')
    parser.add_argument('-sf', metavar='tau_full_step',dest='tau_full_step', type=int, default=0,
                        help='set step for full update')
    parser.add_argument('-i', metavar='initial_type',dest='initial_type', type=int, default=0,
                        help='set initial state type (0:random, 1:FM, 2:AF, \
                        -1 or negative value:from file (default input_Tn.npz))')
    parser.add_argument('--second_ST', action='store_const', const=True, default=False,
                        help='use second order Suzuki Trotter decomposition for ITE')
    parser.add_argument('-o', metavar='output_file',dest='output_file', default="output",
                        help='set output filename prefix for optimized tensor')
    parser.add_argument('-in', metavar='input_file',dest='input_file', default="input_Tn.npz",
                        help='set input filename for initial tensors')
    return parser.parse_args()

def Set_Hamiltonian(Jxy,Jz,hx):
    # AF
#    Jz = 1.0
#    Jxy = 1.0
    Ham = np.zeros((4,4))

    Ham[0,0] = 0.25 * Jz
    Ham[0,1] = -0.125 * hx 
    Ham[0,2] = -0.125 * hx 

    Ham[1,0] = -0.125 * hx
    Ham[1,1] = -0.25 * Jz
    Ham[1,2] = 0.5 * Jxy
    Ham[1,3] = -0.125 * hx

    Ham[2,0] = -0.125 * hx
    Ham[2,1] = 0.5 * Jxy
    Ham[2,2] = -0.25 * Jz
    Ham[2,3] = -0.125 * hx

    Ham[3,1] = -0.125 * hx
    Ham[3,2] = -0.125 * hx
    Ham[3,3] = 0.25 * Jz

    return Ham

def Initialize_Tensors(Tn,seed,initial_type=0):
    np.random.seed(seed)
    ## Random tensors back_ground
#    back_ground_amp = 0.001
    back_ground_amp = 0.0
    Tn_temp = back_ground_amp * (np.random.rand(D,D,D,D,2)-0.5)

    for i in range(0,N_UNIT):
        Tn[i][:] = Tn_temp.copy()

    if initial_type == 1:
        ## ferro
        for i in range(0,N_UNIT):
            Tn[i][0,0,0,0,0]=1.0
            Tn[i][0,0,0,0,1]=0.0
    elif initial_type == 2:
        ## AF
        for i in range(0,N_UNIT):
            ix = i%LX_ori
            iy = i/LX_ori
            if (ix + iy)%2 == 0:
                Tn[i][0,0,0,0,0]=1.0
                Tn[i][0,0,0,0,1]=0.0   
            else:
                Tn[i][0,0,0,0,0]=0.0
                Tn[i][0,0,0,0,1]=1.0
    elif initial_type == 3:
        ## ferro x
        for i in range(0,N_UNIT):
            Tn[i][0,0,0,0,0]=1.0/np.sqrt(2.0)
            Tn[i][0,0,0,0,1]=1.0/np.sqrt(2.0)

def main():
    ## timers
    time_simple_update=0.0
    time_full_update=0.0
    time_env=0.0
    time_obs=0.0

    ## Parameters
    args = parse_args()

    seed = args.seed
#    Jxy = args.J
#    Jz = args.J
    Jxy = args.Jxy
    Jz = args.Jz
    hx = args.hx
    tau = args.tau
    tau_step = args.tau_step
    tau_full = args.tau_full
    tau_full_step = args.tau_full_step
    second_ST = args.second_ST
    initial_type = args.initial_type
    output_prefix=args.output_file
    input_file = args.input_file

    print "## Logs: seed =",seed
    print "## Logs: Jxy =",Jxy
    print "## Logs: Jz =",Jz
    print "## Logs: hx =",hx
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
    Tn=[np.zeros((D,D,D,D,2),dtype=TENSOR_DTYPE)]
    eTt=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]
    eTr=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]
    eTb=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]
    eTl=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]

    C1=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)
    C2=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)
    C3=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)
    C4=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)

    for i in range(1,N_UNIT):
        Tn.append(np.zeros((D,D,D,D,2),dtype=TENSOR_DTYPE))
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
            Tn[i][0:min(Tn[i].shape[0],Tn_in.shape[0]),0:min(Tn[i].shape[1],Tn_in.shape[1]),0:min(Tn[i].shape[2],Tn_in.shape[2]),0:min(Tn[i].shape[3],Tn_in.shape[3]),0:min(Tn[i].shape[4],Tn_in.shape[4])] = Tn_in[0:min(Tn[i].shape[0],Tn_in.shape[0]),0:min(Tn[i].shape[1],Tn_in.shape[1]),0:min(Tn[i].shape[2],Tn_in.shape[2]),0:min(Tn[i].shape[3],Tn_in.shape[3]),0:min(Tn[i].shape[4],Tn_in.shape[4])]

    lambda_tensor = np.ones((N_UNIT,4,D),dtype=float)
    Ham = Set_Hamiltonian(Jxy,Jz,hx)
    s,U = linalg.eigh(Ham)

    op12 =  np.dot(np.dot(U,np.diag(np.exp(-tau * s))),U.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
    op12_2 =  np.dot(np.dot(U,np.diag(np.exp(-tau*0.5 * s))),U.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
#    op12 =  np.dot(np.dot(U,np.diag(np.exp(-1j*tau * s))),U.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
#    op12_2 =  np.dot(np.dot(U,np.diag(np.exp(-1j*tau*0.5 * s))),U.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
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
        Ham = Set_Hamiltonian(hx)
        s,U = linalg.eigh(Ham)

        op12 =  np.dot(np.dot(U,np.diag(np.exp(-tau_full * s))),U.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)

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

    ## Calc physical quantities
    start_env = time.time()
    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)
    time_env += time.time() - start_env



#    for i in range(0,N_UNIT):
#        print "## i eTl[i] ",i,eTl[i]
#        print "## i eTt[i] ",i,eTt[i]
#        print "## i eTr[i] ",i,eTr[i]
#        print "## i eTb[i] ",i,eTb[i]

##    |    |
## --[2]  [3]--
##    |    |
##    |    |
##   [0]--[1]
##    |    |
##
##   2,3
##    |
## 1-[0]-0
##   
##   2,3
##    |
## 1-[1]-0
##   
## 0-[2]-1
##    |
##   2,3
##   
## 0-[3]-1
##    |
##   2,3
##   

    T0b = eTb[0]
    T1b = eTb[1]
#    T2t = eTt[2]
#    T3t = eTt[3]
    T2t = eTt[1]
    T3t = eTt[0]

    T02 = np.tensordot(T0b,T2t,axes=((2,3),(2,3)))
    T13 = np.tensordot(T1b,T3t,axes=((2,3),(2,3)))
#    T0123h = np.tensordot(T02,T13,axes=((0,3),(1,2))).transpose(0,3,1,2).reshape(CHI*CHI,CHI*CHI)
    T0123h = np.tensordot(T02,T13,axes=((0,3),(1,2))).reshape(CHI*CHI,CHI*CHI)

    print "## np.allclose(T0123h,T0123h.T,rtol=1e-6,atol=1e-6)", np.allclose(T0123h,T0123h.T,rtol=1e-6,atol=1e-6)
#    wT0123h = linalg.eigvalsh(T0123h)
    wT0123h = linalg.eigvals(T0123h)
    idx = np.abs(wT0123h).argsort()[::-1]
    wT0123h = wT0123h[idx]
    print "## eigenvalues of T0123h abs",' '.join([str(elem) for elem in np.abs(wT0123h)])
    idx = np.abs(wT0123h.real).argsort()[::-1]
    wT0123h = wT0123h[idx]
#    print "## eigenvalues of T0123h",' '.join([str(elem) for elem in wT0123h])
    print "## eigenvalues of T0123h real",' '.join([str(elem) for elem in wT0123h.real])
    print "## eigenvalues of T0123h imag",' '.join([str(elem) for elem in wT0123h.imag])
    print "## eigenvalues of T0123h imag max",np.max(np.abs(wT0123h.imag))

    T0l = eTl[0]
    T1r = eTr[1]
#    T2l = eTl[2]
#    T3r = eTr[3]
    T2l = eTl[1]
    T3r = eTr[0]

    T23 = np.tensordot(T2l,T3r,axes=((2,3),(2,3)))
    T01 = np.tensordot(T0l,T1r,axes=((2,3),(2,3)))
#    T0123v = np.tensordot(T23,T01,axes=((0,3),(1,2))).transpose(0,3,1,2).reshape(CHI*CHI,CHI*CHI)
    T0123v = np.tensordot(T23,T01,axes=((0,3),(1,2))).reshape(CHI*CHI,CHI*CHI)

    print "## np.allclose(T0123v,T0123v.T,rtol=1e-6,atol=1e-6)", np.allclose(T0123v,T0123v.T,rtol=1e-6,atol=1e-6)
#    wT0123v = linalg.eigvalsh(T0123v)
    wT0123v = linalg.eigvals(T0123v)
    idx = np.abs(wT0123v).argsort()[::-1]
    wT0123v = wT0123v[idx]
    print "## eigenvalues of T0123v abs",' '.join([str(elem) for elem in np.abs(wT0123v)])
    idx = np.abs(wT0123v.real).argsort()[::-1]
    wT0123v = wT0123v[idx]
#    print "## eigenvalues of T0123v",' '.join([str(elem) for elem in wT0123v])
    print "## eigenvalues of T0123v real",' '.join([str(elem) for elem in wT0123v.real])
    print "## eigenvalues of T0123v imag",' '.join([str(elem) for elem in wT0123v.imag])
    print "## eigenvalues of T0123v imag max",np.max(np.abs(wT0123v.imag))



    op_identity = np.identity(2)

    op_mz = np.zeros((2,2))
    op_mz[0,0] = 0.5
    op_mz[1,1] = -0.5

    op_mx = np.zeros((2,2))
    op_mx[1,0] = 0.5
    op_mx[0,1] = 0.5

    op_my = np.zeros((2,2),dtype=complex)
    op_my[0,1] = 0.5j
    op_my[1,0] = -0.5j    

    mz = np.zeros(N_UNIT)
    mx = np.zeros(N_UNIT)
    my = np.zeros(N_UNIT)

    zz = np.zeros((N_UNIT,2))
    xx = np.zeros((N_UNIT,2))
    yy = np.zeros((N_UNIT,2))

    start_obs = time.time()

    for i in range(0,N_UNIT):        
        norm = Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_identity)
        mz[i] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_mz)/norm)
        mx[i] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_mx)/norm)
        my[i] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_my)/norm)

        print "## hx,i,norm,mx,my,mz,sqrt(sum m^2) ",hx,i,norm,mx[i],my[i],mz[i],np.sqrt(mx[i]**2+my[i]**2+mz[i]**2)

    for num in range(0,N_UNIT):
        ## x direction
        num_j = NN_Tensor[num,2]
        norm_x = Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_identity,op_identity)
        zz[num,0] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_mz,op_mz)/norm_x)
        xx[num,0] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_mx,op_mx)/norm_x)
        yy[num,0] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_my,op_my)/norm_x)

        ## y direction
        num_j = NN_Tensor[num,3]
        norm_y = Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_identity,op_identity)
        zz[num,1] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_mz,op_mz)/norm_y)
        xx[num,1] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_mx,op_mx)/norm_y)
        yy[num,1] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_my,op_my)/norm_y)

        print "## hx,num,norm_x,norm_y,xx[0],xx[1],yy[0],yy[1],zz[0],zz[1] ",hx,num,norm_x,norm_y,xx[num,0],xx[num,1],yy[num,0],yy[num,1],zz[num,0],zz[num,1]

    time_obs += time.time() - start_obs

#    Energy_x = np.sum(xx[:,0]+yy[:,0]+zz[:,0])/N_UNIT
#    Energy_y = np.sum(yy[:,1]+yy[:,1]+zz[:,1])/N_UNIT
    Energy_x = (np.sum(xx[:,0]+yy[:,0])*Jxy+np.sum(zz[:,0])*Jz)/N_UNIT
    Energy_y = (np.sum(xx[:,1]+yy[:,1])*Jxy+np.sum(zz[:,1])*Jz)/N_UNIT
    Mag_abs = np.sum(np.sqrt(mx**2+my**2+mz**2))/N_UNIT

#    print hx, -np.sum(xx+yy+zz)/N_UNIT - hx * np.sum(mx)/N_UNIT,np.sum(mz)/N_UNIT,np.sum(mx)/N_UNIT,np.sum(zz)/N_UNIT
#    print hx,Energy_x+Energy_y,Mag_abs,Energy_x,Energy_y
    print hx,Energy_x+Energy_y - hx * np.sum(mx)/N_UNIT,Mag_abs,Energy_x,Energy_y
    print "## hx,mx,mz0mz1",hx,np.sum(mx)/N_UNIT,0.5*(np.sum(zz[:,0])+np.sum(zz[:,1]))/N_UNIT

    print "## time simple update=",time_simple_update
    print "## time full update=",time_full_update
    print "## time environment=",time_env
    print "## time observable=",time_obs

if __name__ == "__main__":
    main()
