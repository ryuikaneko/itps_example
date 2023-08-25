# coding:utf-8
import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import time
import argparse

## import basic routines
from PEPS_Basics import *
from PEPS_Basics_Added import *
from PEPS_Parameters import *
from Square_lattice_CTM import *

def parse_args():
    parser = argparse.ArgumentParser(description='iPEPS simulation for TFI model')
#    parser = argparse.ArgumentParser(description='iPEPS simulation for Heisenberg model')
    parser.add_argument('-s', metavar='seed',dest='seed', type=int, default=11,
                        help='set random seed')
#
##    parser.add_argument('-J', metavar='J',dest='J', type=float, default=1.0,
##                        help='set Heisenberg coupling J')
##    parser.add_argument('-Jxy', metavar='Jxy',dest='Jxy', type=float, default=1.0,
##                        help='set XY coupling Jxy')
#    parser.add_argument('-Jxy', metavar='Jxy',dest='Jxy', type=float, default=0.0,
#                        help='set XY coupling Jxy')
##    parser.add_argument('-Jz', metavar='Jz',dest='Jz', type=float, default=1.0,
##                        help='set Ising coupling Jz')
#    parser.add_argument('-Jz', metavar='Jz',dest='Jz', type=float, default=-1.0,
#                        help='set Ising coupling Jz')
##    parser.add_argument('-hx', metavar='hx',dest='hx', type=float, default=0.0,
##                        help='set magnetic field hx')
#    parser.add_argument('-hx', metavar='hx',dest='hx', type=float, default=1.5,
#                        help='set magnetic field hx')
#
    parser.add_argument('-V',metavar='V',dest='V',type=float,default=1.0,help='set V')
    parser.add_argument('-Omega',metavar='Omega',dest='Omega',type=float,default=0.01,help='set Omega')
    parser.add_argument('-Delta',metavar='Delta',dest='Delta',type=float,default=2.0,help='set Delta')
#
    parser.add_argument('-indexnum',metavar='indexnum',dest='indexnum',type=int,default=0,help='set indexnum')
#
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

## + Jxy: AF
## + Jz: AF
## - hx: favors |right>
## - hz: favors |up>
def Set_Hamiltonian(Jxy,Jz,hx,hz):
    z = 4
    hx = hx * 1.0/z
    hz = hz * 1.0/z
#
    I = np.array([[1,0],[0,1]])
    Sp = np.array([[0,1],[0,0]])
    Sm = np.array([[0,0],[1,0]])
    Sx = 0.5*np.array([[0,1],[1,0]])
    Sy = 0.5*np.array([[0,-1j],[1j,0]])
    Sz = 0.5*np.array([[1,0],[0,-1]])
#
    H = + Jxy * 0.5*(np.kron(Sp,Sm)+np.kron(Sm,Sp)) \
        + Jz * np.kron(Sz,Sz) \
        - hx * (np.kron(I,Sx)+np.kron(Sx,I)) \
        - hz * (np.kron(I,Sz)+np.kron(Sz,I))
    return H

#def Set_Hamiltonian(Jxy,Jz,hx):
#    # AF
##    Jz = 1.0
##    Jxy = 1.0
#    Ham = np.zeros((4,4))
#
#    Ham[0,0] = 0.25 * Jz
#    Ham[0,1] = -0.125 * hx
#    Ham[0,2] = -0.125 * hx
#
#    Ham[1,0] = -0.125 * hx
#    Ham[1,1] = -0.25 * Jz
#    Ham[1,2] = 0.5 * Jxy
#    Ham[1,3] = -0.125 * hx
#
#    Ham[2,0] = -0.125 * hx
#    Ham[2,1] = 0.5 * Jxy
#    Ham[2,2] = -0.25 * Jz
#    Ham[2,3] = -0.125 * hx
#
#    Ham[3,1] = -0.125 * hx
#    Ham[3,2] = -0.125 * hx
#    Ham[3,3] = 0.25 * Jz
#
#    return Ham

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
    elif initial_type == 4:
        ## 1z2x
        for i in range(0,N_UNIT):
            ix = i%LX_ori
            iy = i/LX_ori
            if (ix + iy)%2 == 0:
                Tn[i][0,0,0,0,0]=1.0
                Tn[i][0,0,0,0,1]=0.0
            else:
                Tn[i][0,0,0,0,0]=1.0/np.sqrt(2.0)
                Tn[i][0,0,0,0,1]=-1.0/np.sqrt(2.0)
    elif initial_type == 5:
        ## ferro, all down
        for i in range(0,N_UNIT):
            Tn[i][0,0,0,0,0]=0.0
            Tn[i][0,0,0,0,1]=1.0
    elif initial_type == 6:
        ## ferro -x
        for i in range(0,N_UNIT):
            Tn[i][0,0,0,0,0]=1.0/np.sqrt(2.0)
            Tn[i][0,0,0,0,1]=-1.0/np.sqrt(2.0)

def main():
    ## timers
    time_simple_update=0.0
    time_full_update=0.0
    time_env=0.0
    time_obs=0.0

    ## Parameters
    args = parse_args()

    seed = args.seed
##    Jxy = args.J
##    Jz = args.J
#    Jxy = args.Jxy
#    Jz = args.Jz
#    hx = args.hx
#
    V = args.V
    Omega = args.Omega
    Delta = args.Delta
    indexnum = args.indexnum
#----
# original
#    Jxy = 0.0
#    Jz = V
#    hx = -0.5*Omega ### !!! half negative Omega !!!
#    hz = Delta
#----
# experiment
    Dim = 2
    Jxy = 0.0
    Jz = V
    hx = - Omega
    hz = Delta - V*Dim
#----
    tau = args.tau
    tau_step = args.tau_step
    tau_full = args.tau_full
    tau_full_step = args.tau_full_step
    second_ST = args.second_ST
    initial_type = args.initial_type
    output_prefix=args.output_file
    input_file = args.input_file

    print "## Logs: seed =",seed
    print "## Logs: V =",V
    print "## Logs: Omega =",Omega
    print "## Logs: Delta =",Delta
    print "## Logs: Jxy =",Jxy
    print "## Logs: Jz =",Jz
    print "## Logs: hx =",hx
    print "## Logs: hz =",hz
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
    Ham = Set_Hamiltonian(Jxy,Jz,hx,hz)
    s,U = linalg.eigh(Ham)

#    op12 =  np.dot(np.dot(U,np.diag(np.exp(-tau * s))),U.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
#    op12_2 =  np.dot(np.dot(U,np.diag(np.exp(-tau*0.5 * s))),U.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
    op12 =  np.dot(np.dot(U,np.diag(np.exp(-1j*tau * s))),U.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
    op12_2 =  np.dot(np.dot(U,np.diag(np.exp(-1j*tau*0.5 * s))),U.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
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
        Ham = Set_Hamiltonian(Jxy,Jz,hx,hz)
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

    zz_11 = np.zeros((N_UNIT,4))
    zz_x_2 = np.zeros(N_UNIT)
    zz_x_3 = np.zeros(N_UNIT)
    zz_x_4 = np.zeros(N_UNIT)
    zz_x_5 = np.zeros(N_UNIT)
    zz_y_2 = np.zeros(N_UNIT)
    zz_y_3 = np.zeros(N_UNIT)
    zz_y_4 = np.zeros(N_UNIT)
    zz_y_5 = np.zeros(N_UNIT)

    xx_11 = np.zeros((N_UNIT,4))
    xx_x_2 = np.zeros(N_UNIT)
    xx_x_3 = np.zeros(N_UNIT)
    xx_x_4 = np.zeros(N_UNIT)
    xx_x_5 = np.zeros(N_UNIT)
    xx_y_2 = np.zeros(N_UNIT)
    xx_y_3 = np.zeros(N_UNIT)
    xx_y_4 = np.zeros(N_UNIT)
    xx_y_5 = np.zeros(N_UNIT)

    start_obs = time.time()

    for i in range(0,N_UNIT):
        norm = Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_identity)
        mz[i] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_mz)/norm)
        mx[i] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_mx)/norm)
        my[i] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_my)/norm)

        print "## V,Omega,Delta,hx,hz,i,norm,mx,my,mz,sqrt(sum m^2) ",V,Omega,Delta,hx,hz,i,norm,mx[i],my[i],mz[i],np.sqrt(mx[i]**2+my[i]**2+mz[i]**2)

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

        print "## V,Omega,Delta,hx,hz,num,norm_x,norm_y,xx[0],xx[1],yy[0],yy[1],zz[0],zz[1] ",V,Omega,Delta,hx,hz,num,norm_x,norm_y,xx[num,0],xx[num,1],yy[num,0],yy[num,1],zz[num,0],zz[num,1]



    for i in range(0,N_UNIT):
        ##
        ## direction 0
        ##
        ## k--l
        ## |  |
        ## j--i
        ##
        ## n.n.n = i-->k
        ##
        j = NN_Tensor[i,0]
        k = NNN_Tensor[i,0]
        l = NN_Tensor[i,1]
        norm = Contract_four_sites(C1[k],C2[l],C3[i],C4[j],eTt[k],eTt[l],eTr[l],eTr[i],eTb[i],eTb[j],eTl[j],eTl[k],Tn[k],Tn[l],Tn[i],Tn[j],op_identity,op_identity,op_identity,op_identity)
        zz_11[i,0] = np.real(Contract_four_sites(C1[k],C2[l],C3[i],C4[j],eTt[k],eTt[l],eTr[l],eTr[i],eTb[i],eTb[j],eTl[j],eTl[k],Tn[k],Tn[l],Tn[i],Tn[j],op_mz,op_identity,op_mz,op_identity)/norm)
        ##
        ## direction 1
        ##
        ## j--k
        ## |  |
        ## i--l
        ##
        ## n.n.n = i-->k
        ##
        j = NN_Tensor[i,1]
        k = NNN_Tensor[i,1]
        l = NN_Tensor[i,2]
        norm = Contract_four_sites(C1[j],C2[k],C3[l],C4[i],eTt[j],eTt[k],eTr[k],eTr[l],eTb[l],eTb[i],eTl[i],eTl[j],Tn[j],Tn[k],Tn[l],Tn[i],op_identity,op_identity,op_identity,op_identity)
        zz_11[i,1] = np.real(Contract_four_sites(C1[j],C2[k],C3[l],C4[i],eTt[j],eTt[k],eTr[k],eTr[l],eTb[l],eTb[i],eTl[i],eTl[j],Tn[j],Tn[k],Tn[l],Tn[i],op_identity,op_mz,op_identity,op_mz)/norm)
        ##
        ## direction 2
        ##
        ## i--j
        ## |  |
        ## l--k
        ##
        ## n.n.n = i-->k
        ##
        j = NN_Tensor[i,2]
        k = NNN_Tensor[i,2]
        l = NN_Tensor[i,3]
        norm = Contract_four_sites(C1[i],C2[j],C3[k],C4[l],eTt[i],eTt[j],eTr[j],eTr[k],eTb[k],eTb[l],eTl[l],eTl[i],Tn[i],Tn[j],Tn[k],Tn[l],op_identity,op_identity,op_identity,op_identity)
        zz_11[i,2] = np.real(Contract_four_sites(C1[i],C2[j],C3[k],C4[l],eTt[i],eTt[j],eTr[j],eTr[k],eTb[k],eTb[l],eTl[l],eTl[i],Tn[i],Tn[j],Tn[k],Tn[l],op_mz,op_identity,op_mz,op_identity)/norm)
        ##
        ## direction 3
        ##
        ## l--i
        ## |  |
        ## k--j
        ##
        ## n.n.n = i-->k
        ##
        j = NN_Tensor[i,3]
        k = NNN_Tensor[i,3]
        l = NN_Tensor[i,0]
        norm = Contract_four_sites(C1[l],C2[i],C3[j],C4[k],eTt[l],eTt[i],eTr[i],eTr[j],eTb[j],eTb[k],eTl[k],eTl[l],Tn[l],Tn[i],Tn[j],Tn[k],op_identity,op_identity,op_identity,op_identity)
        zz_11[i,3] = np.real(Contract_four_sites(C1[l],C2[i],C3[j],C4[k],eTt[l],eTt[i],eTr[i],eTr[j],eTb[j],eTb[k],eTl[k],eTl[l],Tn[l],Tn[i],Tn[j],Tn[k],op_identity,op_mz,op_identity,op_mz)/norm)
    for num in range(0,N_UNIT):
        for nnn in range(4):
            print "## V,Omega,Delta,hx,hz,num,nnn,norm,zz_11[num,nnn]",V,Omega,Delta,hx,hz,num,nnn,norm,zz_11[num,nnn]



    for i in range(0,N_UNIT):
        ##
        ## direction 0
        ##
        ## k--l
        ## |  |
        ## j--i
        ##
        ## n.n.n = i-->k
        ##
        j = NN_Tensor[i,0]
        k = NNN_Tensor[i,0]
        l = NN_Tensor[i,1]
        norm = Contract_four_sites(C1[k],C2[l],C3[i],C4[j],eTt[k],eTt[l],eTr[l],eTr[i],eTb[i],eTb[j],eTl[j],eTl[k],Tn[k],Tn[l],Tn[i],Tn[j],op_identity,op_identity,op_identity,op_identity)
        xx_11[i,0] = np.real(Contract_four_sites(C1[k],C2[l],C3[i],C4[j],eTt[k],eTt[l],eTr[l],eTr[i],eTb[i],eTb[j],eTl[j],eTl[k],Tn[k],Tn[l],Tn[i],Tn[j],op_mx,op_identity,op_mx,op_identity)/norm)
        ##
        ## direction 1
        ##
        ## j--k
        ## |  |
        ## i--l
        ##
        ## n.n.n = i-->k
        ##
        j = NN_Tensor[i,1]
        k = NNN_Tensor[i,1]
        l = NN_Tensor[i,2]
        norm = Contract_four_sites(C1[j],C2[k],C3[l],C4[i],eTt[j],eTt[k],eTr[k],eTr[l],eTb[l],eTb[i],eTl[i],eTl[j],Tn[j],Tn[k],Tn[l],Tn[i],op_identity,op_identity,op_identity,op_identity)
        xx_11[i,1] = np.real(Contract_four_sites(C1[j],C2[k],C3[l],C4[i],eTt[j],eTt[k],eTr[k],eTr[l],eTb[l],eTb[i],eTl[i],eTl[j],Tn[j],Tn[k],Tn[l],Tn[i],op_identity,op_mx,op_identity,op_mx)/norm)
        ##
        ## direction 2
        ##
        ## i--j
        ## |  |
        ## l--k
        ##
        ## n.n.n = i-->k
        ##
        j = NN_Tensor[i,2]
        k = NNN_Tensor[i,2]
        l = NN_Tensor[i,3]
        norm = Contract_four_sites(C1[i],C2[j],C3[k],C4[l],eTt[i],eTt[j],eTr[j],eTr[k],eTb[k],eTb[l],eTl[l],eTl[i],Tn[i],Tn[j],Tn[k],Tn[l],op_identity,op_identity,op_identity,op_identity)
        xx_11[i,2] = np.real(Contract_four_sites(C1[i],C2[j],C3[k],C4[l],eTt[i],eTt[j],eTr[j],eTr[k],eTb[k],eTb[l],eTl[l],eTl[i],Tn[i],Tn[j],Tn[k],Tn[l],op_mx,op_identity,op_mx,op_identity)/norm)
        ##
        ## direction 3
        ##
        ## l--i
        ## |  |
        ## k--j
        ##
        ## n.n.n = i-->k
        ##
        j = NN_Tensor[i,3]
        k = NNN_Tensor[i,3]
        l = NN_Tensor[i,0]
        norm = Contract_four_sites(C1[l],C2[i],C3[j],C4[k],eTt[l],eTt[i],eTr[i],eTr[j],eTb[j],eTb[k],eTl[k],eTl[l],Tn[l],Tn[i],Tn[j],Tn[k],op_identity,op_identity,op_identity,op_identity)
        xx_11[i,3] = np.real(Contract_four_sites(C1[l],C2[i],C3[j],C4[k],eTt[l],eTt[i],eTr[i],eTr[j],eTb[j],eTb[k],eTl[k],eTl[l],Tn[l],Tn[i],Tn[j],Tn[k],op_identity,op_mx,op_identity,op_mx)/norm)
    for num in range(0,N_UNIT):
        for nnn in range(4):
            print "## V,Omega,Delta,hx,hz,num,nnn,norm,xx_11[num,nnn]",V,Omega,Delta,hx,hz,num,nnn,norm,xx_11[num,nnn]



    for i in range(0,N_UNIT):
        #
        # x=2
        # i-j-k
        #
        j = FAR_X_Tensor[i,1]
        k = FAR_X_Tensor[i,2]
        norm = \
        Contract_scalar_3x1(\
         C1[i],eTt[i],eTt[j],eTt[k], C2[k],\
        eTl[i], Tn[i], Tn[j], Tn[k],eTr[k],\
         C4[i],eTb[i],eTb[j],eTb[k], C3[k],\
        op_identity,op_identity,op_identity\
        )
        zz_x_2[i] = np.real(\
        Contract_scalar_3x1(\
         C1[i],eTt[i],eTt[j],eTt[k], C2[k],\
        eTl[i], Tn[i], Tn[j], Tn[k],eTr[k],\
         C4[i],eTb[i],eTb[j],eTb[k], C3[k],\
        op_mz,op_identity,op_mz\
        )\
        /norm)
        print "## V,Omega,Delta,hx,hz,num,distx,disty,norm,zz",\
        V,Omega,Delta,hx,hz,i,2,0,norm,\
        zz_x_2[i]
        #
        # x=3
        # i-j-k-l
        #
        j = FAR_X_Tensor[i,1]
        k = FAR_X_Tensor[i,2]
        l = FAR_X_Tensor[i,3]
        norm = \
        Contract_scalar_4x1(\
         C1[i],eTt[i],eTt[j],eTt[k],eTt[l], C2[l],\
        eTl[i], Tn[i], Tn[j], Tn[k], Tn[l],eTr[l],\
         C4[i],eTb[i],eTb[j],eTb[k],eTb[l], C3[l],\
        op_identity,op_identity,op_identity,op_identity\
        )
        zz_x_3[i] = np.real(\
        Contract_scalar_4x1(\
         C1[i],eTt[i],eTt[j],eTt[k],eTt[l], C2[l],\
        eTl[i], Tn[i], Tn[j], Tn[k], Tn[l],eTr[l],\
         C4[i],eTb[i],eTb[j],eTb[k],eTb[l], C3[l],\
        op_mz,op_identity,op_identity,op_mz\
        )\
        /norm)
        print "## V,Omega,Delta,hx,hz,num,distx,disty,norm,zz",\
        V,Omega,Delta,hx,hz,i,3,0,norm,\
        zz_x_3[i]
        #
        # x=4
        # i-i1-i2-i3-i4
        #
        i1 = FAR_X_Tensor[i,1]
        i2 = FAR_X_Tensor[i,2]
        i3 = FAR_X_Tensor[i,3]
        i4 = FAR_X_Tensor[i,4]
        norm = \
        Contract_scalar_5x1(\
         C1[i],eTt[i],eTt[i1],eTt[i2],eTt[i3],eTt[i4], C2[i4],\
        eTl[i], Tn[i], Tn[i1], Tn[i2], Tn[i3], Tn[i4],eTr[i4],\
         C4[i],eTb[i],eTb[i1],eTb[i2],eTb[i3],eTb[i4], C3[i4],\
        op_identity,op_identity,op_identity,op_identity,op_identity\
        )
        zz_x_4[i] = np.real(\
        Contract_scalar_5x1(\
         C1[i],eTt[i],eTt[i1],eTt[i2],eTt[i3],eTt[i4], C2[i4],\
        eTl[i], Tn[i], Tn[i1], Tn[i2], Tn[i3], Tn[i4],eTr[i4],\
         C4[i],eTb[i],eTb[i1],eTb[i2],eTb[i3],eTb[i4], C3[i4],\
        op_mz,op_identity,op_identity,op_identity,op_mz\
        )\
        /norm)
        print "## V,Omega,Delta,hx,hz,num,distx,disty,norm,zz",\
        V,Omega,Delta,hx,hz,i,4,0,norm,\
        zz_x_4[i]
        #
        # x=5
        # i-i1-i2-i3-i4-i5
        #
        i1 = FAR_X_Tensor[i,1]
        i2 = FAR_X_Tensor[i,2]
        i3 = FAR_X_Tensor[i,3]
        i4 = FAR_X_Tensor[i,4]
        i5 = FAR_X_Tensor[i,5]
        norm = \
        Contract_scalar_6x1(\
         C1[i],eTt[i],eTt[i1],eTt[i2],eTt[i3],eTt[i4],eTt[i5], C2[i5],\
        eTl[i], Tn[i], Tn[i1], Tn[i2], Tn[i3], Tn[i4], Tn[i5],eTr[i5],\
         C4[i],eTb[i],eTb[i1],eTb[i2],eTb[i3],eTb[i4],eTb[i5], C3[i5],\
        op_identity,op_identity,op_identity,op_identity,op_identity,op_identity\
        )
        zz_x_5[i] = np.real(\
        Contract_scalar_6x1(\
         C1[i],eTt[i],eTt[i1],eTt[i2],eTt[i3],eTt[i4],eTt[i5], C2[i5],\
        eTl[i], Tn[i], Tn[i1], Tn[i2], Tn[i3], Tn[i4], Tn[i5],eTr[i5],\
         C4[i],eTb[i],eTb[i1],eTb[i2],eTb[i3],eTb[i4],eTb[i5], C3[i5],\
        op_mz,op_identity,op_identity,op_identity,op_identity,op_mz\
        )\
        /norm)
        print "## V,Omega,Delta,hx,hz,num,distx,disty,norm,zz",\
        V,Omega,Delta,hx,hz,i,5,0,norm,\
        zz_x_5[i]
        #
        # y=2
        # k
        # |
        # j
        # |
        # i
        #
        j = FAR_Y_Tensor[i,1]
        k = FAR_Y_Tensor[i,2]
        norm = \
        Contract_scalar_1x3(\
         C1[k],eTt[k], C2[k],\
        eTl[k], Tn[k],eTr[k],\
        eTl[j], Tn[j],eTr[j],\
        eTl[i], Tn[i],eTr[i],\
         C4[i],eTb[i], C3[i],\
        op_identity,\
        op_identity,\
        op_identity\
        )
        zz_y_2[i] = np.real(\
        Contract_scalar_1x3(\
         C1[k],eTt[k], C2[k],\
        eTl[k], Tn[k],eTr[k],\
        eTl[j], Tn[j],eTr[j],\
        eTl[i], Tn[i],eTr[i],\
         C4[i],eTb[i], C3[i],\
        op_mz,\
        op_identity,\
        op_mz\
        )\
        /norm)
        print "## V,Omega,Delta,hx,hz,num,distx,disty,norm,zz",\
        V,Omega,Delta,hx,hz,i,0,2,norm,\
        zz_y_2[i]
        #
        # y=3
        # l
        # |
        # k
        # |
        # j
        # |
        # i
        #
        j = FAR_Y_Tensor[i,1]
        k = FAR_Y_Tensor[i,2]
        l = FAR_Y_Tensor[i,3]
        norm = \
        Contract_scalar_1x4(\
         C1[l],eTt[l], C2[l],\
        eTl[l], Tn[l],eTr[l],\
        eTl[k], Tn[k],eTr[k],\
        eTl[j], Tn[j],eTr[j],\
        eTl[i], Tn[i],eTr[i],\
         C4[i],eTb[i], C3[i],\
        op_identity,\
        op_identity,\
        op_identity,\
        op_identity\
        )
        zz_y_3[i] = np.real(\
        Contract_scalar_1x4(\
         C1[l],eTt[l], C2[l],\
        eTl[l], Tn[l],eTr[l],\
        eTl[k], Tn[k],eTr[k],\
        eTl[j], Tn[j],eTr[j],\
        eTl[i], Tn[i],eTr[i],\
         C4[i],eTb[i], C3[i],\
        op_mz,\
        op_identity,\
        op_identity,\
        op_mz\
        )\
        /norm)
        print "## V,Omega,Delta,hx,hz,num,distx,disty,norm,zz",\
        V,Omega,Delta,hx,hz,i,0,3,norm,\
        zz_y_3[i]
        #
        # y=4
        # i4
        # |
        # i3
        # |
        # i2
        # |
        # i1
        # |
        # i
        #
        i1 = FAR_Y_Tensor[i,1]
        i2 = FAR_Y_Tensor[i,2]
        i3 = FAR_Y_Tensor[i,3]
        i4 = FAR_Y_Tensor[i,4]
        norm = \
        Contract_scalar_1x5(\
         C1[i4],eTt[i4], C2[i4],\
        eTl[i4], Tn[i4],eTr[i4],\
        eTl[i3], Tn[i3],eTr[i3],\
        eTl[i2], Tn[i2],eTr[i2],\
        eTl[i1], Tn[i1],eTr[i1],\
         eTl[i],  Tn[i], eTr[i],\
          C4[i], eTb[i],  C3[i],\
        op_identity,\
        op_identity,\
        op_identity,\
        op_identity,\
        op_identity\
        )
        zz_y_4[i] = np.real(\
        Contract_scalar_1x5(\
         C1[i4],eTt[i4], C2[i4],\
        eTl[i4], Tn[i4],eTr[i4],\
        eTl[i3], Tn[i3],eTr[i3],\
        eTl[i2], Tn[i2],eTr[i2],\
        eTl[i1], Tn[i1],eTr[i1],\
         eTl[i],  Tn[i], eTr[i],\
          C4[i], eTb[i],  C3[i],\
        op_mz,\
        op_identity,\
        op_identity,\
        op_identity,\
        op_mz\
        )\
        /norm)
        print "## V,Omega,Delta,hx,hz,num,distx,disty,norm,zz",\
        V,Omega,Delta,hx,hz,i,0,4,norm,\
        zz_y_4[i]
        #
        # y=5
        # i5
        # |
        # i4
        # |
        # i3
        # |
        # i2
        # |
        # i1
        # |
        # i
        #
        i1 = FAR_Y_Tensor[i,1]
        i2 = FAR_Y_Tensor[i,2]
        i3 = FAR_Y_Tensor[i,3]
        i4 = FAR_Y_Tensor[i,4]
        i5 = FAR_Y_Tensor[i,5]
        norm = \
        Contract_scalar_1x6(\
         C1[i5],eTt[i5], C2[i5],\
        eTl[i5], Tn[i5],eTr[i5],\
        eTl[i4], Tn[i4],eTr[i4],\
        eTl[i3], Tn[i3],eTr[i3],\
        eTl[i2], Tn[i2],eTr[i2],\
        eTl[i1], Tn[i1],eTr[i1],\
         eTl[i],  Tn[i], eTr[i],\
          C4[i], eTb[i],  C3[i],\
        op_identity,\
        op_identity,\
        op_identity,\
        op_identity,\
        op_identity,\
        op_identity\
        )
        zz_y_5[i] = np.real(\
        Contract_scalar_1x6(\
         C1[i5],eTt[i5], C2[i5],\
        eTl[i5], Tn[i5],eTr[i5],\
        eTl[i4], Tn[i4],eTr[i4],\
        eTl[i3], Tn[i3],eTr[i3],\
        eTl[i2], Tn[i2],eTr[i2],\
        eTl[i1], Tn[i1],eTr[i1],\
         eTl[i],  Tn[i], eTr[i],\
          C4[i], eTb[i],  C3[i],\
        op_mz,\
        op_identity,\
        op_identity,\
        op_identity,\
        op_identity,\
        op_mz\
        )\
        /norm)
        print "## V,Omega,Delta,hx,hz,num,distx,disty,norm,zz",\
        V,Omega,Delta,hx,hz,i,0,5,norm,\
        zz_y_5[i]



    for i in range(0,N_UNIT):
        #
        # x=2
        # i-j-k
        #
        j = FAR_X_Tensor[i,1]
        k = FAR_X_Tensor[i,2]
        norm = \
        Contract_scalar_3x1(\
         C1[i],eTt[i],eTt[j],eTt[k], C2[k],\
        eTl[i], Tn[i], Tn[j], Tn[k],eTr[k],\
         C4[i],eTb[i],eTb[j],eTb[k], C3[k],\
        op_identity,op_identity,op_identity\
        )
        xx_x_2[i] = np.real(\
        Contract_scalar_3x1(\
         C1[i],eTt[i],eTt[j],eTt[k], C2[k],\
        eTl[i], Tn[i], Tn[j], Tn[k],eTr[k],\
         C4[i],eTb[i],eTb[j],eTb[k], C3[k],\
        op_mx,op_identity,op_mx\
        )\
        /norm)
        print "## V,Omega,Delta,hx,hz,num,distx,disty,norm,xx",\
        V,Omega,Delta,hx,hz,i,2,0,norm,\
        xx_x_2[i]
        #
        # x=3
        # i-j-k-l
        #
        j = FAR_X_Tensor[i,1]
        k = FAR_X_Tensor[i,2]
        l = FAR_X_Tensor[i,3]
        norm = \
        Contract_scalar_4x1(\
         C1[i],eTt[i],eTt[j],eTt[k],eTt[l], C2[l],\
        eTl[i], Tn[i], Tn[j], Tn[k], Tn[l],eTr[l],\
         C4[i],eTb[i],eTb[j],eTb[k],eTb[l], C3[l],\
        op_identity,op_identity,op_identity,op_identity\
        )
        xx_x_3[i] = np.real(\
        Contract_scalar_4x1(\
         C1[i],eTt[i],eTt[j],eTt[k],eTt[l], C2[l],\
        eTl[i], Tn[i], Tn[j], Tn[k], Tn[l],eTr[l],\
         C4[i],eTb[i],eTb[j],eTb[k],eTb[l], C3[l],\
        op_mx,op_identity,op_identity,op_mx\
        )\
        /norm)
        print "## V,Omega,Delta,hx,hz,num,distx,disty,norm,xx",\
        V,Omega,Delta,hx,hz,i,3,0,norm,\
        xx_x_3[i]
        #
        # x=4
        # i-i1-i2-i3-i4
        #
        i1 = FAR_X_Tensor[i,1]
        i2 = FAR_X_Tensor[i,2]
        i3 = FAR_X_Tensor[i,3]
        i4 = FAR_X_Tensor[i,4]
        norm = \
        Contract_scalar_5x1(\
         C1[i],eTt[i],eTt[i1],eTt[i2],eTt[i3],eTt[i4], C2[i4],\
        eTl[i], Tn[i], Tn[i1], Tn[i2], Tn[i3], Tn[i4],eTr[i4],\
         C4[i],eTb[i],eTb[i1],eTb[i2],eTb[i3],eTb[i4], C3[i4],\
        op_identity,op_identity,op_identity,op_identity,op_identity\
        )
        xx_x_4[i] = np.real(\
        Contract_scalar_5x1(\
         C1[i],eTt[i],eTt[i1],eTt[i2],eTt[i3],eTt[i4], C2[i4],\
        eTl[i], Tn[i], Tn[i1], Tn[i2], Tn[i3], Tn[i4],eTr[i4],\
         C4[i],eTb[i],eTb[i1],eTb[i2],eTb[i3],eTb[i4], C3[i4],\
        op_mx,op_identity,op_identity,op_identity,op_mx\
        )\
        /norm)
        print "## V,Omega,Delta,hx,hz,num,distx,disty,norm,xx",\
        V,Omega,Delta,hx,hz,i,4,0,norm,\
        xx_x_4[i]
        #
        # x=5
        # i-i1-i2-i3-i4-i5
        #
        i1 = FAR_X_Tensor[i,1]
        i2 = FAR_X_Tensor[i,2]
        i3 = FAR_X_Tensor[i,3]
        i4 = FAR_X_Tensor[i,4]
        i5 = FAR_X_Tensor[i,5]
        norm = \
        Contract_scalar_6x1(\
         C1[i],eTt[i],eTt[i1],eTt[i2],eTt[i3],eTt[i4],eTt[i5], C2[i5],\
        eTl[i], Tn[i], Tn[i1], Tn[i2], Tn[i3], Tn[i4], Tn[i5],eTr[i5],\
         C4[i],eTb[i],eTb[i1],eTb[i2],eTb[i3],eTb[i4],eTb[i5], C3[i5],\
        op_identity,op_identity,op_identity,op_identity,op_identity,op_identity\
        )
        xx_x_5[i] = np.real(\
        Contract_scalar_6x1(\
         C1[i],eTt[i],eTt[i1],eTt[i2],eTt[i3],eTt[i4],eTt[i5], C2[i5],\
        eTl[i], Tn[i], Tn[i1], Tn[i2], Tn[i3], Tn[i4], Tn[i5],eTr[i5],\
         C4[i],eTb[i],eTb[i1],eTb[i2],eTb[i3],eTb[i4],eTb[i5], C3[i5],\
        op_mx,op_identity,op_identity,op_identity,op_identity,op_mx\
        )\
        /norm)
        print "## V,Omega,Delta,hx,hz,num,distx,disty,norm,xx",\
        V,Omega,Delta,hx,hz,i,5,0,norm,\
        xx_x_5[i]
        #
        # y=2
        # k
        # |
        # j
        # |
        # i
        #
        j = FAR_Y_Tensor[i,1]
        k = FAR_Y_Tensor[i,2]
        norm = \
        Contract_scalar_1x3(\
         C1[k],eTt[k], C2[k],\
        eTl[k], Tn[k],eTr[k],\
        eTl[j], Tn[j],eTr[j],\
        eTl[i], Tn[i],eTr[i],\
         C4[i],eTb[i], C3[i],\
        op_identity,\
        op_identity,\
        op_identity\
        )
        xx_y_2[i] = np.real(\
        Contract_scalar_1x3(\
         C1[k],eTt[k], C2[k],\
        eTl[k], Tn[k],eTr[k],\
        eTl[j], Tn[j],eTr[j],\
        eTl[i], Tn[i],eTr[i],\
         C4[i],eTb[i], C3[i],\
        op_mx,\
        op_identity,\
        op_mx\
        )\
        /norm)
        print "## V,Omega,Delta,hx,hz,num,distx,disty,norm,xx",\
        V,Omega,Delta,hx,hz,i,0,2,norm,\
        xx_y_2[i]
        #
        # y=3
        # l
        # |
        # k
        # |
        # j
        # |
        # i
        #
        j = FAR_Y_Tensor[i,1]
        k = FAR_Y_Tensor[i,2]
        l = FAR_Y_Tensor[i,3]
        norm = \
        Contract_scalar_1x4(\
         C1[l],eTt[l], C2[l],\
        eTl[l], Tn[l],eTr[l],\
        eTl[k], Tn[k],eTr[k],\
        eTl[j], Tn[j],eTr[j],\
        eTl[i], Tn[i],eTr[i],\
         C4[i],eTb[i], C3[i],\
        op_identity,\
        op_identity,\
        op_identity,\
        op_identity\
        )
        xx_y_3[i] = np.real(\
        Contract_scalar_1x4(\
         C1[l],eTt[l], C2[l],\
        eTl[l], Tn[l],eTr[l],\
        eTl[k], Tn[k],eTr[k],\
        eTl[j], Tn[j],eTr[j],\
        eTl[i], Tn[i],eTr[i],\
         C4[i],eTb[i], C3[i],\
        op_mx,\
        op_identity,\
        op_identity,\
        op_mx\
        )\
        /norm)
        print "## V,Omega,Delta,hx,hz,num,distx,disty,norm,xx",\
        V,Omega,Delta,hx,hz,i,0,3,norm,\
        xx_y_3[i]
        #
        # y=4
        # i4
        # |
        # i3
        # |
        # i2
        # |
        # i1
        # |
        # i
        #
        i1 = FAR_Y_Tensor[i,1]
        i2 = FAR_Y_Tensor[i,2]
        i3 = FAR_Y_Tensor[i,3]
        i4 = FAR_Y_Tensor[i,4]
        norm = \
        Contract_scalar_1x5(\
         C1[i4],eTt[i4], C2[i4],\
        eTl[i4], Tn[i4],eTr[i4],\
        eTl[i3], Tn[i3],eTr[i3],\
        eTl[i2], Tn[i2],eTr[i2],\
        eTl[i1], Tn[i1],eTr[i1],\
         eTl[i],  Tn[i], eTr[i],\
          C4[i], eTb[i],  C3[i],\
        op_identity,\
        op_identity,\
        op_identity,\
        op_identity,\
        op_identity\
        )
        xx_y_4[i] = np.real(\
        Contract_scalar_1x5(\
         C1[i4],eTt[i4], C2[i4],\
        eTl[i4], Tn[i4],eTr[i4],\
        eTl[i3], Tn[i3],eTr[i3],\
        eTl[i2], Tn[i2],eTr[i2],\
        eTl[i1], Tn[i1],eTr[i1],\
         eTl[i],  Tn[i], eTr[i],\
          C4[i], eTb[i],  C3[i],\
        op_mx,\
        op_identity,\
        op_identity,\
        op_identity,\
        op_mx\
        )\
        /norm)
        print "## V,Omega,Delta,hx,hz,num,distx,disty,norm,xx",\
        V,Omega,Delta,hx,hz,i,0,4,norm,\
        xx_y_4[i]
        #
        # y=5
        # i5
        # |
        # i4
        # |
        # i3
        # |
        # i2
        # |
        # i1
        # |
        # i
        #
        i1 = FAR_Y_Tensor[i,1]
        i2 = FAR_Y_Tensor[i,2]
        i3 = FAR_Y_Tensor[i,3]
        i4 = FAR_Y_Tensor[i,4]
        i5 = FAR_Y_Tensor[i,5]
        norm = \
        Contract_scalar_1x6(\
         C1[i5],eTt[i5], C2[i5],\
        eTl[i5], Tn[i5],eTr[i5],\
        eTl[i4], Tn[i4],eTr[i4],\
        eTl[i3], Tn[i3],eTr[i3],\
        eTl[i2], Tn[i2],eTr[i2],\
        eTl[i1], Tn[i1],eTr[i1],\
         eTl[i],  Tn[i], eTr[i],\
          C4[i], eTb[i],  C3[i],\
        op_identity,\
        op_identity,\
        op_identity,\
        op_identity,\
        op_identity,\
        op_identity\
        )
        xx_y_5[i] = np.real(\
        Contract_scalar_1x6(\
         C1[i5],eTt[i5], C2[i5],\
        eTl[i5], Tn[i5],eTr[i5],\
        eTl[i4], Tn[i4],eTr[i4],\
        eTl[i3], Tn[i3],eTr[i3],\
        eTl[i2], Tn[i2],eTr[i2],\
        eTl[i1], Tn[i1],eTr[i1],\
         eTl[i],  Tn[i], eTr[i],\
          C4[i], eTb[i],  C3[i],\
        op_mx,\
        op_identity,\
        op_identity,\
        op_identity,\
        op_identity,\
        op_mx\
        )\
        /norm)
        print "## V,Omega,Delta,hx,hz,num,distx,disty,norm,xx",\
        V,Omega,Delta,hx,hz,i,0,5,norm,\
        xx_y_5[i]



#    Energy_x = np.sum(xx[:,0]+yy[:,0]+zz[:,0])/N_UNIT
#    Energy_y = np.sum(yy[:,1]+yy[:,1]+zz[:,1])/N_UNIT
    Energy_x = (np.sum(xx[:,0]+yy[:,0])*Jxy+np.sum(zz[:,0])*Jz)/N_UNIT
    Energy_y = (np.sum(xx[:,1]+yy[:,1])*Jxy+np.sum(zz[:,1])*Jz)/N_UNIT
    Mag_abs = np.sum(np.sqrt(mx**2+my**2+mz**2))/N_UNIT

#    print hx, -np.sum(xx+yy+zz)/N_UNIT - hx * np.sum(mx)/N_UNIT,np.sum(mz)/N_UNIT,np.sum(mx)/N_UNIT,np.sum(zz)/N_UNIT
#    print hx,Energy_x+Energy_y,Mag_abs,Energy_x,Energy_y
#    print hx,Energy_x+Energy_y - hx * np.sum(mx)/N_UNIT,Mag_abs,Energy_x,Energy_y
#    print V,Omega,Delta,hx,hz,Energy_x+Energy_y - hx * np.sum(mx)/N_UNIT - hz * np.sum(mz)/N_UNIT,Mag_abs,Energy_x,Energy_y
#    print V,Omega,Delta,hx,hz,Energy_x+Energy_y - hx * np.sum(mx)/N_UNIT - hz * np.sum(mz)/N_UNIT,Mag_abs,Energy_x,Energy_y,mz[0],mz[1],mx[0],mx[1]
    eshift = 0.25*V*Dim - 0.5*Delta
    print V,Omega,Delta,hx,hz,Energy_x+Energy_y - hx * np.sum(mx)/N_UNIT - hz * np.sum(mz)/N_UNIT + eshift,Mag_abs,Energy_x,Energy_y,mz[0],mz[1],mx[0],mx[1]
    print "## indexnum,V,Omega,Delta,hx,hz,ene,mag,ex,ey,mz0,mz1,mx0,mx1",indexnum,V,Omega,Delta,hx,hz,Energy_x+Energy_y - hx * np.sum(mx)/N_UNIT - hz * np.sum(mz)/N_UNIT + eshift,Mag_abs,Energy_x,Energy_y,mz[0],mz[1],mx[0],mx[1]
#    print "## V,Omega,Delta,hx,hz,mx,mz,mz0mz1",V,Omega,Delta,hx,hz,np.sum(mx)/N_UNIT,np.sum(mz)/N_UNIT,0.5*(np.sum(zz[:,0])+np.sum(zz[:,1]))/N_UNIT
    print "## indexnum,V,Omega,Delta,hx,hz,mx,mz,mz0mz1",indexnum,V,Omega,Delta,hx,hz,np.sum(mx)/N_UNIT,np.sum(mz)/N_UNIT,0.5*(np.sum(zz[:,0])+np.sum(zz[:,1]))/N_UNIT



    ave_zz_dist01 = (np.sum(zz))/N_UNIT/2
    ave_zz_dist02 = (np.sum(zz_x_2) + np.sum(zz_y_2))/N_UNIT/2
    ave_zz_dist03 = (np.sum(zz_x_3) + np.sum(zz_y_3))/N_UNIT/2
    ave_zz_dist04 = (np.sum(zz_x_4) + np.sum(zz_y_4))/N_UNIT/2
    ave_zz_dist05 = (np.sum(zz_x_5) + np.sum(zz_y_5))/N_UNIT/2
    ave_zz_dist11 = (np.sum(zz_11))/N_UNIT/4
    print "## indexnum,V,Omega,Delta,hx,hz,\
ave_zz_dist01,ave_zz_dist02,ave_zz_dist03,\
ave_zz_dist04,ave_zz_dist05,\
ave_zz_dist11,ave_zz_dist22,ave_zz_dist33\
",\
        indexnum,\
        V,Omega,Delta,hx,hz,\
        ave_zz_dist01,ave_zz_dist02,ave_zz_dist03,\
        ave_zz_dist04,ave_zz_dist05,\
        ave_zz_dist11



    ave_zz_dist00 = np.sum(mz)/N_UNIT * np.sum(mz)/N_UNIT
    ave_zz_dist01_c = ave_zz_dist01 - ave_zz_dist00
    ave_zz_dist02_c = ave_zz_dist02 - ave_zz_dist00
    ave_zz_dist03_c = ave_zz_dist03 - ave_zz_dist00
    ave_zz_dist04_c = ave_zz_dist04 - ave_zz_dist00
    ave_zz_dist05_c = ave_zz_dist05 - ave_zz_dist00
    ave_zz_dist11_c = ave_zz_dist11 - ave_zz_dist00
    print "## indexnum,V,Omega,Delta,hx,hz,\
ave_zz_dist01_c,ave_zz_dist02_c,ave_zz_dist03_c,\
ave_zz_dist04_c,ave_zz_dist05_c,\
ave_zz_dist11_c,ave_zz_dist22_c,ave_zz_dist33_c\
",\
        indexnum,\
        V,Omega,Delta,hx,hz,\
        ave_zz_dist01_c,ave_zz_dist02_c,ave_zz_dist03_c,\
        ave_zz_dist04_c,ave_zz_dist05_c,\
        ave_zz_dist11_c



    ave_xx_dist01 = (np.sum(xx))/N_UNIT/2
    ave_xx_dist02 = (np.sum(xx_x_2) + np.sum(xx_y_2))/N_UNIT/2
    ave_xx_dist03 = (np.sum(xx_x_3) + np.sum(xx_y_3))/N_UNIT/2
    ave_xx_dist04 = (np.sum(xx_x_4) + np.sum(xx_y_4))/N_UNIT/2
    ave_xx_dist05 = (np.sum(xx_x_5) + np.sum(xx_y_5))/N_UNIT/2
    ave_xx_dist11 = (np.sum(xx_11))/N_UNIT/4
    print "## indexnum,V,Omega,Delta,hx,hz,\
ave_xx_dist01,ave_xx_dist02,ave_xx_dist03,\
ave_xx_dist04,ave_xx_dist05,\
ave_xx_dist11,ave_xx_dist22,ave_xx_dist33\
",\
        indexnum,\
        V,Omega,Delta,hx,hz,\
        ave_xx_dist01,ave_xx_dist02,ave_xx_dist03,\
        ave_xx_dist04,ave_xx_dist05,\
        ave_xx_dist11



    ave_xx_dist00 = np.sum(mx)/N_UNIT * np.sum(mx)/N_UNIT
    ave_xx_dist01_c = ave_xx_dist01 - ave_xx_dist00
    ave_xx_dist02_c = ave_xx_dist02 - ave_xx_dist00
    ave_xx_dist03_c = ave_xx_dist03 - ave_xx_dist00
    ave_xx_dist04_c = ave_xx_dist04 - ave_xx_dist00
    ave_xx_dist05_c = ave_xx_dist05 - ave_xx_dist00
    ave_xx_dist11_c = ave_xx_dist11 - ave_xx_dist00
    print "## indexnum,V,Omega,Delta,hx,hz,\
ave_xx_dist01_c,ave_xx_dist02_c,ave_xx_dist03_c,\
ave_xx_dist04_c,ave_xx_dist05_c,\
ave_xx_dist11_c,ave_xx_dist22_c,ave_xx_dist33_c\
",\
        indexnum,\
        V,Omega,Delta,hx,hz,\
        ave_xx_dist01_c,ave_xx_dist02_c,ave_xx_dist03_c,\
        ave_xx_dist04_c,ave_xx_dist05_c,\
        ave_xx_dist11_c



    time_obs += time.time() - start_obs

    print "## time simple update=",time_simple_update
    print "## time full update=",time_full_update
    print "## time environment=",time_env
    print "## time observable=",time_obs

if __name__ == "__main__":
    main()
