# coding:utf-8
## Constant Parameters for PEPS
import numpy as np
import fractions

## Lattice
# At least one of the LX_ori and LY_ori must be even.
#LX_ori = 2
#LY_ori = 2
LX_ori = 2
LY_ori = 1
N_UNIT = LX_ori * LY_ori

## true periodicity
## definition of slide when we apply the slided bounday
## User have to set correct (acceptable) valuses
LX_diff = LX_ori
LY_diff = LY_ori

## Following condition is tentative. It works only for limited situations.
if (LX_ori%2 ==1):
    ## skew boundary along x direction    
    ## fractions.gcd returns Greatest Common Divisor
    LX = (LY_ori / fractions.gcd(LY_ori, LX_diff)) * LX_ori
    LY = LY_ori
elif (LY_ori%2 ==1):
    ## skew boundary along x direction    
    ## fractions.gcd returns Greatest Common Divisor
    LX = LX_ori
    LY = (LX_ori / fractions.gcd(LX_ori, LY_diff)) * LY_ori
else:
    LX = LX_ori
    LY = LY_ori

#if (LX_ori%2 ==1):
#    ## skew boundary along x direction    
#    LX = (LY_ori/LX_diff) * LX_ori
#    LY = LY_ori
#elif (LY_ori%2 ==1):
#    ## skew boundary along x direction    
#    LX = LX_ori
#    LY = (LX_ori/LY_diff) * LY_ori
#else:
#    LX = LX_ori
#    LY = LY_ori
    
## Tensors
#D = 2
D = 3
#D = 4
#D = 5
#D = 6
#CHI = D**2
CHI = 3*D**2//2
#TENSOR_DTYPE=np.dtype(float)
TENSOR_DTYPE=np.dtype(complex)

## Debug
#Debug_flag = False
Debug_flag = True

## Simple update
Inverse_lambda_cut = 1e-12

## Full update
Full_Inverse_precision = 1e-12
Full_Convergence_Epsilon = 1e-12
Full_max_iteration = 1000
#Full_max_iteration = 10000
Full_Gauge_Fix = True
Full_Use_FFU = True

## Environment
#Inverse_projector_cut = 1e-12
#Inverse_projector_cut = 1e-8
Inverse_projector_cut = 1e-6
Inverse_Env_cut = Inverse_projector_cut
#CTM_Convergence_Epsilon = 1e-10
#CTM_Convergence_Epsilon = 1e-8
#CTM_Convergence_Epsilon = 1e-7
CTM_Convergence_Epsilon = 1e-6
Max_CTM_Iteration = 100
#Max_CTM_Iteration = 10000
#CTM_Projector_corner = False
CTM_Projector_corner = True
Use_Partial_SVD = False
Use_Interporative_SVD = False
