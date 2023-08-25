# coding:utf-8
import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import scipy.linalg.interpolative
from PEPS_Parameters import *

def Contract_scalar_1x1(\
    t0_2,t1_2,t2_2,\
    t0_1,t1_1,t2_1,\
    t0_0,t1_0,t2_0,\
    o1_1\
    ):
    ##############################
    # ./input/input_Lx1Ly1.dat
    ##############################
    # (o1_1*(t1_1.conj()*((t2_1*(t2_0*t1_0))*(t1_1*((t0_0*t0_1)*(t0_2*(t2_2*t1_2)))))))
    # cpu_cost= 6.04e+10  memory= 4.0004e+08
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o1_1, np.tensordot(
            t1_1.conj(), np.tensordot(
                np.tensordot(
                    t2_1, np.tensordot(
                        t2_0, t1_0, ([1], [0])
                    ), ([1], [0])
                ), np.tensordot(
                    t1_1, np.tensordot(
                        np.tensordot(
                            t0_0, t0_1, ([1], [0])
                        ), np.tensordot(
                            t0_2, np.tensordot(
                                t2_2, t1_2, ([0], [1])
                            ), ([1], [1])
                        ), ([1], [0])
                    ), ([0, 1], [1, 4])
                ), ([0, 1, 3, 4], [5, 0, 3, 1])
            ), ([0, 1, 2, 3], [3, 4, 0, 1])
        ), ([0, 1], [1, 0])
    )

def Contract_scalar_1x2(\
    t0_3,t1_3,t2_3,\
    t0_2,t1_2,t2_2,\
    t0_1,t1_1,t2_1,\
    t0_0,t1_0,t2_0,\
    o1_2,\
    o1_1\
    ):
    ##############################
    # ./input/input_Lx1Ly2.dat
    ##############################
    # (o1_1*(t1_1.conj()*((t0_1*(t0_0*t1_0))*(t1_1*((t2_0*t2_1)*(t2_2*(t1_2.conj()*((o1_2*t1_2)*(t0_2*(t0_3*(t2_3*t1_3)))))))))))
    # cpu_cost= 1.204e+11  memory= 4.0209e+08
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o1_1, np.tensordot(
            t1_1.conj(), np.tensordot(
                np.tensordot(
                    t0_1, np.tensordot(
                        t0_0, t1_0, ([0], [1])
                    ), ([0], [0])
                ), np.tensordot(
                    t1_1, np.tensordot(
                        np.tensordot(
                            t2_0, t2_1, ([0], [1])
                        ), np.tensordot(
                            t2_2, np.tensordot(
                                t1_2.conj(), np.tensordot(
                                    np.tensordot(
                                        o1_2, t1_2, ([0], [4])
                                    ), np.tensordot(
                                        t0_2, np.tensordot(
                                            t0_3, np.tensordot(
                                                t2_3, t1_3, ([0], [1])
                                            ), ([1], [1])
                                        ), ([1], [0])
                                    ), ([1, 2], [1, 4])
                                ), ([0, 1, 4], [4, 6, 0])
                            ), ([0, 2, 3], [5, 2, 0])
                        ), ([1], [0])
                    ), ([1, 2], [4, 1])
                ), ([0, 1, 3, 4], [6, 0, 3, 1])
            ), ([0, 1, 2, 3], [0, 4, 3, 1])
        ), ([0, 1], [1, 0])
    )

def Contract_scalar_1x3(\
    t0_4,t1_4,t2_4,\
    t0_3,t1_3,t2_3,\
    t0_2,t1_2,t2_2,\
    t0_1,t1_1,t2_1,\
    t0_0,t1_0,t2_0,\
    o1_3,\
    o1_2,\
    o1_1\
    ):
    ##############################
    # ./input/input_Lx1Ly3.dat
    ##############################
    # (o1_2*(t1_2*((t2_2*(t2_1*(t1_1*((o1_1*t1_1.conj())*(t0_1*(t0_0*(t2_0*t1_0)))))))*(t1_2.conj()*(t0_2*(t0_3*(t1_3*((o1_3*t1_3.conj())*(t2_3*(t0_4*(t2_4*t1_4)))))))))))
    # cpu_cost= 1.804e+11  memory= 5.0206e+08
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o1_2, np.tensordot(
            t1_2, np.tensordot(
                np.tensordot(
                    t2_2, np.tensordot(
                        t2_1, np.tensordot(
                            t1_1, np.tensordot(
                                np.tensordot(
                                    o1_1, t1_1.conj(), ([1], [4])
                                ), np.tensordot(
                                    t0_1, np.tensordot(
                                        t0_0, np.tensordot(
                                            t2_0, t1_0, ([1], [0])
                                        ), ([0], [1])
                                    ), ([0], [0])
                                ), ([1, 4], [2, 5])
                            ), ([0, 3, 4], [4, 6, 0])
                        ), ([1, 2, 3], [5, 1, 3])
                    ), ([1], [0])
                ), np.tensordot(
                    t1_2.conj(), np.tensordot(
                        t0_2, np.tensordot(
                            t0_3, np.tensordot(
                                t1_3, np.tensordot(
                                    np.tensordot(
                                        o1_3, t1_3.conj(), ([1], [4])
                                    ), np.tensordot(
                                        t2_3, np.tensordot(
                                            t0_4, np.tensordot(
                                                t2_4, t1_4, ([0], [1])
                                            ), ([1], [1])
                                        ), ([0], [1])
                                    ), ([2, 3], [5, 2])
                                ), ([1, 2, 4], [6, 4, 0])
                            ), ([1, 2, 3], [5, 0, 2])
                        ), ([1], [0])
                    ), ([0, 1], [2, 4])
                ), ([0, 2, 4, 5], [6, 0, 1, 3])
            ), ([0, 1, 2, 3], [3, 4, 0, 1])
        ), ([0, 1], [0, 1])
    )

def Contract_scalar_1x4(\
    t0_5,t1_5,t2_5,\
    t0_4,t1_4,t2_4,\
    t0_3,t1_3,t2_3,\
    t0_2,t1_2,t2_2,\
    t0_1,t1_1,t2_1,\
    t0_0,t1_0,t2_0,\
    o1_4,\
    o1_3,\
    o1_2,\
    o1_1\
    ):
    ##############################
    # ./input/input_Lx1Ly4.dat
    ##############################
    # (o1_1*(t1_1.conj()*((t1_0*(t2_0*t2_1))*(t1_1*((t0_0*t0_1)*(t0_2*(t1_2.conj()*((o1_2*t1_2)*(t2_2*(t2_3*(t1_3*((o1_3*t1_3.conj())*(t0_3*(t0_4*(t1_4.conj()*((t1_4*o1_4)*(t2_4*(t0_5*(t1_5*t2_5)))))))))))))))))))
    # cpu_cost= 2.404e+11  memory= 4.0617e+08
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o1_1, np.tensordot(
            t1_1.conj(), np.tensordot(
                np.tensordot(
                    t1_0, np.tensordot(
                        t2_0, t2_1, ([0], [1])
                    ), ([0], [0])
                ), np.tensordot(
                    t1_1, np.tensordot(
                        np.tensordot(
                            t0_0, t0_1, ([1], [0])
                        ), np.tensordot(
                            t0_2, np.tensordot(
                                t1_2.conj(), np.tensordot(
                                    np.tensordot(
                                        o1_2, t1_2, ([0], [4])
                                    ), np.tensordot(
                                        t2_2, np.tensordot(
                                            t2_3, np.tensordot(
                                                t1_3, np.tensordot(
                                                    np.tensordot(
                                                        o1_3, t1_3.conj(), ([1], [4])
                                                    ), np.tensordot(
                                                        t0_3, np.tensordot(
                                                            t0_4, np.tensordot(
                                                                t1_4.conj(), np.tensordot(
                                                                    np.tensordot(
                                                                        t1_4, o1_4, ([4], [0])
                                                                    ), np.tensordot(
                                                                        t2_4, np.tensordot(
                                                                            t0_5, np.tensordot(
                                                                                t1_5, t2_5, ([1], [0])
                                                                            ), ([1], [0])
                                                                        ), ([0], [3])
                                                                    ), ([1, 2], [4, 1])
                                                                ), ([1, 2, 4], [6, 4, 2])
                                                            ), ([1, 2, 3], [5, 2, 0])
                                                        ), ([1], [0])
                                                    ), ([1, 2], [2, 3])
                                                ), ([0, 1, 4], [4, 5, 0])
                                            ), ([0, 2, 3], [5, 0, 2])
                                        ), ([0], [0])
                                    ), ([2, 3], [3, 1])
                                ), ([1, 2, 4], [5, 4, 0])
                            ), ([1, 2, 3], [5, 2, 0])
                        ), ([1], [0])
                    ), ([0, 1], [1, 4])
                ), ([0, 1, 3, 4], [3, 1, 6, 0])
            ), ([0, 1, 2, 3], [3, 4, 1, 0])
        ), ([0, 1], [1, 0])
    )

def Contract_scalar_1x5(\
    t0_6,t1_6,t2_6,\
    t0_5,t1_5,t2_5,\
    t0_4,t1_4,t2_4,\
    t0_3,t1_3,t2_3,\
    t0_2,t1_2,t2_2,\
    t0_1,t1_1,t2_1,\
    t0_0,t1_0,t2_0,\
    o1_5,\
    o1_4,\
    o1_3,\
    o1_2,\
    o1_1\
    ):
    ##############################
    # ./input/input_Lx1Ly5.dat
    ##############################
    # (o1_2*(t1_2.conj()*((t0_2*(t0_1*(t1_1.conj()*((o1_1*t1_1)*(t2_1*(t0_0*(t2_0*t1_0)))))))*(t1_2*(t2_2*(t0_3*(t1_3.conj()*((t1_3*o1_3)*(t2_3*(t0_4*(t1_4.conj()*((o1_4*t1_4)*(t2_4*(t0_5*(t1_5.conj()*((o1_5*t1_5)*(t2_5*(t0_6*(t2_6*t1_6)))))))))))))))))))
    # cpu_cost= 3.004e+11  memory= 5.0206e+08
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o1_2, np.tensordot(
            t1_2.conj(), np.tensordot(
                np.tensordot(
                    t0_2, np.tensordot(
                        t0_1, np.tensordot(
                            t1_1.conj(), np.tensordot(
                                np.tensordot(
                                    o1_1, t1_1, ([0], [4])
                                ), np.tensordot(
                                    t2_1, np.tensordot(
                                        t0_0, np.tensordot(
                                            t2_0, t1_0, ([1], [0])
                                        ), ([0], [1])
                                    ), ([1], [1])
                                ), ([3, 4], [1, 4])
                            ), ([2, 3, 4], [4, 6, 0])
                        ), ([0, 2, 3], [5, 2, 0])
                    ), ([0], [0])
                ), np.tensordot(
                    t1_2, np.tensordot(
                        t2_2, np.tensordot(
                            t0_3, np.tensordot(
                                t1_3.conj(), np.tensordot(
                                    np.tensordot(
                                        t1_3, o1_3, ([4], [0])
                                    ), np.tensordot(
                                        t2_3, np.tensordot(
                                            t0_4, np.tensordot(
                                                t1_4.conj(), np.tensordot(
                                                    np.tensordot(
                                                        o1_4, t1_4, ([0], [4])
                                                    ), np.tensordot(
                                                        t2_4, np.tensordot(
                                                            t0_5, np.tensordot(
                                                                t1_5.conj(), np.tensordot(
                                                                    np.tensordot(
                                                                        o1_5, t1_5, ([0], [4])
                                                                    ), np.tensordot(
                                                                        t2_5, np.tensordot(
                                                                            t0_6, np.tensordot(
                                                                                t2_6, t1_6, ([0], [1])
                                                                            ), ([1], [1])
                                                                        ), ([0], [1])
                                                                    ), ([2, 3], [4, 1])
                                                                ), ([1, 2, 4], [6, 4, 0])
                                                            ), ([1, 2, 3], [5, 2, 0])
                                                        ), ([0], [3])
                                                    ), ([2, 3], [5, 1])
                                                ), ([1, 2, 4], [6, 4, 0])
                                            ), ([1, 2, 3], [5, 2, 0])
                                        ), ([0], [3])
                                    ), ([1, 2], [5, 1])
                                ), ([1, 2, 4], [6, 4, 2])
                            ), ([1, 2, 3], [5, 2, 0])
                        ), ([0], [3])
                    ), ([1, 2], [5, 1])
                ), ([0, 1, 4, 5], [5, 0, 1, 3])
            ), ([0, 1, 2, 3], [0, 4, 3, 1])
        ), ([0, 1], [1, 0])
    )

def Contract_scalar_1x6(\
    t0_7,t1_7,t2_7,\
    t0_6,t1_6,t2_6,\
    t0_5,t1_5,t2_5,\
    t0_4,t1_4,t2_4,\
    t0_3,t1_3,t2_3,\
    t0_2,t1_2,t2_2,\
    t0_1,t1_1,t2_1,\
    t0_0,t1_0,t2_0,\
    o1_6,\
    o1_5,\
    o1_4,\
    o1_3,\
    o1_2,\
    o1_1\
    ):
    ##############################
    # ./input/input_Lx1Ly6.dat
    ##############################
    # (o1_3*(t1_3.conj()*((t0_3*(t2_2*(t1_2*((t1_2.conj()*o1_2)*(t0_2*(t2_1*(t1_1.conj()*((o1_1*t1_1)*(t0_1*(t0_0*(t2_0*t1_0)))))))))))*(t1_3*(t2_3*(t0_4*(t1_4.conj()*((o1_4*t1_4)*(t2_4*(t0_5*(t1_5*((o1_5*t1_5.conj())*(t2_5*(t2_6*(t1_6.conj()*((t1_6*o1_6)*(t0_6*(t0_7*(t2_7*t1_7)))))))))))))))))))
    # cpu_cost= 3.604e+11  memory= 5.041e+08
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o1_3, np.tensordot(
            t1_3.conj(), np.tensordot(
                np.tensordot(
                    t0_3, np.tensordot(
                        t2_2, np.tensordot(
                            t1_2, np.tensordot(
                                np.tensordot(
                                    t1_2.conj(), o1_2, ([4], [1])
                                ), np.tensordot(
                                    t0_2, np.tensordot(
                                        t2_1, np.tensordot(
                                            t1_1.conj(), np.tensordot(
                                                np.tensordot(
                                                    o1_1, t1_1, ([0], [4])
                                                ), np.tensordot(
                                                    t0_1, np.tensordot(
                                                        t0_0, np.tensordot(
                                                            t2_0, t1_0, ([1], [0])
                                                        ), ([0], [1])
                                                    ), ([0], [0])
                                                ), ([1, 4], [1, 4])
                                            ), ([0, 3, 4], [4, 6, 0])
                                        ), ([1, 2, 3], [5, 3, 1])
                                    ), ([0], [3])
                                ), ([0, 3], [2, 4])
                            ), ([0, 3, 4], [4, 6, 2])
                        ), ([1, 2, 3], [5, 1, 3])
                    ), ([0], [3])
                ), np.tensordot(
                    t1_3, np.tensordot(
                        t2_3, np.tensordot(
                            t0_4, np.tensordot(
                                t1_4.conj(), np.tensordot(
                                    np.tensordot(
                                        o1_4, t1_4, ([0], [4])
                                    ), np.tensordot(
                                        t2_4, np.tensordot(
                                            t0_5, np.tensordot(
                                                t1_5, np.tensordot(
                                                    np.tensordot(
                                                        o1_5, t1_5.conj(), ([1], [4])
                                                    ), np.tensordot(
                                                        t2_5, np.tensordot(
                                                            t2_6, np.tensordot(
                                                                t1_6.conj(), np.tensordot(
                                                                    np.tensordot(
                                                                        t1_6, o1_6, ([4], [0])
                                                                    ), np.tensordot(
                                                                        t0_6, np.tensordot(
                                                                            t0_7, np.tensordot(
                                                                                t2_7, t1_7, ([0], [1])
                                                                            ), ([1], [1])
                                                                        ), ([1], [0])
                                                                    ), ([0, 1], [1, 4])
                                                                ), ([0, 1, 4], [4, 6, 2])
                                                            ), ([0, 2, 3], [5, 2, 0])
                                                        ), ([0], [0])
                                                    ), ([2, 3], [3, 2])
                                                ), ([1, 2, 4], [5, 4, 0])
                                            ), ([1, 2, 3], [5, 0, 2])
                                        ), ([0], [3])
                                    ), ([2, 3], [4, 1])
                                ), ([1, 2, 4], [6, 4, 0])
                            ), ([1, 2, 3], [5, 2, 0])
                        ), ([0], [3])
                    ), ([1, 2], [5, 1])
                ), ([0, 1, 3, 4], [5, 0, 3, 1])
            ), ([0, 1, 2, 3], [0, 4, 3, 1])
        ), ([0, 1], [1, 0])
    )

def Contract_scalar_2x1(\
    t0_2,t1_2,t2_2,t3_2,\
    t0_1,t1_1,t2_1,t3_1,\
    t0_0,t1_0,t2_0,t3_0,\
    o1_1,o2_1\
    ):
    ##############################
    # ./input/input_Lx2Ly1.dat
    ##############################
    # (o1_1*(t1_1.conj()*((t0_1*(t0_2*t1_2))*(t1_1*((t0_0*t1_0)*(t2_0*(t2_1.conj()*((o2_1*t2_1)*(t2_2*(t3_0*(t3_1*t3_2)))))))))))
    # cpu_cost= 1.204e+11  memory= 4.0209e+08
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o1_1, np.tensordot(
            t1_1.conj(), np.tensordot(
                np.tensordot(
                    t0_1, np.tensordot(
                        t0_2, t1_2, ([1], [0])
                    ), ([1], [0])
                ), np.tensordot(
                    t1_1, np.tensordot(
                        np.tensordot(
                            t0_0, t1_0, ([0], [1])
                        ), np.tensordot(
                            t2_0, np.tensordot(
                                t2_1.conj(), np.tensordot(
                                    np.tensordot(
                                        o2_1, t2_1, ([0], [4])
                                    ), np.tensordot(
                                        t2_2, np.tensordot(
                                            t3_0, np.tensordot(
                                                t3_1, t3_2, ([0], [1])
                                            ), ([0], [0])
                                        ), ([1], [3])
                                    ), ([2, 3], [1, 4])
                                ), ([1, 2, 4], [4, 6, 0])
                            ), ([0, 2, 3], [5, 3, 1])
                        ), ([1], [0])
                    ), ([2, 3], [4, 1])
                ), ([0, 1, 3, 4], [3, 0, 6, 1])
            ), ([0, 1, 2, 3], [0, 1, 4, 3])
        ), ([0, 1], [1, 0])
    )

def Contract_scalar_2x2(\
    t0_3,t1_3,t2_3,t3_3,\
    t0_2,t1_2,t2_2,t3_2,\
    t0_1,t1_1,t2_1,t3_1,\
    t0_0,t1_0,t2_0,t3_0,\
    o1_2,o2_2,\
    o1_1,o2_1\
    ):
    ##############################
    # ./input/input_Lx2Ly2.dat
    ##############################
    # (o1_2*(t1_2.conj()*((t0_2*(t0_3*t1_3))*(t1_2*((t2_2.conj()*((o2_2*t2_2)*(t2_3*(t3_3*t3_2))))*((t1_1*((t1_1.conj()*o1_1)*(t1_0*(t0_0*t0_1))))*(t2_1.conj()*((o2_1*t2_1)*(t2_0*(t3_0*t3_1))))))))))
    # cpu_cost= 2.2004e+12  memory= 6.0008e+08
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o1_2, np.tensordot(
            t1_2.conj(), np.tensordot(
                np.tensordot(
                    t0_2, np.tensordot(
                        t0_3, t1_3, ([1], [0])
                    ), ([1], [0])
                ), np.tensordot(
                    t1_2, np.tensordot(
                        np.tensordot(
                            t2_2.conj(), np.tensordot(
                                np.tensordot(
                                    o2_2, t2_2, ([0], [4])
                                ), np.tensordot(
                                    t2_3, np.tensordot(
                                        t3_3, t3_2, ([1], [0])
                                    ), ([1], [0])
                                ), ([2, 3], [1, 4])
                            ), ([1, 2, 4], [4, 6, 0])
                        ), np.tensordot(
                            np.tensordot(
                                t1_1, np.tensordot(
                                    np.tensordot(
                                        t1_1.conj(), o1_1, ([4], [1])
                                    ), np.tensordot(
                                        t1_0, np.tensordot(
                                            t0_0, t0_1, ([1], [0])
                                        ), ([1], [0])
                                    ), ([0, 3], [5, 2])
                                ), ([0, 3, 4], [6, 4, 2])
                            ), np.tensordot(
                                t2_1.conj(), np.tensordot(
                                    np.tensordot(
                                        o2_1, t2_1, ([0], [4])
                                    ), np.tensordot(
                                        t2_0, np.tensordot(
                                            t3_0, t3_1, ([0], [1])
                                        ), ([0], [0])
                                    ), ([3, 4], [4, 1])
                                ), ([2, 3, 4], [6, 4, 0])
                            ), ([1, 3, 4], [2, 0, 4])
                        ), ([1, 3, 5], [3, 4, 5])
                    ), ([2, 3], [1, 3])
                ), ([0, 1, 3, 4], [6, 0, 4, 1])
            ), ([0, 1, 2, 3], [0, 1, 3, 4])
        ), ([0, 1], [1, 0])
    )

def Contract_scalar_2x3(\
    t0_4,t1_4,t2_4,t3_4,\
    t0_3,t1_3,t2_3,t3_3,\
    t0_2,t1_2,t2_2,t3_2,\
    t0_1,t1_1,t2_1,t3_1,\
    t0_0,t1_0,t2_0,t3_0,\
    o1_3,o2_3,\
    o1_2,o2_2,\
    o1_1,o2_1\
    ):
    ##############################
    # ./input/input_Lx2Ly3.dat
    ##############################
    # (o2_1*(t2_1.conj()*((t2_0*(t3_0*t3_1))*(t2_1*((t1_1*((o1_1*t1_1.conj())*(t1_0*(t0_0*t0_1))))*(t3_2*(t2_2.conj()*((o2_2*t2_2)*(t1_2.conj()*((t1_2*o1_2)*(t0_2*((t2_3*((t2_3.conj()*o2_3)*(t2_4*(t3_4*t3_3))))*(t1_3.conj()*((o1_3*t1_3)*(t1_4*(t0_4*t0_3))))))))))))))))
    # cpu_cost= 1.22004e+13  memory= 3.02011e+10
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o2_1, np.tensordot(
            t2_1.conj(), np.tensordot(
                np.tensordot(
                    t2_0, np.tensordot(
                        t3_0, t3_1, ([0], [1])
                    ), ([0], [0])
                ), np.tensordot(
                    t2_1, np.tensordot(
                        np.tensordot(
                            t1_1, np.tensordot(
                                np.tensordot(
                                    o1_1, t1_1.conj(), ([1], [4])
                                ), np.tensordot(
                                    t1_0, np.tensordot(
                                        t0_0, t0_1, ([1], [0])
                                    ), ([1], [0])
                                ), ([1, 4], [5, 2])
                            ), ([0, 3, 4], [6, 4, 0])
                        ), np.tensordot(
                            t3_2, np.tensordot(
                                t2_2.conj(), np.tensordot(
                                    np.tensordot(
                                        o2_2, t2_2, ([0], [4])
                                    ), np.tensordot(
                                        t1_2.conj(), np.tensordot(
                                            np.tensordot(
                                                t1_2, o1_2, ([4], [0])
                                            ), np.tensordot(
                                                t0_2, np.tensordot(
                                                    np.tensordot(
                                                        t2_3, np.tensordot(
                                                            np.tensordot(
                                                                t2_3.conj(), o2_3, ([4], [1])
                                                            ), np.tensordot(
                                                                t2_4, np.tensordot(
                                                                    t3_4, t3_3, ([1], [0])
                                                                ), ([1], [0])
                                                            ), ([1, 2], [2, 5])
                                                        ), ([1, 2, 4], [4, 6, 2])
                                                    ), np.tensordot(
                                                        t1_3.conj(), np.tensordot(
                                                            np.tensordot(
                                                                o1_3, t1_3, ([0], [4])
                                                            ), np.tensordot(
                                                                t1_4, np.tensordot(
                                                                    t0_4, t0_3, ([0], [1])
                                                                ), ([0], [0])
                                                            ), ([1, 2], [4, 1])
                                                        ), ([0, 1, 4], [6, 4, 0])
                                                    ), ([0, 2, 4], [2, 0, 4])
                                                ), ([1], [5])
                                            ), ([0, 1], [1, 7])
                                        ), ([0, 1, 4], [4, 8, 2])
                                    ), ([1, 2], [2, 5])
                                ), ([0, 1, 4], [3, 7, 0])
                            ), ([0, 2, 3], [7, 2, 0])
                        ), ([0, 2, 5], [4, 3, 5])
                    ), ([0, 1], [0, 5])
                ), ([0, 1, 3, 4], [4, 1, 5, 0])
            ), ([0, 1, 2, 3], [3, 4, 1, 0])
        ), ([0, 1], [1, 0])
    )

def Contract_scalar_2x4(\
    t0_5,t1_5,t2_5,t3_5,\
    t0_4,t1_4,t2_4,t3_4,\
    t0_3,t1_3,t2_3,t3_3,\
    t0_2,t1_2,t2_2,t3_2,\
    t0_1,t1_1,t2_1,t3_1,\
    t0_0,t1_0,t2_0,t3_0,\
    o1_4,o2_4,\
    o1_3,o2_3,\
    o1_2,o2_2,\
    o1_1,o2_1\
    ):
    ##############################
    # ./input/input_Lx2Ly4.dat
    ##############################
    # (o2_4*(t2_4*((t2_5*(t3_5*t3_4))*(t2_4.conj()*((t1_4*((o1_4*t1_4.conj())*(t0_4*(t0_5*t1_5))))*(t0_3*(t1_3.conj()*((o1_3*t1_3)*(t2_3.conj()*((o2_3*t2_3)*(t3_3*(t0_2*(t1_2.conj()*((o1_2*t1_2)*(t2_2.conj()*((t2_2*o2_2)*(t3_2*((t1_1.conj()*((o1_1*t1_1)*(t1_0*(t0_0*t0_1))))*(t2_1*((o2_1*t2_1.conj())*(t3_1*(t3_0*t2_0))))))))))))))))))))))
    # cpu_cost= 2.22004e+13  memory= 3.02032e+10
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o2_4, np.tensordot(
            t2_4, np.tensordot(
                np.tensordot(
                    t2_5, np.tensordot(
                        t3_5, t3_4, ([1], [0])
                    ), ([1], [0])
                ), np.tensordot(
                    t2_4.conj(), np.tensordot(
                        np.tensordot(
                            t1_4, np.tensordot(
                                np.tensordot(
                                    o1_4, t1_4.conj(), ([1], [4])
                                ), np.tensordot(
                                    t0_4, np.tensordot(
                                        t0_5, t1_5, ([1], [0])
                                    ), ([1], [0])
                                ), ([1, 2], [2, 5])
                            ), ([0, 1, 4], [4, 6, 0])
                        ), np.tensordot(
                            t0_3, np.tensordot(
                                t1_3.conj(), np.tensordot(
                                    np.tensordot(
                                        o1_3, t1_3, ([0], [4])
                                    ), np.tensordot(
                                        t2_3.conj(), np.tensordot(
                                            np.tensordot(
                                                o2_3, t2_3, ([0], [4])
                                            ), np.tensordot(
                                                t3_3, np.tensordot(
                                                    t0_2, np.tensordot(
                                                        t1_2.conj(), np.tensordot(
                                                            np.tensordot(
                                                                o1_2, t1_2, ([0], [4])
                                                            ), np.tensordot(
                                                                t2_2.conj(), np.tensordot(
                                                                    np.tensordot(
                                                                        t2_2, o2_2, ([4], [0])
                                                                    ), np.tensordot(
                                                                        t3_2, np.tensordot(
                                                                            np.tensordot(
                                                                                t1_1.conj(), np.tensordot(
                                                                                    np.tensordot(
                                                                                        o1_1, t1_1, ([0], [4])
                                                                                    ), np.tensordot(
                                                                                        t1_0, np.tensordot(
                                                                                            t0_0, t0_1, ([1], [0])
                                                                                        ), ([1], [0])
                                                                                    ), ([1, 4], [4, 1])
                                                                                ), ([0, 3, 4], [6, 4, 0])
                                                                            ), np.tensordot(
                                                                                t2_1, np.tensordot(
                                                                                    np.tensordot(
                                                                                        o2_1, t2_1.conj(), ([1], [4])
                                                                                    ), np.tensordot(
                                                                                        t3_1, np.tensordot(
                                                                                            t3_0, t2_0, ([1], [0])
                                                                                        ), ([1], [0])
                                                                                    ), ([3, 4], [2, 5])
                                                                                ), ([2, 3, 4], [4, 6, 0])
                                                                            ), ([1, 3, 4], [2, 0, 5])
                                                                        ), ([1], [5])
                                                                    ), ([2, 3], [1, 6])
                                                                ), ([2, 3, 4], [4, 8, 2])
                                                            ), ([3, 4], [2, 6])
                                                        ), ([2, 3, 4], [3, 7, 0])
                                                    ), ([0, 2, 3], [7, 2, 0])
                                                ), ([1], [5])
                                            ), ([3, 4], [1, 7])
                                        ), ([2, 3, 4], [4, 8, 0])
                                    ), ([3, 4], [2, 7])
                                ), ([2, 3, 4], [3, 8, 0])
                            ), ([0, 2, 3], [7, 2, 0])
                        ), ([1, 3, 4], [2, 1, 0])
                    ), ([0, 3], [1, 3])
                ), ([0, 2, 3, 5], [4, 0, 6, 1])
            ), ([0, 1, 2, 3], [3, 0, 1, 4])
        ), ([0, 1], [0, 1])
    )

def Contract_scalar_2x5(\
    t0_6,t1_6,t2_6,t3_6,\
    t0_5,t1_5,t2_5,t3_5,\
    t0_4,t1_4,t2_4,t3_4,\
    t0_3,t1_3,t2_3,t3_3,\
    t0_2,t1_2,t2_2,t3_2,\
    t0_1,t1_1,t2_1,t3_1,\
    t0_0,t1_0,t2_0,t3_0,\
    o1_5,o2_5,\
    o1_4,o2_4,\
    o1_3,o2_3,\
    o1_2,o2_2,\
    o1_1,o2_1\
    ):
    ##############################
    # ./input/input_Lx2Ly5.dat
    ##############################
    # (o1_2*(t1_2*((t0_2*((t1_1*((t1_1.conj()*o1_1)*(t1_0*(t0_0*t0_1))))*(t2_1*((o2_1*t2_1.conj())*(t2_0*(t3_0*t3_1))))))*(t1_2.conj()*(t2_2.conj()*((o2_2*t2_2)*(t3_2*(t0_3*(t1_3.conj()*((t1_3*o1_3)*(t2_3*((t2_3.conj()*o2_3)*(t3_3*(t0_4*(t1_4*((t1_4.conj()*o1_4)*(t2_4*((t2_4.conj()*o2_4)*(t3_4*((t2_5*((o2_5*t2_5.conj())*(t2_6*(t3_6*t3_5))))*(t1_5.conj()*((o1_5*t1_5)*(t0_5*(t0_6*t1_6))))))))))))))))))))))))
    # cpu_cost= 3.22004e+13  memory= 4.00042e+10
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o1_2, np.tensordot(
            t1_2, np.tensordot(
                np.tensordot(
                    t0_2, np.tensordot(
                        np.tensordot(
                            t1_1, np.tensordot(
                                np.tensordot(
                                    t1_1.conj(), o1_1, ([4], [1])
                                ), np.tensordot(
                                    t1_0, np.tensordot(
                                        t0_0, t0_1, ([1], [0])
                                    ), ([1], [0])
                                ), ([0, 3], [5, 2])
                            ), ([0, 3, 4], [6, 4, 2])
                        ), np.tensordot(
                            t2_1, np.tensordot(
                                np.tensordot(
                                    o2_1, t2_1.conj(), ([1], [4])
                                ), np.tensordot(
                                    t2_0, np.tensordot(
                                        t3_0, t3_1, ([0], [1])
                                    ), ([0], [0])
                                ), ([3, 4], [5, 2])
                            ), ([2, 3, 4], [6, 4, 0])
                        ), ([1, 3, 4], [0, 2, 4])
                    ), ([0], [2])
                ), np.tensordot(
                    t1_2.conj(), np.tensordot(
                        t2_2.conj(), np.tensordot(
                            np.tensordot(
                                o2_2, t2_2, ([0], [4])
                            ), np.tensordot(
                                t3_2, np.tensordot(
                                    t0_3, np.tensordot(
                                        t1_3.conj(), np.tensordot(
                                            np.tensordot(
                                                t1_3, o1_3, ([4], [0])
                                            ), np.tensordot(
                                                t2_3, np.tensordot(
                                                    np.tensordot(
                                                        t2_3.conj(), o2_3, ([4], [1])
                                                    ), np.tensordot(
                                                        t3_3, np.tensordot(
                                                            t0_4, np.tensordot(
                                                                t1_4, np.tensordot(
                                                                    np.tensordot(
                                                                        t1_4.conj(), o1_4, ([4], [1])
                                                                    ), np.tensordot(
                                                                        t2_4, np.tensordot(
                                                                            np.tensordot(
                                                                                t2_4.conj(), o2_4, ([4], [1])
                                                                            ), np.tensordot(
                                                                                t3_4, np.tensordot(
                                                                                    np.tensordot(
                                                                                        t2_5, np.tensordot(
                                                                                            np.tensordot(
                                                                                                o2_5, t2_5.conj(), ([1], [4])
                                                                                            ), np.tensordot(
                                                                                                t2_6, np.tensordot(
                                                                                                    t3_6, t3_5, ([1], [0])
                                                                                                ), ([1], [0])
                                                                                            ), ([2, 3], [2, 5])
                                                                                        ), ([1, 2, 4], [4, 6, 0])
                                                                                    ), np.tensordot(
                                                                                        t1_5.conj(), np.tensordot(
                                                                                            np.tensordot(
                                                                                                o1_5, t1_5, ([0], [4])
                                                                                            ), np.tensordot(
                                                                                                t0_5, np.tensordot(
                                                                                                    t0_6, t1_6, ([1], [0])
                                                                                                ), ([1], [0])
                                                                                            ), ([1, 2], [1, 4])
                                                                                        ), ([0, 1, 4], [4, 6, 0])
                                                                                    ), ([0, 2, 4], [2, 0, 5])
                                                                                ), ([0], [2])
                                                                            ), ([1, 2], [4, 2])
                                                                        ), ([1, 2, 4], [5, 4, 2])
                                                                    ), ([1, 2], [5, 2])
                                                                ), ([1, 2, 4], [7, 3, 2])
                                                            ), ([1, 2, 3], [7, 0, 2])
                                                        ), ([0], [5])
                                                    ), ([1, 2], [7, 2])
                                                ), ([1, 2, 4], [8, 4, 2])
                                            ), ([1, 2], [6, 0])
                                        ), ([1, 2, 4], [8, 4, 2])
                                    ), ([1, 2, 3], [7, 2, 0])
                                ), ([0], [5])
                            ), ([2, 3], [6, 1])
                        ), ([1, 2, 4], [8, 4, 0])
                    ), ([1, 2], [6, 0])
                ), ([0, 2, 4, 5, 6, 7], [7, 0, 1, 5, 3, 6])
            ), ([0, 1, 2, 3], [0, 4, 3, 1])
        ), ([0, 1], [0, 1])
    )

def Contract_scalar_3x1(\
    t0_2,t1_2,t2_2,t3_2,t4_2,\
    t0_1,t1_1,t2_1,t3_1,t4_1,\
    t0_0,t1_0,t2_0,t3_0,t4_0,\
    o1_1,o2_1,o3_1\
    ):
    ##############################
    # ./input/input_Lx3Ly1.dat
    ##############################
    # (o2_1*(t2_1*((t2_2*(t1_2*(t1_1*((o1_1*t1_1.conj())*(t1_0*(t0_0*(t0_2*t0_1)))))))*(t2_1.conj()*(t2_0*(t3_2*(t3_1*((o3_1*t3_1.conj())*(t3_0*(t4_0*(t4_2*t4_1)))))))))))
    # cpu_cost= 1.804e+11  memory= 5.0206e+08
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o2_1, np.tensordot(
            t2_1, np.tensordot(
                np.tensordot(
                    t2_2, np.tensordot(
                        t1_2, np.tensordot(
                            t1_1, np.tensordot(
                                np.tensordot(
                                    o1_1, t1_1.conj(), ([1], [4])
                                ), np.tensordot(
                                    t1_0, np.tensordot(
                                        t0_0, np.tensordot(
                                            t0_2, t0_1, ([0], [1])
                                        ), ([1], [1])
                                    ), ([1], [0])
                                ), ([1, 4], [5, 2])
                            ), ([0, 3, 4], [6, 4, 0])
                        ), ([0, 2, 3], [5, 0, 2])
                    ), ([0], [0])
                ), np.tensordot(
                    t2_1.conj(), np.tensordot(
                        t2_0, np.tensordot(
                            t3_2, np.tensordot(
                                t3_1, np.tensordot(
                                    np.tensordot(
                                        o3_1, t3_1.conj(), ([1], [4])
                                    ), np.tensordot(
                                        t3_0, np.tensordot(
                                            t4_0, np.tensordot(
                                                t4_2, t4_1, ([1], [0])
                                            ), ([0], [1])
                                        ), ([0], [0])
                                    ), ([3, 4], [5, 2])
                                ), ([2, 3, 4], [6, 4, 0])
                            ), ([1, 2, 3], [5, 1, 3])
                        ), ([0], [3])
                    ), ([2, 3], [5, 2])
                ), ([0, 2, 4, 5], [5, 1, 0, 3])
            ), ([0, 1, 2, 3], [1, 0, 4, 3])
        ), ([0, 1], [0, 1])
    )

def Contract_scalar_3x2(\
    t0_3,t1_3,t2_3,t3_3,t4_3,\
    t0_2,t1_2,t2_2,t3_2,t4_2,\
    t0_1,t1_1,t2_1,t3_1,t4_1,\
    t0_0,t1_0,t2_0,t3_0,t4_0,\
    o1_2,o2_2,o3_2,\
    o1_1,o2_1,o3_1\
    ):
    ##############################
    # ./input/input_Lx3Ly2.dat
    ##############################
    # (o2_1*(t2_1.conj()*((t2_0*((t1_2.conj()*((o1_2*t1_2)*(t1_3*(t0_3*t0_2))))*(t1_1*((o1_1*t1_1.conj())*(t0_1*(t0_0*t1_0))))))*(t2_1*(t2_2.conj()*((o2_2*t2_2)*(t2_3*((t3_2*((t3_2.conj()*o3_2)*(t4_2*(t4_3*t3_3))))*(t3_1.conj()*((t3_1*o3_1)*(t4_1*(t4_0*t3_0))))))))))))
    # cpu_cost= 1.22004e+13  memory= 4.00001e+10
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o2_1, np.tensordot(
            t2_1.conj(), np.tensordot(
                np.tensordot(
                    t2_0, np.tensordot(
                        np.tensordot(
                            t1_2.conj(), np.tensordot(
                                np.tensordot(
                                    o1_2, t1_2, ([0], [4])
                                ), np.tensordot(
                                    t1_3, np.tensordot(
                                        t0_3, t0_2, ([0], [1])
                                    ), ([0], [0])
                                ), ([1, 2], [4, 1])
                            ), ([0, 1, 4], [6, 4, 0])
                        ), np.tensordot(
                            t1_1, np.tensordot(
                                np.tensordot(
                                    o1_1, t1_1.conj(), ([1], [4])
                                ), np.tensordot(
                                    t0_1, np.tensordot(
                                        t0_0, t1_0, ([0], [1])
                                    ), ([0], [0])
                                ), ([1, 4], [2, 5])
                            ), ([0, 3, 4], [4, 6, 0])
                        ), ([1, 3, 5], [2, 0, 4])
                    ), ([1], [5])
                ), np.tensordot(
                    t2_1, np.tensordot(
                        t2_2.conj(), np.tensordot(
                            np.tensordot(
                                o2_2, t2_2, ([0], [4])
                            ), np.tensordot(
                                t2_3, np.tensordot(
                                    np.tensordot(
                                        t3_2, np.tensordot(
                                            np.tensordot(
                                                t3_2.conj(), o3_2, ([4], [1])
                                            ), np.tensordot(
                                                t4_2, np.tensordot(
                                                    t4_3, t3_3, ([0], [1])
                                                ), ([0], [0])
                                            ), ([1, 2], [5, 2])
                                        ), ([1, 2, 4], [6, 4, 2])
                                    ), np.tensordot(
                                        t3_1.conj(), np.tensordot(
                                            np.tensordot(
                                                t3_1, o3_1, ([4], [0])
                                            ), np.tensordot(
                                                t4_1, np.tensordot(
                                                    t4_0, t3_0, ([1], [0])
                                                ), ([1], [0])
                                            ), ([2, 3], [1, 4])
                                        ), ([2, 3, 4], [4, 6, 2])
                                    ), ([1, 3, 4], [3, 1, 4])
                                ), ([1], [2])
                            ), ([2, 3], [1, 3])
                        ), ([1, 2, 4], [4, 5, 0])
                    ), ([1, 2], [3, 6])
                ), ([0, 1, 3, 4, 5, 6], [8, 1, 3, 5, 6, 0])
            ), ([0, 1, 2, 3], [1, 3, 4, 0])
        ), ([0, 1], [1, 0])
    )

def Contract_scalar_3x3(\
    t0_4,t1_4,t2_4,t3_4,t4_4,\
    t0_3,t1_3,t2_3,t3_3,t4_3,\
    t0_2,t1_2,t2_2,t3_2,t4_2,\
    t0_1,t1_1,t2_1,t3_1,t4_1,\
    t0_0,t1_0,t2_0,t3_0,t4_0,\
    o1_3,o2_3,o3_3,\
    o1_2,o2_2,o3_2,\
    o1_1,o2_1,o3_1\
    ):
    ##############################
    # ./input/input_Lx3Ly3.dat
    ##############################
    # (o3_1*(t3_1.conj()*((t4_1*(t4_0*t3_0))*(t3_1*(t2_0*(t2_1*((t2_1.conj()*o2_1)*((t1_1.conj()*((o1_1*t1_1)*(t0_1*(t0_0*t1_0))))*(t0_2*(t1_2*((t1_2.conj()*o1_2)*(t2_2*((t2_2.conj()*o2_2)*(t3_2*((o3_2*t3_2.conj())*(t4_2*((t1_3*((o1_3*t1_3.conj())*(t0_3*(t0_4*t1_4))))*(t2_3.conj()*((t2_3*o2_3)*(t2_4*(t3_3*((t3_3.conj()*o3_3)*(t3_4*(t4_4*t4_3))))))))))))))))))))))))
    # cpu_cost= 1.6102e+15  memory= 3.0002e+12
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o3_1, np.tensordot(
            t3_1.conj(), np.tensordot(
                np.tensordot(
                    t4_1, np.tensordot(
                        t4_0, t3_0, ([1], [0])
                    ), ([1], [0])
                ), np.tensordot(
                    t3_1, np.tensordot(
                        t2_0, np.tensordot(
                            t2_1, np.tensordot(
                                np.tensordot(
                                    t2_1.conj(), o2_1, ([4], [1])
                                ), np.tensordot(
                                    np.tensordot(
                                        t1_1.conj(), np.tensordot(
                                            np.tensordot(
                                                o1_1, t1_1, ([0], [4])
                                            ), np.tensordot(
                                                t0_1, np.tensordot(
                                                    t0_0, t1_0, ([0], [1])
                                                ), ([0], [0])
                                            ), ([1, 4], [1, 4])
                                        ), ([0, 3, 4], [4, 6, 0])
                                    ), np.tensordot(
                                        t0_2, np.tensordot(
                                            t1_2, np.tensordot(
                                                np.tensordot(
                                                    t1_2.conj(), o1_2, ([4], [1])
                                                ), np.tensordot(
                                                    t2_2, np.tensordot(
                                                        np.tensordot(
                                                            t2_2.conj(), o2_2, ([4], [1])
                                                        ), np.tensordot(
                                                            t3_2, np.tensordot(
                                                                np.tensordot(
                                                                    o3_2, t3_2.conj(), ([1], [4])
                                                                ), np.tensordot(
                                                                    t4_2, np.tensordot(
                                                                        np.tensordot(
                                                                            t1_3, np.tensordot(
                                                                                np.tensordot(
                                                                                    o1_3, t1_3.conj(), ([1], [4])
                                                                                ), np.tensordot(
                                                                                    t0_3, np.tensordot(
                                                                                        t0_4, t1_4, ([1], [0])
                                                                                    ), ([1], [0])
                                                                                ), ([1, 2], [2, 5])
                                                                            ), ([0, 1, 4], [4, 6, 0])
                                                                        ), np.tensordot(
                                                                            t2_3.conj(), np.tensordot(
                                                                                np.tensordot(
                                                                                    t2_3, o2_3, ([4], [0])
                                                                                ), np.tensordot(
                                                                                    t2_4, np.tensordot(
                                                                                        t3_3, np.tensordot(
                                                                                            np.tensordot(
                                                                                                t3_3.conj(), o3_3, ([4], [1])
                                                                                            ), np.tensordot(
                                                                                                t3_4, np.tensordot(
                                                                                                    t4_4, t4_3, ([1], [0])
                                                                                                ), ([1], [0])
                                                                                            ), ([1, 2], [2, 5])
                                                                                        ), ([1, 2, 4], [4, 6, 2])
                                                                                    ), ([1], [4])
                                                                                ), ([1, 2], [1, 3])
                                                                            ), ([1, 2, 4], [4, 6, 2])
                                                                        ), ([0, 2, 5], [2, 0, 4])
                                                                    ), ([0], [7])
                                                                ), ([2, 3], [9, 2])
                                                            ), ([1, 2, 4], [10, 4, 0])
                                                        ), ([1, 2], [8, 2])
                                                    ), ([1, 2, 4], [10, 3, 2])
                                                ), ([1, 2], [8, 2])
                                            ), ([1, 2, 4], [9, 3, 2])
                                        ), ([1, 2, 3], [9, 0, 2])
                                    ), ([0, 2, 4], [2, 1, 0])
                                ), ([0, 1], [0, 4])
                            ), ([0, 1, 4], [3, 5, 2])
                        ), ([1, 2, 3], [4, 1, 3])
                    ), ([0, 1], [1, 3])
                ), ([0, 1, 3, 4], [6, 0, 3, 1])
            ), ([0, 1, 2, 3], [3, 4, 0, 1])
        ), ([0, 1], [1, 0])
    )

def Contract_scalar_3x4(\
    t0_5,t1_5,t2_5,t3_5,t4_5,\
    t0_4,t1_4,t2_4,t3_4,t4_4,\
    t0_3,t1_3,t2_3,t3_3,t4_3,\
    t0_2,t1_2,t2_2,t3_2,t4_2,\
    t0_1,t1_1,t2_1,t3_1,t4_1,\
    t0_0,t1_0,t2_0,t3_0,t4_0,\
    o1_4,o2_4,o3_4,\
    o1_3,o2_3,o3_3,\
    o1_2,o2_2,o3_2,\
    o1_1,o2_1,o3_1\
    ):
    ##############################
    # ./input/input_Lx3Ly4.dat
    ##############################
    # (o2_2*(t2_2*((t1_2.conj()*((t1_2*o1_2)*(t0_2*((t3_1.conj()*((t3_1*o3_1)*(t4_1*(t4_0*t3_0))))*(t2_1*((t2_1.conj()*o2_1)*(t2_0*(t1_1*((o1_1*t1_1.conj())*(t0_1*(t0_0*t1_0)))))))))))*(t2_2.conj()*(t3_2*((o3_2*t3_2.conj())*(t4_2*(t0_3*(t1_3.conj()*((t1_3*o1_3)*(t2_3*((t2_3.conj()*o2_3)*(t3_3*((o3_3*t3_3.conj())*(t4_3*((t1_4*((t1_4.conj()*o1_4)*(t0_4*(t0_5*t1_5))))*(t2_4.conj()*((o2_4*t2_4)*(t2_5*(t3_4.conj()*((o3_4*t3_4)*(t4_4*(t4_5*t3_5)))))))))))))))))))))))
    # cpu_cost= 3.0102e+15  memory= 5e+12
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o2_2, np.tensordot(
            t2_2, np.tensordot(
                np.tensordot(
                    t1_2.conj(), np.tensordot(
                        np.tensordot(
                            t1_2, o1_2, ([4], [0])
                        ), np.tensordot(
                            t0_2, np.tensordot(
                                np.tensordot(
                                    t3_1.conj(), np.tensordot(
                                        np.tensordot(
                                            t3_1, o3_1, ([4], [0])
                                        ), np.tensordot(
                                            t4_1, np.tensordot(
                                                t4_0, t3_0, ([1], [0])
                                            ), ([1], [0])
                                        ), ([2, 3], [1, 4])
                                    ), ([2, 3, 4], [4, 6, 2])
                                ), np.tensordot(
                                    t2_1, np.tensordot(
                                        np.tensordot(
                                            t2_1.conj(), o2_1, ([4], [1])
                                        ), np.tensordot(
                                            t2_0, np.tensordot(
                                                t1_1, np.tensordot(
                                                    np.tensordot(
                                                        o1_1, t1_1.conj(), ([1], [4])
                                                    ), np.tensordot(
                                                        t0_1, np.tensordot(
                                                            t0_0, t1_0, ([0], [1])
                                                        ), ([0], [0])
                                                    ), ([1, 4], [2, 5])
                                                ), ([0, 3, 4], [4, 6, 0])
                                            ), ([1], [5])
                                        ), ([0, 3], [6, 2])
                                    ), ([0, 3, 4], [6, 4, 2])
                                ), ([0, 2, 5], [3, 1, 4])
                            ), ([0], [7])
                        ), ([0, 3], [1, 8])
                    ), ([0, 3, 4], [4, 10, 2])
                ), np.tensordot(
                    t2_2.conj(), np.tensordot(
                        t3_2, np.tensordot(
                            np.tensordot(
                                o3_2, t3_2.conj(), ([1], [4])
                            ), np.tensordot(
                                t4_2, np.tensordot(
                                    t0_3, np.tensordot(
                                        t1_3.conj(), np.tensordot(
                                            np.tensordot(
                                                t1_3, o1_3, ([4], [0])
                                            ), np.tensordot(
                                                t2_3, np.tensordot(
                                                    np.tensordot(
                                                        t2_3.conj(), o2_3, ([4], [1])
                                                    ), np.tensordot(
                                                        t3_3, np.tensordot(
                                                            np.tensordot(
                                                                o3_3, t3_3.conj(), ([1], [4])
                                                            ), np.tensordot(
                                                                t4_3, np.tensordot(
                                                                    np.tensordot(
                                                                        t1_4, np.tensordot(
                                                                            np.tensordot(
                                                                                t1_4.conj(), o1_4, ([4], [1])
                                                                            ), np.tensordot(
                                                                                t0_4, np.tensordot(
                                                                                    t0_5, t1_5, ([1], [0])
                                                                                ), ([1], [0])
                                                                            ), ([0, 1], [2, 5])
                                                                        ), ([0, 1, 4], [4, 6, 2])
                                                                    ), np.tensordot(
                                                                        t2_4.conj(), np.tensordot(
                                                                            np.tensordot(
                                                                                o2_4, t2_4, ([0], [4])
                                                                            ), np.tensordot(
                                                                                t2_5, np.tensordot(
                                                                                    t3_4.conj(), np.tensordot(
                                                                                        np.tensordot(
                                                                                            o3_4, t3_4, ([0], [4])
                                                                                        ), np.tensordot(
                                                                                            t4_4, np.tensordot(
                                                                                                t4_5, t3_5, ([0], [1])
                                                                                            ), ([0], [0])
                                                                                        ), ([2, 3], [4, 1])
                                                                                    ), ([1, 2, 4], [6, 4, 0])
                                                                                ), ([1], [5])
                                                                            ), ([2, 3], [1, 5])
                                                                        ), ([1, 2, 4], [4, 5, 0])
                                                                    ), ([0, 2, 5], [2, 0, 4])
                                                                ), ([0], [7])
                                                            ), ([2, 3], [8, 2])
                                                        ), ([1, 2, 4], [10, 4, 0])
                                                    ), ([1, 2], [8, 2])
                                                ), ([1, 2, 4], [10, 3, 2])
                                            ), ([1, 2], [7, 0])
                                        ), ([1, 2, 4], [9, 4, 2])
                                    ), ([1, 2, 3], [9, 2, 0])
                                ), ([0], [7])
                            ), ([2, 3], [9, 2])
                        ), ([1, 2, 4], [10, 4, 0])
                    ), ([1, 2], [9, 2])
                ), ([0, 1, 2, 4, 5, 6, 7, 9], [8, 0, 9, 7, 5, 4, 6, 1])
            ), ([0, 1, 2, 3], [0, 4, 3, 1])
        ), ([0, 1], [0, 1])
    )

def Contract_scalar_4x1(\
    t0_2,t1_2,t2_2,t3_2,t4_2,t5_2,\
    t0_1,t1_1,t2_1,t3_1,t4_1,t5_1,\
    t0_0,t1_0,t2_0,t3_0,t4_0,t5_0,\
    o1_1,o2_1,o3_1,o4_1\
    ):
    ##############################
    # ./input/input_Lx4Ly1.dat
    ##############################
    # (o1_1*(t1_1.conj()*((t1_2*(t0_2*t0_1))*(t1_1*((t0_0*t1_0)*(t2_0*(t2_1.conj()*((o2_1*t2_1)*(t2_2*(t3_0*(t3_1*((o3_1*t3_1.conj())*(t3_2*(t4_2*(t4_1.conj()*((t4_1*o4_1)*(t4_0*(t5_0*(t5_2*t5_1)))))))))))))))))))
    # cpu_cost= 2.404e+11  memory= 4.0617e+08
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o1_1, np.tensordot(
            t1_1.conj(), np.tensordot(
                np.tensordot(
                    t1_2, np.tensordot(
                        t0_2, t0_1, ([0], [1])
                    ), ([0], [0])
                ), np.tensordot(
                    t1_1, np.tensordot(
                        np.tensordot(
                            t0_0, t1_0, ([0], [1])
                        ), np.tensordot(
                            t2_0, np.tensordot(
                                t2_1.conj(), np.tensordot(
                                    np.tensordot(
                                        o2_1, t2_1, ([0], [4])
                                    ), np.tensordot(
                                        t2_2, np.tensordot(
                                            t3_0, np.tensordot(
                                                t3_1, np.tensordot(
                                                    np.tensordot(
                                                        o3_1, t3_1.conj(), ([1], [4])
                                                    ), np.tensordot(
                                                        t3_2, np.tensordot(
                                                            t4_2, np.tensordot(
                                                                t4_1.conj(), np.tensordot(
                                                                    np.tensordot(
                                                                        t4_1, o4_1, ([4], [0])
                                                                    ), np.tensordot(
                                                                        t4_0, np.tensordot(
                                                                            t5_0, np.tensordot(
                                                                                t5_2, t5_1, ([1], [0])
                                                                            ), ([0], [1])
                                                                        ), ([0], [0])
                                                                    ), ([2, 3], [4, 1])
                                                                ), ([2, 3, 4], [6, 4, 2])
                                                            ), ([1, 2, 3], [5, 3, 1])
                                                        ), ([1], [0])
                                                    ), ([2, 3], [2, 3])
                                                ), ([1, 2, 4], [4, 5, 0])
                                            ), ([0, 2, 3], [5, 1, 3])
                                        ), ([1], [3])
                                    ), ([2, 3], [1, 4])
                                ), ([1, 2, 4], [4, 6, 0])
                            ), ([0, 2, 3], [5, 3, 1])
                        ), ([1], [0])
                    ), ([2, 3], [4, 1])
                ), ([0, 1, 3, 4], [6, 1, 3, 0])
            ), ([0, 1, 2, 3], [1, 0, 4, 3])
        ), ([0, 1], [1, 0])
    )

def Contract_scalar_4x2(\
    t0_3,t1_3,t2_3,t3_3,t4_3,t5_3,\
    t0_2,t1_2,t2_2,t3_2,t4_2,t5_2,\
    t0_1,t1_1,t2_1,t3_1,t4_1,t5_1,\
    t0_0,t1_0,t2_0,t3_0,t4_0,t5_0,\
    o1_2,o2_2,o3_2,o4_2,\
    o1_1,o2_1,o3_1,o4_1\
    ):
    ##############################
    # ./input/input_Lx4Ly2.dat
    ##############################
    # (o4_2*(t4_2*((t4_3*(t5_3*t5_2))*(t4_2.conj()*((t4_1.conj()*((t4_1*o4_1)*(t4_0*(t5_0*t5_1))))*(t3_3*(t3_2*((o3_2*t3_2.conj())*(t3_1.conj()*((o3_1*t3_1)*(t3_0*(t2_0*(t2_1*((o2_1*t2_1.conj())*(t2_2.conj()*((o2_2*t2_2)*(t2_3*((t1_1.conj()*((o1_1*t1_1)*(t0_1*(t0_0*t1_0))))*(t1_2.conj()*((o1_2*t1_2)*(t0_2*(t0_3*t1_3))))))))))))))))))))))
    # cpu_cost= 2.22004e+13  memory= 3.02032e+10
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o4_2, np.tensordot(
            t4_2, np.tensordot(
                np.tensordot(
                    t4_3, np.tensordot(
                        t5_3, t5_2, ([1], [0])
                    ), ([1], [0])
                ), np.tensordot(
                    t4_2.conj(), np.tensordot(
                        np.tensordot(
                            t4_1.conj(), np.tensordot(
                                np.tensordot(
                                    t4_1, o4_1, ([4], [0])
                                ), np.tensordot(
                                    t4_0, np.tensordot(
                                        t5_0, t5_1, ([0], [1])
                                    ), ([0], [0])
                                ), ([2, 3], [4, 1])
                            ), ([2, 3, 4], [6, 4, 2])
                        ), np.tensordot(
                            t3_3, np.tensordot(
                                t3_2, np.tensordot(
                                    np.tensordot(
                                        o3_2, t3_2.conj(), ([1], [4])
                                    ), np.tensordot(
                                        t3_1.conj(), np.tensordot(
                                            np.tensordot(
                                                o3_1, t3_1, ([0], [4])
                                            ), np.tensordot(
                                                t3_0, np.tensordot(
                                                    t2_0, np.tensordot(
                                                        t2_1, np.tensordot(
                                                            np.tensordot(
                                                                o2_1, t2_1.conj(), ([1], [4])
                                                            ), np.tensordot(
                                                                t2_2.conj(), np.tensordot(
                                                                    np.tensordot(
                                                                        o2_2, t2_2, ([0], [4])
                                                                    ), np.tensordot(
                                                                        t2_3, np.tensordot(
                                                                            np.tensordot(
                                                                                t1_1.conj(), np.tensordot(
                                                                                    np.tensordot(
                                                                                        o1_1, t1_1, ([0], [4])
                                                                                    ), np.tensordot(
                                                                                        t0_1, np.tensordot(
                                                                                            t0_0, t1_0, ([0], [1])
                                                                                        ), ([0], [0])
                                                                                    ), ([1, 4], [1, 4])
                                                                                ), ([0, 3, 4], [4, 6, 0])
                                                                            ), np.tensordot(
                                                                                t1_2.conj(), np.tensordot(
                                                                                    np.tensordot(
                                                                                        o1_2, t1_2, ([0], [4])
                                                                                    ), np.tensordot(
                                                                                        t0_2, np.tensordot(
                                                                                            t0_3, t1_3, ([1], [0])
                                                                                        ), ([1], [0])
                                                                                    ), ([1, 2], [1, 4])
                                                                                ), ([0, 1, 4], [4, 6, 0])
                                                                            ), ([0, 2, 4], [1, 3, 4])
                                                                        ), ([0], [5])
                                                                    ), ([1, 2], [7, 1])
                                                                ), ([0, 1, 4], [8, 4, 0])
                                                            ), ([1, 2], [5, 1])
                                                        ), ([0, 1, 4], [7, 5, 0])
                                                    ), ([1, 2, 3], [7, 1, 3])
                                                ), ([1], [0])
                                            ), ([1, 4], [3, 1])
                                        ), ([0, 3, 4], [5, 4, 0])
                                    ), ([1, 4], [5, 0])
                                ), ([0, 3, 4], [7, 4, 0])
                            ), ([0, 2, 3], [7, 0, 2])
                        ), ([0, 2, 4], [3, 4, 5])
                    ), ([0, 3], [5, 0])
                ), ([0, 2, 3, 5], [5, 0, 4, 1])
            ), ([0, 1, 2, 3], [4, 0, 1, 3])
        ), ([0, 1], [0, 1])
    )

def Contract_scalar_4x3(\
    t0_4,t1_4,t2_4,t3_4,t4_4,t5_4,\
    t0_3,t1_3,t2_3,t3_3,t4_3,t5_3,\
    t0_2,t1_2,t2_2,t3_2,t4_2,t5_2,\
    t0_1,t1_1,t2_1,t3_1,t4_1,t5_1,\
    t0_0,t1_0,t2_0,t3_0,t4_0,t5_0,\
    o1_3,o2_3,o3_3,o4_3,\
    o1_2,o2_2,o3_2,o4_2,\
    o1_1,o2_1,o3_1,o4_1\
    ):
    ##############################
    # ./input/input_Lx4Ly3.dat
    ##############################
    # (o1_2*(t1_2*((t0_2*(t1_1*((o1_1*t1_1.conj())*(t0_1*(t0_0*t1_0)))))*(t1_2.conj()*((t1_3*((o1_3*t1_3.conj())*(t0_3*(t0_4*t1_4))))*(t2_4*(t2_3*((t2_3.conj()*o2_3)*(t2_2*((o2_2*t2_2.conj())*(t2_1*((t2_1.conj()*o2_1)*(t2_0*(t3_4*(t3_3.conj()*((o3_3*t3_3)*(t3_2.conj()*((t3_2*o3_2)*(t3_1.conj()*((t3_1*o3_1)*(t3_0*((t4_3.conj()*((o4_3*t4_3)*(t5_3*(t5_4*t4_4))))*(t4_2*((t4_2.conj()*o4_2)*(t5_2*(t4_1.conj()*((t4_1*o4_1)*(t4_0*(t5_0*t5_1)))))))))))))))))))))))))))))
    # cpu_cost= 3.0102e+15  memory= 3.0101e+12
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o1_2, np.tensordot(
            t1_2, np.tensordot(
                np.tensordot(
                    t0_2, np.tensordot(
                        t1_1, np.tensordot(
                            np.tensordot(
                                o1_1, t1_1.conj(), ([1], [4])
                            ), np.tensordot(
                                t0_1, np.tensordot(
                                    t0_0, t1_0, ([0], [1])
                                ), ([0], [0])
                            ), ([1, 4], [2, 5])
                        ), ([0, 3, 4], [4, 6, 0])
                    ), ([0], [4])
                ), np.tensordot(
                    t1_2.conj(), np.tensordot(
                        np.tensordot(
                            t1_3, np.tensordot(
                                np.tensordot(
                                    o1_3, t1_3.conj(), ([1], [4])
                                ), np.tensordot(
                                    t0_3, np.tensordot(
                                        t0_4, t1_4, ([1], [0])
                                    ), ([1], [0])
                                ), ([1, 2], [2, 5])
                            ), ([0, 1, 4], [4, 6, 0])
                        ), np.tensordot(
                            t2_4, np.tensordot(
                                t2_3, np.tensordot(
                                    np.tensordot(
                                        t2_3.conj(), o2_3, ([4], [1])
                                    ), np.tensordot(
                                        t2_2, np.tensordot(
                                            np.tensordot(
                                                o2_2, t2_2.conj(), ([1], [4])
                                            ), np.tensordot(
                                                t2_1, np.tensordot(
                                                    np.tensordot(
                                                        t2_1.conj(), o2_1, ([4], [1])
                                                    ), np.tensordot(
                                                        t2_0, np.tensordot(
                                                            t3_4, np.tensordot(
                                                                t3_3.conj(), np.tensordot(
                                                                    np.tensordot(
                                                                        o3_3, t3_3, ([0], [4])
                                                                    ), np.tensordot(
                                                                        t3_2.conj(), np.tensordot(
                                                                            np.tensordot(
                                                                                t3_2, o3_2, ([4], [0])
                                                                            ), np.tensordot(
                                                                                t3_1.conj(), np.tensordot(
                                                                                    np.tensordot(
                                                                                        t3_1, o3_1, ([4], [0])
                                                                                    ), np.tensordot(
                                                                                        t3_0, np.tensordot(
                                                                                            np.tensordot(
                                                                                                t4_3.conj(), np.tensordot(
                                                                                                    np.tensordot(
                                                                                                        o4_3, t4_3, ([0], [4])
                                                                                                    ), np.tensordot(
                                                                                                        t5_3, np.tensordot(
                                                                                                            t5_4, t4_4, ([0], [1])
                                                                                                        ), ([0], [0])
                                                                                                    ), ([2, 3], [4, 1])
                                                                                                ), ([1, 2, 4], [6, 4, 0])
                                                                                            ), np.tensordot(
                                                                                                t4_2, np.tensordot(
                                                                                                    np.tensordot(
                                                                                                        t4_2.conj(), o4_2, ([4], [1])
                                                                                                    ), np.tensordot(
                                                                                                        t5_2, np.tensordot(
                                                                                                            t4_1.conj(), np.tensordot(
                                                                                                                np.tensordot(
                                                                                                                    t4_1, o4_1, ([4], [0])
                                                                                                                ), np.tensordot(
                                                                                                                    t4_0, np.tensordot(
                                                                                                                        t5_0, t5_1, ([0], [1])
                                                                                                                    ), ([0], [0])
                                                                                                                ), ([2, 3], [4, 1])
                                                                                                            ), ([2, 3, 4], [6, 4, 2])
                                                                                                        ), ([1], [5])
                                                                                                    ), ([2, 3], [2, 4])
                                                                                                ), ([2, 3, 4], [4, 7, 2])
                                                                                            ), ([1, 3, 4], [3, 1, 4])
                                                                                        ), ([0], [7])
                                                                                    ), ([2, 3], [9, 1])
                                                                                ), ([2, 3, 4], [10, 4, 2])
                                                                            ), ([2, 3], [8, 3])
                                                                        ), ([2, 3, 4], [10, 4, 2])
                                                                    ), ([3, 4], [8, 3])
                                                                ), ([2, 3, 4], [9, 4, 0])
                                                            ), ([1, 2, 3], [9, 3, 1])
                                                        ), ([0], [7])
                                                    ), ([2, 3], [8, 2])
                                                ), ([2, 3, 4], [10, 4, 2])
                                            ), ([3, 4], [8, 3])
                                        ), ([2, 3, 4], [10, 4, 0])
                                    ), ([2, 3], [8, 3])
                                ), ([2, 3, 4], [10, 4, 2])
                            ), ([1, 2, 3], [9, 1, 3])
                        ), ([0, 2, 5], [1, 2, 0])
                    ), ([1, 2], [1, 4])
                ), ([0, 2, 4, 5, 6, 7], [4, 0, 6, 1, 7, 8])
            ), ([0, 1, 2, 3], [0, 3, 4, 1])
        ), ([0, 1], [0, 1])
    )

def Contract_scalar_5x1(\
    t0_2,t1_2,t2_2,t3_2,t4_2,t5_2,t6_2,\
    t0_1,t1_1,t2_1,t3_1,t4_1,t5_1,t6_1,\
    t0_0,t1_0,t2_0,t3_0,t4_0,t5_0,t6_0,\
    o1_1,o2_1,o3_1,o4_1,o5_1\
    ):
    ##############################
    # ./input/input_Lx5Ly1.dat
    ##############################
    # (o2_1*(t2_1.conj()*((t2_2*(t1_0*(t1_1.conj()*((o1_1*t1_1)*(t1_2*(t0_0*(t0_1*t0_2)))))))*(t2_1*(t2_0*(t3_2*(t3_1.conj()*((t3_1*o3_1)*(t3_0*(t4_0*(t4_1.conj()*((o4_1*t4_1)*(t4_2*(t5_0*(t5_1.conj()*((o5_1*t5_1)*(t5_2*(t6_0*(t6_2*t6_1)))))))))))))))))))
    # cpu_cost= 3.004e+11  memory= 5.0206e+08
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o2_1, np.tensordot(
            t2_1.conj(), np.tensordot(
                np.tensordot(
                    t2_2, np.tensordot(
                        t1_0, np.tensordot(
                            t1_1.conj(), np.tensordot(
                                np.tensordot(
                                    o1_1, t1_1, ([0], [4])
                                ), np.tensordot(
                                    t1_2, np.tensordot(
                                        t0_0, np.tensordot(
                                            t0_1, t0_2, ([1], [0])
                                        ), ([1], [0])
                                    ), ([0], [3])
                                ), ([1, 2], [4, 1])
                            ), ([0, 1, 4], [6, 4, 0])
                        ), ([1, 2, 3], [5, 3, 1])
                    ), ([0], [3])
                ), np.tensordot(
                    t2_1, np.tensordot(
                        t2_0, np.tensordot(
                            t3_2, np.tensordot(
                                t3_1.conj(), np.tensordot(
                                    np.tensordot(
                                        t3_1, o3_1, ([4], [0])
                                    ), np.tensordot(
                                        t3_0, np.tensordot(
                                            t4_0, np.tensordot(
                                                t4_1.conj(), np.tensordot(
                                                    np.tensordot(
                                                        o4_1, t4_1, ([0], [4])
                                                    ), np.tensordot(
                                                        t4_2, np.tensordot(
                                                            t5_0, np.tensordot(
                                                                t5_1.conj(), np.tensordot(
                                                                    np.tensordot(
                                                                        o5_1, t5_1, ([0], [4])
                                                                    ), np.tensordot(
                                                                        t5_2, np.tensordot(
                                                                            t6_0, np.tensordot(
                                                                                t6_2, t6_1, ([1], [0])
                                                                            ), ([0], [1])
                                                                        ), ([1], [1])
                                                                    ), ([2, 3], [1, 4])
                                                                ), ([1, 2, 4], [4, 6, 0])
                                                            ), ([0, 2, 3], [5, 3, 1])
                                                        ), ([1], [3])
                                                    ), ([2, 3], [1, 5])
                                                ), ([1, 2, 4], [4, 6, 0])
                                            ), ([0, 2, 3], [5, 3, 1])
                                        ), ([0], [0])
                                    ), ([2, 3], [4, 1])
                                ), ([2, 3, 4], [5, 4, 2])
                            ), ([1, 2, 3], [5, 3, 1])
                        ), ([0], [3])
                    ), ([2, 3], [5, 1])
                ), ([0, 1, 3, 5], [5, 1, 3, 0])
            ), ([0, 1, 2, 3], [1, 0, 4, 3])
        ), ([0, 1], [1, 0])
    )

def Contract_scalar_5x2(\
    t0_3,t1_3,t2_3,t3_3,t4_3,t5_3,t6_3,\
    t0_2,t1_2,t2_2,t3_2,t4_2,t5_2,t6_2,\
    t0_1,t1_1,t2_1,t3_1,t4_1,t5_1,t6_1,\
    t0_0,t1_0,t2_0,t3_0,t4_0,t5_0,t6_0,\
    o1_2,o2_2,o3_2,o4_2,o5_2,\
    o1_1,o2_1,o3_1,o4_1,o5_1\
    ):
    ##############################
    # ./input/input_Lx5Ly2.dat
    ##############################
    # (o3_1*(t3_1*((t3_0*(t4_0*(t4_1.conj()*((o4_1*t4_1)*(t4_2.conj()*((o4_2*t4_2)*(t4_3*((t5_2*((o5_2*t5_2.conj())*(t6_2*(t6_3*t5_3))))*(t5_1.conj()*((t5_1*o5_1)*(t5_0*(t6_0*t6_1))))))))))))*(t3_1.conj()*(t3_2*((t3_2.conj()*o3_2)*(t3_3*(t2_3*(t2_2*((t2_2.conj()*o2_2)*(t2_1*((o2_1*t2_1.conj())*(t2_0*((t1_2*((t1_2.conj()*o1_2)*(t0_2*(t0_3*t1_3))))*(t1_1*((t1_1.conj()*o1_1)*(t0_1*(t0_0*t1_0))))))))))))))))))
    # cpu_cost= 3.22004e+13  memory= 5.00021e+10
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o3_1, np.tensordot(
            t3_1, np.tensordot(
                np.tensordot(
                    t3_0, np.tensordot(
                        t4_0, np.tensordot(
                            t4_1.conj(), np.tensordot(
                                np.tensordot(
                                    o4_1, t4_1, ([0], [4])
                                ), np.tensordot(
                                    t4_2.conj(), np.tensordot(
                                        np.tensordot(
                                            o4_2, t4_2, ([0], [4])
                                        ), np.tensordot(
                                            t4_3, np.tensordot(
                                                np.tensordot(
                                                    t5_2, np.tensordot(
                                                        np.tensordot(
                                                            o5_2, t5_2.conj(), ([1], [4])
                                                        ), np.tensordot(
                                                            t6_2, np.tensordot(
                                                                t6_3, t5_3, ([0], [1])
                                                            ), ([0], [0])
                                                        ), ([2, 3], [5, 2])
                                                    ), ([1, 2, 4], [6, 4, 0])
                                                ), np.tensordot(
                                                    t5_1.conj(), np.tensordot(
                                                        np.tensordot(
                                                            t5_1, o5_1, ([4], [0])
                                                        ), np.tensordot(
                                                            t5_0, np.tensordot(
                                                                t6_0, t6_1, ([0], [1])
                                                            ), ([0], [0])
                                                        ), ([2, 3], [4, 1])
                                                    ), ([2, 3, 4], [6, 4, 2])
                                                ), ([1, 3, 4], [3, 1, 5])
                                            ), ([1], [2])
                                        ), ([2, 3], [1, 3])
                                    ), ([1, 2, 4], [4, 5, 0])
                                ), ([2, 3], [3, 6])
                            ), ([1, 2, 4], [4, 7, 0])
                        ), ([0, 2, 3], [7, 3, 1])
                    ), ([0], [0])
                ), np.tensordot(
                    t3_1.conj(), np.tensordot(
                        t3_2, np.tensordot(
                            np.tensordot(
                                t3_2.conj(), o3_2, ([4], [1])
                            ), np.tensordot(
                                t3_3, np.tensordot(
                                    t2_3, np.tensordot(
                                        t2_2, np.tensordot(
                                            np.tensordot(
                                                t2_2.conj(), o2_2, ([4], [1])
                                            ), np.tensordot(
                                                t2_1, np.tensordot(
                                                    np.tensordot(
                                                        o2_1, t2_1.conj(), ([1], [4])
                                                    ), np.tensordot(
                                                        t2_0, np.tensordot(
                                                            np.tensordot(
                                                                t1_2, np.tensordot(
                                                                    np.tensordot(
                                                                        t1_2.conj(), o1_2, ([4], [1])
                                                                    ), np.tensordot(
                                                                        t0_2, np.tensordot(
                                                                            t0_3, t1_3, ([1], [0])
                                                                        ), ([1], [0])
                                                                    ), ([0, 1], [2, 5])
                                                                ), ([0, 1, 4], [4, 6, 2])
                                                            ), np.tensordot(
                                                                t1_1, np.tensordot(
                                                                    np.tensordot(
                                                                        t1_1.conj(), o1_1, ([4], [1])
                                                                    ), np.tensordot(
                                                                        t0_1, np.tensordot(
                                                                            t0_0, t1_0, ([0], [1])
                                                                        ), ([0], [0])
                                                                    ), ([0, 3], [2, 5])
                                                                ), ([0, 3, 4], [4, 6, 2])
                                                            ), ([1, 3, 4], [0, 2, 4])
                                                        ), ([1], [5])
                                                    ), ([1, 4], [7, 2])
                                                ), ([0, 3, 4], [8, 4, 0])
                                            ), ([0, 3], [6, 2])
                                        ), ([0, 3, 4], [7, 3, 2])
                                    ), ([0, 2, 3], [7, 0, 2])
                                ), ([0], [0])
                            ), ([0, 1], [4, 2])
                        ), ([0, 1, 4], [5, 4, 2])
                    ), ([0, 1], [6, 3])
                ), ([0, 2, 3, 5, 6, 7], [8, 1, 0, 5, 3, 6])
            ), ([0, 1, 2, 3], [4, 3, 1, 0])
        ), ([0, 1], [0, 1])
    )

def Contract_scalar_6x1(\
    t0_2,t1_2,t2_2,t3_2,t4_2,t5_2,t6_2,t7_2,\
    t0_1,t1_1,t2_1,t3_1,t4_1,t5_1,t6_1,t7_1,\
    t0_0,t1_0,t2_0,t3_0,t4_0,t5_0,t6_0,t7_0,\
    o1_1,o2_1,o3_1,o4_1,o5_1,o6_1\
    ):
    ##############################
    # ./input/input_Lx6Ly1.dat
    ##############################
    # (o3_1*(t3_1.conj()*((t3_0*(t2_2*(t2_1*((t2_1.conj()*o2_1)*(t2_0*(t1_0*(t1_1.conj()*((o1_1*t1_1)*(t1_2*(t0_0*(t0_2*t0_1)))))))))))*(t3_1*(t3_2*(t4_0*(t4_1.conj()*((o4_1*t4_1)*(t4_2*(t5_0*(t5_1*((o5_1*t5_1.conj())*(t5_2*(t6_0*(t6_1.conj()*((t6_1*o6_1)*(t6_2*(t7_0*(t7_2*t7_1)))))))))))))))))))
    # cpu_cost= 3.604e+11  memory= 5.041e+08
    # final_bond_order ()
    ##############################
    return np.tensordot(
        o3_1, np.tensordot(
            t3_1.conj(), np.tensordot(
                np.tensordot(
                    t3_0, np.tensordot(
                        t2_2, np.tensordot(
                            t2_1, np.tensordot(
                                np.tensordot(
                                    t2_1.conj(), o2_1, ([4], [1])
                                ), np.tensordot(
                                    t2_0, np.tensordot(
                                        t1_0, np.tensordot(
                                            t1_1.conj(), np.tensordot(
                                                np.tensordot(
                                                    o1_1, t1_1, ([0], [4])
                                                ), np.tensordot(
                                                    t1_2, np.tensordot(
                                                        t0_0, np.tensordot(
                                                            t0_2, t0_1, ([0], [1])
                                                        ), ([1], [1])
                                                    ), ([0], [1])
                                                ), ([1, 2], [4, 1])
                                            ), ([0, 1, 4], [6, 4, 0])
                                        ), ([1, 2, 3], [5, 3, 1])
                                    ), ([1], [0])
                                ), ([0, 3], [3, 2])
                            ), ([0, 3, 4], [5, 4, 2])
                        ), ([0, 2, 3], [5, 0, 2])
                    ), ([1], [3])
                ), np.tensordot(
                    t3_1, np.tensordot(
                        t3_2, np.tensordot(
                            t4_0, np.tensordot(
                                t4_1.conj(), np.tensordot(
                                    np.tensordot(
                                        o4_1, t4_1, ([0], [4])
                                    ), np.tensordot(
                                        t4_2, np.tensordot(
                                            t5_0, np.tensordot(
                                                t5_1, np.tensordot(
                                                    np.tensordot(
                                                        o5_1, t5_1.conj(), ([1], [4])
                                                    ), np.tensordot(
                                                        t5_2, np.tensordot(
                                                            t6_0, np.tensordot(
                                                                t6_1.conj(), np.tensordot(
                                                                    np.tensordot(
                                                                        t6_1, o6_1, ([4], [0])
                                                                    ), np.tensordot(
                                                                        t6_2, np.tensordot(
                                                                            t7_0, np.tensordot(
                                                                                t7_2, t7_1, ([1], [0])
                                                                            ), ([0], [1])
                                                                        ), ([1], [1])
                                                                    ), ([1, 2], [1, 4])
                                                                ), ([1, 2, 4], [4, 6, 2])
                                                            ), ([0, 2, 3], [5, 3, 1])
                                                        ), ([1], [3])
                                                    ), ([2, 3], [2, 4])
                                                ), ([1, 2, 4], [4, 6, 0])
                                            ), ([0, 2, 3], [5, 1, 3])
                                        ), ([1], [3])
                                    ), ([2, 3], [1, 4])
                                ), ([1, 2, 4], [4, 6, 0])
                            ), ([0, 2, 3], [5, 3, 1])
                        ), ([1], [3])
                    ), ([1, 2], [1, 5])
                ), ([0, 1, 3, 4], [5, 1, 3, 0])
            ), ([0, 1, 2, 3], [1, 3, 4, 0])
        ), ([0, 1], [1, 0])
    )

