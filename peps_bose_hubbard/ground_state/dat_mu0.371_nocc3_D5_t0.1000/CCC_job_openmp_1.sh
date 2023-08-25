#!/bin/bash
#$ -cwd
#$ -V -S /bin/bash
#$ -N test
#$ -pe smp 1
#$ -q dp.q
##$ -q dp5.q

#source scl_source enable python27

#export OMP_NUM_THREADS=$PBS_NP

#cd $PBS_O_WORKDIR

#----

## conda create -n py2 python=2.7 numpy scipy matplotlib numba pip

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate
conda activate py2

#python=python2.7
python=python

#----

#cp _output_Tn_*.npz input_Tn.npz

#----

prog=Model_Bose_Hubbard_Square.py

date

#init=-1
#init=0 ## rnd
#init=1 ## MI
#init=2 ## CDW
init=3
steps=100

timemax=1.0
#ts=0.0128
ts=0.0512
Nsteps=120
part=0.1000
parU=1.0
parmu=0.371

ipad=000000
  # calculate expectation value only
  ${python} ${prog} \
  -i ${init} -part ${part} -parU ${parU} -parmu ${parmu} -ts ${ts} -ss 0 -sf 0 --second_ST \
  > output_phys_quant_${ipad}
  cp output_Tn.npz output_Tn_${ipad}.npz
  mv output_Tn.npz input_Tn.npz
  date

#eneold=`grep -v "#" output_phys_quant_${ipad} | awk '{print $2}'`
eneold=`grep "t,U,mu,z,nocc,ene,ene_t,ene_U,ene_mu" output_phys_quant_${ipad} | awk '{print $8}'`

cnt=1
for i in \
`seq -f "%06g" 1 ${Nsteps}`
do
  printf -v ipad "%06g" ${cnt}
  # simple update
  ${python} ${prog} \
  -i -1 -part ${part} -parU ${parU} -parmu ${parmu} -ts ${ts} -ss ${steps} -sf 0 --second_ST \
  > output_phys_quant_${ipad}
  cp output_Tn.npz output_Tn_${ipad}.npz
  mv output_Tn.npz input_Tn.npz
#
#  enenew=`grep -v "#" output_phys_quant_${ipad} | awk '{print $2}'`
  enenew=`grep "t,U,mu,z,nocc,ene,ene_t,ene_U,ene_mu" output_phys_quant_${ipad} | awk '{print $8}'`
  eneoldadd=`echo "${eneold}+0.000001" | bc -l`
  echo ${cnt} ${eneold} ${eneoldadd} ${enenew}
  if (( $(echo "${enenew} > ${eneoldadd}" | bc -l) )); then
    echo "### higher energy, recalculate at ${cnt} step"
    j=${cnt}
    jm=$((j-1))
    jmm=$((j-2))
    printf -v jpad "%06g" ${j}
    printf -v jmpad "%06g" ${jm}
    printf -v jmmpad "%06g" ${jmm}
    tmpdate=`date +%s%3N`
#    tmpdate=`gdate +%s%3N` ## Mac
    mv output_phys_quant_${jpad} _output_phys_quant_${jpad}_${tmpdate}
    mv output_Tn_${jpad}.npz _output_Tn_${jpad}.npz_${tmpdate}
    mv output_phys_quant_${jmpad} _output_phys_quant_${jmpad}_${tmpdate}
    mv output_Tn_${jmpad}.npz _output_Tn_${jmpad}.npz_${tmpdate}
    ts=`echo "${ts}/2" | bc -l`
    echo "### set dt ${ts}"
    cp output_Tn_${jmmpad}.npz input_Tn.npz
#    eneold=`grep -v "#" output_phys_quant_${jmmpad} | awk '{print $2}'`
    eneold=`grep "t,U,mu,z,nocc,ene,ene_t,ene_U,ene_mu" output_phys_quant_${jmmpad} | awk '{print $8}'`
    cnt=${jm}
  else
    eneold=${enenew}
    cnt=$((cnt+1))
  fi
  date
done

#----

conda deactivate
