#!/bin/bash
#$ -cwd
#$ -V -S /bin/bash
##$ -N test
#$ -N test_PEPS
#$ -pe smp 1
#$ -q dp2.q
##$ -q dp.q
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

init=-1
#init=1 ## MI
steps=1
ts=0.005

Nsteps=400
part=1.0
parU=19.6
parmudivU=0.371
parmu=`echo "${parU}*${parmudivU}" | bc -l`
D=5
nocc=3

i=000364
#cpdir=../../old_10_dat_01_dynamics_nocc${nocc}_U${parU}_mu0.371__init_MI/dat_nocc${nocc}_U${parU}_D${D}_mu0.371/
cpdir=../dat_nocc${nocc}_U${parU}_D${D}_mu0.371_opt/
inputfile=output_Tn_${i}.npz
cp ${cpdir}/${inputfile} .
  # calculate expectation value only
  ${python} ${prog} \
  -i ${init} -part ${part} -parU ${parU} -parmu ${parmu} -ts ${ts} -ss 0 -sf 0 --second_ST \
  -in ${inputfile} \
  > output_phys_quant_${i}
#  cp output_Tn.npz output_Tn_${i}.npz
#  mv output_Tn.npz input_Tn.npz
  date

#for i in \
#`seq -f "%06g" 1 ${Nsteps}`
#do
#  # simple update
#  ${python} ${prog} \
#  -i -1 -part ${part} -parU ${parU} -parmu ${parmu} -ts ${ts} -ss ${steps} -sf 0 --second_ST \
#  > output_phys_quant_${i}
#  cp output_Tn.npz output_Tn_${i}.npz
#  mv output_Tn.npz input_Tn.npz
#  date
#done

rm ${inputfile}

#----

conda deactivate
