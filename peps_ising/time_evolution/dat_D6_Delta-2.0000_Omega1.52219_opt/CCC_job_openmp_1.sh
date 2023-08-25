#!/bin/bash
#$ -cwd
#$ -V -S /bin/bash
#$ -N test
#$ -pe smp 1
#$ -q dp8.q
##$ -q dp.q
##$ -q dp2.q
##$ -q dp3.q
##$ -q dp4.q
##$ -q dp5.q
##$ -q dp6.q
##$ -q dp7.q
##$ -q dp8.q

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

#cp _output_Tn_060.npz input_Tn.npz

#----

prog_opt=Model_TFI_Square_opt.py

date

#init=-1
#init=1 ## FM z
#init=2 ## AF z
#init=3 ## FM x
#init=4 ## 1z2x
#init=5 ## all down
init=6 ## FM -x
steps=1
#steps=100
#steps=1000
ts=0.005

V=-1.0
Omega=1.52219
Delta=-2.0000

i=000000
  # calculate expectation value only
  ${python} ${prog_opt} \
  -i ${init} -V ${V} -Omega ${Omega} -Delta ${Delta} -ts ${ts} -ss 0 -sf 0 --second_ST \
  > output_phys_quant_${i}
  cp output_Tn.npz output_Tn_${i}.npz
  mv output_Tn.npz input_Tn.npz
  date

for i in \
`seq -f "%06g" 1 2000`
do
  # simple update
  ${python} ${prog_opt} \
  -i -1 -V ${V} -Omega ${Omega} -Delta ${Delta} -ts ${ts} -ss ${steps} -sf 0 --second_ST \
  > output_phys_quant_${i}
  cp output_Tn.npz output_Tn_${i}.npz
  mv output_Tn.npz input_Tn.npz
  date
done

#----

conda deactivate
