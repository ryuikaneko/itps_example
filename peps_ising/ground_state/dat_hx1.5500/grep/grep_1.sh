#!/bin/bash

for hx in \
1.5500
#`seq -f%.4f 0.0000 0.0001 3.00001`
do

for D in \
2 3
do

file1=output_phys_quant_000000
output1=dat_D${D}_hx${hx}
#output2=dat_GSnum_D${D}_hx${hx}
output3=dat_mag_D${D}_hx${hx}
output4=dat_corrlength_D${D}_hx${hx}

echo -ne "" > ${output1}
#echo -ne "" > ${output2}
echo -ne "# D Chi hx mx mz\n" > ${output3}
echo -ne "# D Chi hx l1/l0 l2/l0 l3/l0 l4/l0 l5/l0 l6/l0\n" > ${output4}

for ratio in \
`seq -f%.1f 0.0 0.5 4.01`
do

Chi=`echo "${D}*${D}*${ratio}/1" | bc`

dir1=../ChiOverD2_D${D}_${ratio}
file=${dir1}/${file1}
if [ -e ${file} ]; then
  echo "${file}"
  grep -v "^#" ${file} | \
    awk '{print '${D}','${Chi}','${hx}',$2,$3,$4,$5}' >> ${output1}
  grep "## hx,i,norm,mx,my,mz,sqrt(sum m^2)" ${file} | \
    sed 's/.*m^2)//g' | awk '$2==0{print '${D}','${Chi}','${hx}',$4,$6}' >> ${output3}
  grep "## eigenvalues of T0123h abs" ${file} | \
    sed 's/.*abs//g' | awk '{print '${D}','${Chi}','${hx}',$2/$1,$3/$1,$4/$1,$5/$1,$6/$1,$7/$1}' >> ${output4}
fi

done
done
done
