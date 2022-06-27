read -p 'Enter root filename:' rootname

for f in Data/$rootname*_Param.dat;
do   
    arrIN=(${f//./ })
    arrIN1=(${arrIN[0]//_/ })
    arrIN2=(${arrIN1[0]//// })
    ./Chem_Evol ${arrIN2[1]}
    #echo 
done


