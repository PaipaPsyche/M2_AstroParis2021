read -p 'Enter root filename:' rootname

for f in Data/$rootname*
do   
    echo "deleted $f"
    rm $f
done


