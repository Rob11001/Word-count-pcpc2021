#!/bin/bash
# chmod +rx
# Allows to send all file from source directory to dest directory, renaming it. It sends them to all ip addresses in conf_file
# Usage: ./flood-files.sh conf_file source_dir dest_dir [index] 

#TODO: vedere come risolvere il problema dell'invio di più file
#TODO: Vedere come testare

file=$1
shift
source=$1
shift
dest=$1
shift
index=0
if [ $1 > 0 ] ; then
  index=$1
fi

# Reads the IP addresses from the file(first word in the line)
i=0
ips[0]=""
while read line; do
  ips[i]=$(grep -o "^[^ ]*" <<< $line)
  i=$((i + 1))
done < $file

# Executes: scp source/file.txt pcpc@ip:path/file[index].txt
for ip in ${ips[@]}; do
  for file in $source/*; do
    filename=${file%.*}
    filename=${filename:((${#source} + 1))}
    echo "scp '${file}' pcpc@${ip}:'${dest}/${filename}${index}.txt'" #vedere se cambiare la destinazione
  done
done
