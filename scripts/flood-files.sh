#!/bin/bash

# Allows to send all file from source directory to dest directory, renaming it. It sends them to all ip addresses in conf_file
# Usage: ./flood-files.sh conf_file source_dir dest_dir [flag] 

# Utilities:
# 1. chmod +rx
# 2. rename 's/.txt/0.txt/' *.txt

if [[ $1 = "" ]] ; then
  echo "Usage: ./flood-files.sh conf_file source_dir dest_dir [flag]"
  exit
fi
file=$1

if [[ $2 = "" ]] ; then
  echo "Usage: ./flood-files.sh conf_file source_dir dest_dir [flag]"
  exit
fi
source=$2

if [[ $3 = "" ]] ; then
  echo "Usage: ./flood-files.sh conf_file source_dir dest_dir [flag]"
  exit
fi
dest=$3

flag=0
if [[ $4 > 0 ]] ; then
  flag=$4
fi

# Reads the IP addresses from the file(if first word in the line)
i=0
ips[0]=""
while read line; do
  ips[i]=$(grep -o "^[^ ]*" <<< $line)
  i=$((i + 1))
done < $file

# Creates a new directory
if [ $flag = 1 ] ; then
  mkdir $dest
fi

# Executes: scp -r source/* pcpc@ip:path/
for ip in ${ips[@]}; do
  if [ $flag = 1 ] ; then
    echo `scp -r $dest pcpc@${ip}:`
  fi
  echo `scp -r ${source}/* pcpc@${ip}:${dest}/ &`
done
