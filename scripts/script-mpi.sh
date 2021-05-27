#!/bin/bash

# Allows the execution of a bash script with ssh in background for all IP addresses in conf_file
# The outputs are redirected in log[i].txt for every ip i.

# Usage: ./script-mpi.sh conf_file.txt key.pem script.sh
# Note: use ssh first for every node
if [[ $1 = "" ]] ; then
  echo "./script-mpi.sh conf_file.txt key.pem script.sh" 
  exit
fi
file=$1

if [[ $2 = "" ]] ; then
  echo "./script-mpi.sh conf_file.txt key.pem script.sh" 
  exit
fi
key=$2

if [[ $3 = "" ]] ; then
  echo "./script-mpi.sh conf_file.txt key.pem script.sh" 
  exit
fi
script=$3

i=0
ips[0]=""
while read line; do
  ips[i]=$(grep -o "^[^ ]*" <<< $line)
  i=$((i + 1))
done < $file

# Vedere se il check connection work
i=1
for ip in ${ips[@]}; do
  echo `ssh -q -i ${key} ubuntu@${ip} exit` # checks connection
  echo `ssh -i ${key} ubuntu@${ip} "bash -s" < ${script}  &> log${i}.txt &`
  i=$((i + 1))
done



