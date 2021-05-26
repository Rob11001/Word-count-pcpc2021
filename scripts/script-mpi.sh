#!/bin/bash

#Usage: ./script-mpi.sh conf_file.txt key.pem script.sh

file=$1
shift
key=$1
shift
script=$1

i=0
ips[0]=""
while read line; do
  ips[i]=$(grep -o "^[^ ]*" <<< $line)
  i=$((i + 1))
done < $file

# Vedere come parallelizzare
i=1
for ip in ${ips[@]}; do
  # vedere se il flag f funziona e se & è necessario
  echo `ssh -f -i ${key} ubuntu@${ip} \"bash -s\" < ${script}  &> log${i}.txt &`
  i=$((i + 1))
done



