#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <jobid>"
  exit 1
fi

jobid=$1

srun --jobid "$jobid" --overlap --pty bash