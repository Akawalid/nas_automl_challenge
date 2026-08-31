#!/bin/bash

# Manually add miniconda to PATH. Don't know why the .basrc is not correctly sourced
# export PATH="/home/tao/${USER}/miniconda3/bin:$PATH"

#bash "${HOME}/adapt_conda.sh"

PORT=27028

WORKDIR="/home/tau/${USER}/codebase/experimental_grow/"
cd $WORKDIR

echo -e "\nStarting Mlflow Server on port ${PORT} on the $(hostname) server."
echo -e "\nSSH tunnel command : "
echo -e "\n==========- RUN IN YOUR COMPUTER TERMINAL -============\n"
echo -e "ssh -NfL ${PORT}:$(hostname):${PORT} margaret"
echo -e "\nTerminate the process by running\n"
echo -e "sudo netstat -lnp | grep ${PORT}"
echo -e "\n and kill\n"

mlflow server --host 0.0.0.0 --port ${PORT} --no-serve-artifacts
