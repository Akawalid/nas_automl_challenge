#!/bin/bash

unset XDG_RUNTIME_DIR           # see https://github.com/jupyter/notebook/issues/1411

# Manually add miniconda to PATH. Don't know why the .basrc is not correctly sourced
# export PATH="/home/tao/${USER}/miniconda3/bin:$PATH"

#bash "${HOME}/adapt_conda.sh"

PORT=27029

WORKDIR="/home/tau/${USER}/codebase/"
cd $WORKDIR

echo -e "\nStarting Jupyter Notebook on port ${PORT} on the $(hostname) server."
echo -e "\nSSH tunnel command : "
echo -e "\n==========- RUN IN YOUR COMPUTER TERMINAL -============\n"
echo -e "ssh -NfL ${PORT}:$(hostname):${PORT} ${USER}@titanic"
echo -e "\nThen you can use the url with the token given by jupyter\n"
echo -e "\nTerminate the process by running\n"
echo -e "sudo netstat -lnp | grep ${PORT}"
echo -e "\n and kill\n"

jupyter-notebook --no-browser --port=${PORT} --ip='*'  # Open for all ip address = dangerous ?
