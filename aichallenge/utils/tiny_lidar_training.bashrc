# shellcheck shell=bash

source /etc/skel/.bashrc

training_dir="${AIC_TINY_LIDAR_TRAINING_DIR:-/aichallenge/ml_workspace/tiny_lidar_net_pytorch}"
cd "${training_dir}" || return
PS1="\[\e]0;TinyLiDARNet Training: \w\a\]\[\033[01;35m\](TinyLiDARNet Training)\[\033[00m\] \u@\h:\[\033[01;34m\]\w\[\033[00m\]\$ "

alias tln-pipeline='./run_pipeline.sh'
alias tln-status='./run_pipeline.sh --list'
alias tln-checkpoints='/aichallenge/utils/select_tiny_lidar_checkpoint.bash --list'
