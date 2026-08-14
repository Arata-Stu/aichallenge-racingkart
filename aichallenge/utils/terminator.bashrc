# shellcheck shell=bash

source /etc/skel/.bashrc

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"

aic_update_prompt()
{
    PS1="\[\e]0;AI Challenge Domain ${ROS_DOMAIN_ID}: \w\a\]\[\033[01;36m\](Domain ${ROS_DOMAIN_ID})\[\033[00m\] \u@\h:\[\033[01;34m\]\w\[\033[00m\]\$ "
}

aic_domain()
{
    local domain_id="${1:-}"
    local max_domain="${AIC_PLAYER_COUNT:-4}"
    if ! [[ "${domain_id}" =~ ^[0-9]+$ ]] || \
       (( domain_id < 0 || domain_id > max_domain )); then
        echo "Usage: aic_domain <0-${max_domain}>" >&2
        return 2
    fi
    export ROS_DOMAIN_ID="${domain_id}"
    aic_update_prompt
    echo "[INFO] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
}

aic_update_prompt
