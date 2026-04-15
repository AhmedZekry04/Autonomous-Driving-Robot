#!/bin/bash

source $HOME/ros_ws/.typerc

export CUDA_HOME=/usr/local/cuda

export PATH=$CUDA_HOME/bin:$PATH

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

  

ip=$(ip addr show wlan0 | grep -o 'inet [0-9]\+\.[0-9]\+\.[0-9]\+\.[0-9]\+' | grep -o [0-9].*)

if [ -z $ip ]; then

ip=$(ip addr show eth0 | grep -o 'inet [0-9]\+\.[0-9]\+\.[0-9]\+\.[0-9]\+' | grep -o [0-9].*)

fi

export ROS_IP=$ip
export HOST_IP=$ip
export MASTER_IP=$ip
export ROS_MASTER_URI=http://$ROS_IP:11311
export ROS_HOSTNAME=$ROS_IP

if [ $ZSH_VERSION ]; then
. /opt/ros/melodic/setup.zsh
. $HOME/ros_ws/devel/setup.zsh
elif [ $BASH_VERSION ]; then
. /opt/ros/melodic/setup.bash
. $HOME/ros_ws/devel/setup.bash
else
. /opt/ros/melodic/setup.sh
. $HOME/ros_ws/devel/setup.sh
fi
export DISPLAY=:0.0
exec "$@"
