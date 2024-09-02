#!/bin/bash
# This file contains bash commands that will be executed at the end of the container build process,
# after all system packages and programming language specific package have been installed.
#
# Note: This file may be removed if you don't need to use it

# Add Docker's official GPG key:
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the docker repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install Docker-out-of-Docker
sudo apt-get install -y docker-ce-cli
cat <<EOM | sudo tee /etc/profile.d/docker-out-of-docker.sh > /dev/null
if ! groups workbench | grep docker > /dev/null; then
    docker_gid=\$(stat -c %g /var/host-run/docker.sock)
    sudo groupadd -g \$docker_gid docker
    sudo usermod -aG docker workbench
fi
export DOCKER_HOST="unix:///var/host-run/docker.sock"
EOM

# Grant user sudo access
echo "workbench ALL=(ALL) NOPASSWD:ALL" | \
    sudo tee /etc/sudoers.d/00-workbench > /dev/null


# clean up
sudo apt-get autoremove -y
sudo rm -rf /var/cache/apt