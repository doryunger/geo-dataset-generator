#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "No working NVIDIA driver on this instance -- install one first." >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sudo sh
fi
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
    echo "Installing the NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
fi
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "Checking that containers can see the GPU..."
sudo docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L

echo "Instance ready. Next: fill in deploy/.env, then run deploy/deploy.sh"
