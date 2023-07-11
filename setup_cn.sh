#!/bin/bash
echo "{ \"event\": \"install-start\" }"
if [ ! -d "venv" ]; then
    echo "{ \"event\": \"create-venv\" }"
    python -m venv venv
else
    echo "venv folder already exists, skipping creation..."
fi
echo "{ \"event\": \"activate-venv\" }"
source venv/bin/activate
echo "{ \"event\": \"install-requirements\" }"
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
if pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --use-pep517 --upgrade -r requirements_macos_cn.txt; then
    echo "Successfully installed requirements"
    echo "{ \"event\": \"install-success\" }"
    exit 0
else
    echo "Failed to install requirements"
    echo "{ \"event\": \"install-failure\" }"
    exit 1
fi