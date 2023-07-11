echo { "event": "install-start" }
IF NOT EXIST venv (
    echo { "event": "create-venv" }
    python -m venv venv
) ELSE (
    echo venv folder already exists, skipping creation...
)
echo { "event": "activate-venv" }
call .\venv\Scripts\activate.bat
echo { "event": "install-requirements" }
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 || (echo {"event":"install-failed"} & exit /b)
pip install --use-pep517 --upgrade -r requirements.txt || (echo {"event":"install-failed"} & exit /b)
pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl || (echo {"event":"install-failed"} & exit /b)

copy /y .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
copy /y .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
copy /y .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py
echo { "event": "install-success" }