echo { "event": "install-start" }
if not defined PYTHON (set PYTHON=python)
if ["%SKIP_VENV%"] == ["1"] goto :skip_venv
IF NOT EXIST venv (
    echo { "event": "create-venv" }
    python -m venv venv
) ELSE (
    echo venv folder already exists, skipping creation...
)
echo { "event": "activate-venv" }
call .\venv\Scripts\activate.bat
:skip_venv
echo { "event": "install-requirements" }
%PYTHON% -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 || (echo {"event":"install-failed"} & exit /b)
%PYTHON% -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --use-pep517 --upgrade -r requirements_cn.txt || (echo {"event":"install-failed"} & exit /b)
%PYTHON% -m pip install -U -I --no-deps https://huggingface.co/takayamaaren/xformers_build_pack/resolve/main/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl || (echo {"event":"install-failed"} & exit /b)

if ["%SKIP_VENV%"] == ["1"] goto :copy_to_python
copy /y .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
copy /y .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
copy /y .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py
:copy_to_python
echo %PYH%
copy /y .\bitsandbytes_windows\*.dll %PYH%\Lib\site-packages\bitsandbytes\
copy /y .\bitsandbytes_windows\cextension.py %PYH%\Lib\site-packages\bitsandbytes\cextension.py
copy /y .\bitsandbytes_windows\main.py %PYH%\Lib\site-packages\bitsandbytes\cuda_setup\main.py
echo { "event": "install-success" }