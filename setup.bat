echo { "event": "install-start" }
if not defined PYTHON (set PYTHON=python)
if ["%SKIP_VENV%"] == ["1"] goto :skip_venv
IF NOT EXIST venv (
    echo { "event": "create-venv" }
    %PYTHON% -m venv venv
) ELSE (
    echo venv folder already exists, skipping creation...
)
echo { "event": "activate-venv" }
call .\venv\Scripts\activate.bat
:skip_venv
echo { "event": "install-requirements" }
%PYTHON% -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 || (echo {"event":"install-failed"} & exit /b)
%PYTHON% -m pip install --use-pep517 --upgrade -r requirements.txt || (echo {"event":"install-failed"} & exit /b)
%PYTHON% -m pip install -U xformers || (echo {"event":"install-failed"} & exit /b)

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