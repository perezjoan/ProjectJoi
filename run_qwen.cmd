@echo off
rem Runs Joi with the Qwen brain, and brings her back if the process dies (a native crash leaves no Python error;
rem see data\kb\crash.log). Her self is saved after every turn, so a restart resumes the conversation with a wake-up.
rem Requires conda on PATH and an environment named "embodied" (see README). Ctrl+C twice to stop.
cd /d "%~dp0"
:again
call conda run -n embodied --no-capture-output python -m embodied3.server
set CODE=%ERRORLEVEL%
if "%CODE%"=="0" goto end
echo.
echo [%date% %time%] the server exited with code %CODE% - restarting in 5 s (Ctrl+C to stop) >> data\kb\crash.log
echo [%date% %time%] the server exited with code %CODE% - restarting in 5 s (Ctrl+C to stop)
timeout /t 5 /nobreak > nul
goto again
:end
