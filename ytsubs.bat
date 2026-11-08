@echo off
rem ============================================================
rem  ytsubs - THIS IS THE FILE TO RUN. Just double-click it.
rem
rem  First time: type your link when asked.
rem  Next times: put the link into YT_SOURCE below and just
rem  double-click - it will not ask anything.
rem
rem  Videos already exported are skipped, so re-running is cheap
rem  and continues where the last run stopped.
rem
rem  Command line still works, arguments are passed through:
rem    ytsubs.bat --video https://youtu.be/ID --format srt
rem ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem --------------- SETTINGS (fill in once, optional) ----------
rem  leave YT_SOURCE empty to be asked for the link every time
set "YT_KIND=--channel"
set "YT_SOURCE="
set "YT_LANGS=ru,en"
set "YT_FORMAT=txt"
set "YT_WORKERS=4"
set "YT_LIMIT="
rem  account (fewer 429 errors): a cookies file next to this .bat
set "YT_COOKIES=cookies.txt"
rem  ...or read cookies from an installed browser. Firefox works on Windows;
rem  Chrome/Edge do not (their new cookie encryption cannot be decrypted).
set "YT_BROWSER=firefox"
rem  proxy for a rate-limited IP, e.g. socks5://127.0.0.1:1080
set "YT_PROXY="
rem ------------------------------------------------------------

if not "%~1"=="" goto :cli
if not "!YT_SOURCE!"=="" goto :settings

echo ============================================
echo   ytsubs - YouTube transcript exporter
echo ============================================
echo.
echo Type a channel, playlist or video link:
echo   https://www.youtube.com/@channel
echo   https://www.youtube.com/playlist?list=LIST_ID
echo   https://www.youtube.com/watch?v=VIDEO_ID
echo.
echo Tip: save this link into YT_SOURCE at the top of this file
echo      and next time it starts right away.
echo.
set "YTURL="
set /p "YTURL=Link (Enter to quit): "
if "!YTURL!"=="" goto :end
set "YTURL=!YTURL: =!"
if "!YTURL!"=="" goto :end

set "YT_KIND=--channel"
if not "!YTURL:list=!"=="!YTURL!" set "YT_KIND=--playlist"
if not "!YTURL:watch?v=!"=="!YTURL!" set "YT_KIND=--video"
if not "!YTURL:youtu.be/=!"=="!YTURL!" set "YT_KIND=--video"
set "YT_SOURCE=!YTURL!"

set "FMT="
set /p "FMT=Format [txt/json/srt/vtt] (Enter = txt): "
if not "!FMT!"=="" set "YT_FORMAT=!FMT: =!"

:settings
set "ARGS=!YT_KIND! "!YT_SOURCE!" --langs !YT_LANGS! --format !YT_FORMAT! --workers !YT_WORKERS!"
if not "!YT_LIMIT!"=="" set "ARGS=!ARGS! --limit !YT_LIMIT!"
if not "!YT_PROXY!"=="" set "ARGS=!ARGS! --proxy !YT_PROXY!"
if exist "!YT_COOKIES!" (
    set "ARGS=!ARGS! --cookies "!YT_COOKIES!""
) else if not "!YT_BROWSER!"=="" (
    set "ARGS=!ARGS! --cookies-from-browser !YT_BROWSER!"
)
echo [ytsubs] !ARGS!
echo.
goto :execute

:cli
set "ARGS=%*"

:execute
python "%~dp0ytsubs.py" !ARGS!
set "RC=!ERRORLEVEL!"
echo.
echo [ytsubs] exit code: !RC!

:end
if not defined RC set "RC=0"
rem Keep the window open when double-clicked, close it when called with args.
if "%~1"=="" pause
endlocal & exit /b %RC%
