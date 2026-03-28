@echo off
title OGS Bot Starter - %date%
:: --- CONFIGURATION ---
set BOT_FOLDER=D:\Code\GNN\interactive
set BRIDGE_FOLDER=D:\Code\GNN\interactive\gtp2ogs-main
:: ---------------------

echo [1/2] Checking Python GTP Interface...
if not exist "%BOT_FOLDER%\gtp_interface.py" (
    echo ERROR: Could not find gtp_interface.py in %BOT_FOLDER%
    pause
    exit
)

echo [2/2] Launching OGS Bridge...
cd /d "%BRIDGE_FOLDER%"

:: Run the bridge. It will automatically call your python script 
:: based on what you put in gtp2ogs-main/config.json
node dist/gtp2ogs.js --config config.json5

pause