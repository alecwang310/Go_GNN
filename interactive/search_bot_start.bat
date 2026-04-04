@echo off
title OGS Search Bot Starter - %date%
:: --- CONFIGURATION ---
set BOT_FOLDER=D:\Code\GNN\interactive
set BRIDGE_FOLDER=D:\Code\GNN\interactive\gtp2ogs-main
:: ---------------------

echo [1/2] Checking Python GTP Search Interface...
if not exist "%BOT_FOLDER%\gtp_interface_search.py" (
    echo ERROR: Could not find gtp_interface_search.py in %BOT_FOLDER%
    pause
    exit
)

echo [2/2] Launching OGS Bridge (Search Bot)...
cd /d "%BRIDGE_FOLDER%"

node dist/gtp2ogs.js --config config_search.json5

pause
