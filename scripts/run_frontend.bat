@echo off
title Radiology AI - Frontend
cd /d "%~dp0..\frontend"
npm run dev
if errorlevel 1 pause
