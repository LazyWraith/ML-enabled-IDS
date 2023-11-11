@echo off

start "" /b python.exe dt-tune.py
start "" /b python.exe gnb-tune.py
start "" /b python.exe knn-tune.py
start "" /b python.exe lr-tune.py
start "" /b python.exe lsvc-tune.py
start "" /b python.exe rf-tune.py
start "" /b python.exe xgb-tune.py

pause