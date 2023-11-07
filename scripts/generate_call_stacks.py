__author__ = "Saumik"
__date__ = "10/31/2023"

""" 
1. C:\Users\SaumikDana\AppData\Local\pypoetry\Cache\virtualenvs\asicore-sfrFScr6-py3.9\Scripts\python.exe .\generate_call_stack.py
2. On the Other terminal
    a. Get the process ID using 
        wmic process where "name='python.exe'" get ProcessId,CommandLine
    b. Replace PID with the process ID in the command
        C:\Users\SaumikDana\AppData\Local\pypoetry\Cache\virtualenvs\asicore-sfrFScr6-py3.9\Scripts\py-spy.exe record --pid <> --output app_driver.svg

"""

import time

time.sleep(30)  # 30-second delay

file = '.\\drivers\\app_driver.py'
# file = '.\\drivers\\acq_driver.py'
# file = '.\\drivers\\map_driver.py'
# file = '.\\drivers\\rec_driver.py'

with open(file, 'r') as f:
    code = f.read()
    exec(code)
