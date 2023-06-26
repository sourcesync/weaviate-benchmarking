#!/bin/bash

# try to clear any SSD caches

sudo hdparm -F /dev/sda

python3 test.py


