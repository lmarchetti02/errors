import os

# creates necessary directories
try:
    os.mkdir("./log")
except Exception as e:
    print(e)
