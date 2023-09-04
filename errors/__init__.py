import os

# creates necessary directories
try:
    os.mkdir("./log")
except Exception as _:
    print("La directory '/log' esiste già. ")
