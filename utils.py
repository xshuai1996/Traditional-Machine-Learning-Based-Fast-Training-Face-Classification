import psutil

def memory_print():
    print("MEMORY USAGE:", psutil.virtual_memory().used / 1024 / 1024 / 1024, "GB.")
