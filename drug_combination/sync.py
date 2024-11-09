import os, sys

filename = sys.argv[1]
with open(filename, 'a') as f:  # Open the file
    os.fsync(f)             # fsync the file descriptor
