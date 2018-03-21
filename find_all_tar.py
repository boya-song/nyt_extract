import tarfile
import os

def findtar(dname):
    inodes = [os.path.join(dname, f) for f in os.listdir(dname)]
    files = [f for f in inodes if os.path.isfile(f)]
    dirs = [f for f in inodes if os.path.isdir(f)]
    for d in dirs:
        findtar(d)
    for f in files:
        if (f.endswith("tar")):
            print("Found: " +  f)

nyt_dir = '/Users/Boya/Desktop/Courses/MATERIAL/nyt_corpus/data'
findtar(nyt_dir)
