import tarfile
import os

def extract(dname):
    inodes = [os.path.join(dname, f) for f in os.listdir(dname)]
    files = [f for f in inodes if os.path.isfile(f)]
    dirs = [f for f in inodes if os.path.isdir(f)]
    for d in dirs:
        extract(d)
    for f in files:
        print("Untar: " +  f)
        if (f.endswith("tgz")):
            tar = tarfile.open(f, "r:gz")
            tar.extractall(path=dname)
            tar.close()

nyt_dir = '/Users/Boya/Desktop/Courses/MATERIAL/nyt_corpus/data/1987'
extract(nyt_dir)
