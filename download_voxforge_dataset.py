import urllib.request
import os
import tarfile

DATA_SRC="http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit"
DATA_TGZ="tgz"
DATA_EXTRACT="extracted"
VOXFORGE_FILE_LIST="part_voxforge_files.lst"

try:
    # Create target Directory
    os.mkdir(DATA_TGZ)
    print("Directory " , DATA_TGZ ,  " Created ") 
except FileExistsError:
    print("Directory " , DATA_TGZ ,  " already exists")

try:
    # Create target Directory
    os.mkdir(DATA_EXTRACT)
    print("Directory " , DATA_EXTRACT ,  " Created ") 
except FileExistsError:
    print("Directory " , DATA_EXTRACT ,  " already exists")

with open(VOXFORGE_FILE_LIST,'r') as fp:
    for line in fp:
        filename = line.rstrip()
        if filename:
            src_url = DATA_SRC + "/" + filename
            print(src_url)
            dst = os.path.join(DATA_TGZ,filename)
            urllib.request.urlretrieve(src_url,dst)
            print('downloading:',dst)
            if os.path.isfile(dst):
                tfile = tarfile.open(dst,'r:gz')
                tfile.extractall(DATA_EXTRACT)
print('\nDownloading is complete.')




