import av
import os 
from tqdm import tqdm

file_dir = '/root/zhanghanlei/Datasets/IEMOCAP/video/Ses02M_impro08_F001.mp4'
av.open(file_dir, 'r')

# files = os.listdir(dirs)

# for file in tqdm(files, desc = 'iteration'):
#     container = av.open(os.path.join(dirs, file), 'r')
