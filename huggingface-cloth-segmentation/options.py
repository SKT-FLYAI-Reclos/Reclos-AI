import os.path as osp
import os


class parser(object):
    def __init__(self):
        
        # self.output = "../data/"  # output image folder path  ./Output # 변환
        self.logs_dir = './logs'
        self.device = 'cuda:0'

opt = parser()