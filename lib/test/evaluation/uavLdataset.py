import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


def get_num(str):
    i  =0
    while str[i] == '0':
        i+=1
    num = str[ i : -4]
    #(str,num)
    return int( num )


class UAVL_Dataset(BaseDataset):
    """
    DTBDataset dataset.
    """
    def __init__(self):
        super().__init__()
        #self.base_path = self.env_settings.dtb_path
        self.base_path = '/media/suixin/D/DataSet2/UAV123'
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'uav', ground_truth_rect[init_omit:,:] )

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        """seq name"""
        name_list = os.listdir(self.base_path + '/UAV123/anno/UAV20L/' )
        sequence_info_list = []
        """get ground_truth and img path"""
        for seq_name in name_list:
            if seq_name == 'att':
                continue
            seq_name = seq_name[: -4]
            img_path =   'UAV123/data_seq/UAV123/'+seq_name
            anno_path =  'UAV123/anno/UAV20L/' +  seq_name + '.txt'
            Frames = os.listdir( self.base_path  + '/'+ img_path  )
            Frames_num = [ get_num(Frames[i]) for i in range( len(Frames) )]
            EndFrame = max( Frames_num  )
            startFrame = min( Frames_num  )
            sequence_info_list.append(
            {"name":seq_name , "path": img_path, "startFrame":  startFrame , "endFrame": EndFrame , "nz": 6,
             "ext": "jpg", "anno_path": anno_path }
            )
        return  sequence_info_list




