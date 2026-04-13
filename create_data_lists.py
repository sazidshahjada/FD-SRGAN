from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['/media/iot/HDD2TB/coco2017/train2017',
                                     '/media/iot/HDD2TB/coco2017/test2017'],
                      test_folders=['/media/iot/HDD2TB/sr_eval/BSDS100',
                                    '/media/iot/HDD2TB/sr_eval/Set5/original',
                                    '/media/iot/HDD2TB/sr_eval/Set14/original'],
                      min_size=100,
                      output_folder='./')
