from torch.utils import data as data
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import os
from pathlib import Path
import random
import numpy as np
import torch
import sys
import diff_esim_torch as esim_torch
# print(esim_torch.__file__)
import cv2

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file,
                                    recursive_glob)
from basicsr.data.event_util import events_to_voxel_grid, events_to_voxel_grid_pytorch, events_to_voxel_grid_haley, voxel_norm
from basicsr.data.transforms import augment, triple_random_crop,  random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, voxel2voxeltensor, padding, get_root_logger
from torch.utils.data.dataloader import default_collate


def get_stats(ten, name):
    tensor = torch.tensor(ten)
    print(name, tensor.min().cpu().numpy(), tensor.max().cpu().numpy(), tensor.mean().cpu().numpy(), tensor.std().cpu().numpy() )
    print()

class GoProSharpEventRecurrentDatasetESIM_imagesonly(data.Dataset):
    """GoPro dataset for training recurrent networks for sharp image interpolation.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot (str): Data root path.
            io_backend (dict): IO backend type and other kwarg.
            num_end_interpolation (int): Number of sharp frames to reconstruct in each blurry image.
            num_inter_interpolation (int): Number of sharp frames to interpolate between two blurry images.
            phase (str): 'train' or 'test'

            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    Returns:
        two image (no paired voxels)
        corresponding voxels in the range
    """

    def __init__(self, opt):
        super(GoProSharpEventRecurrentDatasetESIM_imagesonly, self).__init__()
        self.debug = opt["debug"]
        self.device = torch.cuda.current_device()
        self.opt = opt
        self.dataroot = Path(opt['dataroot'])
        self.m = opt['num_end_interpolation']
        assert self.m == 1, 'num of frames must be 1 for sharp image interpolation!'
        self.n = opt['num_inter_interpolation']
        self.split = 'train' if opt['phase']=='train' else 'test' # train or test
        self.norm_voxel = opt.get('norm_voxel', True)
        
        self.small_set = opt.get('small_set', False)
        self.linear_regime = opt.get('linear', False)
        self.new_log_regime = opt.get('new_log', False)
        self.use_non_overlapping_frames = opt.get('use_non_overlapping_frames', False) # mainly for testing. 
        # print(self.linear_regime)
        # sys.exit()
        
        ## the sequence names
        train_video_list = [
            'GOPR0372_07_00', 'GOPR0374_11_01', 'GOPR0378_13_00', 'GOPR0384_11_01', 'GOPR0384_11_04', 'GOPR0477_11_00', 'GOPR0868_11_02', 'GOPR0884_11_00', 
            'GOPR0372_07_01', 'GOPR0374_11_02', 'GOPR0379_11_00', 'GOPR0384_11_02', 'GOPR0385_11_00', 'GOPR0857_11_00', 'GOPR0871_11_01', 'GOPR0374_11_00', 
            'GOPR0374_11_03', 'GOPR0380_11_00', 'GOPR0384_11_03', 'GOPR0386_11_00', 'GOPR0868_11_01', 'GOPR0881_11_00']
        test_video_list = [
            'GOPR0384_11_00', 'GOPR0385_11_01', 'GOPR0410_11_00', 'GOPR0862_11_00', 'GOPR0869_11_00', 'GOPR0881_11_01', 'GOPR0384_11_05', 'GOPR0396_11_00', 
            'GOPR0854_11_00', 'GOPR0868_11_00', 'GOPR0871_11_00']
        
        if self.small_set:
            train_video_list = ['GOPR0372_07_00']
            test_video_list = ['GOPR0384_11_00']
        
        video_list = train_video_list if self.split=='train' else test_video_list
        self.setLength = self.n + 2
        self.imageSeqsPath = [] 
        self.timestamps = []

        for video in video_list:
            timestamp_file_path = os.path.join(self.dataroot, self.split, video, "timestamps.txt")
            timestamps_list = self.txt_to_list(timestamp_file_path)
            frames  = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, video, 'imgs'), suffix='.png'))  # all sharp frames in one video

            # print('DEBUG: frames:{}'.format(frames))
            # del(frames[0]) # del the first image of the seq, because we want to use the event before first image
            

            if self.use_non_overlapping_frames: # ok so this is like 1-9, 9-18, 18-27 etc
                n_sets = len(frames)//self.setLength
                videoInputs = [frames[i:i + self.setLength] for i in range(0, len(frames), self.setLength)]
                timeInputs = [timestamps_list[i:i + self.setLength] for i in range(0, len(timestamps_list), self.setLength)]
                videoInputs = [[os.path.join(self.dataroot, self.split, video, 'imgs', f) for f in group] for group in videoInputs] # GOPR0372_07_00/xxx.png ...
                valid_videoInputs = [videoInput_i for videoInput_i in videoInputs if len(videoInput_i) == self.setLength]
                self.imageSeqsPath.extend(valid_videoInputs)# list of lists of inputs
                self.timestamps.extend(timeInputs)# list of lists of inputs
                # for i in range(len(self.imageSeqsPath)):
                #     print(self.imageSeqsPath[i])
                #     print()
                # print("--------------------------------")
                # print(len(self.imageSeqsPath))
            else: # this one was used for training. 1-9 10-19 20-29 etc
                n_sets = (len(frames) - self.setLength)//(self.n + 1)  + 1
                videoInputs = [frames[(self.n +1)*i:(self.n +1)*i+self.setLength] for i in range(n_sets)]
                timeInputs = [timestamps_list[(self.n +1)*i:(self.n +1)*i+self.setLength] for i in range(n_sets)]
                videoInputs = [[os.path.join(self.dataroot, self.split, video, 'imgs', f) for f in group] for group in videoInputs] # GOPR0372_07_00/xxx.png ...
                self.imageSeqsPath.extend(videoInputs)# list of lists of inputs
                self.timestamps.extend(timeInputs)# list of lists of inputs
                # for i in range(len(self.imageSeqsPath)):
                #     print(self.imageSeqsPath[i])
                #     print()
                # print("--------------------------------")
                # print(len(self.imageSeqsPath))
        
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        # temporal augmentation configs
        if not self.small_set:
            self.random_reverse = opt.get('random_reverse', False)
            logger = get_root_logger()
            logger.info(f'Temporal augmentation: random reverse is {self.random_reverse}.')
    
    def txt_to_list(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            timestamps = []
            for line in lines:
                timestamps.append(np.float64(line.strip()))
        return timestamps
    
    def get_seq(self):
        return self.imageSeqsPath

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']

        if self.debug:
            index = 0
            # print(self.imageSeqsPath[index])

        all_image_paths = self.imageSeqsPath[index]
        timestamps = torch.from_numpy(np.array(self.timestamps[index]))#.to(self.device) # seconds
        # timestamps_ns = (timestamps * 1e9).astype("int64")
        # timestamps_ns = torch.from_numpy(timestamps_ns).to(self.device) # nanoseconds

        if not self.debug and not self.small_set:
            if self.random_reverse and random.random() < 0.5:
                all_image_paths.reverse()


        # read for event making -- ESIM reading
        # all_images, grayscale, log
        log_images = []
        for im_path in all_image_paths:
            # if not self.linear_regime:
            image = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)           # 0 255
            if self.linear_regime:
                log_image = image.astype("float32") / 255 
            else:
                if self.new_log_regime:
                    log_image = np.log(image.astype("float32") / 255 + 1)    # -5 0 ???
                else:
                    log_image = np.log(image.astype("float32") / 255 + 1e-5)    # -5 0 ???
            log_image = log_image[:,:,np.newaxis]
            log_images.append(log_image)
        

        # read for reconstruction -- REFID reading
        # lq
        # gt
        input_idx = [0,-1]
        gt_idx = list(range(1, self.setLength-1))
        image_paths = [all_image_paths[idx] for idx in input_idx] 
        gt_paths = [all_image_paths[idx] for idx in gt_idx]
        
        if not self.debug and not self.small_set:
            if self.random_reverse and random.random() < 0.5:
                image_paths.reverse()
                gt_paths.reverse()
            # TODO: reverse event

        ## Read blur and gt sharps
        img_lqs = [] # these are the two intensity frames
        img_gts = [] # these are the frames between the intensity lqs
        for image_path in image_paths:
            # get LQ
            img_bytes = self.file_client.get(image_path)  # 'lq'
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)
            
            # all_imgs.append(img_lq)

        for gt_path in gt_paths:
            # get GT
            img_bytes = self.file_client.get(gt_path)    # 'gt'
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)
        
            # all_imgs.append(img_gts)
        

        # if gt_size is not None and not self.debug:
        # print(img_gts[0].shape, img_lqs[0].shape, log_images[0].shape)
        # sys.exit()
        # print("gt", type(log_images))
        # print("img_lqs", type(img_lqs))
        # print("log_images", type(log_images))
        # sys.exit()
        if gt_size is not None: 
            img_gts, img_lqs, log_images = triple_random_crop(img_gts, img_lqs, log_images, gt_size, scale, gt_paths[0],self.debug)
        
        # print(type(img_gts[0]), type(img_lqs[0]), type(log_images[0]), "------------------------")
        # sys.exit()
        # print(img_gts[0].shape, img_lqs[0].shape, log_images[0].shape )
        # sys.exit()

        # augmentation - flip, rotate
        num_lq = len(img_lqs) if isinstance(img_lqs, list) else 1
        num_gt = len(img_gts) if isinstance(img_gts, list) else 1
        num_log = len(log_images) if isinstance(log_images, list) else 1

        img_lqs.extend(img_gts)
        img_lqs.extend(log_images) if isinstance(log_images,list) else img_lqs.append(log_images) # [img_lqs, img_gts, voxels]
        # for i in range(len(img_lqs)):
        #     print(i, img_lqs[i].shape)
        
        if not self.debug and not self.small_set:
            img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])
        else:
            img_results = img_lqs

        img_results = img2tensor(img_results) # hwc -> chw
        img_lqs = torch.stack(img_results[:num_lq], dim=0) # t,c,h,w
        img_gts = torch.stack(img_results[num_lq:num_lq+num_gt], dim=0)
        img_logs = torch.stack(img_results[num_lq+num_gt:], dim=0)
        
        # ## Norm voxel
        # if self.norm_voxel:
        #     for voxel in voxels_list:
        #         voxel = voxel_norm(voxel)


        blur0_path = all_image_paths[0]
        seq = blur0_path.split(f'{self.split}/')[1].split('/')[0]
        origin_index = os.path.basename(blur0_path).split('.')[0]
        
        
    
        # print('img_lqs', img_lqs.size()) # img_lqs torch.Size([2, 3, 256, 256])
        # print('img_gts', img_gts.size()) # img_gts torch.Size([15, 3, 256, 256])
        # print('log_images', img_logs.size()) # log_images torch.Size([17, 1, 256, 256])
        # print('timestamps', timestamps.size()) # timestamps torch.Size([17])
        # print('seq', seq)
        # print('origin_index', origin_index.size())
        # sys.exit()
        return {'lq': img_lqs, 'gt': img_gts, 'log_images': img_logs, 'img_times': timestamps, 'seq': seq, 'origin_index': origin_index}
        

    def __len__(self):
        return len(self.imageSeqsPath)

### Sun Peng
class GoProSharpEventRecurrentDataset(data.Dataset):
    """GoPro dataset for training recurrent networks for sharp image interpolation.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot (str): Data root path.
            io_backend (dict): IO backend type and other kwarg.
            num_end_interpolation (int): Number of sharp frames to reconstruct in each blurry image.
            num_inter_interpolation (int): Number of sharp frames to interpolate between two blurry images.
            phase (str): 'train' or 'test'

            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    Returns:
        two image (no paired voxels)
        corresponding voxels in the range
    """

    def __init__(self, opt):
        super(GoProSharpEventRecurrentDataset, self).__init__()
        self.debug = opt["debug"]
        self.opt = opt
        self.dataroot = Path(opt['dataroot'])
        self.m = opt['num_end_interpolation']
        assert self.m == 1, 'num of frames must be 1 for sharp image interpolation!'
        self.n = opt['num_inter_interpolation']
        self.num_input_blur = 2
        self.num_input_gt = 2*self.m + self.n
        self.num_bins = self.n + 1
        self.split = 'train' if opt['phase']=='train' else 'test' # train or test
        self.norm_voxel = opt.get('norm_voxel', True)
        self.one_voxel_flg = opt.get('one_voxel_flag', True)
        self.return_deblur_voxel = opt.get('return_deblur_voxel', False) # false for sharp image interpolation
        self.return_deblur_voxel = self.return_deblur_voxel and self.one_voxel_flg
        self.refrac = opt["refrac"]

        ## the sequence names
        train_video_list = [
            'GOPR0372_07_00', 'GOPR0374_11_01', 'GOPR0378_13_00', 'GOPR0384_11_01', 'GOPR0384_11_04', 'GOPR0477_11_00', 'GOPR0868_11_02', 'GOPR0884_11_00', 
            'GOPR0372_07_01', 'GOPR0374_11_02', 'GOPR0379_11_00', 'GOPR0384_11_02', 'GOPR0385_11_00', 'GOPR0857_11_00', 'GOPR0871_11_01', 'GOPR0374_11_00', 
            'GOPR0374_11_03', 'GOPR0380_11_00', 'GOPR0384_11_03', 'GOPR0386_11_00', 'GOPR0868_11_01', 'GOPR0881_11_00']
        test_video_list = [
            'GOPR0384_11_00', 'GOPR0385_11_01', 'GOPR0410_11_00', 'GOPR0862_11_00', 'GOPR0869_11_00', 'GOPR0881_11_01', 'GOPR0384_11_05', 'GOPR0396_11_00', 
            'GOPR0854_11_00', 'GOPR0868_11_00', 'GOPR0871_11_00']
        
        self.small_set = opt.get('small_set', False)
        if self.small_set:
            train_video_list = ['GOPR0372_07_00']
            test_video_list = ['GOPR0384_11_00']
        video_list = train_video_list if self.split=='train' else test_video_list
        self.setLength = self.n + 2
        self.imageSeqsPath = [] 
        self.eventSeqsPath = [] # list of lists of event frames
        ### Formate file lists
        # print(video_list)
        # sys.exit()
        for video in video_list:
            # print('PATH')
            # print(os.path.join(self.dataroot, self.split, video, 'gt'))
            # print(os.path.join(self.dataroot, self.split, video, 'imgs'))
            # assert os.path.exists(os.path.join(self.dataroot, self.split, video, 'imgs'))
            # sys.exit()
            ## frames
            frames  = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, video, 'imgs'), suffix='.png'))  # all sharp frames in one video
            # print('DEBUG: frames:{}'.format(frames))
            # del(frames[0]) # del the first image of the seq, because we want to use the event before first image
            n_sets = (len(frames) - self.setLength)//(self.n + 1)  + 1

            videoInputs = [frames[(self.n +1)*i:(self.n +1)*i+self.setLength] for i in range(n_sets)]
            videoInputs = [[os.path.join(self.dataroot, self.split, video, 'imgs', f) for f in group] for group in videoInputs] # GOPR0372_07_00/xxx.png ...
            
            self.imageSeqsPath.extend(videoInputs)# list of lists of paired blur input, e.g.:
            # [['GOPR0372_07_00/blur/000328.png', 'GOPR0372_07_00/blur/000342.png'],
            #  ['GOPR0372_07_00/blur/000342.png', 'GOPR0372_07_00/blur/000356.png']]
            ## events
            event_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split+'_events', video), suffix='.npz'))  # all sharp frames in one video
            # events
            eventInputs = [event_frames[(self.n +1)*i :(self.n +1)*i+self.setLength -1] for i in range(n_sets)]
            if self.refrac:
                # print("REFRAC")
                # sys.exit()
                eventInputs = [[os.path.join(self.dataroot, self.split+'_events', video, f) for f in group] for group in eventInputs] # GOPR0372_07_00/xxx.png ...
            else:
                print("NO REFRAC")
                # sys.exit()
                eventInputs = [[os.path.join(self.dataroot, self.split+'_events_no_refrac', video, f) for f in group] for group in eventInputs] # GOPR0372_07_00/xxx.png ...
            self.eventSeqsPath.extend(eventInputs)
            # print(len(eventInputs))
            # sys.exit()


        assert len(self.imageSeqsPath)==len(self.eventSeqsPath), 'The number of sharp/interpo: {}/{} does not match.'
        
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        # temporal augmentation configs
        self.random_reverse = opt.get('random_reverse', False)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation: random reverse is {self.random_reverse}.')


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']

        if self.debug:
            index = 0
            # print(self.imageSeqsPath[index])

        all_image_paths = self.imageSeqsPath[index]
        event_paths = self.eventSeqsPath[index]
        
        input_idx = [0,-1]
        gt_idx = list(range(1, self.setLength-1))
        image_paths = [all_image_paths[idx] for idx in input_idx] 
        gt_paths = [all_image_paths[idx] for idx in gt_idx]
        assert len(event_paths) == len(gt_paths)+1, 'The length error'
        # print("[DEBUG]: len of event_paths:{}".format(len(event_paths))) # 8
        # print("[DEBUG]: len of gt_paths:{}".format(len(gt_paths))) # 7
        # random reverse

        if not self.debug and not self.small_set:
            if self.random_reverse and random.random() < 0.5:
                image_paths.reverse()
                gt_paths.reverse()
            # TODO: reverse event

        ## Read blur and gt sharps
        img_lqs = [] # these are the two intensity frames
        img_gts = [] # these are the frames between the intensity lqs
        for image_path in image_paths:
            # get LQ
            img_bytes = self.file_client.get(image_path)  # 'lq'
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)
            # all_imgs.append(img_lq)

        for gt_path in gt_paths:
            # get GT
            img_bytes = self.file_client.get(gt_path)    # 'gt'
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)
            # all_imgs.append(img_gts)

        #--------------------------------------------------------------
        
        h_lq, w_lq, _ = img_lqs[0].shape
        ## Read event and convert to voxel grid:
        # print(len(event_paths))
        # sys.exit()
        events = [np.load(event_path) for event_path in event_paths]
        # print("num events " , len(events))
        # sys.exit()
        # npz -> ndarray
        voxels = []
        if self.one_voxel_flg:
            all_quad_event_array = np.zeros((0,4)).astype(np.float32)
            for event in events:
                x = event['x'].astype(np.float32)[:,np.newaxis]
                y = event['y'].astype(np.float32)[:,np.newaxis]
                t = event['t'].astype(np.float32)[:,np.newaxis]
                p = event['p'].astype(np.float32)[:,np.newaxis]
                # print(x.shape, np.sum(np.nonzero(x)))
                # print(y.shape, np.sum(np.nonzero(y)))
                # print(t.shape, t.mean())
                # print(p.shape, np.sum(np.nonzero(p)), np.unique(p))
                # sys.exit()
                this_quad_event_array = np.concatenate((t,x,y,p),axis=1) # N,4
                # print("before" , all_quad_event_array.shape, this_quad_event_array.shape)
                all_quad_event_array = np.concatenate((all_quad_event_array, this_quad_event_array), axis=0)
                # print("after", all_quad_event_array.shape )
            voxel = events_to_voxel_grid(all_quad_event_array, num_bins=self.num_bins, width=w_lq, height=h_lq, return_format='HWC')
            # Voxel Norm
            # if self.norm_voxel:
            #     voxel = voxel_norm(voxel)
            print(f'voxel shape', voxel.shape)
            voxels.append(voxel) # len=1, shape:h,w,num_bins
            print(len(voxels), "LEN VOXELS")

            

            # num_bins,h,w
        else:
            for i in range(len(events)):
                x = events[i]['x'].astype(np.float32)[:,np.newaxis]
                y = events[i]['y'].astype(np.float32)[:,np.newaxis]
                t = events[i]['t'].astype(np.float32)[:,np.newaxis]
                p = events[i]['p'].astype(np.float32)[:,np.newaxis]
                this_quad_event_array = np.concatenate((t,x,y,p),axis=1) # N,4
                if i == 0:
                    last_quad_event_array = this_quad_event_array
                elif i >=1:
                    two_quad_event_array = np.concatenate((last_quad_event_array, this_quad_event_array), axis=0)
                    sub_voxel = events_to_voxel_grid(two_quad_event_array, num_bins=2,width=w_lq, height=h_lq, return_format='HWC')
                    voxels.append(sub_voxel)
                    last_quad_event_array = this_quad_event_array
            # print("voxel grid ", sub_voxel.size(), len(voxels))
                # len=2m+n+1, each with shape: h,w,2
        # Voxel: list of chw or hwc
        # randomly crop
        # voxel shape: h,w,c
        #----------------------------------------------------------------


        if gt_size is not None:
            img_gts, img_lqs, voxels = triple_random_crop(img_gts, img_lqs, voxels, gt_size, scale, gt_paths[0], self.debug)

        # augmentation - flip, rotate
        num_lq = len(img_lqs) if isinstance(img_lqs, list) else 1
        num_gt = len(img_gts) if isinstance(img_gts, list) else 1
        num_voxel = len(voxels) if isinstance(voxels, list) else 1

        img_lqs.extend(img_gts)
        img_lqs.extend(voxels) if isinstance(voxels,list) else img_lqs.append(voxels) # [img_lqs, img_gts, voxels]
        if not self.debug and not self.small_set:
            img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])
        else:
            img_results = img_lqs

        img_results = img2tensor(img_results) # hwc -> chw
        img_lqs = torch.stack(img_results[:num_lq], dim=0) # t,c,h,w
        # print(img_lqs.size())
        # sys.exit()
        img_gts = torch.stack(img_results[num_lq:num_lq+num_gt], dim=0)
        
        voxels_list = img_results[num_lq+num_gt:]
        ## Norm voxel
        
        if self.norm_voxel:
            print("___________________NORMALIZING_________________")
            for voxel in voxels_list:
                print("BEFORE NORMALIZATION VOXEL ", voxel.min(), voxel.mean(), voxel.max()) 
                voxel = voxel_norm(voxel)
                print("AFTER NORMALIZATION VOXEL ", voxel.min(), voxel.mean(), voxel.max()) 

        for voxel in voxels_list:
            print("SHOULD BE NORMALIZED VOXEL ", voxel.min(), voxel.mean(), voxel.max()) 

        # make deblur voxel
        if self.return_deblur_voxel:
            # print('DEBUG: Total_voxel.shape:{}'.format(voxels_list[0].shape))
            # print('DEBUG: left_deblur_voxel.shape:{}'.format(left_deblur_voxel.shape))
            # print('DEBUG: right_deblur_voxel.shape:{}'.format(right_deblur_voxel.shape))

            left_lq = img_lqs[0,:,:,:]
            right_lq = img_lqs[1,:,:,:]
            # print('DEBUG: img_lqs.shape:{}'.format(img_lqs.shape))
            _, h, w = left_lq.shape
            left_deblur_voxel = torch.zeros((10,h,w)) # 10 for 11 making blur
            right_deblur_voxel = torch.zeros((10,h,w))
            img_lqs = torch.cat((left_lq, left_deblur_voxel, right_lq, right_deblur_voxel), dim=0) # c,h,w

        voxels = torch.stack(voxels_list, dim=0) # t,c,h,w

        # reshape of the voxel tensor   1, num_bins, h, w -> t, 2, h, w
        if self.one_voxel_flg:
            voxels = voxels.squeeze(0)
            all_voxel = []
            for i in range(voxels.shape[0]-1):
                sub_voxel = voxels[i:i+2, :, :]
                all_voxel.append(sub_voxel)
            voxels = torch.stack(all_voxel, dim=0)


        # img_gts: (t, c, h, w)
        # voxels: (t, num_bins (2), h, w)
        blur0_path = image_paths[0]
        # print('blur0_path:{}'.format(blur0_path))
        seq = blur0_path.split(f'{self.split}/')[1].split('/')[0]
        origin_index = os.path.basename(blur0_path).split('.')[0]
        

        # print(voxels.size(), "voxels size")
        # get_stats(voxels, "voxels")
        # sys.exit()
        if self.split == 'train':
            return {'lq': img_lqs, 'gt': img_gts, 'voxel': voxels, 'seq': seq, 'origin_index': origin_index}
        else:
            return {'lq': img_lqs, 'gt': img_gts, 'voxel': voxels, 'seq': seq, 'origin_index': origin_index}

    def __len__(self):
        return len(self.imageSeqsPath)
    
    def get_seq(self):
        return self.imageSeqsPath


class GoProSharpwithVoxelEventRecurrentDataset(data.Dataset):
    """GoPro dataset for training recurrent networks for sharp image interpolation.
        For the two sharp image, also include correspoding voxels (for deblur)

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot (str): Data root path.
            io_backend (dict): IO backend type and other kwarg.
            num_end_interpolation (int): Number of sharp frames to reconstruct in each blurry image.
            num_inter_interpolation (int): Number of sharp frames to interpolate between two blurry images.
            phase (str): 'train' or 'test'

            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    Returns:
        two sharp img (and paired voxels, cat in c)
        corresponding voxels in the range

    """

    def __init__(self, opt):
        super(GoProSharpwithVoxelEventRecurrentDataset, self).__init__()
        self.opt = opt
        self.dataroot = Path(opt['dataroot'])
        self.m = opt['num_end_interpolation']
        assert self.m == 1, 'num of frames must be 1 for sharp image interpolation!'
        self.n = opt['num_inter_interpolation']  ## m=1, n=7 or 15 for sharp image interpolation
        self.num_input_blur = 2
        self.num_input_gt = 2*self.m + self.n
        self.num_bins = self.n + 1
        self.split = 'train' if opt['phase']=='train' else 'test' # train or test
        self.norm_voxel = opt.get('norm_voxel', True)
        self.one_voxel_flg = opt.get('one_voxel_flag', True)
        self.return_deblur_voxel = opt.get('return_deblur_voxel', False) # false for sharp image interpolation
        self.return_deblur_voxel = self.return_deblur_voxel and self.one_voxel_flg

        ## the sequence names
        train_video_list = [
            'GOPR0372_07_00', 'GOPR0374_11_01', 'GOPR0378_13_00', 'GOPR0384_11_01', 'GOPR0384_11_04', 'GOPR0477_11_00', 'GOPR0868_11_02', 'GOPR0884_11_00', 
            'GOPR0372_07_01', 'GOPR0374_11_02', 'GOPR0379_11_00', 'GOPR0384_11_02', 'GOPR0385_11_00', 'GOPR0857_11_00', 'GOPR0871_11_01', 'GOPR0374_11_00', 
            'GOPR0374_11_03', 'GOPR0380_11_00', 'GOPR0384_11_03', 'GOPR0386_11_00', 'GOPR0868_11_01', 'GOPR0881_11_00']
        test_video_list = [
            'GOPR0384_11_00', 'GOPR0385_11_01', 'GOPR0410_11_00', 'GOPR0862_11_00', 'GOPR0869_11_00', 'GOPR0881_11_01', 'GOPR0384_11_05', 'GOPR0396_11_00', 
            'GOPR0854_11_00', 'GOPR0868_11_00', 'GOPR0871_11_00']
        video_list = train_video_list if self.split=='train' else test_video_list
        self.setLength = self.n + 2
        self.imageSeqsPath = [] 
        self.eventSeqsPath = [] # list of lists of event frames
        ### Formate file lists
        for video in video_list:
            ## frames
            frames  = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, video, 'gt'), suffix='.png'))  # all sharp frames in one video
            # print('DEBUG: frames:{}'.format(frames))
            # del(frames[0]) # del the first image of the seq, because we want to use the event before first image
            n_sets = (len(frames) - self.setLength)//(self.n + 1)  + 1

            videoInputs = [frames[(self.n +1)*i:(self.n +1)*i+self.setLength] for i in range(n_sets)]
            videoInputs = [[os.path.join(self.dataroot, self.split, video, 'gt', f) for f in group] for group in videoInputs] # GOPR0372_07_00/xxx.png ...
            self.imageSeqsPath.extend(videoInputs)# list of lists of paired blur input, e.g.:
            # [['GOPR0372_07_00/blur/000328.png', 'GOPR0372_07_00/blur/000342.png'],
            #  ['GOPR0372_07_00/blur/000342.png', 'GOPR0372_07_00/blur/000356.png']]
            ## events
            event_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split+'_event', video), suffix='.npz'))  # all sharp frames in one video
            # events
            eventInputs = [event_frames[(self.n +1)*i :(self.n +1)*i+self.setLength -1] for i in range(n_sets)]
            eventInputs = [[os.path.join(self.dataroot, self.split+'_event', video, f) for f in group] for group in eventInputs] # GOPR0372_07_00/xxx.png ...
            self.eventSeqsPath.extend(eventInputs)


        assert len(self.imageSeqsPath)==len(self.eventSeqsPath), 'The number of sharp/interpo: {}/{} does not match.'
        
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        # temporal augmentation configs
        self.random_reverse = opt.get('random_reverse', False)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation: random reverse is {self.random_reverse}.')


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']

        all_image_paths = self.imageSeqsPath[index]
        event_paths = self.eventSeqsPath[index]
        
        input_idx = [0,-1]
        gt_idx = list(range(1, self.setLength-1))
        image_paths = [all_image_paths[idx] for idx in input_idx]
        gt_paths = [all_image_paths[idx] for idx in gt_idx]
        assert len(event_paths) == len(gt_paths)+1, 'The length error'
        # print("[DEBUG]: len of event_paths:{}".format(len(event_paths))) # 8
        # print("[DEBUG]: len of gt_paths:{}".format(len(gt_paths))) # 7
        # random reverse
        if self.random_reverse and random.random() < 0.5:
            image_paths.reverse()
            gt_paths.reverse()
            # TODO: reverse event

        ## Read blur and gt sharps
        img_lqs = []
        img_gts = []
        for image_path in image_paths:
            # get LQ
            img_bytes = self.file_client.get(image_path)  # 'lq'
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

        for gt_path in gt_paths:
            # get GT
            img_bytes = self.file_client.get(gt_path)    # 'gt'
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        h_lq, w_lq, _ = img_lqs[0].shape
        ## Read event and convert to voxel grid:
        events = [np.load(event_path) for event_path in event_paths]
        # npz -> ndarray
        voxels = []
        if self.one_voxel_flg:
            all_quad_event_array = np.zeros((0,4)).astype(np.float32)
            for event in events:
                x = event['x'].astype(np.float32)[:,np.newaxis]
                y = event['y'].astype(np.float32)[:,np.newaxis]
                t = event['timestamp'].astype(np.float32)[:,np.newaxis]
                p = event['polarity'].astype(np.float32)[:,np.newaxis]
                this_quad_event_array = np.concatenate((t,x,y,p),axis=1) # N,4
                all_quad_event_array = np.concatenate((all_quad_event_array, this_quad_event_array), axis=0)
            voxel = events_to_voxel_grid(all_quad_event_array, num_bins=self.num_bins, width=w_lq, height=h_lq, return_format='HWC')
            # Voxel Norm
            # if self.norm_voxel:
            #     voxel = voxel_norm(voxel)
            voxels.append(voxel) # len=1, shape:h,w,num_bins

            # num_bins,h,w
        else:
            for i in range(len(events)):
                x = events[i]['x'].astype(np.float32)[:,np.newaxis]
                y = events[i]['y'].astype(np.float32)[:,np.newaxis]
                t = events[i]['timestamp'].astype(np.float32)[:,np.newaxis]
                p = events[i]['polarity'].astype(np.float32)[:,np.newaxis]
                this_quad_event_array = np.concatenate((t,x,y,p),axis=1) # N,4
                if i == 0:
                    last_quad_event_array = this_quad_event_array
                elif i >=1:
                    two_quad_event_array = np.concatenate((last_quad_event_array, this_quad_event_array), axis=0)
                    sub_voxel = events_to_voxel_grid(two_quad_event_array, num_bins=2,width=w_lq, height=h_lq, return_format='HWC')
                    voxels.append(sub_voxel)
                    last_quad_event_array = this_quad_event_array
                # len=2m+n+1, each with shape: h,w,2
        # Voxel: list of chw or hwc
        # randomly crop
        # voxel shape: h,w,c
        img_gts, img_lqs, voxels = triple_random_crop(img_gts, img_lqs, voxels, gt_size, scale, gt_paths[0])

        # augmentation - flip, rotate
        num_lq = len(img_lqs) if isinstance(img_lqs, list) else 1
        num_gt = len(img_gts) if isinstance(img_gts, list) else 1
        num_voxel = len(voxels) if isinstance(voxels, list) else 1

        img_lqs.extend(img_gts)
        img_lqs.extend(voxels) if isinstance(voxels,list) else img_lqs.append(voxels) # [img_lqs, img_gts, voxels]

        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results) # hwc -> chw
        img_lqs = torch.stack(img_results[:num_lq], dim=0) # t,c,h,w
        img_gts = torch.stack(img_results[num_lq:num_lq+num_gt], dim=0)
        voxels_list = img_results[num_lq+num_gt:]
        ## Norm voxel
        if self.norm_voxel:
            for voxel in voxels_list:
                voxel = voxel_norm(voxel)

        # make deblur voxel
        if self.return_deblur_voxel:
            left_deblur_voxel = voxels_list[0][1:self.m, :, :]
            right_deblur_voxel = voxels_list[0][self.m+2 + self.n:, :, :]
            # print('DEBUG: Total_voxel.shape:{}'.format(voxels_list[0].shape))
            # print('DEBUG: left_deblur_voxel.shape:{}'.format(left_deblur_voxel.shape))
            # print('DEBUG: right_deblur_voxel.shape:{}'.format(right_deblur_voxel.shape))
            # left_deblur_voxel = left_deblur_voxel.unsqueeze(0) # 1,c,h,w
            # right_deblur_voxel = right_deblur_voxel.unsqueeze(0)
            left_lq = img_lqs[0,:,:,:]
            right_lq = img_lqs[1,:,:,:]
            img_lqs = torch.cat((left_lq, left_deblur_voxel, right_lq, right_deblur_voxel), dim=0) # c,h,w

        voxels = torch.stack(voxels_list, dim=0) # t,c,h,w

        # reshape of the voxel tensor   1, num_bins, h, w -> t, 2, h, w
        if self.one_voxel_flg:
            voxels = voxels.squeeze(0)
            all_voxel = []
            for i in range(voxels.shape[0]-1):
                sub_voxel = voxels[i:i+2, :, :]
                all_voxel.append(sub_voxel)
            voxels = torch.stack(all_voxel, dim=0)

        # print('DEBUG: lq.shape:{}'.format(img_lqs.shape))
        # print('DEBUG: gt.shape:{}'.format(img_gts.shape))
        # print('DEBUG: voxel.shape:{}'.format(voxels.shape))

        # img_lqs: (t, c, h, w)  t=2
        # img_lqs: (3*2+(m-1)*2, h, w) if return_deblur_voxel

        # img_gts: (t, c, h, w)  t=skip
        # voxels: (t, num_bins (2), h, w)
        if self.split == 'train':
            return {'lq': img_lqs, 'gt': img_gts, 'voxel': voxels}
        else:
            blur0_path = image_paths[0]
            # print('blur0_path:{}'.format(blur0_path))
            seq = blur0_path.split('test/')[1].split('/')[0]
            origin_index = os.path.basename(blur0_path).split('.')[0]
            # print("DEBUG:0 seq:" + f'{seq}')
            # seq = seq[0]
            # print("DEBUG:1 seq:" + f'{seq}')
            return {'lq': img_lqs, 'gt': img_gts, 'voxel': voxels, 'seq': seq, 'origin_index': origin_index}


    def __len__(self):
        return len(self.imageSeqsPath)


class BsergbSharpEventRecurrentDataset(data.Dataset):
    """Bsergb dataset for training recurrent networks for sharp image interpolation.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot (str): Data root path.
            io_backend (dict): IO backend type and other kwarg.
            num_end_interpolation (int): Number of sharp frames to reconstruct in each blurry image.
            num_inter_interpolation (int): Number of sharp frames to interpolate between two blurry images.
            phase (str): 'train' or 'test'

            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(BsergbSharpEventRecurrentDataset, self).__init__()
        self.opt = opt
        self.dataroot = opt['dataroot']
        self.m = opt['num_end_interpolation']
        assert self.m == 1, 'num of frames must be 1 for sharp image interpolation!'
        self.n = opt['num_inter_interpolation']
        self.num_input_blur = 2
        self.num_input_gt = 2*self.m + self.n
        self.num_bins = self.n + 1
        if opt['phase']=='train':
            self.split = '3_TRAINING'
        elif opt['phase']=='val':
            self.split = '2_VALIDATION'
        elif opt['phase']=='test':
            self.split = '1_TEST'
        self.norm_voxel = opt.get('norm_voxel', True)
        self.one_voxel_flg = opt.get('one_voxel_flag', True)
        self.return_deblur_voxel = opt.get('return_deblur_voxel', False) # false for sharp image interpolation
        self.return_deblur_voxel = self.return_deblur_voxel and self.one_voxel_flg

        ## the sequence names
        video_list = os.listdir(os.path.join(self.dataroot, self.split))
        # print('debug: video list:{}'.format(video_list))
        self.setLength = self.n + 2
        self.imageSeqsPath = [] 
        self.eventSeqsPath = [] # list of lists of event frames
        ### Formate file lists
        for video in video_list:
            ## frames
            # print('debug: dir:{}'.format(os.path.join(self.dataroot, self.split, video, 'images')))
            frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, video, 'images'), suffix='.png'))  # all sharp frames in one video
            if len(frames)==0:
                print('Warning: 0 frames in {}'.format(os.path.join(self.dataroot, self.split, video, 'images')))
                continue
            frames.pop() # del the last image because image is 1 more than the events
            # print('DEBUG: frames:{}'.format(frames))
            # del(frames[0]) # del the first image of the seq, because we want to use the event before first image
            n_sets = (len(frames) - self.setLength)//(self.n + 1)  + 1

            videoInputs = [frames[(self.n +1)*i:(self.n +1)*i+self.setLength] for i in range(n_sets)]
            videoInputs = [[os.path.join(self.dataroot, self.split, video, 'images', f) for f in group] for group in videoInputs] # GOPR0372_07_00/xxx.png ...
            self.imageSeqsPath.extend(videoInputs)# list of lists of paired blur input, e.g.:
            # [['GOPR0372_07_00/blur/000328.png', 'GOPR0372_07_00/blur/000342.png'],
            #  ['GOPR0372_07_00/blur/000342.png', 'GOPR0372_07_00/blur/000356.png']]
            ## events
            event_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, video, 'events'), suffix='.npz'))  # all sharp frames in one video
            # events
            eventInputs = [event_frames[(self.n +1)*i :(self.n +1)*i+self.setLength -1] for i in range(n_sets)]
            eventInputs = [[os.path.join(self.dataroot, self.split, video, 'events', f) for f in group] for group in eventInputs] # GOPR0372_07_00/xxx.png ...
            self.eventSeqsPath.extend(eventInputs)


        assert len(self.imageSeqsPath)==len(self.eventSeqsPath), 'The number of sharp/interpo: {}/{} does not match.'
        
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        # temporal augmentation configs
        self.random_reverse = opt.get('random_reverse', False)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation: random reverse is {self.random_reverse}.')


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']

        all_image_paths = self.imageSeqsPath[index]
        event_paths = self.eventSeqsPath[index]
        
        input_idx = [0,-1]
        gt_idx = list(range(1, self.setLength-1))
        image_paths = [all_image_paths[idx] for idx in input_idx]
        gt_paths = [all_image_paths[idx] for idx in gt_idx]
        assert len(event_paths) == len(gt_paths)+1, \
            'The length error, num of events:{}, num of images:{}'.format(len(event_paths), len(gt_paths))
        print("[DEBUG]: image_paths:{}".format(image_paths)) # 8
        print("[DEBUG]: gt_paths:{}".format(gt_paths)) # 7
        print("[DEBUG]: event_paths:{}".format(event_paths)) # 8

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            image_paths.reverse()
            gt_paths.reverse()
            # TODO: reverse event

        ## Read blur and gt sharps
        img_lqs = []
        img_gts = []
        for image_path in image_paths:
            # get LQ
            img_bytes = self.file_client.get(image_path)  # 'lq'
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

        for gt_path in gt_paths:
            # get GT
            img_bytes = self.file_client.get(gt_path)    # 'gt'
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        h_lq, w_lq, _ = img_lqs[0].shape
        ## Read event and convert to voxel grid:
        events = [np.load(event_path) for event_path in event_paths]
        # npz -> ndarray
        voxels = []
        if self.one_voxel_flg:
            all_quad_event_array = np.zeros((0,4)).astype(np.float32)
            for event in events:
                ## alignmen for BSERGB dataset
                x = event['x'].astype(np.float32)[:,np.newaxis]
                x = x/32
                x[x>=w_lq]=w_lq-1
                ## alignmen for BSERGB dataset
                y = event['y'].astype(np.float32)[:,np.newaxis]
                y = y/32
                y[y>=h_lq]=h_lq-1

                t = event['timestamp'].astype(np.float32)[:,np.newaxis]
                p = event['polarity'].astype(np.float32)[:,np.newaxis]
                this_quad_event_array = np.concatenate((t,x,y,p),axis=1) # N,4
                all_quad_event_array = np.concatenate((all_quad_event_array, this_quad_event_array), axis=0)
            voxel = events_to_voxel_grid(all_quad_event_array, num_bins=self.num_bins, width=w_lq, height=h_lq, return_format='HWC')
            # Voxel Norm
            # if self.norm_voxel:
            #     voxel = voxel_norm(voxel)
            voxels.append(voxel) # len=1, shape:h,w,num_bins

            # num_bins,h,w
        else:
            for i in range(len(events)):
                x = events[i]['x'].astype(np.float32)[:,np.newaxis]
                x = ((x/65535)*970*2.11)
                y = events[i]['y'].astype(np.float32)[:,np.newaxis]
                y = ((y/65535)*625*3.28)

                t = events[i]['timestamp'].astype(np.float32)[:,np.newaxis]
                p = events[i]['polarity'].astype(np.float32)[:,np.newaxis]
                this_quad_event_array = np.concatenate((t,x,y,p),axis=1) # N,4
                if i == 0:
                    last_quad_event_array = this_quad_event_array
                elif i >=1:
                    two_quad_event_array = np.concatenate((last_quad_event_array, this_quad_event_array), axis=0)
                    sub_voxel = events_to_voxel_grid(two_quad_event_array, num_bins=2,width=w_lq, height=h_lq, return_format='HWC')
                    voxels.append(sub_voxel)
                    last_quad_event_array = this_quad_event_array
                # len=2m+n+1, each with shape: h,w,2
        # Voxel: list of chw or hwc
        # randomly crop
        # voxel shape: h,w,c
        img_gts, img_lqs, voxels = triple_random_crop(img_gts, img_lqs, voxels, gt_size, scale, gt_paths[0])

        # augmentation - flip, rotate
        num_lq = len(img_lqs) if isinstance(img_lqs, list) else 1
        num_gt = len(img_gts) if isinstance(img_gts, list) else 1
        num_voxel = len(voxels) if isinstance(voxels, list) else 1

        img_lqs.extend(img_gts)
        img_lqs.extend(voxels) if isinstance(voxels,list) else img_lqs.append(voxels) # [img_lqs, img_gts, voxels]

        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results) # hwc -> chw
        img_lqs = torch.stack(img_results[:num_lq], dim=0) # t,c,h,w
        img_gts = torch.stack(img_results[num_lq:num_lq+num_gt], dim=0)
        voxels_list = img_results[num_lq+num_gt:]
        ## Norm voxel
        if self.norm_voxel:
            for voxel in voxels_list:
                voxel = voxel_norm(voxel)

        # make deblur voxel
        if self.return_deblur_voxel:
            left_deblur_voxel = voxels_list[0][1:self.m, :, :]
            right_deblur_voxel = voxels_list[0][self.m+2 + self.n:, :, :]
            # print('DEBUG: Total_voxel.shape:{}'.format(voxels_list[0].shape))
            # print('DEBUG: left_deblur_voxel.shape:{}'.format(left_deblur_voxel.shape))
            # print('DEBUG: right_deblur_voxel.shape:{}'.format(right_deblur_voxel.shape))
            # left_deblur_voxel = left_deblur_voxel.unsqueeze(0) # 1,c,h,w
            # right_deblur_voxel = right_deblur_voxel.unsqueeze(0)
            left_lq = img_lqs[0,:,:,:]
            right_lq = img_lqs[1,:,:,:]
            img_lqs = torch.cat((left_lq, left_deblur_voxel, right_lq, right_deblur_voxel), dim=0) # c,h,w

        voxels = torch.stack(voxels_list, dim=0) # t,c,h,w

        # reshape of the voxel tensor   1, num_bins, h, w -> t, 2, h, w
        if self.one_voxel_flg:
            voxels = voxels.squeeze(0)
            all_voxel = []
            for i in range(voxels.shape[0]-1):
                sub_voxel = voxels[i:i+2, :, :]
                all_voxel.append(sub_voxel)
            voxels = torch.stack(all_voxel, dim=0)

        # print('DEBUG: lq.shape:{}'.format(img_lqs.shape))
        # print('DEBUG: gt.shape:{}'.format(img_gts.shape))
        # print('DEBUG: voxel.shape:{}'.format(voxels.shape))

        # img_lqs: (t, c, h, w)
        # img_lqs: (3*2+(m-1)*2, h, w) if return_deblur_voxel

        # img_gts: (t, c, h, w)
        # voxels: (t, num_bins (2), h, w)
        if self.split == 'train':
            return {'lq': img_lqs, 'gt': img_gts, 'voxel': voxels}
        else:
            blur0_path = image_paths[0]
            print('blur0_path:{}'.format(blur0_path))
            seq = blur0_path.split('test')[1].split('blur')[0].strip('/')
            # print("DEBUG:0 seq:" + f'{seq}')
            # seq = seq[0]
            # print("DEBUG:1 seq:" + f'{seq}')
            return {'lq': img_lqs, 'gt': img_gts, 'voxel': voxels, 'seq': seq}



    def __len__(self):
        return len(self.imageSeqsPath)
