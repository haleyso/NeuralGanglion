import random
import math
import cv2
import numpy as np
import logging
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf

from diff_esim_torch.utils.lrgc_model_utils import lstkConv
from diff_esim_torch.utils.quantization_utils import differentiable_event_generation, FloorStraightThroughEstimator, CeilStraightThroughEstimator
from diff_esim_torch.utils.lrgc_emulator_utils import  generate_shot_noise,  check_nans_infs, initialize_identity, event_preprocess_pytorch
from diff_esim_torch.utils.emulator_utils import rescale_intensity_frame, lin_log, subtract_leak_current, low_pass_filter
import sys

def print_stats(tensor, name):
    print(name, tensor.min().detach().cpu().numpy(), tensor.max().detach().cpu().numpy(), tensor.mean().detach().cpu().numpy(), tensor.std().detach().cpu().numpy())

class GeneralEventEmulator(nn.Module):
    def __init__(self, opts, device):
        super(GeneralEventEmulator, self).__init__()
        
        self.debug = opts.get("debug", False)
        self.debug_nans = False # print infs and nans
        self.use_old_encoder = True
        self.device = device
        self.num_bins = opts["num_bins"] # for voxel grid
        self.output_mode = opts["output_mode"] # raw or voxel_grid
        self.output_bandwidth = opts["output_bandwidth"] # essentially max number events
        self.num_kinds_of_events = opts["output_channels"]

        
        self.spatially_vary = opts["mixed_threshold"]
        self.spatially_vary_k = opts.get("mixed_kernels", False)
        self.log_temporal = opts.get("log_temporal", False)
        if self.log_temporal:
            self.multiplier_curr = nn.Parameter(torch.tensor(1.0))
            self.multiplier_mem = nn.Parameter(torch.tensor(1.0))
        self.new_mem_update = opts.get("new_mem_update", False)
        self.event_offset = 1e-5
        if self.log_temporal:
            self.event_offset = 1.0

        self.learn_thresholds = opts["enable_threshold_learning"]
        # self.gradient_test_debug = nn.Parameter(torch.tensor(1.0))
        self.normalize_voxel = opts["normalize_voxel"]
        # event mode
        event_mode = opts["event_mode"]
        self.event_mode = event_mode

        if self.num_kinds_of_events > 1:
            assert self.event_mode == 'lrgc'
        
        if event_mode in ['dvs']:
            wts = torch.ones(1, 1, 1, 1)
            self.current_kernel = nn.Conv2d(1, 1, 1, bias=False, padding=0)
            self.current_kernel.weight.requires_grad=False
            self.current_kernel.weight.copy_(wts)

            self.mem_kernel = nn.Conv2d(1, 1, 1, bias=False, padding=0)
            self.mem_kernel.weight.requires_grad=False
            self.mem_kernel.weight.copy_(-1*wts)
            
        
        elif event_mode == 'csdvs_delbruck':
            # kernel applied after the differencing log images.
            # They said for the conv2d, zero padding pulls down the border pixels,
            # so they use replication padding to reduce this effect on border.
            # wts = torch.tensor([[[[0,  1,  0],
            #                       [1, -4,  1],
            #                       [0,  1,  0]]]], dtype=torch.float64)
            # above is the kernel from v2e's csdvs code, but it operates on the inverted input
            wts = torch.tensor([[[  [0,  -1,   0],
                                    [-1,  4,  -1],
                                    [0,  -1,   0]]]], dtype=torch.float64)
            self.diff_kernel = nn.Conv2d(1, 1, 3, bias=False, padding=1, padding_mode="replicate", dtype=torch.float64)
            self.diff_kernel.weight.requires_grad=False
            self.diff_kernel.weight.copy_(wts)
            with torch.no_grad():
                self.current_kernel = self.diff_kernel

        elif event_mode == 'csdvs':
            assert opts["kernel_size"]%2 ==1 # odd kernel
            P1 = (opts["kernel_size"]-1)//2
            wts = 1/(opts["kernel_size"]**2) * torch.ones(1, 1, opts["kernel_size"], opts["kernel_size"], dtype=torch.float64)
            wts[:,:,wts.size()[2]//2,wts.size()[3]//2 ] = 1.0
            
            self.current_kernel = nn.Conv2d(1, 1, opts["kernel_size"], bias=False, padding=P1, padding_mode="replicate", dtype=torch.float64)
            self.mem_kernel = nn.Conv2d(1, 1, opts["kernel_size"], bias=False, padding=P1, padding_mode="replicate", dtype=torch.float64)
            self.current_kernel.weight.requires_grad=False
            self.mem_kernel.weight.requires_grad=False
            self.current_kernel.weight.copy_(wts)
            self.mem_kernel.weight.copy_(-1*wts)

        elif event_mode == 'csdvs_new':
            assert opts["kernel_size"]%2 ==1 # odd kernel
            P1 = (opts["kernel_size"]-1)//2
            wts = -1/(opts["kernel_size"]**2) * torch.ones(1, 1, opts["kernel_size"], opts["kernel_size"], dtype=torch.float64)
            wts[:,:,wts.size()[2]//2,wts.size()[3]//2 ] = 1.0
            
            self.current_kernel = nn.Conv2d(1, 1, opts["kernel_size"], bias=False, padding=P1, padding_mode="replicate", dtype=torch.float64)
            self.mem_kernel = nn.Conv2d(1, 1, opts["kernel_size"], bias=False, padding=P1, padding_mode="replicate", dtype=torch.float64)
            self.current_kernel.weight.requires_grad=False
            self.mem_kernel.weight.requires_grad=False
            self.current_kernel.weight.copy_(wts)
            self.mem_kernel.weight.copy_(-1*wts)
        
        elif event_mode == 'csdvs_new_new':
            assert opts["kernel_size"]%2 ==1 # odd kernel
            P1 = (opts["kernel_size"]-1)//2
            wts = (-1/(opts["kernel_size"]**2-1)) * torch.ones(1, 1, opts["kernel_size"], opts["kernel_size"], dtype=torch.float64)
            wts[:,:,wts.size()[2]//2,wts.size()[3]//2 ] = 1.0
            
            self.current_kernel = nn.Conv2d(1, 1, opts["kernel_size"], bias=False, padding=P1, padding_mode="replicate", dtype=torch.float64)
            self.mem_kernel = nn.Conv2d(1, 1, opts["kernel_size"], bias=False, padding=P1, padding_mode="replicate", dtype=torch.float64)
            self.current_kernel.weight.requires_grad=False
            self.mem_kernel.weight.requires_grad=False
            self.current_kernel.weight.copy_(wts)
            self.mem_kernel.weight.copy_(-1*wts)

                
        elif event_mode == 'lrgc': # learned rgc
            P1 = (opts["kernel_size"]-1)//2
            if self.spatially_vary_k:                
                self.current_kernel = lstkConv(self.num_kinds_of_events, self.num_kinds_of_events, opts["kernel_size"], bias=opts["bias"], padding=P1, padding_mode=opts["padding_mode"] , dtype=torch.float64, mode='cur', groups=self.num_kinds_of_events)
                self.mem_kernel     = lstkConv(self.num_kinds_of_events, self.num_kinds_of_events, opts["kernel_size"], padding=P1, bias=opts["bias"], padding_mode=opts["padding_mode"] , dtype=torch.float64, mode='mem', groups=self.num_kinds_of_events)

                print("---------KERNELS INITIALIZED- spatially varying--------")
                print(self.current_kernel.event_conv.weight.shape, self.mem_kernel.event_conv.weight.shape)
            else:
                self.current_kernel = nn.Conv2d(self.num_kinds_of_events, self.num_kinds_of_events, opts["kernel_size"],  bias=opts["bias"], padding=P1, padding_mode=opts["padding_mode"] , dtype=torch.float64, groups=self.num_kinds_of_events)
                self.mem_kernel     = nn.Conv2d(self.num_kinds_of_events, self.num_kinds_of_events, opts["kernel_size"], padding=P1, bias=opts["bias"], padding_mode=opts["padding_mode"] , dtype=torch.float64, groups=self.num_kinds_of_events)
                                # if multichannel events, then each event kind has a separate memory frame of past event
                if not self.debug:
                    initialize_identity(self.current_kernel.weight, kernel_size=opts["kernel_size"], dtype=torch.float64, mode='cur')
                    initialize_identity(self.mem_kernel.weight,kernel_size=opts["kernel_size"], dtype=torch.float64, mode='mem')

                print("---------KERNELS INITIALIZED- not spatially varying--------")
                print(self.current_kernel.weight.shape, self.mem_kernel.weight.shape)
        elif event_mode == 'diff_of_gaussians':
            self.sigma_1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float64, requires_grad=True, device=self.device))
            self.sigma_2 = nn.Parameter(torch.tensor(2.0, dtype=torch.float64, requires_grad=True, device=self.device))
        
        elif event_mode == 'diff_of_gaussians_amp':
            self.sigma_1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float64, requires_grad=True, device=self.device))
            self.sigma_2 = nn.Parameter(torch.tensor(2.0, dtype=torch.float64, requires_grad=True, device=self.device))
            self.amp_1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float64, requires_grad=True, device=self.device))
            self.amp_2 = nn.Parameter(torch.tensor(1.0, dtype=torch.float64, requires_grad=True, device=self.device))

        elif event_mode == 'NoEvents':
            print("RUNNING W NO EVENTS.")
        else:
            print(f"Event mode {event_mode} not supported yet.")
            sys.exit()
        
        self.output_precision = opts["output_precision"]
        self.pos_thres_nominal = torch.tensor(opts["pos_thres"], dtype=torch.float64, requires_grad=self.learn_thresholds, device=self.device).repeat(self.num_kinds_of_events) # nominal threshold of triggering positive event in log intensity.
        self.neg_thres_nominal = torch.tensor(opts["neg_thres"], dtype=torch.float64, requires_grad=self.learn_thresholds, device=self.device).repeat(self.num_kinds_of_events) # nominal threshold of triggering negative event in log intensity.
        
        if self.learn_thresholds:
            self.pos_thres_nominal = nn.Parameter(self.pos_thres_nominal )
            self.neg_thres_nominal = nn.Parameter(self.neg_thres_nominal )

        self.sigma_thres = opts["sigma_thres"]                   # std deviation of threshold in log intensity.

        # mixed_threshold_params
        # spatially varying thresholds grid 2x2
        
        if self.spatially_vary: # thresholds
            self.pl = opts.get('pl', 1)
            self.ps = opts.get('ps', 1)
            self.p_pos = torch.tensor([[self.ps, self.pl],[self.pl, self.pl]], dtype=torch.float64, requires_grad=self.learn_thresholds, device=self.device).unsqueeze(0).repeat(self.num_kinds_of_events, 1, 1) # positive thresholds
            self.p_neg = torch.tensor([[self.ps, self.pl],[self.pl, self.pl]], dtype=torch.float64, requires_grad=self.learn_thresholds, device=self.device).unsqueeze(0).repeat(self.num_kinds_of_events, 1, 1) # negative thresholds

            if self.learn_thresholds:
                self.p_pos = nn.Parameter(self.p_pos)
                self.p_neg = nn.Parameter(self.p_neg)
                print('learned: ', self.p_pos.shape)


        
        self.refractory_period_s = torch.tensor(opts["refractory_period_s"], dtype=torch.float64, device=self.device)

        self.gen_events = differentiable_event_generation()
        self.floor_STE = FloorStraightThroughEstimator()
        self.ceil_STE = CeilStraightThroughEstimator()

        # output properties
        self.show_input = None

        # event stats
        self.reset()
        self.base_log_frame = None
        self.lp_log_frame0 = None
        self.num_events_total = 0
        self.num_events_on = 0
        self.num_events_off = 0
        self.t_previous = 0  # time of previous frame
        
    def _init(self, frame_log, Tr_frames):
        '''Initialise base frame and other parameters
        frame_log: torch.tensor                                 torch.Size([2, 1, 256, 256])
        Tr_frames: refractory period in the format as frames    torch.Size([2, 1, 1, 256, 256])
        '''
        
        # base_frame are memorized lin_log pixel values        
        self.base_log_frame = frame_log.repeat(1,self.num_kinds_of_events, 1, 1) if self.base_log_frame is None else self.base_log_frame  #lin_log(first_frame_linear)
        

        # initialize first stage of 2nd order IIR to first input  
        self.lp_log_frame0 = self.base_log_frame.clone().detach() if self.lp_log_frame0 is None else self.base_log_frame
        
        # positive and negative thresholds
        self.pos_thres_real_nominal = self.pos_thres_nominal.view(1, -1, 1, 1)*torch.ones(self.base_log_frame.size(), dtype=torch.float64, device=self.device)
        self.neg_thres_real_nominal = self.neg_thres_nominal.view(1, -1, 1, 1)*torch.ones(self.base_log_frame.size(), dtype=torch.float64, device=self.device)
        
        # slightly spatially varying thresholds.
        if self.sigma_thres > 0:
            self.pos_thres_noise = torch.normal(0, self.sigma_thres, size=self.pos_thres_real_nominal.size(), device=self.device, requires_grad=False)
            self.neg_thres_noise = torch.normal(0, self.sigma_thres, size=self.neg_thres_real_nominal.size(), device=self.device, requires_grad=False)

        # refractory period
        if (self.refractory_period_s > 0).any():
            self.timestamp_mem = torch.add(torch.zeros_like(self.base_log_frame.squeeze(2)), -Tr_frames.squeeze(2))
        else:
            self.timestamp_mem = torch.zeros_like(self.base_log_frame)
        

    def reset(self):
        '''resets so that next use will reinitialize the base frame
        '''
        self.num_events_total = 0
        self.num_events_on = 0
        self.num_events_off = 0
        self.base_log_frame = None
        self.lp_log_frame0 = None  # lowpass stage 0

        
    def _show(self, inp: np.ndarray):
        inp = np.array(inp.cpu().data[0,0,:,:].numpy())
        min = np.min(inp)
        norm = (np.max(inp) - min)
        if norm == 0:
            print('image is blank, max-min=0')
            norm = 1
        img = ((inp - min) / norm)
        cv2.imshow(self.show_input, img) #__name__+':'+
        cv2.waitKey(30) 
        
    def gaussian_kernel(self, size, sigma):
        x = torch.arange(size, device=self.device, dtype=torch.float64) - size // 2
        y = torch.arange(size, device=self.device, dtype=torch.float64) - size // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel/kernel.sum() # normalize
        kernel = kernel.unsqueeze(0).unsqueeze(0) # add batch and channel dimensions

        return kernel
    

    def apply_diff_of_gaussians(self, frame, sigma_1, sigma_2, amp_1=1, amp_2=1):

        # create two gaussians with the given sigmas
        k1 = int(6 * sigma_1) + 1  # Kernel size heuristic
        k2 = int(6 * sigma_2) + 1  # Kernel size heuristic

        if k1 % 2 == 0:
            k1 += 1
        if k2 % 2 == 0:
            k2 += 1

        gaussian_1 = amp_1*self.gaussian_kernel(k1, sigma_1)
        gaussian_2 = amp_2*self.gaussian_kernel(k2, sigma_2)

        # apply the gaussians to the input frame
        gaussian_1_applied = torch.nn.functional.conv2d(frame, gaussian_1, padding=k1//2)
        gaussian_2_applied = torch.nn.functional.conv2d(frame, gaussian_2, padding=k2//2)

        diff_of_gaussians = gaussian_1_applied - gaussian_2_applied

        return diff_of_gaussians

    def forward(self, frames, t_frames): #new_frames, rescaled_org_frames, t_frames
        
        # frames come in as a list of 9 of size(b, t, 3, 256, 256)
        # want the voxels to exit as #b, 7, 2, 256, 256

        batch_size, num_frames, channels, height, width = frames.size()

        if self.event_mode == 'NoEvents':
            events = torch.zeros((batch_size, num_frames-2, 1, 2, height, width), device=frames.device)
            # events = torch.zeros((batch_size, num_frames-2, 2, height, width), device=frames.device)
            total_num_events = torch.zeros((1), device=frames.device)
            event_count = torch.zeros((1), device=frames.device)
            return  events, total_num_events, event_count
        


        with torch.no_grad(): # Don't pass gradients while setting up time variables, rescaled frames, and voxel grid
            
            # t_float_frames (T length list with frame timestamps in seconds) # [b,len_seq] 
            if t_frames.size()[1] == 2:
                t_float_frames = torch.linspace(start=t_frames[0,0], end=t_frames[0,-1], steps=num_frames, dtype=torch.float64, device=self.device)
            else:
                t_float_frames = t_frames[0]
            
            # everything in units of seconds
            # print("t_frames ", t_frames) # tensor([[0.0000, 0.0042, 0.0083, 0.0125, 0.0167, 0.0208, 0.0250, 0.0292, 0.0333]]
            
            frame_duration = t_frames[0,1]-t_frames[0,0]  
            # print("frame duration ", frame_duration) # 0.0042
            bin_duration = (num_frames-1)/(self.num_bins-1) * frame_duration    
            # print("bin duration ", bin_duration)
            t_bins = torch.linspace(start=t_float_frames[0], end=bin_duration*(self.num_bins-1)+t_float_frames[0], steps=self.num_bins, dtype=torch.float64, device=self.device) 
            # print("t_bins ",t_bins)
            
            # self.refractory_period_s is the refractory period in seconds size: b,1,h,w
            t_refrac = torch.ones(batch_size, self.num_kinds_of_events, height, width, device=frames.device) * self.refractory_period_s
            frames = frames.squeeze(2)
            # print("after frames", frames.size() ) # after frames torch.Size([2, 9, 256, 256])



            if self.base_log_frame is None:
                self._init(frames[:,0:1,:,:], t_refrac) # timestamp is set to negative refractory time
                self.t_previous = t_frames[0,0] # time of previous intensity frame
            
            frames_filtered = frames

            
            if self.output_mode == 'voxel_grid':
                                        # batch, 7, 2 bins, h, w
                # events = torch.zeros((batch_size, num_frames-2, 2, height, width), devices=self.device)
                events = torch.zeros((batch_size, num_frames-2, self.num_kinds_of_events, 2, height, width), device=self.device) # now that we have multiple kinds of events 
            else:
                print("Have not implemented other output mode. Please use Voxel Grid")
                sys.exit()


        # set thresholds patterns
        b, num_e, h, w = self.base_log_frame.size()
        
        self.base_log_frame  = self.base_log_frame.type(torch.float64)
        if self.new_mem_update: 
            if self.event_mode == 'diff_of_gaussians':
                self.base_log_frame = self.apply_diff_of_gaussians(self.base_log_frame, self.sigma_1, self.sigma_2)
            elif self.event_mode == 'diff_of_gaussians_amp':
                self.base_log_frame = self.apply_diff_of_gaussians(self.base_log_frame, self.sigma_1, self.sigma_2, self.amp_1, self.amp_2)
            else:
                self.base_log_frame = self.current_kernel(self.base_log_frame)
            if self.log_temporal:
                # unclamped = torch.log(self.base_log_frame+self.event_offset)
                unclamped = torch.log(torch.clamp(self.base_log_frame+self.event_offset, min=1e-8)) # HS: added this min = 1e-8 to avoid nans
                self.base_log_frame = torch.clamp(unclamped, min=0.0).detach() + unclamped - unclamped.detach()
        
        if self.spatially_vary:
            # print(self.pos_thres_nominal.view(1, -1, 1, 1).size(), self.p_pos.repeat(b,1,round(h/2), round(w/2))[:,:,:h, :w].size())
            pos_thres = self.pos_thres_nominal.view(1, -1, 1, 1)*self.p_pos.repeat(b,1,round(h/2), round(w/2))[:,:,:h, :w]
            # print(pos_thres.size())
            # sys.exit()
            neg_thres = self.neg_thres_nominal.view(1, -1, 1, 1)*self.p_neg.repeat(b,1,round(h/2), round(w/2))[:,:,:h, :w]
        else:
            pos_thres = self.pos_thres_real_nominal #.view(1, -1, 1, 1) * torch.ones_like(self.base_log_frame)
            neg_thres = self.neg_thres_real_nominal #.view(1, -1, 1, 1) * torch.ones_like(self.base_log_frame)
        

        if self.sigma_thres > 0:
            pos_thres = self.pos_thres_noise + pos_thres
            neg_thres = self.neg_thres_noise + neg_thres
            
        # to avoid the situation where the threshold is too small. STE
        pos_thres = torch.clamp(pos_thres, min=0.01).detach() + pos_thres - pos_thres.detach()
        neg_thres = torch.clamp(neg_thres, min=0.01).detach() + neg_thres - neg_thres.detach()


        total_num_events = 0
        event_count = 0
        bin_minus_index = 0
        bin_plus_index  = 1

        duration = (self.num_bins-1)/(num_frames-1)
        # print(f'self.num_bins {self.num_bins}')
        # print(f'num_frames {num_frames}')
        # print(f'durati {self.num_bins-1} {num_frames-1}')
        # print("DURATION", duration, self.num_bins, num_frames, t_bins.size())
        # sys.exit()
        # 0    1    2    3    4    5    6    7    8
        # 0,0  0,1 
        #      1,0  1,1
        #           2,0  2,1 
        #                3,0  3,1
        #                     4,0  4,1 
        #                          5,0  5,1
        #                               6,0  6,1 
        
        # print_stats(self.base_log_frame, "self.base_log_frame")
        for n in range(1, num_frames-1):
            # print("n", n, int(bin_minus_index), int(bin_plus_index) )

            with torch.no_grad(): # Don't pass gradients while applying pre kernel noise to frames
                if self.debug:
                    print("DEBUG", n)
                    new_frame = (frames_filtered[:,0:1,...]).type(torch.float64).repeat(1, self.num_kinds_of_events, 1, 1)
                else:
                    new_frame = (frames_filtered[:,n:n+1,...]).type(torch.float64).repeat(1, self.num_kinds_of_events, 1, 1)
                # compute time difference between this and the previous frame
                delta_time = t_float_frames[n] - self.t_previous
                
      
            if self.event_mode == 'dvs':
                diff_frame = new_frame - self.base_log_frame 

            elif self.event_mode == 'csdvs_delbruck':
                diffed = new_frame  - self.base_log_frame
                diff_frame = self.diff_kernel(diffed)

            elif self.event_mode in ['lrgc','csdvs', 'diff_of_gaussians', 'csdvs_new', 'csdvs_new_new', 'diff_of_gaussians_amp']:

                if self.event_mode == 'diff_of_gaussians':
                    # DOG difference of gaussians: apply two gaussians to the input frame and take the difference.
                    curr_out = self.apply_diff_of_gaussians(new_frame, self.sigma_1, self.sigma_2)
                elif self.event_mode == 'diff_of_gaussians_amp':
                    curr_out = self.apply_diff_of_gaussians(new_frame, self.sigma_1, self.sigma_2, self.amp_1, self.amp_2)

                else:
                    # apply the spatio-temporal kernel
                    curr_out = self.current_kernel(new_frame)
                
                
                # new update -- kernel applied
                if self.new_mem_update:
                    
                    if self.log_temporal:
                        curr_out = torch.clamp(curr_out, min=1e-8).detach() + curr_out - curr_out.detach() # HS: added this min = 1e-8 to avoid nans
                        intermed = torch.log(curr_out + self.event_offset)
                        check_nans_infs(curr_out, "curr_out", self.debug_nans)
                        # check_nans_infs(curr_out.grad, "curr_out gradients", self.debug_nans)
                        if self.spatially_vary_k:
                            check_nans_infs(self.current_kernel.event_conv.weight, "self.current_kernel", self.debug_nans)
                        else:
                            check_nans_infs(self.current_kernel.weight, "self.current_kernel", self.debug_nans)
                        check_nans_infs(intermed, "intermed before clamp", self.debug_nans)
                        intermed = torch.clamp(intermed, min=0.0).detach() + intermed - intermed.detach()
                        check_nans_infs(self.multiplier_mem,"self.multiplier_mem", self.debug_nans)
                        diff_frame = self.multiplier_curr * intermed - self.base_log_frame.clone()* self.multiplier_mem 
                        check_nans_infs(intermed, "intermed after clamp", self.debug_nans)
                        check_nans_infs( self.base_log_frame, "base_log_frame", self.debug_nans)
                        check_nans_infs(diff_frame, "diff_frame", self.debug_nans)
                    else:
                        diff_frame = curr_out - self.base_log_frame
                # old update
                else:
                    mem_out = self.mem_kernel(self.base_log_frame)
                    if self.log_temporal:
                        diff_frame = self.multiplier_curr * (torch.log(torch.clamp(curr_out, min=1e-8))) - self.multiplier_mem * mem_out # HS: added this min = 1e-8 to avoid nans
                    else:
                        diff_frame = curr_out - mem_out


            if self.show_input:
                if self.show_input == 'new_frame':
                    self._show(new_frame)
                elif self.show_input == 'baseLogFrame':
                    self._show(self.base_log_frame)
                elif self.show_input == 'lpLogFrame0':
                    self._show(self.lp_log_frame0)
                elif self.show_input == 'lpLogFrame1':
                    self._show(self.lp_log_frame1)
                elif self.show_input == 'diff_frame':
                    self._show(diff_frame)



            # 
            bin_minus_time = t_bins[int(bin_minus_index)]
            bin_plus_time = t_bins[int(bin_plus_index)]
            FrameStartTime = t_float_frames[n-1] 
            FrameEndTime = t_float_frames[n]
            pr = FrameStartTime - self.timestamp_mem  # time of last generated event in previous sequence 

            # Continuous Conv -> Events with differentiable proxy
            # generate polarity mask: each pixel is in [-1,0,1]
            pol = self.gen_events.apply(diff_frame, 0.0) # 1e-6 is the tolerance
            threshold = torch.where(diff_frame >= 0, torch.clamp(pos_thres, min=0.01).detach() + pos_thres - pos_thres.detach(), -(torch.clamp(neg_thres, min=0.01).detach() + neg_thres - neg_thres.detach())) 

            # Discrete Computation for alpha_r <= FrameEndTime - FrameStartTime - beta_r (event fired)
            eps = 1e-9
            alpha   =  (FrameEndTime-FrameStartTime)/((diff_frame+eps*torch.where(diff_frame>=0, 1, -1))/threshold) 
            alpha_r = (self.floor_STE.apply(self.refractory_period_s/alpha) + 1)*alpha  #  spacing of events given a refractory period.
            beta_r  = (self.floor_STE.apply((self.refractory_period_s - pr)/alpha) ) * alpha    # time offset to the 1st event allowed after refractory period of the previous frame
            beta_r  = torch.where(beta_r>0, beta_r, 0) # HS: changed this. now no nans.
            num_events_r = self.floor_STE.apply((FrameEndTime-FrameStartTime-beta_r) / alpha_r)
            num_events_r = torch.where(alpha_r <= FrameEndTime - FrameStartTime - beta_r, num_events_r, 0) 

            # print_stats((self.refractory_period_s - pr), "(self.refractory_period_s - pr)")
            # print_stats(beta_r, "beta_r")
        
            # Continuous Computation for backward pass for alpha_r <= FrameEndTime - FrameStartTime - beta_r (event fired)      
            continuous_alpha_r = self.refractory_period_s + alpha # for gradient approximation
            continuous_beta_r = self.refractory_period_s - pr
            continuous_beta_r = torch.where(continuous_beta_r>0, continuous_beta_r, 0) 
            continuous_num_events_r = (FrameEndTime - FrameStartTime - torch.max(self.refractory_period_s - pr, torch.zeros_like(pr)) )/(continuous_alpha_r+eps)    # for gradient approximation.            

            # Combining continuous gradients and discrete values for alpha_r <= FrameEndTime - FrameStartTime - beta_r (event fired)
            alpha_r = alpha_r.detach() + continuous_alpha_r - continuous_alpha_r.detach()
            beta_r = beta_r.detach() + continuous_beta_r - continuous_beta_r.detach()
            num_events_r = num_events_r.detach() + continuous_num_events_r - continuous_num_events_r.detach()

            # Bin events for pixels with event firings
            bin_minus_frame = pol*(1- beta_r/(bin_plus_time-bin_minus_time))*num_events_r-(alpha_r/(bin_plus_time-bin_minus_time)) * pol  * (num_events_r*(num_events_r+1))/(2)
            bin_plus_frame  = pol*(   beta_r/(bin_plus_time-bin_minus_time))*num_events_r+(alpha_r/(bin_plus_time-bin_minus_time)) * pol  * (num_events_r*(num_events_r+1))/(2)


            # print_stats(pol*(   beta_r/(bin_plus_time-bin_minus_time))*num_events_r, "pol*(   beta_r/(bin_plus_time-bin_minus_time))*num_events_r")
            # print_stats((alpha_r/(bin_plus_time-bin_minus_time)) * pol  * (num_events_r*(num_events_r+1))/(2), "(alpha_r/(bin_plus_time-bin_minus_time)) * pol  * (num_events_r*(num_events_r+1))/(2)")
            
            # print_stats(pol, "pol")
            # Define Gradient for alpha_r <= FrameEndTime - FrameStartTime - beta_r (no events)
            continuous_num_events_r_noevents = diff_frame/threshold
            continuous_bin_minus_frame_noevents = torch.zeros_like(bin_minus_frame)
            continuous_bin_plus_frame_noevents = continuous_num_events_r_noevents

            # Overwrite gradients for pixels with no events to avoid NaN gradients         
            num_events_r = torch.where(alpha_r <= FrameEndTime - FrameStartTime - beta_r, num_events_r, continuous_num_events_r_noevents-continuous_num_events_r_noevents.detach())
            bin_minus_frame = torch.where(alpha_r <= FrameEndTime - FrameStartTime - beta_r, bin_minus_frame, continuous_bin_minus_frame_noevents-continuous_bin_minus_frame_noevents.detach())
            bin_plus_frame = torch.where(alpha_r <= FrameEndTime - FrameStartTime - beta_r, bin_plus_frame, continuous_bin_plus_frame_noevents-continuous_bin_plus_frame_noevents.detach())
            
            # Add up events for sparsity loss
            event_count += num_events_r.abs().sum()

            # batch_size, num_frames-1, self.num_bins, height, width
            events[:,n-1, :, 0, :, :]  = bin_minus_frame
            events[:,n-1, :, 1, :, :]  = bin_plus_frame
            
            with torch.no_grad(): 
                # update timestamp_mem
                self.t_previous = t_float_frames[n]
                self.timestamp_mem = torch.where( num_events_r==0,  self.timestamp_mem,  FrameStartTime + beta_r + alpha_r*num_events_r) # when it fired last

                bin_minus_index = bin_minus_index + duration + 1e-16 
                bin_plus_index  = bin_minus_index + 1

                total_num_events += torch.sum(num_events_r.abs()).detach()
                final_evts_frame = torch.where(num_events_r>0, 1, 0)*num_events_r
                C = torch.mul(pos_thres, pol>0) + torch.mul(neg_thres,pol<0) 
                self.base_log_frame += pol*final_evts_frame*C # this updates the current intensity value by the number of events that was fired.

                check_nans_infs(self.base_log_frame, "self.base_log_frame",self.debug_nans)
                check_nans_infs(self.base_log_frame, "base_log_frame",self.debug_nans)
                
            
            check_nans_infs(self.base_log_frame, "base_log_frame",self.debug_nans)
            check_nans_infs(self.timestamp_mem, "timestamp_mem",self.debug_nans)
            check_nans_infs(events, "events after",self.debug_nans)



        # Add bin plus prev and bin minus next
        for t_index in range(events.shape[1]-1):
            events[:,t_index,:,1,:,:] = events[:,t_index,:,1,:,:] + events[:,t_index+1,:,0,:,:]
            events[:,t_index+1,:,0,:,:] = events[:,t_index,:,1,:,:]

        # normalize:
        if self.normalize_voxel:
            
            events = event_preprocess_pytorch(events, mode='std', filter_hot_pixel=False, debug=None)
            # events = event_preprocess_pytorch(events, mode='std_pytorch', filter_hot_pixel=False, debug=None)

        # print("output events size", events.size())
        # for i in range(events.shape[1]):
        #     print(i, torch.sum(torch.abs(events[:,i,:,:,:,:])))
        # print('b, 7, num kinds events, 2 bins, 256, 256') # why did i say num_bins is 9? alas. ok in interpolation, num_bins is set to 2 + actual number of bins...
        # sys.exit()

        # by the way, there are 2 bins, bin- and bin+, but actually bin- at num_bin6 is the same as bin+ at num_bin5. 
        return events, total_num_events, event_count
    