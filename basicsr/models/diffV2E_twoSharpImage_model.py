import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import make_grid
from einops import rearrange
import sys
import os
import cv2

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from diff_esim_torch.esim_torch import EventSimulator_torch
from diff_esim_torch.general_simulator import GeneralEventEmulator
from basicsr.data.event_util import make_event_preview, make_event_preview_blended


loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

cv2save = True
# save each bin
bins_to_save = [3,4]

class DiffV2E_TwoSharpImages(BaseModel):
    """Base Event-based deblur model for recurrent image deblur and interpolation."""

    def __init__(self, opt):
        super(DiffV2E_TwoSharpImages, self).__init__(opt)

        # define network
        self.v2e_model = GeneralEventEmulator(deepcopy(opt['e_simulator']), device=self.device)
        self.v2e_model = self.model_to_device(self.v2e_model)
        
        self.spatially_varying = opt['e_simulator'].get('mixed_kernels', False)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        

        # load pretrained models
        load_sim_path = self.opt['path'].get('pretrain_network_sim', None)
        load_g_path = self.opt['path'].get('pretrain_network_g', None)
        if load_sim_path is not None:
            print("loading v2e_model from ", load_sim_path)
            self.load_network(self.v2e_model, load_sim_path,
                              self.opt['path'].get('strict_load_sim', True), param_key=self.opt['path'].get('param_key', 'params'))
        if load_g_path is not None:
            print("loading net_g from ", load_g_path)
            self.load_network(self.net_g, load_g_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.v2e_model.train()
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.pixel_type = train_opt['pixel_opt'].pop('type')
            # print('LOSS: pixel_type:{}'.format(self.pixel_type))
            cri_pix_cls = getattr(loss_module, self.pixel_type)

            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('sparsity_opt'):
            sparsity_type = train_opt['sparsity_opt'].pop('type')
            cri_sparsity_cls = getattr(loss_module, sparsity_type)
            self.cri_sparsity = cri_sparsity_cls(
                **train_opt['sparsity_opt']).to(self.device)
        else:
            self.cri_sparsity = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')


        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_encoder_params = []
        optim_encoder_names = []
        optim_params = []
        optim_param_names = []
        optim_params_lowlr = []
        optim_params_lowlr_names = []


        for k, v in self.v2e_model.named_parameters():
            if v.requires_grad:
                optim_encoder_params.append(v)
                optim_encoder_names.append(k)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if k.startswith('module.offsets') or k.startswith('module.dcns'):
                    optim_params_lowlr.append(v)
                    optim_params_lowlr_names.append(k)
                else:
                    optim_params.append(v)
                    optim_param_names.append(k)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        ratio = 0.1

        optim_type = train_opt['optim'].pop('type')
        

        # print("optim_encoder_names: ", optim_encoder_names)
        # print("optim_encoder_params: ", optim_encoder_params)
        # print("optim_encoder_params lr: ", train_opt['optim_v2e']['lr'])
        
        # print()
        # print("optim_params_lowlr_names ", optim_params_lowlr_names)
        # print("optim_encoder_params lr: ", train_opt['optim']['lr'] * ratio)
        # print()
        # print("optim_param_names ", optim_param_names)
        # print("optim_encoder_params lr: ", train_opt['optim']['lr'])
        # print()
        # sys.exit()
        if 'weight_decay' not in train_opt['optim_v2e'].keys():
            wd_encoder = 0.0
        else:
            wd_encoder = train_opt['optim_v2e']['weight_decay']

        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim']['lr'] * ratio}, {'params': optim_encoder_params, 'lr': train_opt['optim_v2e']['lr'], 'weight_decay': wd_encoder }],
                                                **train_opt['optim'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim']['lr'] * ratio}, {'params': optim_encoder_params, 'lr': train_opt['optim_v2e']['lr'], 'weight_decay': wd_encoder }],
                                                **train_opt['optim'])

        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supported yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):

        self.lq = data["lq"].to(self.device)
        self.gt = data["gt"].to(self.device)
        self.log_images = data["log_images"].to(self.device)
        self.timestamps = data["img_times"].to(self.device)
        self.seq_name = data["seq"]
        self.origin_index = data["origin_index"][0]
        self.voxel = None
        

    def transpose(self, t, trans_idx):
        # print('transpose jt .. ', t.size())
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return torch.rot90(t, trans_idx % 4, [2, 3])

    def transpose_inverse(self, t, trans_idx):
        # print( 'inverse transpose .. t', t.size())
        t = torch.rot90(t, 4 - trans_idx % 4, [2, 3])
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return t

    def grids_voxel(self):
        b, c, h, w = self.voxel.size()
        self.original_size_voxel = self.voxel.size()
        assert b == 1
        crop_size = self.opt['val'].get('crop_size')
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)

        # print('step_i, stepj', step_i, step_j)
        # exit(0)


        parts = []
        idxes = []

        # cnt_idx = 0

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True


            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt['val'].get('trans_num', 1)):
                    parts.append(self.transpose(self.voxel[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                    idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                    # cnt_idx += 1
                j = j + step_j
            i = i + step_i
        if self.opt['val'].get('random_crop_num', 0) > 0:
            for _ in range(self.opt['val'].get('random_crop_num')):
                import random
                i = random.randint(0, h-crop_size)
                j = random.randint(0, w-crop_size)
                trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
                parts.append(self.transpose(self.voxel[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})


        self.origin_voxel = self.voxel
        self.voxel = torch.cat(parts, dim=0)
        print('----------parts voxel .. ', len(parts), self.voxel.size())
        self.idxes = idxes

    def grids(self):
        b, c, h, w = self.lq.size()  # lq is after data augment (for example, crop, if have)
        self.original_size = self.lq.size()
        assert b == 1
        crop_size = self.opt['val'].get('crop_size')
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)


        # print('step_i, stepj', step_i, step_j)
        # exit(0)


        parts = []
        idxes = []

        # cnt_idx = 0

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True


            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt['val'].get('trans_num', 1)):
                    parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                    idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                    # cnt_idx += 1
                j = j + step_j
            i = i + step_i
        if self.opt['val'].get('random_crop_num', 0) > 0:
            for _ in range(self.opt['val'].get('random_crop_num')):
                import random
                i = random.randint(0, h-crop_size)
                j = random.randint(0, w-crop_size)
                trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
                parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})


        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        # print('parts .. ', len(parts), self.lq.size())
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size).to(self.device)
        b, c, h, w = self.original_size

        print('...', self.device)

        count_mt = torch.zeros((b, 1, h, w)).to(self.device)
        crop_size = self.opt['val'].get('crop_size')

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            trans_idx = each_idx['trans_idx']
            preds[0, :, i:i + crop_size, j:j + crop_size] += self.transpose_inverse(self.output[cnt, :, :, :].unsqueeze(0), trans_idx).squeeze(0)
            count_mt[0, 0, i:i + crop_size, j:j + crop_size] += 1.

        self.output = preds / count_mt
        self.lq = self.origin_lq
        self.voxel = self.origin_voxel

    def optimize_parameters(self, current_iter, tb_logger):
        # print("optimize_parameters ..")
        # sys.exit()
        self.optimizer_g.zero_grad()

        # generate events
        self.v2e_model.reset()    
        # print(print(self.log_images.dtype)) # torch32
        # print(print(self.voxel.dtype))

        events, total_num_events, event_count_continuous = self.v2e_model(self.log_images, self.timestamps)
        
        self.voxel = events.detach() # b, number_frames-2, num_kind_events, bin- and bin +, h, w
        # print(self.voxel.size())
        # sys.exit()    
        pred = self.net_g(x=self.lq, event=events)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            # b, t, 3, 256, 256
            # print(self.gt.dtype, pred.dtype) # torch32 torch32
            # sys.exit()
            l_pix = self.cri_pix(pred, self.gt)             
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        if self.cri_sparsity:
            l_sparsity = 0.
            l_sparsity = self.cri_sparsity(event_count_continuous, torch.zeros_like(event_count_continuous))
            l_total += l_sparsity
            loss_dict['l_sparsity'] = l_sparsity

        # loss_dict['numEvents'] = total_num_events.detach()
        loss_dict['cNumEvents'] = event_count_continuous.detach() # the correct one

        # perceptual loss
        # if self.cri_perceptual:
        #
        #
        #     l_percep, l_style = self.cri_perceptual(self.output, self.gt)
        #
        #     if l_percep is not None:
        #         l_total += l_percep
        #         loss_dict['l_percep'] = l_percep
        #     if l_style is not None:
        #         l_total += l_style
        #         loss_dict['l_style'] = l_style


        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())
        l_total.backward()

        for name, param in self.v2e_model.named_parameters():
            if param.grad is not None:
                if current_iter % 100 == 0:  # Log every 100 iterations
                    # print("LOGGING GRADIENTS AND PARAMETERS")
                    tb_logger.add_scalar(f'gradients/{name}_max', param.grad.abs().max(), current_iter)
                    tb_logger.add_scalar(f'gradients/{name}_mean', param.grad.abs().mean(), current_iter)
                    tb_logger.add_scalar(f'parameters/{name}_max', param.abs().max(), current_iter)
                    tb_logger.add_scalar(f'parameters/{name}_mean', param.abs().mean(), current_iter)

        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        grad_clip_value = self.opt['train'].get('grad_clip_value', 0.01) 
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.v2e_model.parameters(), grad_clip_value)
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), grad_clip_value)
        
        for name, param in self.v2e_model.named_parameters():
            if param.grad is not None:
                # print(f"Parameter: {name}, Gradient norm: {torch.norm(param.grad)}")
                if torch.isnan(param.grad).any():
                    param.grad = torch.zeros_like(param.grad)
                    print(f"Parameter v2e_model: {name}, Gradient is NaN, set to zero")
                if torch.isinf(param.grad).any():
                    param.grad = torch.zeros_like(param.grad)
                    print(f"Parameter v2e_model: {name}, Gradient is Inf, set to zero")
                assert not torch.isnan(param.grad).any(), f"Parameter v2e_model: {name}, Gradient is NaN"
                assert not torch.isinf(param.grad).any(), f"Parameter v2e_model: {name}, Gradient is Inf"
            else:
                None
                # print(f"Parameter: {name}, Gradient: None")
        for name, param in self.net_g.named_parameters():
            if param.grad is not None:
                # print(f"Parameter: {name}, Gradient norm: {torch.norm(param.grad)}")
                if torch.isnan(param.grad).any():
                    param.grad = torch.zeros_like(param.grad)
                    print(f"Parameter net_g: {name}, Gradient is NaN, set to zero")
                if torch.isinf(param.grad).any():
                    param.grad = torch.zeros_like(param.grad)
                    print(f"Parameter net_g: {name}, Gradient is Inf, set to zero")
                assert not torch.isnan(param.grad).any(), f"Parameter net_g: {name}, Gradient is NaN"
                assert not torch.isinf(param.grad).any(), f"Parameter net_g: {name}, Gradient is Inf"
            else:
                None
        
        self.optimizer_g.step()


        with torch.no_grad():
            if current_iter % 200000 == 0 or current_iter==0:
                # voxel: torch.Size([batch, 7 frames, num_kinds_events, 2 bins, h, w])

                # # first events
                # bb, ff, nke, bbin, h, w = self.voxel.size()
                # first_events = torch.cat([self.voxel[0, 0, i:i+1, 0, :, :] for i in range(nke)], dim=-1)
                # first_events =  make_event_preview(first_events.cpu().numpy(), mode='red-blue', num_bins_to_show=-1).transpose(2,0,1)
                # tb_logger.add_image('TRAIN first events', first_events, current_iter)
                
                # # last events
                # last_events = torch.cat([self.voxel[0, -1, i:i+1, 1, :, :] for i in range(nke)], dim=-1)
                # last_events =  make_event_preview(last_events.cpu().numpy(), mode='red-blue', num_bins_to_show=-1).transpose(2,0,1)
                # tb_logger.add_image('TRAIN last events', last_events, current_iter)

                # middle events, all kinds of events
                bb, ff, nke, bbin, h, w = self.voxel.size()
                mid_events = torch.cat([self.voxel[:, ff//2:ff//2+1, i:i+1, :, :, :] for i in range(nke)], dim=-1)
                all_the_events =  make_event_preview(mid_events[0,0,0,0,:,:].cpu().numpy(), mode='red-blue', num_bins_to_show=-1).transpose(2,0,1)
                tb_logger.add_image('TRAIN event kinds -- midframe', all_the_events, current_iter)

                # full_video_frames_gt 
                gt_frames = torch.cat([self.gt[0, i, :, :, :] for i in range(ff)], dim=-1)
                pred_frames = torch.cat([pred[0, i, :, :, :] for i in range(ff)], dim=-1)
                gt_pred = torch.cat([gt_frames, pred_frames],  dim=-2)
                tb_logger.add_image(f'TRAIN full_video -- gt_pred', gt_pred, current_iter)


                # full_video_frames_gt = torch.concat(full_video_frames_gt, dim=-1)
                # full_video_frames_pred = torch.concat(full_video_frames_pred, dim=-1)
                # full_video_frames = make_grid(torch.concat([full_video_frames_gt, full_video_frames_pred ], dim=-2), nrow=10, normalize=True)
                # tb_logger.add_image('TRAIN full_video_output', full_video_frames, current_iter)


                # if you want to show each frame of each kind of event.
                # bb, ff, nke, bbin, h, w = self.voxel.size()
                # all_events = torch.cat([self.voxel[:, :, i:i+1, :, :, :] for i in range(nke)], dim=-1)
                # all_events = torch.cat([all_events[:, i:i+1, :, :, :, :] for i in range(ff)], dim=-2)
                # event_bins = make_event_preview(all_events[0,0,0,0,:,:].cpu().numpy(), mode='red-blue', num_bins_to_show=-1).transpose(2,0,1)
                # tb_logger.add_image(f'TRAIN all events', event_bins, current_iter)

                    
                # if all_events == None:
                    # all_events = event_bins
                # else:
                    # all_events = torch.concat([all_events, event_bins], dim=-1)
                # all_the_events =  make_event_preview(all_events.cpu().numpy(), mode='red-blue', num_bins_to_show=-1).transpose(2,0,1)
                # tb_logger.add_image('TRAIN all events', all_the_events, current_iter)

                # curr_kernel =  make_event_preview(self.voxel[0,0,0,:,:].cpu().numpy(), mode='red-blue', num_bins_to_show=-1).transpose(2,0,1)
                # tb_logger.add_image('curr kernel', curr_kernel, current_iter)

                # mem_kernel =  make_event_preview(self.voxel[0,0,0,:,:].cpu().numpy(), mode='red-blue', num_bins_to_show=-1).transpose(2,0,1)
                # tb_logger.add_image('mem kernel', mem_kernel, current_iter)


            if current_iter % 25000 == 0 or current_iter==0:
                for name, p in self.v2e_model.named_parameters():
                    if 'current_kernel' in name:
                        # print(p.size()) # torch.Size([8, 1, 5, 5])
                        kernels = torch.cat([F.pad(p, (1,1,1,1), "constant", 0)[i, :, :, :] for i in range(p.size()[0])], dim=-1)
                        tb_logger.add_image('kernel: current', make_grid(kernels.detach().cpu(), nrow=8, normalize=True), current_iter)
                        tb_logger.add_scalar('kernel: current-max', p.max(),current_iter)
                        tb_logger.add_scalar('kernel: current-min', p.min(),current_iter)
                    if 'mem_kernel' in name:
                        kernels = torch.cat([F.pad(p, (1,1,1,1), "constant", 0)[i, :, :, :] for i in range(p.size()[0])], dim=-1)
                        tb_logger.add_image('kernel: memory',  make_grid(kernels.detach().cpu(), nrow=8, normalize=True), current_iter)
                        tb_logger.add_scalar('kernel: memory-max', p.max(),current_iter)
                        tb_logger.add_scalar('kernel: memory-min', p.min(),current_iter)
                    if 'pos_thres' in name:
                        if p.dim == 4:
                            tb_logger.add_image('threshold: pos_thres',  make_grid(p[0:1,:,:,:].detach().cpu(), nrow=8, normalize=False),current_iter)
                    if 'neg_thres' in name:
                        if p.dim == 4:
                            tb_logger.add_image('threshold: neg_thres',  make_grid(p[0:1,:,:,:].detach().cpu(), nrow=8, normalize=False),current_iter)
                    if 'p_pos' in name:
                        # print(p.size()) # num kinds events, 2, 2
                        kernels = torch.cat([F.pad(p, (1,1,1,1), "constant", 0)[i, :, :] for i in range(p.size()[0])], dim=-1)
                        tb_logger.add_image('sv threshold: p_pos',  make_grid(kernels.detach().cpu(), nrow=8, normalize=False),current_iter)
                        tb_logger.add_scalar('sv threshold: p_pos-max', p.max(),current_iter)
                        tb_logger.add_scalar('sv threshold: p_pos-min', p.min(),current_iter)
                    if 'p_neg' in name:
                        kernels = torch.cat([F.pad(p, (1,1,1,1), "constant", 0)[i, :, :] for i in range(p.size()[0])], dim=-1)
                        tb_logger.add_image('sv threshold: p_neg',  make_grid(kernels.detach().cpu(), nrow=8, normalize=False),current_iter)
                        tb_logger.add_scalar('sv threshold: p_neg-max', p.max(),current_iter)
                        tb_logger.add_scalar('sv threshold: p_neg-min', p.min(),current_iter)
                    if 'sigma_1' in name:
                        tb_logger.add_scalar('dog_sigma_1', p.item(),current_iter)
                    if 'sigma_2' in name:
                        tb_logger.add_scalar('dog_sigma_2', p.item(),current_iter)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.v2e_model.eval()
        self.v2e_model.reset()
        self.net_g.eval()
        
        with torch.no_grad():
            n = self.log_images.size(0)  # n: batch size
            outs = []
            m = self.opt['val'].get('max_minibatch', n)  # m is the minibatch, equals to batch size or mini batch size
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n

                events, total_num_events, event_count_continuous = self.v2e_model(self.log_images, self.timestamps)
                self.voxel = events.detach()
                
                # bin_minus = self.voxel[:,:,0,:,:]
                # bin_plus = self.voxel[:,:,1,:,:]
                # print("bin_minus ", bin_minus.min().cpu().numpy(),bin_minus.max().cpu().numpy(),bin_minus.mean().cpu().numpy(),bin_minus.std().cpu().numpy())
                # print("bin_plus ", bin_plus.min().cpu().numpy(),bin_plus.max().cpu().numpy(),bin_plus.mean().cpu().numpy(),bin_plus.std().cpu().numpy())
       
                pred = self.net_g(x = self.lq, event = events)  # mini batch all in 
                outs.append(pred)
                i = j

            self.output = torch.cat(outs, dim=0)  # all mini batch cat in dim0
        self.v2e_model.train()
        self.net_g.train()

    def single_image_inference(self, img, voxel, save_path):
        self.feed_data(data={'lq': img.unsqueeze(dim=0), 'voxel': voxel.unsqueeze(dim=0)})
        if self.opt['val'].get('grids') is not None:
            self.grids()
            self.grids_voxel()

        self.test()

        if self.opt['val'].get('grids') is not None:
            self.grids_inverse()
            # self.grids_inverse_voxel()

        visuals = self.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        imwrite(sr_img, save_path)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        # print("dist_validation ..")
        logger = get_root_logger()
        # logger.info('Only support single GPU validation.')
        import os
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image, save_all_imgs=False):
        # print("nondist_validation ..")
        # sys.exit()
        if save_all_imgs:
            
            eval_folder = os.path.join(self.opt['path']['root'], 'eval', self.opt['name'], f"{self.opt['datasets']['val']['num_inter_interpolation']}skip")
            info_folder = os.path.join(eval_folder, 'info')
            img_folder = os.path.join(eval_folder, 'imgs')
            alpha_blend_folder = os.path.join(self.opt['path']['root'], 'eval', f"{self.opt['datasets']['val']['num_inter_interpolation']}skip",'alpha_blended_gt') # save alpha blended gt
            to_make = [eval_folder, info_folder, img_folder, alpha_blend_folder]
            for folder in to_make:
                if not os.path.exists(folder):
                    os.makedirs(folder)
            self.logger_file = open(os.path.join(info_folder, 'log.txt'), 'w') 
            self.logger_file_kernel = open(os.path.join(info_folder, 'log_kernel.txt'), 'w') 


            for name, p in self.v2e_model.named_parameters():
                if 'current_kernel' in name:
                    self.logger_file_kernel.write(f"current kernel:\n{p}\n")
                    self.logger_file_kernel.write(f"\n")
                if 'mem_kernel' in name:
                    self.logger_file_kernel.write(f"memory kernel:\n{p}\n")
                    self.logger_file_kernel.write(f"\n")
                if 'pos_thres' in name:
                    self.logger_file_kernel.write(f"threshold: pos_thres:\n{p}\n")
                    self.logger_file_kernel.write(f"\n")
                if 'neg_thres' in name:
                    self.logger_file_kernel.write(f"threshold: neg_thres:\n{p}\n")
                    self.logger_file_kernel.write(f"\n")
                if 'p_pos' in name:
                    self.logger_file_kernel.write(f"sv threshold: p_pos:\n{p}\n")
                    self.logger_file_kernel.write(f"\n")
                if 'p_neg' in name:
                    self.logger_file_kernel.write(f"sv threshold: p_neg:\n{p}\n")
                    self.logger_file_kernel.write(f"\n")
                if 'sigma_1' in name:
                    self.logger_file_kernel.write(f"dog sigma_1:\n{p}\n")
                    self.logger_file_kernel.write(f"\n")
                if 'sigma_2' in name:
                    self.logger_file_kernel.write(f"dog sigma_2:\n{p}\n")
                    self.logger_file_kernel.write(f"\n")
                    
            # sys.exit()
            self.logger_file_kernel.close()
            
                    
        
        dataset_name = self.opt.get('name') # !
        save_gt = self.opt['val'].get('save_gt', False)

        self.m = self.opt['datasets']['val'].get('num_end_interpolation')
        self.n = self.opt['datasets']['val'].get('num_inter_interpolation')
        imgs_per_iter_deblur = 2*self.m
        imgs_per_iter_interpo = self.n

        with_metrics = self.opt['val'].get('metrics_interpo') is not None
        if with_metrics:

            self.metric_results_interpo = {
                metric: 0
                for metric in self.opt['val']['metrics_interpo'].keys()
            }
            self.metric_results_interpo['event_count'] = 0

        pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0
        last_seq_name = 'Lei Sun'
        seq_inner_cnt = 0
        # print('DEBUG: val_loader_len:{}'.format(len(dataloader)))

        # total_events = 0

        for idx, val_data in enumerate(dataloader):
            seq_metric_results  = {
                metric: 0
                for metric in self.opt['val']['metrics_interpo'].keys()
            }
            seq_metric_results['event_count'] = 0
            self.v2e_model.reset()
            self.feed_data(val_data)


            if self.seq_name == last_seq_name:
                seq_inner_cnt += 1
                img_name = '{:04d}'.format(seq_inner_cnt)
            else:
                seq_inner_cnt = 0
                img_name = '{:04d}'.format(seq_inner_cnt)
                last_seq_name = self.seq_name

            if self.opt['val'].get('grids') is not None:
                self.grids()
                self.grids_voxel()

            self.test()
            if self.opt['val'].get('grids') is not None:
                self.grids_inverse()
            visuals = self.get_current_visuals()

            # total_events = visuals['voxel'].abs().sum().detach()
            # tb_logger.add_scalar('VAL event count: ', total_events.cpu(), current_iter+idx)  
            # tentative for out of GPU memory
            
            
            del self.log_images
            del self.voxel
            del self.output
            del self.gt
            torch.cuda.empty_cache()
            imgs_per_iter = visuals['result'].size(1)

            self.metric_results_interpo['event_count'] += visuals['voxel'].abs().sum().detach()
            seq_metric_results['event_count'] += visuals['voxel'].abs().sum().detach()
            # self.metric_results_interpo['event_count'] = visuals['voxel'].abs().sum().detach()
            # val data
            # lq torch.Size([1, 2, 3, 720, 1280])
            # gt torch.Size([1, 7, 3, 720, 1280])
            # log_images torch.Size([1, 9, 1, 720, 1280])
            # img_times torch.Size([1, 9])
            # seq ['GOPR0384_11_00']
            # origin_index ['000001']
            # visuals['result'].size(1):  7

            im0 = val_data['lq'][0,0,:,:,:]
            im1 = val_data['lq'][0,1,:,:,:]
            im0_img = tensor2img(im0)
            im1_img = tensor2img(im1) 
            if save_all_imgs:
                alpha_blended_gt = cv2.addWeighted(im0_img, 0.5, im1_img, 0.5, 0)
                file_blended_gt = os.path.join(alpha_blend_folder, f"{val_data['seq'][0]}_start_idx_{val_data['origin_index'][0]}_alpha_blended_gt.png")
                if not os.path.exists(file_blended_gt):
                    if cv2save:
                        cv2.imwrite(file_blended_gt, alpha_blended_gt)

            # sys.exit()
            # print(visuals['result'].size(1), "visuals['result'].size(1)") # 7
            # sys.exit()

            for frame_idx in range(visuals['result'].size(1)):
                img_name = '{}_{:02d}'.format(self.origin_index, frame_idx)
                # print("img_name:", img_name)
                result = visuals['result'][0, frame_idx, :, :, :]
                sr_img = tensor2img([result])  # uint8, bgr
                if 'gt' in visuals:
                    gt = visuals['gt'][0, frame_idx, :, :, :]
                    gt_img = tensor2img([gt])  # uint8, bgr
                

                if with_metrics:
                    # calculate metrics
                    opt_metric_interpo = deepcopy(self.opt['val']['metrics_interpo'])
                    # print(opt_metric_interpo, "opt_metric_interpo")
                    # sys.exit()
                    file_name= f"{val_data['seq'][0]}_start_index_{val_data['origin_index'][0]}_frame_idx_{frame_idx:02d}"
                    ori_file_name = f"{val_data['seq'][0]}_start_index_{val_data['origin_index'][0]}"
                    

                    # print("opt_metric_interpo: ", opt_metric_interpo) 
                    # sys.exit()
                    # print("use_image: ", use_image)
                    # sys.exit()
                    if use_image:
                        # interpo
                        for name, opt_ in opt_metric_interpo.items(): # name: psnr, ...; opt_: type, ...
                            metric_type = opt_.pop('type') # calculate_psnr or calculate_ssim
                            self.metric_results_interpo[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)
                            seq_metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)
                            if save_all_imgs:
                                file_name+= f"_{name}_{getattr(metric_module, metric_type)(sr_img, gt_img, **opt_):.3f}"
                        if save_all_imgs: # and ori_file_name in ['GOPR0410_11_00_start_index_001773', 'GOPR0881_11_01_start_index_003289', 'GOPR0868_11_00_start_index_000273']:
                            if cv2save and frame_idx in bins_to_save:
                                cv2.imwrite(os.path.join(img_folder, file_name+".png"), sr_img) # save each predicted frame
                        # print(file_name+".png")
                    else:
                        # interpo
                        for name, opt_ in opt_metric_interpo.items(): # name: psnr, ...; opt_: type, ...
                            metric_type = opt_.pop('type') # calculate_psnr or calculate_ssim
                            self.metric_results_interpo[name] += getattr(metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)
                            seq_metric_results[name] += getattr(metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)
            if save_all_imgs:
                
                # save info
                self.logger_file.write(f"{val_data['seq'][0]}_start_index_{val_data['origin_index'][0]}\n")
                # print()
                # print("self.metric_results_interpo: ", self.metric_results_interpo)
                # sys.exit()
                for name, value in seq_metric_results.items():
                    if name == 'event_count':
                        self.logger_file.write(f"    total events bw: {value}\n")
                        print(f"    total events bw: {value}")
                    else:
                        self.logger_file.write(f"    avg {name}: {value/(visuals['result'].size(1))}\n")
                        print(f"    avg {name}: {value/(visuals['result'].size(1))}")
                # sys.exit()
                # reset seq_metric_results
                for name in seq_metric_results.keys():
                    seq_metric_results[name] = 0


            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
            cnt += 1
            if (save_img and idx%10000== 0 and tb_logger) or save_all_imgs:
                b, num_frames, c, h, w = visuals['result'].size()
                

                full_video_frames_gt = None
                full_video_frames_pred = None
                for num_frames_i in range(num_frames):
                    if full_video_frames_gt == None:
                        full_video_frames_gt = visuals['gt'][0,num_frames_i:num_frames_i+1,:,:,:]
                        full_video_frames_pred = visuals['result'][0,num_frames_i:num_frames_i+1,:,:,:]
                    else:
                        full_video_frames_gt = torch.concat([full_video_frames_gt, visuals['gt'][0,num_frames_i:num_frames_i+1,:,:,: ]], dim=-1)
                        full_video_frames_pred = torch.concat([full_video_frames_pred, visuals['result'][0,num_frames_i:num_frames_i+1,:,:,: ]], dim=-1)
                full_video_frames = make_grid(torch.concat([full_video_frames_gt, full_video_frames_pred ], dim=-2), nrow=10, normalize=True)
                tb_logger.add_image('VAL full_video_output', full_video_frames, current_iter+idx)

            
                # print(visuals['voxel'].shape) # torch.Size([1, 7, 2, 2, 720, 1280])


                gt_pred = make_grid(torch.concat([visuals['gt'][0,num_frames//2:num_frames//2+1,:,:,:], visuals['result'][0,num_frames//2:num_frames//2+1,:,:,:]], dim=-1), nrow=8, normalize=True)
                tb_logger.add_image('VAL gt/pred', gt_pred, current_iter+idx)
                diff = make_grid(visuals['gt'][0,num_frames//2:num_frames//2+1,:,:,:] - visuals['result'][0,num_frames//2:num_frames//2+1,:,:,:], nrow=8, normalize=True)
                # print("diff:",diff.shape, diff.max(), diff.min())
                # diff_img = tensor2img(diff.abs())
                # cv2.imwrite(os.path.join(img_folder, f"diff_{current_iter+idx}.png"), diff_img)
                # sys.exit()
                tb_logger.add_image('VAL diff', diff.abs(), current_iter+idx)


                # # all events
                # bb, ff, nke, bbin, h, w = visuals['voxel'].size()
                # all_events = torch.cat([visuals['voxel'][:, :, i:i+1, :, :, :] for i in range(nke)], dim=-1)
                # all_events = torch.cat([all_events[:, i:i+1, :, :, :, :] for i in range(ff)], dim=-2)
                # all_the_events =  make_event_preview(all_events[0,0,0,0,:,:].cpu().numpy(), mode='red-blue', num_bins_to_show=-1).transpose(2,0,1)
                # tb_logger.add_image('VAL all kinds of events frame', all_the_events, current_iter+idx)

                # # first events
                # bb, ff, nke, bbin, h, w = visuals['voxel'].size()
                # first_events = torch.cat([visuals['voxel'][0, 0, i:i+1, 0, :, :] for i in range(nke)], dim=-1)
                # first_events =  make_event_preview(first_events.cpu().numpy(), mode='red-blue', num_bins_to_show=-1).transpose(2,0,1)
                # tb_logger.add_image('VAL first events', first_events, current_iter+idx)
                
                # # last events
                # last_events = torch.cat([visuals['voxel'][0, -1, i:i+1, 1, :, :] for i in range(nke)], dim=-1)
                # last_events =  make_event_preview(last_events.cpu().numpy(), mode='red-blue', num_bins_to_show=-1).transpose(2,0,1)
                # tb_logger.add_image('VAL last events', last_events, current_iter+idx)

                # middle events
                bb, ff, nke, bbin, h, w = visuals['voxel'].size()
                mid_events = torch.cat([visuals['voxel'][:, ff//2:ff//2+1, i:i+1, :, :, :] for i in range(nke)], dim=-1)
                # print("mid_events:", mid_events.size()) # ([1, 1, 1, 2, 720, 1280])
                all_the_events =  make_event_preview(mid_events[0,0,0,0,:,:].cpu().numpy(), mode='red-blue', num_bins_to_show=-1).transpose(2,0,1)
                tb_logger.add_image('VAL event kinds -- midframe', all_the_events, current_iter+idx)

                if save_all_imgs: # and ori_file_name in ['GOPR0410_11_00_start_index_001773', 'GOPR0881_11_01_start_index_003289', 'GOPR0868_11_00_start_index_000273']:
                    # print(visuals['voxel'].size()) # torch.Size([1, 7, 1, 2, 720, 1280]) # events_image[0,0,0,0,:,:]
                    # by the way, there are 2 bins, bin- and bin+, but actually bin- at num_bin6 is the same as bin+ at num_bin5. 
                    
                                    # b, number_frames-2, num_kind_events, bin- and bin +, h, w
                    # alpha blend all. like we do in idnet

                    print((visuals['voxel'][:,0,:,1,:,:] - visuals['voxel'][:,1,:,0,:,:]).abs().sum())
                    if self.spatially_varying:
                        print(visuals['voxel'].size())
                        print(visuals['voxel'].min(), visuals['voxel'].max())

                        # separate kinds of events
                        b,fr,nke,bin,h,w = visuals['voxel'].size()
                        assert nke==1 # only one kind of event for this visualization.
                        separated_out_events = torch.zeros(b,fr,4,1,h,w)
                        separated_out_events[:,:,:,:,0::2,0::2] = visuals['voxel'][:,:,:,0:1,0::2,0::2]
                        separated_out_events[:,:,:,:,0::2,1::2] = visuals['voxel'][:,:,:,0:1,0::2,1::2]
                        separated_out_events[:,:,:,:,1::2,0::2] = visuals['voxel'][:,:,:,0:1,1::2,0::2]
                        separated_out_events[:,:,:,:,1::2,1::2] = visuals['voxel'][:,:,:,0:1,1::2,1::2]
                        visuals['voxel'] = separated_out_events
                        print(visuals['voxel'].size())
                        # separated_out_events_img = make_event_preview(separated_out_events[0,0,0,0,:,:].cpu().numpy(), mode='red-blue', num_bins_to_show=-1).transpose(2,0,1)
                    
                    blended_events_list = make_event_preview_blended(visuals['voxel'][0,:,:,0,:,:])
                    if cv2save:
                        for i, blended_events in enumerate(blended_events_list):
                            bgr_image = cv2.cvtColor(blended_events, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(os.path.join(img_folder, f"{ori_file_name}_event_blended_ekind{i}.png"), bgr_image)

                    # for i in range(ff): # for each bin
                    for i in bins_to_save:
                        events_image = torch.cat([visuals['voxel'][:, i:i+1, j:j+1, :, :, :] for j in range(nke)], dim=-1)
                        events_preview = make_event_preview(events_image[0,0,0,0,:,:].cpu().numpy(), mode='red-blue', num_bins_to_show=-1).transpose(2,0,1)
                        events_img = events_preview.transpose(1,2,0)
                        events_bgr = cv2.cvtColor(events_img, cv2.COLOR_RGB2BGR)
                        # print("events_preview:", events_preview.shape, events_preview.max(), events_preview.min())
                        # sys.exit()
                        # print("events_preview:", events_preview.shape, events_preview.max(), events_preview.min())
                    #         event_img = make_event_preview(mid_events[0,0,j,i,:,:].cpu().numpy(), mode='red-blue', num_bins_to_show=-1).transpose(2,0,1)
                        if cv2save: 
                            cv2.imwrite(os.path.join(img_folder, f"{ori_file_name}_event_bin_{i}.png"), events_bgr)
                    # sys.exit()

            
        # ave_events = total_events/len(dataloader)  
        pbar.close()
        if save_all_imgs:
            self.logger_file.close()


        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results_interpo.keys():
                self.metric_results_interpo[metric] /= (cnt * imgs_per_iter)
                current_metric = self.metric_results_interpo[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

        return current_metric # this return result is not important


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):

        # interpo:
        log_str = f'Validation {dataset_name} [interpolation],\t'
        for metric, value in self.metric_results_interpo.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        if tb_logger:
            for metric, value in self.metric_results_interpo.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)


    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['imgs'] = self.log_images.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['voxel'] = self.voxel.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
    

    def save(self, epoch, current_iter):
        self.save_network(self.v2e_model, 'v2e_model', current_iter)
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)