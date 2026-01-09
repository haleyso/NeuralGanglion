import numpy as np
import torch
from basicsr.utils import Timer, CudaTimer
import sys

def events_to_voxel_grid_haley(events, num_bins, width, height, device, return_format='CHW'):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param device: device to use to perform computations
    :return voxel_grid: PyTorch event tensor (on the device specified)
    :param return_format: 'CHW' or 'HWC'
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    with torch.no_grad():
        if isinstance(events, np.ndarray):
            events_torch = torch.from_numpy(events)
        else:
            events_torch = events

        events_torch = events_torch.to(device)
        voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=device).flatten()

        print('size of voxel_grid ', voxel_grid.size()) #  torch.Size([524288])  
        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events_torch[-1, 0]
        first_stamp = events_torch[0, 0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
        ts = events_torch[:, 0]
        xs = events_torch[:, 1].long()
        ys = events_torch[:, 2].long()
        pols = events_torch[:, 3].float()
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = torch.floor(ts)
        tis_long = tis.long()
        dts = ts - tis
        vals_left = pols * (1.0 - dts.float())
        vals_right = pols * dts.float()

        valid_indices = tis < num_bins
        valid_indices &= tis >= 0

        print('indices', xs[valid_indices] + ys[valid_indices]* width + tis_long[valid_indices] * width * height)
        sys.exit()
        voxel_grid.index_add_(dim=0,
                                index=xs[valid_indices] + ys[valid_indices]
                                * width + tis_long[valid_indices] * width * height,
                                source=vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        valid_indices &= tis >= 0

        # voxel_grid.index_add_(dim=0,
        #                         index=xs[valid_indices] + ys[valid_indices] * width
        #                         + (tis_long[valid_indices] + 1) * width * height,
        #                         source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

    if return_format == 'CHW':
        return voxel_grid
    elif return_format == 'HWC':
        return voxel_grid.permute(1,2,0)
    

def events_to_voxel_grid(events, num_bins, width, height, return_format='CHW'):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param return_format: 'CHW' or 'HWC'
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
    # print('size of voxel_grid ', voxel_grid.shape, num_bins, height, width) #  torch.Size([524288])  
    # print('DEBUG: voxel.shape:{}'.format(voxel_grid.shape))

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    # print('last stamp:{}'.format(last_stamp))
    # print('max stamp:{}'.format(events[:, 0].max()))
    # print('timestamp:{}'.format(events[:, 0]))
    # print('polarity:{}'.format(events[:, -1]))
    # sys.exit()
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT # 
    ts = events[:, 0]
    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins # [True True ... True]
    # print('x max:{}'.format(xs[valid_indices].max()))
    # print('y max:{}'.format(ys[valid_indices].max()))
    # print('tix max:{}'.format(tis[valid_indices].max()))
    # print('indices', xs[valid_indices] + ys[valid_indices]* width + tis[valid_indices] * width * height)
    # sys.exit()
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width  ## ! ! !
            + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    if return_format == 'CHW':
        return voxel_grid
    elif return_format == 'HWC':
        return voxel_grid.transpose(1,2,0)


def events_to_voxel_grid_pytorch(events, num_bins, width, height, device, return_format='CHW'):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param device: device to use to perform computations
    :return voxel_grid: PyTorch event tensor (on the device specified)
    :param return_format: 'CHW' or 'HWC'
    """

    # DeviceTimer = CudaTimer if device.type == 'cuda' else Timer
    DeviceTimer = Timer if device=='cpu' else CudaTimer

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    with torch.no_grad():
        if isinstance(events, np.ndarray):
            events_torch = torch.from_numpy(events)
        else:
            events_torch = events
        with DeviceTimer('Events -> Device (voxel grid)'):
            events_torch = events_torch.to(device)

        with DeviceTimer('Voxel grid voting'):
            voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=device).flatten()

            # normalize the event timestamps so that they lie between 0 and num_bins
            last_stamp = events_torch[-1, 0]
            first_stamp = events_torch[0, 0]
            deltaT = last_stamp - first_stamp

            if deltaT == 0:
                deltaT = 1.0

            events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
            ts = events_torch[:, 0]
            xs = events_torch[:, 1].long()
            ys = events_torch[:, 2].long()
            pols = events_torch[:, 3].float()
            pols[pols == 0] = -1  # polarity should be +1 / -1

            tis = torch.floor(ts)
            tis_long = tis.long()
            dts = ts - tis
            vals_left = pols * (1.0 - dts.float())
            vals_right = pols * dts.float()

            valid_indices = tis < num_bins
            valid_indices &= tis >= 0
            voxel_grid.index_add_(dim=0,
                                  index=xs[valid_indices] + ys[valid_indices]
                                  * width + tis_long[valid_indices] * width * height,
                                  source=vals_left[valid_indices])

            valid_indices = (tis + 1) < num_bins
            valid_indices &= tis >= 0

            voxel_grid.index_add_(dim=0,
                                  index=xs[valid_indices] + ys[valid_indices] * width
                                  + (tis_long[valid_indices] + 1) * width * height,
                                  source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

    if return_format == 'CHW':
        return voxel_grid
    elif return_format == 'HWC':
        return voxel_grid.permute(1,2,0)


def voxel_norm(voxel):
    """
    Norm the voxel

    :param voxel: The unnormed voxel grid
    :return voxel: The normed voxel grid
    """
    nonzero_ev = (voxel != 0)
    num_nonzeros = nonzero_ev.sum()
    # print('DEBUG: num_nonzeros:{}'.format(num_nonzeros))
    if num_nonzeros > 0:
        # compute mean and stddev of the **nonzero** elements of the event tensor
        # we do not use PyTorch's default mean() and std() functions since it's faster
        # to compute it by hand than applying those funcs to a masked array
        mean = voxel.sum() / num_nonzeros
        stddev = torch.sqrt((voxel ** 2).sum() / num_nonzeros - mean ** 2)
        mask = nonzero_ev.float()
        voxel = mask * (voxel - mean) / stddev

    return voxel



def voxel_norm_antirs(voxel):
    """
    Norm the voxel

    :param voxel: The unnormed voxel grid
    :return voxel: The normed voxel grid
    """
    nonzero_ev = (voxel != 0)
    num_nonzeros = nonzero_ev.sum()
    # print('DEBUG: num_nonzeros:{}'.format(num_nonzeros))
    if num_nonzeros > 0:
        # compute mean and stddev of the **nonzero** elements of the event tensor
        # we do not use PyTorch's default mean() and std() functions since it's faster
        # to compute it by hand than applying those funcs to a masked array
        mean = voxel.sum() / num_nonzeros
        stddev = torch.sqrt((voxel ** 2).sum() / num_nonzeros - mean ** 2)
        mask = nonzero_ev.float()
        voxel = mask * (voxel - mean) * 4 / stddev

    return voxel


def filter_event(x,y,p,t, s_e_index=[0,6]):
    '''
    s_e_index: include both left and right index
    '''
    t_1=t.squeeze(1)
    uniqw, inverse = np.unique(t_1, return_inverse=True)
    discretized_ts = np.bincount(inverse)
    index_exposure_start = np.sum(discretized_ts[0:s_e_index[0]])
    index_exposure_end = np.sum(discretized_ts[0:s_e_index[1]+1])
    x_1 = x[index_exposure_start:index_exposure_end]
    y_1 = y[index_exposure_start:index_exposure_end]
    p_1 = p[index_exposure_start:index_exposure_end]
    t_1 = t[index_exposure_start:index_exposure_end]
    
    return x_1, y_1, p_1, t_1

def make_event_preview_blended(events):
    # i think b, 15, 288, 384
    # print(events.shape)# [7, 1, 720, 1280]

    blended_outputs = []
    for j in range(events.shape[1]): # for each kind of event
        events_j = events[:,j,:,:]
        events_j = events_j.squeeze().detach().cpu().numpy() # 15, 288, 384
        min_val = events_j.min()
        max_val = events_j.max()

        image_list = []
        for i in range(events_j.shape[0]):
            rgba_i = events_j[i,:,:]

            event_preview = np.zeros((rgba_i.shape[0], rgba_i.shape[1], 3), dtype=np.uint8)
            
            b = event_preview[:, :, 0]
            g = event_preview[:, :, 1]
            r = event_preview[:, :, 2]

            b[rgba_i >= 0] = 255
            r[rgba_i <= 0] = 255
            g[rgba_i == 0] = 255


            # event_preview = np.where(rgba_i ==0, )
            image_list.append(event_preview)


        # If no alpha weights are given, use equal blending
        blended = np.zeros_like(image_list[0], dtype=np.float32)
        alpha_weights = [1 / len(image_list)] * len(image_list)
        alpha_weights = np.array(alpha_weights) / np.sum(alpha_weights)

        # Perform weighted blending
        for img, alpha in zip(image_list, alpha_weights):
            blended += img.astype(np.float32) * alpha
        
        # Clip to valid range and convert back to uint8
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        blended_outputs.append(blended)
    return blended_outputs

def make_event_preview(events, mode='grayscale', num_bins_to_show=-1):
    # print(visuals['voxel'].size()) # torch.Size([1, 7, 1, 2, 720, 1280]) # events_image[0,0,0,0,:,:]
    # print("make_event_preview ..")
    # print(events.shape, events.min(), events.max(), num_bins_to_show)
    # sys.exit()
    # torch.Size([2, 7, num_kinds_events, 2, 256, 256])

    # events: [1 x C x H x W] event numpy or [C x H x W]
    # num_bins_to_show: number of bins of the voxel grid to show. -1 means show all bins.
    if events.ndim == 2:
        events = np.expand_dims(events,axis=0)
        events = np.expand_dims(events,axis=0)
    if events.ndim == 3:    
        events = np.expand_dims(events,axis=0)
    if num_bins_to_show < 0:
        sum_events = np.sum(events[0, :, :, :], axis=0)
    else:
        sum_events = np.sum(events[0, -num_bins_to_show:, :, :], axis=0)

    if mode == 'red-blue':
        # Red-blue mode
        # positive events: blue, negative events: red
        event_preview = np.zeros((sum_events.shape[0], sum_events.shape[1], 3), dtype=np.uint8)
        b = event_preview[:, :, 0]
        r = event_preview[:, :, 2]
        g = event_preview[:, :, 1]
        b[sum_events >= 0] = 255
        r[sum_events <= 0] = 255
        g[sum_events == 0] = 255
    else:
        # Grayscale mode
        # normalize event image to [0, 255] for display
        m, M = -5.0, 5.0
        # M = (sum_events.max() - sum_events.min())/2
        # m = -M
        
        event_preview = np.clip((255.0 * (sum_events - m) / (M - m)), 0, 255).astype(np.uint8)
        # event_preview = np.clip((255.0 * (sum_events - sum_events.min()) / (sum_events.max() - sum_events.min())).astype(np.uint8), 0, 255)

    return event_preview
