import math
import torch
import torch.nn.functional as F
from torch.autograd import Function
import sys

def low_pass_filter(
        log_new_frame,
        lp_log_frame0,
        inten01,
        delta_time,
        cutoff_hz=0):
    """Compute intensity-dependent low-pass filter.
    # Arguments
        log_new_frame: new frame in lin-log representation.
        lp_log_frame0:
        lp_log_frame1:
        inten01:
        delta_time:
        cutoff_hz:
    # Returns
        new_lp_log_frame0
        new_lp_log_frame1
    """
    if cutoff_hz <= 0:
        # unchange
        return log_new_frame

    tau = 1/(math.pi*2*cutoff_hz)
    if inten01 is not None:
        eps = inten01*(delta_time/tau)
        max_eps = torch.max(eps)
        eps = torch.clamp(eps, max=1).detach() + eps - eps.detach()  # keep filter stable. STE just in case
    else:
        eps=delta_time/tau

    new_lp_log_frame0 = (1-eps)*lp_log_frame0+eps*log_new_frame
    # print(new_lp_log_frame0.min(), new_lp_log_frame0.max(), new_lp_log_frame0.mean() )
    # sys.exit()

    return new_lp_log_frame0 #, new_lp_log_frame1

def generate_shot_noise(
        shot_noise_rate_hz,
        delta_time,
        shot_noise_inten_factor,
        inten01,
        pos_thres_pre_prob,
        neg_thres_pre_prob):
    """Generate shot noise.
    :param shot_noise_rate_hz: the rate per pixel in hz
    :param delta_time: the delta time for this frame in seconds
    :param shot_noise_inten_factor: factor to model the slight increase
        of shot noise with intensity when shot noise dominates at low intensity
    :param inten01: the pixel light intensities in this frame; shape is used to generate output
    :param pos_thres_pre_prob: per pixel factor to generate more
        noise from pixels with lower ON threshold: self.pos_thres_nominal/self.pos_thres
    :param neg_thres_pre_prob: same for OFF

    :returns: shot_on_coord, shot_off_coord, each are (h,w) arrays of on and off boolean True for noise events per pixel
    """
    # new shot noise generator, generate for the entire batch of iterations over this frame

    if shot_noise_rate_hz*delta_time>1:
        print(f'shot_noise_rate_hz*delta_time={shot_noise_rate_hz:.2f}*{delta_time:.2g}={shot_noise_rate_hz*delta_time:.2f} is too large, decrease timestamp resolution or sample rate')

    # shot noise factor is the probability of generating an OFF event in this frame (which is tiny typically)
    # we compute it by taking half the total shot noise rate (OFF only),
    # multiplying by the delta time of this frame,
    # and multiplying by the intensity factor
    # division by num_iter is correct if generate_shot_noise is called outside the iteration loop, unless num_iter=1 for calling outside loop
    shot_noise_factor = (
        (shot_noise_rate_hz/2)*delta_time) * \
        ((shot_noise_inten_factor-1)*inten01+1) # =1 for inten=0 and SHOT_NOISE_INTEN_FACTOR for inten=1 # TODO check this logic again, the shot noise rate should increase with intensity but factor is negative here

    # probability for each pixel is
    # dt*rate*nom_thres/actual_thres.
    # That way, the smaller the threshold,
    # the larger the rate
    one_minus_shot_ON_prob_this_sample = \
        1 - shot_noise_factor*pos_thres_pre_prob # ON shot events are generated when uniform sampled random number from range 0-1 is larger than this; the larger shot_noise_factor, the larger the noise rate
    shot_OFF_prob_this_sample = \
        shot_noise_factor*neg_thres_pre_prob # OFF shot events when 0-1 sample less than this

    # for shot noise generate rands from 0-1 for each pixel
    rand01 = torch.rand(
        size=inten01.shape,
        dtype=torch.float32,
        device=inten01.device)  # draw_frame samples

    # precompute all the shot noise cords, gets binary array size of chip
    shot_on_cord = torch.gt(
        rand01, one_minus_shot_ON_prob_this_sample)
    shot_off_cord = torch.lt(
        rand01, shot_OFF_prob_this_sample)

    return shot_on_cord, shot_off_cord


def initialize_identity(tensor, kernel_size, dtype, mode):
    ident = torch.zeros_like(tensor, dtype=dtype)
    ident[:,:,kernel_size//2,kernel_size//2] = torch.tensor(1.0, dtype=dtype)

    # if mode == 'mem':
    #     ident = -ident
    tensor.data.copy_(ident)

def check_nans_infs(x, name, print_output=False):
    if print_output:
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f" {name}: {torch.isnan(x).sum()} nans {torch.isinf(x).sum()} infs")


def event_preprocess_pytorch(event_voxel_grid, mode='std', filter_hot_pixel=True, debug=None):
# Normalize the event tensor (voxel grid) so that
# the mean and stddev of the nonzero values in the tensor are equal to (0.0, 1.0)

    num_bins = event_voxel_grid.shape[0]
    if filter_hot_pixel:
        event_voxel_grid[abs(event_voxel_grid) > 20./num_bins]= 0
    if mode == 'maxmin':
        event_voxel_grid = (event_voxel_grid- event_voxel_grid.min())/(event_voxel_grid.max()- event_voxel_grid.min()+1e-8)
    elif mode == 'std':
        print("__________________NORMALIZING____________")
        print("BEFORE NORMALIZATION", event_voxel_grid.min(), event_voxel_grid.mean(), event_voxel_grid.max())
        nonzero_ev = (event_voxel_grid != 0)
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            # compute mean and stddev of the **nonzero** elements of the event tensor
            # we do not use PyTorch's default mean() and std() functions since it's faster
            # to compute it by hand than applying those funcs to a masked array
            mean = event_voxel_grid.sum() / num_nonzeros
            # mask = nonzero_ev.float()
            stddev = torch.sqrt((event_voxel_grid ** 2).sum() / num_nonzeros - mean ** 2)                
            event_voxel_grid = torch.where( event_voxel_grid != 0, (event_voxel_grid - mean) / (stddev+1e-8), event_voxel_grid)
        print("AFTER NORMALIZATION", event_voxel_grid.min(), event_voxel_grid.mean(), event_voxel_grid.max())
    elif mode == 'std_pytorch':
        if debug:
                event_voxel_grid = debug * event_voxel_grid
        if (event_voxel_grid.abs()).sum() > 0: 
            mean = (event_voxel_grid/(1.0*event_voxel_grid.detach().numel())).sum()
            output_event_voxel_grid = torch.where(event_voxel_grid>0, event_voxel_grid - mean, event_voxel_grid)
            # event_voxel_grid = (event_voxel_grid - mean)
            event_voxel_grid = event_voxel_grid/event_voxel_grid.detach().std()
            # print(f"torch.float64 std {torch.isnan(event_voxel_grid).sum()} nans {torch.isinf(event_voxel_grid).sum()} infs")
    return event_voxel_grid
