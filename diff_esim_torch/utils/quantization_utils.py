import math
import torch
import torch.nn.functional as F
from torch.autograd import Function
import sys
import torch.nn as nn



class differentiable_event_generation(Function):
    @staticmethod
    def forward(ctx, x, tolerance): 
        # Create a mask based on the polarity of the change [-1,0,1]
        pol = torch.sign(x)

        # # Apply tolerance to the mask
        # pol = where(x.abs() <= tolerance, torch.zeros_like(pol), pol) # use binarized hadamard product
        ctx.save_for_backward(x)
        ctx.tolerance = tolerance
        return pol # aka "polarity" 


    @staticmethod # straight through estimator
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        tolerance = ctx.tolerance
        grad_input = grad_output * (1 - torch.tanh(x)**2)

        return grad_input, None
    
''' Neural Sensors '''
''' Function for the binarized hadamard product between weights and inputs'''
def where(cond, x1, x2):
    return cond.float() * x1 + (1 - cond.float()) * x2

class BinarizeHadamardFunction(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        weight_b = where(weight>=0, 1, 0) # binarize weights
        output = input * weight_b

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_variables
        weight_b = where(weight>=0, 1, 0) # binarize weights
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * weight_b
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output * input

        return grad_input, grad_weight

binarize_hadamard = BinarizeHadamardFunction.apply

''' Straight through estimators '''
class SignIncludingZeroStraightThroughFunction(Function):
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        output = input>=0 # includes 0
        return output.float()

    def backward(ctx, grad_output):
        grad_input = torch.clamp(grad_output,-1,1)
        return grad_input

class SignExcludingZeroStraightThroughFunction(Function):
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        output = input>0 # excludes 0
        return output.float()

    def backward(ctx, grad_output):
        grad_input = torch.clamp(grad_output,-1,1)
        return grad_input

sign_incl0 = SignIncludingZeroStraightThroughFunction.apply
sign_excl0 = SignExcludingZeroStraightThroughFunction.apply

# a <= b
def le_st(a,b):
    return sign_incl0(b-a)

#a < b
def lt_st(a,b):
    return sign_excl0(b-a)

# a >= b
def ge_st(a,b):
    return sign_incl0(a-b)

# a > b
def gt_st(a,b):
    return sign_excl0(a-b)

class FloorStraightThroughEstimator(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.clamp(grad_output,-1,1)
        return grad_input

class CeilStraightThroughEstimator(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.ceil(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.clamp(grad_output,-1,1)
        return grad_input


''' Differentiable binary shutter (for HFR)'''
class BinaryBlockShutter(torch.nn.Module):
    '''
    block repeat
    '''

    def __init__(self,shutter_prob, block_size, overlap_per_dim, quant_noise=2./255., learnable=False):
        super(BinaryBlockShutter, self).__init__()
        self.shutter_prob = shutter_prob
        self.block_size = block_size
        self.overlap_per_dim = overlap_per_dim
        self.quant_noise = quant_noise

        self.subblock_size = [0,0,0]
        self.subblock_size[0] = block_size[0]
        self.subblock_size[1] = block_size[1]//overlap_per_dim
        self.subblock_size[2] = block_size[2]//overlap_per_dim

        # weights are randomly drawn betwen [0,1] and then shifted by (1-shutter_prob)
        # then rand_weight is in [-1+shutter_prob,shutter_prob]
        # this means we have shutter_prob chance to be above 0, that is to shutter on!
        rand_weight = torch.rand(self.subblock_size,dtype=torch.float64) - (1-shutter_prob) # the weight is the size of a subblock
        self.weight = nn.Parameter(rand_weight,requires_grad=learnable)

    def getMeasurementMatrix(self):
        weight_tiled = self.weight.repeat(1,self.overlap_per_dim,self.overlap_per_dim)
        measurement_matrix = torch.gt(weight_tiled,0).float()
        return measurement_matrix

    def forward(self, video_block):
        # the weight is the size of a subblock and should be tiled to a whole block
        weight_tiled = self.weight.repeat(1,self.overlap_per_dim,self.overlap_per_dim)

        # B W C H
        measurement = torch.mean( binarize_hadamard(video_block,weight_tiled), 1, keepdim=True)

        measurement = torch.clamp(measurement + self.quant_noise*torch.randn_like(measurement),0,1)
        return measurement

class BinaryGlobalShutter(torch.nn.Module):
    ''' learn one length
    '''

    def __init__(self, shutter_prob, size, quant_noise=2./255., learnable=False):
        super(BinaryGlobalShutter, self).__init__()
        self.shutter_prob = shutter_prob
        self.size = size
        self.quant_noise = quant_noise

        # weights are randomly drawn betwen [0,1] and then shifted by (1-shutter_prob)
        # then rand_weight is in [-1+shutter_prob,shutter_prob]
        # this means we have shutter_prob chance to be above 0, that is to shutter on!
        rand_weight = torch.rand(self.size,dtype=torch.float64) - (1-shutter_prob) # the weight is the size of a subblock
        self.weight = nn.Parameter(rand_weight,requires_grad=learnable)

    def getMeasurementMatrix(self):
        measurement_matrix = torch.gt(self.weight,0).float()
        return measurement_matrix

    def forward(self, video_block):
        # B C W H
        h = self.size[1]
        w = self.size[2]
        measurement = torch.mean( binarize_hadamard(video_block[:,:,0:h,0:w],self.weight), 1, keepdim=True)

        measurement = torch.clamp(measurement + self.quant_noise*torch.randn_like(measurement),0,1)
        return measurement


class BinaryCstrdBlockShutter(torch.nn.Module):
    ''' learn different start times.block
    '''
    def __init__(self,shutter_length, block_size, overlap_per_dim, quant_noise=2./255., learnable=False):
        super(BinaryCstrdBlockShutter, self).__init__()
        self.shutter_length = shutter_length
        self.block_size = block_size
        self.overlap_per_dim = overlap_per_dim
        self.quant_noise = quant_noise

        self.subblock_size = [0,0,0]
        self.subblock_size[0] = block_size[0]
        self.subblock_size[1] = block_size[1]//overlap_per_dim
        self.subblock_size[2] = block_size[2]//overlap_per_dim

        # A start time is randomly chosen between [0 ; block_size[0]-shutter_length] (we will use floor to convert to int)
        rand_start = (self.subblock_size[0]-self.shutter_length+1) \
                    * torch.rand((self.subblock_size[1],self.subblock_size[2]),dtype=torch.float64)
        self.start_params = nn.Parameter(rand_start,requires_grad=learnable)
        #self.start_params.register_hook(print)

        # Producing the time_ranges for all pixels in a block
        time_range = torch.arange(0,self.subblock_size[0],dtype=torch.float64)[:,None,None]
        time_range = time_range.repeat(1,self.subblock_size[1],self.subblock_size[2])
        self.register_buffer('time_range',time_range) # otherwise it does not go on CUDA

    def getMeasurementMatrix(self):
        start_params_int = torch.clamp(torch.floor(self.start_params),0,self.subblock_size[0]-self.shutter_length) # we optimize over start_params
        measurement_matrix =  (self.time_range >= start_params_int) \
                            * (self.time_range  < start_params_int + self.shutter_length)
        measurement_matrix = measurement_matrix.repeat(1,self.overlap_per_dim,self.overlap_per_dim).float()

        return measurement_matrix

    def forward(self, video_block):
        # XXX misses the floor?
        start_params_int = torch.clamp(self.start_params,0,self.subblock_size[0]-self.shutter_length) # we optimize over start_params
        measurement_matrix =  ge_st(self.time_range,start_params_int) \
                            * lt_st(self.time_range,start_params_int + self.shutter_length)
        measurement_matrix_tiled = measurement_matrix.repeat(1,self.overlap_per_dim,self.overlap_per_dim)

        # B C W H
        measurement = torch.mean( video_block*measurement_matrix_tiled, 1, keepdim=True)

        measurement = torch.clamp(measurement + self.quant_noise*torch.randn_like(measurement),0,1)
        return measurement


class BinaryCstrdBlockShutter2(torch.nn.Module):
    def __init__(self, block_size, overlap_per_dim, quant_noise=2./255., learnable=False):
        super(BinaryCstrdBlockShutter2, self).__init__()
        self.block_size = block_size
        self.overlap_per_dim = overlap_per_dim
        self.quant_noise = quant_noise

        self.subblock_size = [0,0,0]
        self.subblock_size[0] = block_size[0]
        self.subblock_size[1] = block_size[1]//overlap_per_dim
        self.subblock_size[2] = block_size[2]//overlap_per_dim

        rand_length = (self.subblock_size[0]-1)*torch.rand((self.subblock_size[1],self.subblock_size[2]),dtype=torch.float64)
        self.shutter_length_params = nn.Parameter(rand_length,requires_grad=learnable)

        # A start time is randomly chosen between [0 ; block_size[0]-shutter_length] (we will use floor to convert to int)
        rand_start = (self.subblock_size[0]-self.shutter_length_params+1) \
                    * torch.rand((self.subblock_size[1],self.subblock_size[2]),dtype=torch.float64)
        self.start_params = nn.Parameter(rand_start,requires_grad=learnable)
        #self.start_params.register_hook(print)

        # Producing the time_ranges for all pixels in a block
        time_range = torch.arange(0,self.subblock_size[0],dtype=torch.float64)[:,None,None]
        time_range = time_range.repeat(1,self.subblock_size[1],self.subblock_size[2])
        self.register_buffer('time_range',time_range) # otherwise it does not go on CUDA

    def getMeasurementMatrix(self):
        #start_params_int = torch.min(torch.max(torch.floor(self.start_params),0),self.subblock_size[0]-self.shutter_length_params)
        start_params_int = torch.clamp(torch.floor(self.start_params),0,self.subblock_size[0]-1) # we optimize over start_params
        measurement_matrix =  (self.time_range >= start_params_int) \
                            * (self.time_range  < start_params_int + self.shutter_length_params)
        measurement_matrix = measurement_matrix.repeat(1,self.overlap_per_dim,self.overlap_per_dim).float()

        return measurement_matrix

    def forward(self, video_block):
        # XXX misses the floor?
        #start_params_int = torch.min(torch.max(torch.floor(self.start_params),0),self.subblock_size[0]-self.shutter_length_params)
        start_params_int = torch.clamp(self.start_params,0,self.subblock_size[0]-1) # we optimize over start_params
        measurement_matrix =  ge_st(self.time_range,start_params_int) \
                            * lt_st(self.time_range,start_params_int + self.shutter_length_params)
        measurement_matrix_tiled = measurement_matrix.repeat(1,self.overlap_per_dim,self.overlap_per_dim)

        # B C W H
        measurement = torch.mean( video_block*measurement_matrix_tiled, 1, keepdim=True)

        measurement = torch.clamp(measurement + self.quant_noise*torch.randn_like(measurement),0,1)
        return measurement


