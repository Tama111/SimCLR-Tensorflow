import numpy as np
import tensorflow as tf

class GaussianBlur2D(object):
    def __init__(self, kernel_size, sigma, chn = 3):
        k = np.arange(-kernel_size//2 + 1, kernel_size//2 + 1)
        x, y = np.meshgrid(k, k)
        
        self.gaussian_kernel = self._gaussian_dist_2d(x, y, sigma).astype('float32')
        self.gaussian_kernel /= tf.math.reduce_sum(self.gaussian_kernel)
        self.gaussian_kernel = tf.repeat(self.gaussian_kernel[..., np.newaxis, np.newaxis], chn, axis = -1)
        
    def _gaussian_dist_2d(self, x, y, sigma):
        return ((1/(2*(sigma**2)*np.pi))*np.exp(-(x**2 + y**2)/(2*(sigma**2))))
    
    def __call__(self, img):
        assert img.dtype == tf.float32
        if len(img.shape) == 3:
            img = tf.expand_dims(img, axis = 0)
        return tf.nn.conv2d(img, self.gaussian_kernel, [1, 1, 1, 1], 'SAME')[0]


class RandomCrop(object):
    def __init__(self, crop_ratio = 0.875, lower_limit = 0.5, resize = True):
        self.crop_ratio = crop_ratio
        self.lower_limit = lower_limit
        self.resize = resize
        self.resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        
    @property
    def get_crop_ratio(self):
        if self.crop_ratio is None:
            crop_ratio = float(tf.random.uniform((1, ), self.lower_limit, 1.0))
        else:
            assert isinstance(self.crop_ratio, float) or isinstance(self.crop_ratio, int)
            assert self.lower_limit < self.crop_ratio
            assert (self.crop_ratio <= 1.0) or (self.crop_ratio > 0.0)
            crop_ratio = self.crop_ratio
        
        return crop_ratio
        
    def __call__(self, img):
        crop_ratio = self.get_crop_ratio
        
        img_h, img_w = img.shape[-3:-1]
        crop_h, crop_w = int(img_h * crop_ratio), int(img_w * crop_ratio)
        
        h_shft = int(np.random.uniform(0, img_h - crop_h, size = (1, )))
        w_shft = int(np.random.uniform(0, img_w - crop_w, size = (1, )))
        
        if len(img.shape) == 3:
            cropped_img = img[h_shft:h_shft+crop_h, w_shft:w_shft+crop_w, :]
        elif len(img.shape) == 4:
            cropped_img = img[:, h_shft:h_shft+crop_h, w_shft:w_shft+crop_w, :]
            
            
        if self.resize:
            return tf.image.resize(cropped_img, [img_h, img_w], self.resize_method)
        return img



class GrayScale(object):
    def __init__(self, keep_channels = True):
        self.keep_channels = keep_channels
        
    def __call__(self, img, return_true = True):
        out = tf.image.rgb_to_grayscale(img)
        out = tf.repeat(out, img.shape[-1], axis = -1) if self.keep_channels else out
        return out
    
    
class LRFlip(object):
    def __init__(self):
        pass
    
    def __call__(self, img):
        return tf.image.random_flip_left_right(img)
    
    
class RandomBrightness(object):
    def __init__(self, delta = None, lower_limit = 0.2):
        self.delta = delta
        self.lower_limit = lower_limit
            
    def __call__(self, img):
        if self.delta is None:
            return img * float(tf.random.uniform((1, ), self.lower_limit, 1.0))
        elif isinstance(self.delta, float):
            return img * delta
        else:
            raise Exception
        

class RandomColorJitter(object):
    def __init__(self):
        self.brightness_delta = None
        self.saturation_range = [0, 50]
        self.contrast_range = [0, 5]
        self.hue_delta = 0.2
        
        self.color_jitter = [RandomBrightness(self.brightness_delta), 
                             lambda x: tf.image.random_saturation(x, self.saturation_range[0], self.saturation_range[1]), 
                             lambda x: tf.image.random_contrast(x, self.contrast_range[0], self.contrast_range[1]), 
                             lambda x: tf.image.random_hue(x, self.hue_delta)]
        
    def __call__(self, img):
        # probs = list(np.random.uniform((4, ), 0, 1))
        # for i, p in enumerate(probs):
        #     if p >= 0.5:
        #         img = self.color_jitter[i](img)
        return self.color_jitter[np.random.choice(np.arange(4))](img)


class DataAugmentPipeline(object):
    def __init__(self, crop_resize, lr_flip, color_jitter, gray_scale, gaussian_blur, crop_resize_p = 1.0, lr_flip_p = 0.5,
                 color_jitter_p = 0.8, gray_scale_p = 0.2, gaussian_blur_p = 0.5):
        
        self.augment = [(crop_resize, crop_resize_p), 
                        (lr_flip, lr_flip_p), 
                        (color_jitter, color_jitter_p), 
                        (gray_scale, gray_scale_p), 
                        (gaussian_blur, gaussian_blur_p)]
        
        
    def __call__(self, img, return_true = True):
        probs = list(np.random.uniform(0.0, 1.0, size = (5, )))
        for i, p in enumerate(probs):
            aug = self.augment[i]
            if p <= aug[1]:
                # try:
                img = aug[0](img)
                # except TypeError:
                # print(img)
                # img = img.map(lambda x, y: (x, aug[0](y)))
                  
        return img

    
