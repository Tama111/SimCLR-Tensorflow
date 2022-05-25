from glob import glob
import tensorflow as tf

class DataLoader(object):
    def __init__(self, data_augment_pipeline, img_size = 224, channels = 3, num_images = 1, 
                 rescale_type = '0|1', batch_size = 1):
        self.img_size = img_size
        self.data_augment_pipeline = data_augment_pipeline
        self.channels = channels
        self.num_images = num_images
        self.rescale_type = rescale_type
        self.batch_size = batch_size
        
    def _load_data(self, images):
        images = tf.io.decode_png(tf.io.read_file(images), channels = self.channels)
        
        images = tf.image.resize(images, [self.img_size, self.img_size], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        images = tf.cast(images, tf.float32)
        
        if self.rescale_type is not None:
            if self.rescale_type == '0|1':
                images /= 255.0
            elif self.rescale_type == '-1|1':
                images = images/127.5 - 1
                
        return images, images
        
    def __call__(self, path, apply_augment = True):
        data = glob(path + '//*.jpg')[:self.num_images]
        
        data = tf.data.Dataset.list_files(data)
        data = data.map(self._load_data, num_parallel_calls = tf.data.AUTOTUNE)
        data = data.shuffle(self.num_images)
        
        ori_data, aug_data = [], []
        for od, ad in data:
            ori_data.append(od)
            aug_data.append(self.data_augment_pipeline(ad))
        
        # if apply_augment:
        #     data = data.map(lambda x, y: (x, self.data_augment_pipeline(y)), num_parallel_calls = tf.data.AUTOTUNE)

        return tf.data.Dataset.from_tensor_slices((ori_data, aug_data)).batch(self.batch_size, drop_remainder = True)
    
