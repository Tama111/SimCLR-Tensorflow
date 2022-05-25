import tensorflow as tf

class ProjectionHead(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.mlp = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units = dim),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(units = dim)
        ])
    
    def call(self, inputs):
        return self.mlp(inputs)



class SimCLR(object):
    def __init__(self, model, out_dim, optimizer, loss):
        if not model.trainable:
            model.trainable = True
        
        self.model = model
        self.projection_head = ProjectionHead(dim = out_dim)
        self.optimizer = optimizer
        self.loss = loss
    
    @tf.function
    def train_step(self, data):
        xi, xj = data
        with tf.GradientTape() as tape:
            hi, hj = self.model(xi), self.model(xj)
            zi, zj = self.projection_head(hi), self.projection_head(hj)
            loss = self.loss(zi, zj)
            
        params = self.model.trainable_variables + self.projection_head.trainable_variables
        grads = tape.gradient(loss, params)
        self.optimizer.apply_gradients(zip(grads, params))
        
        return loss
    
    def train(self, data, epochs = 1):
        losses = []
        for e in range(epochs):
            print(f'Epoch: {e} Starts')
            t_loss = 0
            for d in data:
                loss = self.train_step(d)
                print('.', end = '')
                t_loss += loss
                
            avg_loss = t_loss/len(data)
            losses.append(avg_loss)
            print(f'\nLoss: {avg_loss}')
            print(f'Epoch: {e} Ends.\n')
        
        return losses
