
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa


learning_rate = 0.0001
weight_decay = 0.0001
num_epochs = 50

image_size = 224
batch_size = 12
n_classes = 1

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = L.Dense(units, activation = tf.nn.gelu)(x)
        x = L.Dropout(dropout_rate)(x)
    return x
    
 
 class Patches(L.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images = images,
            sizes = [1, self.patch_size, self.patch_size, 1],
            strides = [1, self.patch_size, self.patch_size, 1],
            rates = [1, 1, 1, 1],
            padding = 'SAME',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

 

class PatchEncoder(L.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = L.Dense(units = projection_dim
                                  , activation=tf.nn.relu
                                 )
        self.position_embedding = L.Embedding(
            input_dim = num_patches, output_dim = projection_dim
        )

    def call(self, patch):
        positions = tf.range(start = 0, limit = self.num_patches, delta = 1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def vision_transformer():
    inputs = L.Input(shape = (image_size, image_size, 4))
    
    patches = Patches(patch_size)(inputs)
    
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        
        x1 = L.LayerNormalization(epsilon = 1e-6)(encoded_patches)
        
        attention_output = L.MultiHeadAttention(num_heads = num_heads, key_dim = projection_dim, dropout = 0.1)(x1, x1)
        
        x2 = L.Add()([attention_output, encoded_patches])
        
        x3 = L.LayerNormalization(epsilon = 1e-6)(x2)
        
        x3 = mlp(x3, hidden_units = transformer_units, dropout_rate = 0.1)
        
        encoded_patches = L.Add()([x3, x2])
        
    representation = L.LayerNormalization(epsilon = 1e-6)(encoded_patches)
    representation = L.Flatten()(representation)
    representation = L.Dropout(0.5)(representation)

    features = mlp(representation, hidden_units = mlp_head_units, dropout_rate = 0.5)
    
    logits = L.Dense(1,activation="softmax")(features)
    
    model = tf.keras.Model(inputs = inputs, outputs = logits)
    
    return model
