#Approved for public release, 20-563

from keras.layers import Conv2D
from keras.layers import Activation,Dense,Flatten,GlobalAveragePooling2D,Dropout,concatenate
from keras.models import Model

from .blocks import Transpose2D_block
from .blocks import Upsample2D_block
from ..utils import get_layer_number, to_tuple

def build_unet(backbone, classes, skip_connection_layers,
               decoder_filters=(256,128,64,32,16),
               upsample_rates=(2,2,2,2,2),
               n_upsample_blocks=5,
               block_type='upsampling',
               activation='tanh',
               use_batchnorm=True, add_height=False):

    input = backbone.input
    x = backbone.output
    
    y = GlobalAveragePooling2D(name='pool1')(x)
    y = Dense(2048, name='fc1')(y)
    y = Dropout(0.5)(y)
    xydir = Dense(2, name='xydir', activation='linear')(y)
    
    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                               for l in skip_connection_layers])

    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        upsample_rate = to_tuple(upsample_rates[i])

        x = up_block(decoder_filters[i], i, upsample_rate=upsample_rate,
                     skip=skip_connection, use_batchnorm=use_batchnorm)(x)
        
    
    x = Conv2D(1, (3,3), padding='same', name='final_conv')(x)
    if add_height:
        height = Dense(1, name='agl', activation='linear')(x)
        x = concatenate([x,height],axis=3)
        
    mag = Dense(1, name='mag', activation='linear')(x)
    
    if add_height:
        model = Model(inputs=input, outputs=[xydir,mag,height])
    else:
        model = Model(inputs=input, outputs=[xydir,mag])
    
    return model
