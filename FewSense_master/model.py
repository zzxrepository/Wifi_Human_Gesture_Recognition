from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Softmax, Dense, Lambda, ZeroPadding2D,Conv2D,MaxPooling2D,Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from Config import getConfig
import tensorflow as tf

config = getConfig()
def learning_rate_schedule(process,init_learning_rate = 0.01,alpha = 10.0 , beta = 0.75):
    num = (process//50)
    return init_learning_rate*(0.1)**num


class models:
    def __init__( self ):
        self.paddingLayer_1 = ZeroPadding2D( padding = 1 )
        self.paddingLayer_2 = ZeroPadding2D( padding = 2 )
        self.mpLayer_1 = MaxPooling2D( pool_size = 3, strides = 1 )
        self.mpLayer_2 = MaxPooling2D( pool_size = 3, strides = 2 )
        self.conv_1 = Conv2D(filters = 96, kernel_size = (11, 5), 
                             strides = 2, input_shape = config.input_shape,
                             padding = 'valid',activation = 'relu')
        self.conv_2 = Conv2D( filters = 256, activation = 'relu', kernel_size = 5, strides = 1, padding = 'valid' )
        self.conv_3 = Conv2D( filters = 384, activation = 'relu', kernel_size = 3, strides = 1, padding = 'valid' )
        self.conv_4 = Conv2D( filters = 384, activation = 'relu', kernel_size = 3, strides = 1, padding = 'valid' )
        self.conv_5 = Conv2D( filters = 256, activation = 'relu', kernel_size = (4, 3), strides = 1, padding = 'valid' )
        self.dp = Dropout( 0.5 )
        # pass
    def buildFeatureExtractor( self ):
        input = Input( config.input_shape, name = 'input_layer' )
        x = self.conv_1( input )
        x = self.mpLayer_1( x )
        x = self.paddingLayer_2( x )
        x = self.conv_2( x )
        x = self.mpLayer_2( x )
        x = self.paddingLayer_1( x )
        x = self.conv_3( x )
        x = self.paddingLayer_1( x )
        x = self.conv_4( x )
        x = self.paddingLayer_1( x )
        x = self.conv_5( x )
        x = self.mpLayer_2( x )
        x = self.dp( x )
        x = Flatten( )( x )
        x = Dense( units = 256, name = 'FC_1' )( x )
        x = Dense( units = 1280, name = 'FC_2' )( x )
        output = Lambda( lambda x: K.l2_normalize( x, axis = -1 ),name = 'lambda_layer' )( x )
        return Model( inputs = input, outputs = output )
    def buildTuneModel( self ,config, isTest:bool = False,feature_extractor = None):
        if isTest:
                feature_extractor = self.buildFeatureExtractor( )
        else:
            assert feature_extractor is not None, 'Using pretrained feature extractor but the path to pretrained model is not specified'
        x = Dense( units = config.N_novel_classes, bias_regularizer = regularizers.l2( 1e-4 ), name = "fine_tune_layer" )(feature_extractor.output )
        output = Softmax( )( x )
        return Model( inputs = feature_extractor.input, outputs = output )


