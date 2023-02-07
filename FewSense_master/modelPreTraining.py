import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from sklearn.metrics.pairwise import cosine_similarity
from Config import getConfig
from model import models
from gestureDataLoader import *
class FSLtest():
    def __init__(self,config):
        self.config = config
        self.modelObj = models()
    def _loadModel(self,applyFinetunedModel:bool = True, useWeightMatrix:bool = False):
        '''
        This function build for load fine tuned model for testing
        :returns pre-trained feature extractor and fine tuned classifier
        '''
        if applyFinetunedModel:
            print( f'loading fine tuned model: {self.config.tunedModel_path}' )
            fine_Tune_model = self.modelObj.buildTuneModel( config = self.config,isTest = True )
            fine_Tune_model.load_weights(self.config.tunedModel_path)
            feature_extractor = fine_Tune_model if useWeightMatrix else Model( inputs = fine_Tune_model.input, outputs = fine_Tune_model.get_layer( 'lambda_layer' ).output )
        else:
            print( f'loading original pretrained feature extractor: {self.config.pretrainedfeatureExtractor_path}' )
            feature_extractor = self.modelObj.buildFeatureExtractor( )
            feature_extractor.load_weights(self.config.pretrainedfeatureExtractor_path)
        return feature_extractor
    def _getOneshotTaskData(self, test_data, test_labels, nway,kshots, mode:str = 'cross_val' ):
        '''
        This function build for n-way 1 shot task
        :param test_data: the Data for testing model
        :param test_labels: corresponding labels
        :param nway: the number of training classes
        :param mode: cross validation or fix the support set classes
        :return: support set : one sample, query set one sample
        '''
        signRange = np.arange( int( np.min( test_labels ) ), int( np.max( test_labels ) + 1 ), 1 )
        # signRange = np.arange( int( 251 ), int( np.max( test_labels ) + 1 ), 1 )
        selected_Sign = np.random.choice( signRange, size=nway, replace=False )
        support_set = [ ]
        query_set = [ ]
        labels = [ ]
        for i in selected_Sign:
            index, _ = np.where( test_labels == i )
            # if mode == 'cross_val':
            selected_samples = np.random.choice( index, size=kshots+1, replace=False )
            n_idx = len(selected_samples)
            support_set.append( test_data[ selected_samples[ 0:n_idx-1 ] ] )
            query_set.append( test_data[ selected_samples[ -1 ] ] )
            labels.append( i )

        return np.concatenate( support_set,axis=0 ), query_set
    def _getTaskData( self,x_test,y_test,nshots):
        num = nshots * (np.max( y_test ) + 1 - np.min( y_test ))
        Support_data = x_test[ 0:num, :, :, : ]
        Query_data = x_test[ num:len( x_test ) + 1, :, :, : ]
        N_sample = int(len(Query_data)/self.config.N_novel_classes)
        selected_samples = np.random.choice( int(N_sample), size = 1, replace = False )
        return Support_data, Query_data[int(selected_samples*self.config.N_novel_classes):int(selected_samples*self.config.N_novel_classes+self.config.N_novel_classes)],
    def _getNShotsEmbedding( self, feature_extractor, support_set ):
        Sign_class = np.arange( 0, self.config.N_novel_classes, 1 )
        Sign_samples = np.arange( 0, len(support_set), self.config.N_novel_classes )
        n_shots_support_embedding = [ ]
        for i in Sign_class:
            n_shots_support_data = [ ]
            for j in Sign_samples:
                n_shots_support_data.append( support_set[ i + j ] )
            n_shots_support_embedding.append(np.mean( feature_extractor.predict( np.asarray( n_shots_support_data ) ), axis = 0 ) )
        n_shots_support_embedding = np.asarray( n_shots_support_embedding )
        return np.asarray( n_shots_support_embedding )
    def signTest(self, test_data, test_labels, feature_extractor,nways=None ):

        '''
        This function build for testing the model performance from two ways to 25 ways
        :param test_data:
        :param test_labels:
        :param N_test_sample:
        :param embedding_model:
        :param isOneShotTask:
        :param mode:
        :return:
        '''
        test_acc = [ ]
        N_test_sample = 1000
        softmax_func = tf.keras.layers.Softmax( )
        # for nway in np.concatenate( (np.arange( 2, 10 ), np.arange( 10, 77, 10 ), np.asarray( [ 76 ] )), axis = 0 ):
        for nway in [nways]:
            print( "Checking %d way accuracy...." % nway )
            correct_count = 0
            for i in range( N_test_sample ):
                support_set, query_set = self._getTaskData( test_data, test_labels,self.config.nshots)
                sample_index = random.randint( 0, nway - 1 )
                # if mode == 'fix' and i == 0:
                #     support_set_embedding = feature_extractor.predict( np.asarray( support_set ) )
                # elif mode == 'cross_val':
                support_set_embedding = self._getNShotsEmbedding( feature_extractor, np.asarray( support_set ) )
                query_set_embedding = feature_extractor.predict( np.expand_dims( query_set[ sample_index ], axis=0 ) )
                sim = cosine_similarity( support_set_embedding, query_set_embedding )
                prob = softmax_func( np.squeeze( sim, -1 ) ).numpy()
                if np.argmax( prob ) == sample_index:
                    correct_count += 1
            acc = (correct_count / N_test_sample) * 100.
            test_acc.append( acc )
            print( "Accuracy %.2f" % acc )
        return test_acc
class PreTrainModel:
    def __init__( self,config):
        self.config = config
        self.modelObj = models( )
    def _splitData( self,x_all, y_all, source: str = '4user' ):
        '''
        This function build for split sign data, 1 to 125 for training, 125 to 150 for one shot learning
        :param x_all:
        :param y_all:
        :param source:
        :return:
        '''
        if source == '4user':
            train_data = np.zeros( (5000, 200, 60, 3) )
            train_labels = np.zeros( (5000, 1), dtype = int )
            unseen_sign_data = np.zeros( (1000, 200, 60, 3) )
            unseen_sign_label = np.zeros( (1000, 1), dtype = int )
            count_tra = 0
            count_test = 0
            for i in np.arange( 0, 6000, 1500 ):
                train_data[ count_tra:count_tra + 1250, :, :, : ] = x_all[ i:i + 1250, :, :, : ]
                train_labels[ count_tra:count_tra + 1250, : ] = y_all[ i:i + 1250, : ]
                unseen_sign_data[ count_test:count_test + 250, :, :, : ] = x_all[ i + 1250:i + 1500, :, :, : ]
                unseen_sign_label[ count_test:count_test + 250, : ] = y_all[ i + 1250:i + 1500, : ]
                count_tra += 1250
                count_test += 250
            idx = np.random.permutation( len( train_labels ) )
            return [ train_data[ idx ], train_labels[ idx ], unseen_sign_data, unseen_sign_label ]
        elif source == 'singleuser':
            x_all = x_all[ 0:1250 ]
            y_all = y_all[ 0:1250 ]
            idx = np.random.permutation( len( y_all ) )
            x_all = x_all[ idx, :, :, : ]
            y_all = y_all[ idx, : ]
        return [ x_all, y_all ]
    def builPretrainModel( self,):
        '''
        This function build for create the pretrain model
        :param mode: select backbone model
        :return: whole model for pre-training and feature extractor
        '''
        self.feature_extractor = self.modelObj.buildFeatureExtractor( )
        self.feature_extractor.trainable = True
        input = Input( self.config.input_shape, name = 'input' )
        x = self.feature_extractor(input)
        x = Dense(
                units = self.config.N_base_classes,
                bias_regularizer = regularizers.l2( 4e-4 ),
                )( x )
        output = Softmax( )( x )
        preTrain_model = Model( inputs = input, outputs = output )
        optimizer = tf.keras.optimizers.Adamax(
                learning_rate = self.config.lr, beta_1 = 0.95, beta_2 = 0.99, epsilon = 1e-09,
                name = 'Adamax'
                )
        preTrain_model.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics = 'acc' )
        preTrain_model.summary( )
        return preTrain_model, self.feature_extractor

def Run_Pretraining(N_train_classes):
    config = getConfig( )
    config.source = 'lab'
    config.train_dir = "/media/b218/HOME/Code_ds/SensingDataset/" + "SignFi/Dataset"
    config.N_base_classes = N_train_classes
    config.lr = 3e-4
    
    # Declare objects
    dataLoadObj = signDataLoader( config = config,)
    preTrain_modelObj = PreTrainModel( config = config )
    # Training params
    lrScheduler = ReduceLROnPlateau( monitor = 'val_loss', factor = 0.1, patience = 20, )
    earlyStop = tf.keras.callbacks.EarlyStopping( monitor = 'val_acc', patience = 50, restore_best_weights = True )
    # Sign recognition
    train_data, train_labels, _, _ = dataLoadObj.getFormatedData( source = config.source)
    train_labels = to_categorical( train_labels - 1, num_classes = int( np.max( train_labels ) ) )
    preTrain_model, feature_extractor = preTrain_modelObj.builPretrainModel(  )
    history = preTrain_model.fit( train_data, train_labels, 
                                 validation_split = 0.05, epochs = 1000, 
                                 callbacks = [ earlyStop, lrScheduler ] )
    val_acc = history.history[ 'val_acc' ]
    return [preTrain_model, feature_extractor,val_acc,config]
def Run_Test(domain,nshots,N_novel_classes=None,applyFinetunedModel=None,nways = None,**kwargs):
    '''
    Fine tune model: This model should be saved with classifier
    Feature extractor: This model should be saved without classifier
    '''
    config = getConfig( )
    # modelObj = models( )
    # config.N_novel_classes = 26
    config.source = domain
    config.train_dir = "/media/b218/HOME/Code_ds/SensingDataset/" + "SignFi/Dataset"
    config.nshots = nshots
    config.N_novel_classes = N_novel_classes
    config.N_base_classes = 276 - config.N_novel_classes

    # config.lr = 3e-4
    if applyFinetunedModel:
        config.tunedModel_path = kwargs['model_path']
    else:
        config.pretrainedfeatureExtractor_path = kwargs['model_path']
    

    dataLoadObj = signDataLoader( config = config )

    _, _, test_data, test_labels = dataLoadObj.getFormatedData(
            source = config.source,
            # isZscore = False
            )
    if type(config.source) == list:
        test_data = test_data[ 1250:1500 ]
        test_labels = test_labels[ 1250:1500 ]
    
    FSLtestObj = FSLtest( config )
    feature_extractor = FSLtestObj._loadModel( applyFinetunedModel, False )
    test_acc = FSLtestObj.signTest( test_data, test_labels, feature_extractor,nways = nways )
    return test_acc
if __name__ == '__main__':
    FE_path = 'FE_model.h5'
    model_folder = 'Saved_Models'
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
        FE_path = os.path.join(model_folder, FE_path)
    preTrain_model, feature_extractor, val_acc, config = Run_Pretraining(200)
    feature_extractor.save(FE_path)
    Run_Test('home',1,76,applyFinetunedModel=0,nways = 26,model_path = FE_path)