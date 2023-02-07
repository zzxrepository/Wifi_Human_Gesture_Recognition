import os.path
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from modelPreTraining import *
from gestureDataLoader import signDataLoader,WidarDataloader
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Softmax, Dense, Reshape, Lambda,Dot,concatenate,ZeroPadding2D,Conv2D,\
    MaxPooling2D,Flatten,Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.metrics.pairwise import cosine_similarity
from Config import getConfig
from methodTesting.t_SNE import *
from scipy.io import savemat,loadmat
from sklearn.metrics import confusion_matrix

class fineTuningSignFi:
    def __init__( self,config, isZscore = False, isiheritance = False ):
        self.nshots = config.nshots
        self.isZscore = isZscore
        self.modelObj = models( )
        self.config = config
        self.trainTestObj = PreTrainModel(config = config )
        self.lrScheduler = tf.keras.callbacks.LearningRateScheduler( self.trainTestObj.scheduler )
        # self.lrScheduler = ReduceLROnPlateau(
        #         monitor = 'val_loss', factor = 0.1,
        #         patience = 20,
        #         )
        self.earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 15, restore_best_weights
        =True, min_delta = 0.0001/2,mode = 'min',verbose=1)
        # self.pretrained_featureExtractor = self._getPreTrainedFeatureExtractor( )
        # self.pretrained_featureExtractor.trainable = True
        if self.nshots == 1:
            self.isOneShotTask = True
        else:
            self.isOneShotTask = False
    def _getSQData( self,nshots:int ):
        '''
        This function build for split support set and query set according to the number of shots
        :param nshots:
        :return:
        '''
        testSign = signDataLoader( config = self.config )

        if type(self.config.source) == list or 'user' in self.config.source:
            _, _, x_test, y_test = testSign.getFormatedData( source = self.config.source,isZscore = self.isZscore )
            x = x_test[ 1250:1500 ]
            y = y_test[ 1250:1500 ]
            num = nshots * 25
            Support_data = x[ 0:num, :, :, : ]
            Support_label = y[0:num,:]
            Query_data = x[num:len(x)+1,:,:,:]
            Query_label = y[num:len(x)+1,:]

        elif 'home' in self.config.source or 'lab' in self.config.source:
            _, _, x_test, y_test, = testSign.getFormatedData( source = self.config.source,isZscore=self.isZscore )
            num = nshots * (np.max(y_test) + 1 - np.min(y_test))
            Support_data = x_test[ 0:num, :, :, : ]
            Support_label = y_test[ 0:num, : ]
            Query_data = x_test[ num:len( x_test ) + 1, :, :, : ]
            Query_label = y_test[ num:len( x_test ) + 1, : ]
        output = {'Support_data':Support_data,
                  'Support_label':Support_label,
                  'Query_data':Query_data,
                  'Query_label':Query_label}
        return output
    def _getValData( self,Query_data,Query_label ):
        '''
        Get the validation data for fine tuning
        :return:
        '''

        val_data = Query_data
        val_label = to_categorical(
                Query_label - np.min( Query_label ), num_classes =
                self.config.n_ft_cls
                )
        return [val_data,val_label]
    def _getDataToTesting(self,query_set,nway,mode:str = 'fix'):
        '''
        Randomly choose one sample from the Query data set
        :param query_set:
        :param nway:
        :return:
        '''
        if mode == 'fix':
            sample_sign = np.random.choice(np.arange(0,len(query_set),self.config.N_novel_classes ),size = 1,
                    replace = False)
            sample_index = random.randint( 0, nway - 1 )
            query_data = np.repeat( query_set[ sample_sign+sample_index ], [ nway ], axis = 0 )
            return [ query_data, sample_index ]
        elif mode == 'random':
            sample_sign = np.random.choice(
                    np.arange( 0, len( query_set ), self.config.N_novel_classes ), size = 2, replace = False
                    )
            sample_index = random.randint( 0, nway - 1 )
            support_data = np.repeat( query_set[ sample_sign[0] + sample_index ], [ nway ], axis = 0 )
            query_data = np.repeat( query_set[ sample_sign[1] + sample_index ], [ nway ], axis = 0 )
            return [ support_data,query_data, sample_index ]
    def _getNShotsEmbedding( self,featureExtractor, Support_data):
        Sign_class = np.arange( 0, self.config.N_novel_classes, 1 )
        # Sign_samples = np.arange( 0, 125, 25 )
        Sign_samples = np.arange( 0, len(Support_data), self.config.N_novel_classes )
        n_shots_support_embedding = [ ]
        for i in Sign_class:
            n_shots_support_data = [ ]
            for j in Sign_samples:
                n_shots_support_data.append( Support_data[ i + j ] )
            n_shots_support_embedding.append(np.mean( featureExtractor.predict( np.asarray( n_shots_support_data ) ), axis = 0 ) )
        n_shots_support_embedding = np.asarray( n_shots_support_embedding )
        return n_shots_support_embedding
    def _getPreTrainedFeatureExtractor( self ):
        '''
        This function build for recreating the feature extractor and load pre-trained weights
        :return: feature extractor
        '''

        trained_featureExtractor = self.modelObj.buildFeatureExtractor( mode='Alexnet' )
        trained_featureExtractor.load_weights(self.config.pretrainedfeatureExtractor_path )
        return trained_featureExtractor
    def _loadFineTunedModel(self,applyFinetunedModel:bool = True, useWeightMatrix:bool = False,isSepRx:bool = False):
        '''
        This function build for load fine tuned model for testing
        :returns pre-trained feature extractor and fine tuned classifier
        '''
        './models/fine_tuning_signfi/fc_fineTuned_250Cls_labTohome_1_shot_without_Zscore.h5'
        if applyFinetunedModel:

            fine_Tune_model = self.modelObj.buildTuneModel( config = self.config,isTest = True )
            # fine_Tune_model = self.modelObj.buildFeatureExtractor(  )
            if not isSepRx:
                print( f'loading fine tuned model: {self.config.tunedModel_path}' )
                print('loading fine tuned model weights}')
                fine_Tune_model.load_weights(self.config.tunedModel_path)
            if useWeightMatrix:
                feature_extractor = fine_Tune_model
            else:
                feature_extractor = Model(
                        inputs = fine_Tune_model.input, outputs = fine_Tune_model.get_layer( 'lambda_layer' ).output
                        )
        elif not applyFinetunedModel:
            print( f'loading original pretrained feature extractor: {self.config.pretrainedfeatureExtractor_path}' )
            feature_extractor = self.modelObj.buildFeatureExtractor(mode = 'Alexnet')
            feature_extractor.load_weights(self.config.pretrainedfeatureExtractor_path)
        '''
        Classifier input: two feature vector
                      output: one probability
        '''
        # cls_intput_Support = Input(feature_extractor.output.shape[1],name = 'Support_input')
        # cls_intput_Query = Input( feature_extractor.output.shape[1], name = 'Query_input' )
        cls_intput_Support = Input(1280,name = 'Support_input')
        cls_intput_Query = Input( 1280, name = 'Query_input' )
        cosSim_layer = Dot( axes = 1, normalize = True )([cls_intput_Support,cls_intput_Query])
        cls_output = Softmax( )( tf.squeeze(cosSim_layer,-1) )
        classifier = Model(inputs = [cls_intput_Support,cls_intput_Query],outputs = cls_output)
        # feature_extractor, classifier = self._configModel(model = self.fine_Tune_model)
        return [feature_extractor, classifier]
    def n_tuning_classes(self,n,data):
        y = np.unique(data['Support_label'])
        FT_label = y[0:n]
        test_label = y[n:len(y)]
        support_classes_idx = np.where( data['Support_label'] == FT_label )[0]
        Query_classes_idx = np.where( data['Query_label'] == FT_label )[0]

        data['Support_data'] = data['Support_data'][support_classes_idx]
        data['Support_label'] = data['Support_label'][support_classes_idx]
        data['Query_data'] = data['Query_data'][Query_classes_idx]
        data['Query_label'] = data['Query_label'][Query_classes_idx]
        return data
    def tuning( self ,init_weights = True,init_bias = False):
        self.pretrained_featureExtractor = self._getPreTrainedFeatureExtractor( )

        # self.pretrained_featureExtractor = Model( inputs = self.pretrained_featureExtractor.input, outputs = self.pretrained_featureExtractor.get_layer( 'Maxpool_3' ).output )
        self.pretrained_featureExtractor.trainable = True
        self.data = self._getSQData( nshots = self.nshots )
        # self.data = self.n_tuning_classes(self.config.n_ft_cls,self.data)
        if 0:
            sup_label_id = np.unique( self.data["Support_label"] )
            # sup_selected_label_id = np.random.choice(sup_label_id,6,replace = False )
            sup_selected_label_id = sup_label_id[0:6]
            sup_idx = np.where( self.data[ 'Support_label' ] == sup_selected_label_id )[ 0 ]
            self.data['Support_data'] = self.data['Support_data'][sup_idx]
            self.data['Support_label'] = self.data['Support_label'][sup_idx]


            que_idx = np.where( self.data[ 'Query_label' ] == sup_selected_label_id )[ 0 ]
            self.data['Query_data'] = self.data['Query_data'][que_idx]
            self.data['Query_label'] = self.data['Query_label'][que_idx]

        fine_Tune_model = self.modelObj.buildTuneModel(
                pretrained_feature_extractor = self.pretrained_featureExtractor,
                isTest = False,config = self.config
                )
        if init_weights:
            if self.nshots == 1:
                weights = np.transpose( self.pretrained_featureExtractor.predict( self.data[ 'Support_data' ] ) )
            elif self.nshots > 1:
                weights = np.transpose(self._getNShotsEmbedding( self.pretrained_featureExtractor,self.data[ 'Support_data' ] ) )
            if init_bias:
                p = fine_Tune_model.predict(self.data['Query_data'])
                bias = np.tile(np.mean(-np.sum( p * np.log(p ),axis = 1 ) ),self.config.N_novel_classes )
            else:
                bias = np.zeros(self.config.N_novel_classes )
            fine_Tune_model.get_layer( 'fine_tune_layer' ).set_weights( [ weights, bias ] )

        val_data, val_label = self._getValData(self.data['Query_data'],self.data['Query_label'] )
        optimizer = tf.keras.optimizers.Adam(
                learning_rate = self.config.lr,
                epsilon = 1e-06,
                )
        # optimizer = tf.keras.optimizers.SGD(
        #         learning_rate = self.config.lr,
        #         momentum = 0.9,
        #         )
        fine_Tune_model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics='acc' )
        idx = np.random.permutation(len(self.data[ 'Support_data' ]))
        start = time.time( )
        self.config.history = fine_Tune_model.fit(
                self.data[ 'Support_data' ][ idx ], to_categorical( self.data[ 'Support_label' ][ idx ] - np.min(
                                self.data[ 'Support_label' ]),num_classes = self.config.n_ft_cls ),
                epochs = 1000,
                # shuffle = True,
                verbose = 0,
                validation_data = (val_data, val_label),
                callbacks = [ self.earlyStop, self.lrScheduler ]
                )
        end = time.time( )
        print( f'feature extractor tuning time is: {end - start:.2f}' )
        return fine_Tune_model
    def _mapToNways(self,Support_data,query_set,query_label,nway):
        query_label = np.argmax( query_label, axis = 1 )
        label = np.unique(query_label)
        selected_sign = np.unique(np.random.choice(label,size = nway,replace=False))
        Support_data = Support_data[selected_sign,:,:,:]
        # selected_data_idx = np.where(query_label == selected_sign)
        index = np.random.choice(np.arange(0,len(query_set),self.config.N_novel_classes ),size = 1,
                    replace = False)
        query_data = query_set[index+selected_sign,:,:,:]
        sample_index = random.randint( 0, nway - 1 )
        Query_data = np.repeat(np.expand_dims(query_data[sample_index],axis=0),[nway],axis = 0)
        return [ Support_data, Query_data,sample_index]
    def test( self, nway,applyFinetunedModel:bool=True ):
        # self.pretrained_featureExtractor = self._getPreTrainedFeatureExtractor( )
        # self.pretrained_featureExtractor.trainable = False
        # softmax_func = tf.keras.layers.Softmax( )
        self.data = self._getSQData( nshots = self.nshots )
        N_test_sample = 100
        feature_extractor, classifier = self._loadFineTunedModel( applyFinetunedModel )
        # load Support and Query dataset
        query_set, query_label = self._getValData(self.data['Query_data'],self.data['Query_label'] )
        Support_data = self.data[ 'Support_data' ]
        Support_label = self.data[ 'Support_label' ]
        test_acc = [ ]
        for i in range( 74,76 ):
            nway = i
            correct_count = 0
            print( f'................................Checking {nway} ways accuracy................................' )
            if self.isOneShotTask:
                for i in range(N_test_sample):
                    Selected_Support_data, Selected_Query_data, sample_index = self._mapToNways(Support_data,query_set,
                            query_label,nway)
                    Support_set_embedding = feature_extractor.predict( Selected_Support_data )
                    # Query_data, sample_index = self._getDataToTesting( query_set = query_set, nway = nway )
                    Query_set_embedding = feature_extractor.predict( Selected_Query_data )
                    prob_classifier = classifier.predict([Support_set_embedding,Query_set_embedding])
                    if np.argmax( prob_classifier ) == sample_index:
                        correct_count += 1
                        print( f'The number of correct: {correct_count}, The number of test count {i}' )
                acc = (correct_count / N_test_sample) * 100.
                test_acc.append( acc )
                print( "Accuracy %.2f" % acc )
            if not self.isOneShotTask:
                Support_set_embedding = self._getNShotsEmbedding( feature_extractor,Support_data )
                for i in range(N_test_sample):
                    Query_data, sample_index = self._getDataToTesting( query_set = query_set, nway = nway )
                    Query_set_embedding = feature_extractor.predict( Query_data )
                    prob_classifier = classifier.predict( [ Support_set_embedding, Query_set_embedding ] )
                    if np.argmax( prob_classifier ) == sample_index:
                        correct_count += 1
                        print( f'The number of correct: {correct_count}, The number of test count {i}' )
                acc = (correct_count / N_test_sample) * 100.
                test_acc.append( acc )
                print( "Accuracy %.2f" % acc )
        return test_acc
class fineTuningWidar(fineTuningSignFi ):
    def __init__( self,config,isMultiDomain:bool = False,isiheritance=False ):
        super().__init__(config = config, isiheritance = True, )
        self.isMultiDomain = isMultiDomain
        # self.config = config
        if not isiheritance:
            self.WidarDataLoaderObj = WidarDataloader(config = config, isMultiDomain = True )
        # self.selected_gesture_samples_data,self.x,self.y = self.WidarDataLoaderObj.x,self.WidarDataLoaderObj.x,self.WidarDataLoaderObj.y
            self.config = config
            self.nshots = config.nshots
            self.nshots_per_domain = config.nshots_per_domain
            # self.nshots_per_domain = int(self.nshots/5)
            self.nways = config.N_novel_classes
            self.pretrained_featureExtractor = self._getPreTrainedFeatureExtractor( )
            self.pretrained_featureExtractor.trainable = True
            self.initializer = tf.keras.initializers.RandomUniform( minval = 0., maxval = 1. )
            self.fine_Tune_model = self.modelObj.buildTuneModel(
                    pretrained_feature_extractor = self.pretrained_featureExtractor,
                    isTest = False, config = self.config
                    )
            if isMultiDomain:
                self.WidarDataLoaderObjMulti = WidarDataloader(
                        isMultiDomain = isMultiDomain,
                        config = config
                        )
            else:
                self.WidarDataLoaderObj = WidarDataloader(
                        isMultiDomain = isMultiDomain,
                        config = config
                        )
    def _getoutput( self, feature_extractor ):
        return Model(inputs = feature_extractor.input,outputs = feature_extractor.get_layer('lambda_layer').output)
    def _getNShotsEmbedding( self,feature_extractor,Support_set ):
        Support_set_embedding_all = feature_extractor.predict( Support_set )
        Support_set_embedding = []
        if self.isMultiDomain:
            n = len(self.config.domain_selection)
            for i in range( self.nways ):
                Support_set_embedding.append(
                        np.mean(
                                Support_set_embedding_all[ i * n * self.nshots:i * n * self.nshots + n * self.nshots ],
                                axis = 0
                                )
                        )
        else:
            # n=self.nshots
            for i in range( self.nways ):
                Support_set_embedding.append(
                        np.mean(
                                Support_set_embedding_all[ i * self.nshots:i * self.nshots + self.nshots ],
                                axis = 0
                                )
                        )

        return np.asarray(Support_set_embedding)
    def tuning( self,init_weights = True,init_bias = False,isTest:bool = False,num_sample_per_gesture = 20):
        self.data = self.WidarDataLoaderObj.getSQDataForTest(
                nshots = self.nshots, mode = 'fix',
                isTest = isTest,
                Best = None,num_sample_per_gesture = num_sample_per_gesture
                )
        # self.data = self.WidarDataLoaderObjMulti.getMultiDomainSQDataForTest(
        #         nshots_per_domain = self.nshots_per_domain, isTest = False
        #         )
        self.pretrained_featureExtractor.load_weights(self.config.pretrainedfeatureExtractor_path)
        if init_weights:
            if self.nshots == 1:
                weights = np.transpose( self.pretrained_featureExtractor.predict( self.data[ 'Support_data' ] ) )
            elif self.nshots > 1:
                weights = np.transpose(
                        self._getNShotsEmbedding( self.pretrained_featureExtractor, self.data[ 'Support_data' ] )
                        )
            if init_bias:
                p = self.fine_Tune_model.predict( self.data[ 'Query_data' ] )
                bias = np.tile( np.mean( -np.sum( p * np.log( p ), axis = 1 ) ), self.config.N_novel_classes )
            else:
                bias = np.zeros( self.config.N_novel_classes )
            self.fine_Tune_model.get_layer( 'fine_tune_layer' ).set_weights( [ weights, bias ] )
            # self.config.weight = self.fine_Tune_model.get_layer( 'FC_1' ).get_weights( )
        val_data, val_label = self.data[ 'Val_data' ], to_categorical(
                self.data[ 'Val_label' ], num_classes
                = self.config.N_novel_classes
                )
        optimizer = tf.keras.optimizers.Adam(
                learning_rate = self.config.lr,
                # momentum = 0.9,
                epsilon = 1e-06,
                )
        self.fine_Tune_model.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics = 'acc' )
        idx = np.random.permutation( len( self.data[ 'Support_data' ] ) )
        history = self.fine_Tune_model.fit(
                self.data[ 'Support_data' ][ idx ], to_categorical(self.data[ 'Support_label' ][ idx ] , num_classes
                = self.config.N_novel_classes ),
                epochs = 1000,
                # shuffle = True,
                validation_data = (val_data, val_label),
                callbacks = [ self.earlyStop, self.lrScheduler ]
                )
        return [self.fine_Tune_model,self.data['record'],history]
    def tuningMultiRx( self ):
        data = self.WidarDataLoaderObj.getMultiDomainSQDataForTest( nshots_per_domain = self.config.nshots, isTest = False )
        Support_data = [ ]
        keys = list( data[ 'Support_data' ].keys( ) )
        [ Support_data.append( np.concatenate( data[ 'Support_data' ][ keys[ j ] ], axis = 0 ) ) for j in
          range( len( keys ) ) ]
        s_data_array = np.concatenate( Support_data, axis = 0 )

        self.pretrained_featureExtractor.load_weights( self.config.pretrainedfeatureExtractor_path )
        start = time.time( )
        weights = np.transpose(
                self._getNShotsEmbedding( self.pretrained_featureExtractor, s_data_array )
                )
        end = time.time( )
        timecost = (end - start)
        print( f'The inference time 1 is: {timecost:.2f}' )
        bias = np.zeros( self.config.N_novel_classes )
        self.fine_Tune_model.get_layer( 'fine_tune_layer' ).set_weights( [ weights, bias ] )
        optimizer = tf.keras.optimizers.Adam(
                learning_rate = self.config.lr,
                # momentum = 0.9,
                epsilon = 1e-06,
                )
        self.fine_Tune_model.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics = 'acc' )
        val_data, val_label = data[ 'Val_data' ], to_categorical(
                data[ 'Val_label' ], num_classes
                = self.config.N_novel_classes
                )

        idx = np.random.permutation( len( s_data_array ) )
        start = time.time( )
        history = self.fine_Tune_model.fit(
                s_data_array[ idx ],
                to_categorical(data[ 'Support_label' ][ idx ], num_classes = self.config.N_novel_classes),
                epochs = 1000,
                verbose = 0,
                # shuffle = True,
                validation_data = (val_data, val_label),
                callbacks = [ self.earlyStop, self.lrScheduler ]
                )
        end = time.time( )
        timecost = (end - start)
        print( f'The inference time 2 is: {timecost:.2f}' )
        return [self.fine_Tune_model,data['record'],history]
    def test( self, applyFinetunedModel:bool):
        self.feature_extractor, self.classifier = self._loadFineTunedModel(
                applyFinetunedModel = applyFinetunedModel,useWeightMatrix = True
                )
        print(f'check for {self.nshots} shots '
              f'accuracy......................................................................')
        N_test_sample = 600
        correct_count = 0
        test_acc = [ ]
        y_true = []
        y_pred = []
        label_true = []
        n = 6
        feature_extractor = self._getoutput(self.feature_extractor)
        classifier = self.classifier
        # Support_set_embedding = np.transpose(self.feature_extractor.get_layer('fine_tune_layer').get_weights()[0])
        # Support_set_embedding = feature_extractor.predict()
        for i in range( N_test_sample ):
            self.data = self.WidarDataLoaderObj.getSQDataForTest(
                    nshots = self.nshots, mode = 'fix',
                    isTest = True, Best = self.config.record
                    )
            # self.data = self.WidarDataLoaderObjMulti.getMultiDomainSQDataForTest(
            #         nshots_per_domain = self.nshots_per_domain, isTest = True, Best = config.record
            #         )
            # Support_set_embedding = matrix
            Query_set = self.data['Query_data']
            Support_set = self.data['Support_data']
            if applyFinetunedModel:
                Support_set_embedding = np.transpose(
                        self.feature_extractor.get_layer( 'fine_tune_layer' ).get_weights( )[ 0 ]
                        )
            else:
                Support_set_embedding = self._getNShotsEmbedding(feature_extractor,Support_set)
            gesture_type_idx = i%6
            Query_sample = np.repeat( np.expand_dims( Query_set[ gesture_type_idx ], axis = 0 ), n, axis = 0 )
            Query_set_embedding = feature_extractor.predict( Query_sample )
            # model = self._getoutput( feature_extractor )
            prob_classifier = classifier.predict( [ Support_set_embedding, Query_set_embedding ] )
            y_true.append(gesture_type_idx)
            label_true.append(self.data['Query_label'][gesture_type_idx][0])
            y_pred.append( np.argmax(prob_classifier))
            if np.argmax( prob_classifier ) == gesture_type_idx:
                correct_count += 1
            print( f'The number of correct: {correct_count}, The number of test count {i}' )
        acc = (correct_count / N_test_sample) * 100.
        test_acc.append( acc )
        print( "Accuracy %.2f" % acc )
        return test_acc,[y_true,y_pred],label_true
    def testMultiRx( self,applyFinetunedModel:bool,useWeightMatrix:bool = True,n_Rx:int = 6):
        # n_Rx = 5
        self.feature_extractor, _ = self._loadFineTunedModel(
                applyFinetunedModel = applyFinetunedModel,useWeightMatrix = useWeightMatrix)
        print(f'check for {self.nshots} shots {n_Rx} Receivers'
              f'accuracy......................................................................')
        N_test_sample,correct_count,test_acc,y_true,y_pred,label_true, n = 600,0,[],[],[],[],6
        softmax_func = tf.keras.layers.Softmax( )
        if useWeightMatrix:
            weights = np.transpose(
                    self.feature_extractor.get_layer( 'fine_tune_layer' ).get_weights( )[ 0 ]
                    )
            feature_extractor = self._getoutput( self.feature_extractor )
        for i in range( N_test_sample ):
            Query_data_selection = []
            data = self.WidarDataLoaderObj.getMultiDomainSQDataForTest( 1, False )
            Support_data,Query_data,Support_embedding,Query_embedding,sim,sim_mean = [],[],[],[],[],[]
            keys = list( data[ 'Support_data' ].keys( ) )
            [Support_data.append( np.concatenate( data[ 'Support_data' ][ keys[ j ] ], axis = 0 )) for j in range(len(keys))]
            # random selection of specific receiver
            selected_Rx_idx = np.unique(np.random.choice(np.arange(0,6),n_Rx,replace = False))
            # selection of specific receiver
            # selected_Rx_idx = np.arange(0,n_Rx)
            for g in range(len(Support_data)):
                Support_data[g] = Support_data[g][selected_Rx_idx]
            if not useWeightMatrix:
                feature_extractor = self.feature_extractor
                '''gestures, 6 receivers CSI embedding'''
                [Support_embedding.append(feature_extractor.predict(Support_data[j])) for j in range(len(Support_data))]
            g_idx = np.random.choice( np.arange( 0, len( keys ) ), 1, replace = False )[ 0 ]
            sample_idx = np.random.choice( np.arange( 0, len(data['Query_data'][keys[g_idx]][0]) ), 1,replace = False)[ 0 ]
            buffer = data['Query_data'][keys[g_idx]]
            '''select one sample for the selected gesture, corresponding to six receivers'''
            [Query_data.append(buffer[ant][sample_idx]) for ant in range(len(data['Query_data'][keys[g_idx]]))]
            [Query_data_selection.append(Query_data[ant]) for ant in selected_Rx_idx]
            Query_embedding.append( feature_extractor.predict( np.asarray( Query_data_selection ) ) )
            if useWeightMatrix:
                p = []
                buf_sim = [ cosine_similarity( weights, np.expand_dims( Query_embedding[ 0 ][ ant ], axis = 0 ) )
                            for ant in range( len(Query_embedding[0])) ]
                [p.append(np.expand_dims(softmax_func( np.squeeze(buf_sim[tt],axis=-1) ).numpy( ),axis = 0 )) for tt in range(len(buf_sim) )]
                p = np.sum(np.concatenate( p, axis = 0 ),axis=0)
            else:
                for g_sim in range(len(Support_embedding ) ):
                    [sim.append(cosine_similarity( np.expand_dims(Support_embedding[ g_sim ][ ant ],axis=0 ),np.expand_dims(
                            Query_embedding[0][ant],axis=0) )) for ant in range(len(Query_embedding[0]))]
                    sim_mean.append(np.mean(sim))
                p = softmax_func(sim_mean).numpy( )
            if np.argmax( p ) == g_idx:
                correct_count += 1
        acc = (correct_count / N_test_sample) * 100.
        print( "Accuracy %.2f" % acc )
        return acc
    def GetparamForMulRx(self,path_idx:int):
        FE, _ = self._loadFineTunedModel(
                applyFinetunedModel = True, useWeightMatrix = True, isSepRx = True
                )
        FE.load_weights( self.config.tunedModel_path[ path_idx ] )
        feature_extractor = self._getoutput( FE )
        weights = np.transpose(
                FE.get_layer( 'fine_tune_layer' ).get_weights( )[ 0 ]
                )
        return feature_extractor,weights
    def testMultiRxSep( self, N_Rx:int ):
        f1, w1 = self.GetparamForMulRx( path_idx = 0 )
        f2, w2 = self.GetparamForMulRx( path_idx = 1 )
        f3, w3 = self.GetparamForMulRx( path_idx = 2 )
        f4, w4 = self.GetparamForMulRx( path_idx = 3 )
        f5, w5 = self.GetparamForMulRx( path_idx = 4 )
        f6, w6 = self.GetparamForMulRx( path_idx = 5 )
        feature_extractor = [f1,f2,f3,f4,f5,f6]
        weights=[w1,w2,w3,w4,w5,w6]
        N_test_sample, correct_count, test_acc, y_true, y_pred, label_true, n = 600, 0, [ ], [ ], [ ], [ ], 6
        print(self.config.tunedModel_path)
        softmax_func = tf.keras.layers.Softmax( )
        print(f'check for {self.nshots} shots, {N_Rx}_Receivers '
              f'accuracy......................................................................')
        for i in range( N_test_sample ):
            data = self.WidarDataLoaderObj.getMultiDomainSQDataForTest( self.config.nshots, True,
                    Best = self.config.record )
            Support_data, Query_data, Support_embedding, Query_embedding, sim, sim_mean,p = [ ], [ ], [ ], [ ], [ ], \
                                                                                            [ ],[ ]
            keys = list( data[ 'Support_data' ].keys( ) )
            '''gestures, 6 receivers CSI data'''
            # [ Support_data.append( np.concatenate( data[ 'Support_data' ][ keys[ j ] ], axis = 0 ) ) for j in range(len( keys ) ) ]
            # '''gestures, 6 receivers CSI embedding'''
            # for j in range( len( Support_data ) ):
            #     data_buf = Support_data[ j ]
            #     emb_buf = []
            #     for f_idx in range( len( data_buf)):
            #         emb_buf.append(feature_extractor[f_idx].predict( np.expand_dims(data_buf[f_idx],axis=0) ))
            # Support_embedding.append( np.concatenate(emb_buf,axis=0) )
            g_idx = np.random.choice( np.arange( 0, len( keys ) ), 1, replace = False )[ 0 ]
            sample_idx = np.random.choice( np.arange( 0, len(data['Query_data'][keys[g_idx]][0]) ), 1,replace = False)[ 0 ]
            buffer = data['Query_data'][keys[g_idx]]
            ant_selection = np.random.choice( np.arange( 0, 6), N_Rx, replace = False)
            '''select one sample for the selected gesture, corresponding to six receivers'''
            [Query_data.append(buffer[ant][sample_idx]) for ant in range(len(data['Query_data'][keys[g_idx]]))]
            '''select antenna'''
            Query_data_selection = []
            weights_selection = []
            for at in ant_selection:
                Query_data_selection.append(Query_data[at])
                weights_selection.append(weights[at])
            start = time.time( )

            [Query_embedding.append(feature_extractor[ant_selection[ant]].predict(np.expand_dims(
                    Query_data_selection[ant],axis = 0))) for ant in range(len(Query_data_selection))]

            end = time.time( )
            timecost = (end - start) / N_Rx
            print( f'The inference time is: {timecost:.2f}' )
            for w in range(len(Query_embedding)):
                sim.append(cosine_similarity( weights_selection[w], Query_embedding[ w ]) )
            for g in range( len( sim ) ):
                p.append(softmax_func(  np.squeeze(sim[ g ] )).numpy())
            p = np.sum( np.asarray(p), axis = 0 )
            if np.argmax( p ) == g_idx:
                correct_count += 1
        acc = (correct_count / N_test_sample) * 100.
        print( "Accuracy %.2f" % acc )
        return acc
class fineTuningWiAR(fineTuningWidar):
    def __init__( self,config,idx_user):
        super( ).__init__( config = config, isMultiDomain = False,isiheritance = True )
        self.config = config
        self.wiar = WiARdataLoader( config, data_path = f'E:\\Sensing_project\\Cross_dataset\\WiAR\\volunteer_{idx_user}' )
        self.data = self.wiar.data
        self.label = self.wiar.label
        self.config.N_novel_classes = len(self.data)
        self.nways = self.config.N_novel_classes
        self.pretrained_featureExtractor = self._getPreTrainedFeatureExtractor( )
        self.pretrained_featureExtractor.trainable = True
        self.fine_Tune_model = self.modelObj.buildTuneModel(
                pretrained_feature_extractor = self.pretrained_featureExtractor,
                isTest = False, config = self.config
                )
    def tuning( self):
        self.pretrained_featureExtractor.load_weights(self.config.pretrainedfeatureExtractor_path)
        self.pretrained_featureExtractor.trainable = True
        self.data = self.wiar.getSQDataForTest()
        weights = np.transpose(
                self._getNShotsEmbedding( self.pretrained_featureExtractor, self.data[ 'Support_data' ] )
                )
        bias = np.zeros( self.config.N_novel_classes )
        self.fine_Tune_model.get_layer( 'fine_tune_layer' ).set_weights( [ weights, bias ] )
        val_data, val_label = self.data[ 'Val_data' ], to_categorical(
                self.data[ 'Val_label' ], num_classes
                = self.config.N_novel_classes
                )
        optimizer = tf.keras.optimizers.Adam(
                learning_rate = self.config.lr,
                # momentum = 0.9,
                epsilon = 1e-06,
                )

        self.fine_Tune_model.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics = 'acc' )
        idx = np.random.permutation( len( self.data[ 'Support_data' ] ) )
        history = self.fine_Tune_model.fit(
                self.data[ 'Support_data' ][ idx ], to_categorical(self.data[ 'Support_label' ][ idx ] , num_classes
                = self.config.N_novel_classes ),
                epochs = 1000,
                verbose = 0,
                validation_data = (val_data, val_label),
                callbacks = [ self.earlyStop, self.lrScheduler ]
                )
        return [self.fine_Tune_model,self.data['record'],history]
def tuningSignFi(preTrain_model_path,nshots,source = 'home',n_ft_cls=None):
    config = getConfig( )
    config.source = source
    config.nshots = nshots
    # config.n_ft_cls = n_ft_cls
    # config.N_novel_classes = 25
    # config.N_base_classes = 276 - config.N_novel_classes

    config.N_novel_classes = n_ft_cls
    config.N_base_classes = 276 - config.N_novel_classes
    # config.lr = 4e-4
    # config.lr = 1e-3
    # user 1 - 0.7e-3, 2,3 - 0.65e-3, 4 -
    # config.lr = 1e-3
    config.lr = 6.5e-4
    # config.lr = 3e-4
    config.train_dir = 'D:\Matlab\SignFi\Dataset'
    # config.tunedModel_path = f'./models/Publication_related/signFi_finetuned_model_{config.nshots}_shots_' \
    #                          f'{config.N_novel_classes}_ways_256_1280.h5'
    # config.pretrainedfeatureExtractor_path = 'a.h5'
    config.pretrainedfeatureExtractor_path = preTrain_model_path
    tuningSignFiObj = fineTuningSignFi(config,isZscore = False)
    fine_Tune_model = tuningSignFiObj.tuning(init_weights = True,init_bias = False)
    return fine_Tune_model,config
def testingSignFi(path,mode,N_train_classes,environment:str):
    config = getConfig( )
    config.nshots = 1
    config.train_dir = 'D:\Matlab\SignFi\Dataset'
    config.source = 'home'
    config.N_base_classes = 200
    config.N_novel_classes = 76
    all_path = os.listdir( f'./models/pretrained_feature_extractors/' )
    # for i, path in enumerate( all_path ):
    #     n = re.findall( r'\d+', all_path[ i ] )[ 2 ]
    #     if int( n ) == 200:
    #         config.N_base_classes = int( n )
    #         config.N_novel_classes = 276 - config.N_base_classes
    #         print( f'{n} in environment {config.source}' )
    # config.pretrainedfeatureExtractor_path = './models/pretrained_feature_extractors/' + path
    config.tunedModel_path = path
    tuningSignFiObj = fineTuningSignFi( config, isZscore = False )
    acc_all = tuningSignFiObj.test( nway = None, applyFinetunedModel = True )

    # config = getConfig( )
    # modelObj = models( )
    #
    # config.source = environment
    # config.train_dir = 'D:\Matlab\SignFi\Dataset'
    # config.N_base_classes = N_train_classes
    # # config.lr = 3e-4
    # config.pretrainedfeatureExtractor_path = path
    # # Declare objects
    # config.tunedModel_path = 'fe.h5'
    # dataLoadObj = signDataLoader( config = config )
    # # preTrain_modelObj = PreTrainModel( config = config )
    # train_data, train_labels, test_data, test_labels = dataLoadObj.getFormatedData(
    #         source = config.source,
    #         isZscore = False
    #         )
    # feature_extractor = modelObj.buildFeatureExtractor( mode = mode )
    # feature_extractor.load_weights(config.pretrainedfeatureExtractor_path )
    # fineTuningSignFiObj = fineTuningSignFi( config )
    # test_acc = fineTuningSignFiObj.test(test_data, test_labels, 1000, feature_extractor)
    return acc_all
def tuningWiar(nshots,idx_user):
    config = getConfig()
    config.nshots = nshots
    config.lr = 1e-4
    config.pretrainedfeatureExtractor_path = './models/Publication_related/FE/a.h5'
    wiarFTObj = fineTuningWiAR(config = config,idx_user=idx_user)
    val_acc = 0.0
    for i in range(50):
        fine_Tune_model,record,history = wiarFTObj.tuning()
        print("=========================================================iteration: "
              "%d=========================================================" % i)
        print('Tuning %d shots model' % nshots)
        currentAcc = np.max( history.history[ 'val_acc' ])
        print(f'Accuracy is {currentAcc}')
        if val_acc < np.max(history.history['val_acc']):
            val_acc = np.max(history.history['val_acc'])
            config.tunedModel_path = f'.\\models\\Publication_related\\wiar_FT_{config.nshots}shots_' \
                                     f'{val_acc:0.2f}_User_{idx_user}.h5'
            fine_Tune_model.save_weights(config.tunedModel_path)
            print(f'Updated record is: {record}')
            mdic = {'record':record,
                    'val_acc':val_acc}
            config.setMatSavePath(
                    f".\\Sample_index\\Publication_related\\sample_index_record_for_{config.nshots}_shots_User_{idx_user}.mat"
                    )
            savemat( config.matPath, mdic )
    return val_acc