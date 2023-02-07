#%%
import os
import random
import re
import copy
import scipy.io as sio
import numpy as np
# from Preprocess.SignalPreprocess import *
from scipy import stats
from Config import getConfig
from scipy.io import savemat,loadmat
class WiARdataLoader:
	def __init__(self,config,data_path):
		self.config = config
		self.data_path = data_path
		self.filename = os.listdir( data_path )
		self.data, self.label = self._loaddataNlabels( )
	def _loaddataNlabels( self ):
		data = []
		label = []
		for count, currentPath in enumerate(self.filename):
			currentPath = os.path.join(self.data_path,currentPath)
			data_amp = sio.loadmat( currentPath )[ 'csiAmplitude' ]
			data_phase = sio.loadmat( currentPath )[ 'csiPhase' ]
			data.append( np.concatenate( (data_amp, data_phase), axis = 1 ) )
			label.append(int( re.findall( r'\d+', self.filename[count] )[0]))
		data, label = np.asarray( data ), np.asarray( label )
		classes = np.unique( label )
		cls = { }
		out_label = []
		for i in classes:
			idx = np.where( label == i )[ 0 ]
			cls[ f'act_{i-1}'] = data[idx]
			out_label.append(label[idx] - 1)

		return cls, np.concatenate( out_label )
	def getSQDataForTest( self ):

		gesture_type = list( self.data.keys( ) )
		num_sample_per_gesture = len(self.data[gesture_type[0]])
		num_val = num_sample_per_gesture - self.config.nshots
		support_set = [ ]
		query_set = [ ]
		support_label = [ ]
		query_label = [ ]
		record = [ ]
		Val_set = np.zeros( (len(gesture_type) * num_val, 200, 60, 3))
		Val_set_label = [ ]
		for count, gesture in enumerate( gesture_type ):
			idx_list = np.arange( 0, num_sample_per_gesture )
			shots_idx = np.random.choice( idx_list, self.config.nshots, replace = False )
			for i in shots_idx:
				idx_list = np.delete( idx_list, np.where( idx_list == i ) )
				support_set.append( self.data[ gesture ][ i ] )
				support_label.append( count )
			sample_idx = np.random.choice( idx_list, 1, replace = False )[ 0 ]
			query_set.append( self.data[ gesture ][ sample_idx ] )
			query_label.append( count )
			Val_set[ count * num_val:count * num_val + num_val, :, :, : ] = self.data[ gesture ][ idx_list ]
			[ Val_set_label.append( count ) for i in range( num_val ) ]
			record.append( shots_idx )
		Support_data = np.asarray( support_set )
		Support_label = np.expand_dims( np.asarray( support_label ), axis = 1 )
		Query_data = np.asarray( query_set )
		Query_label = np.expand_dims( np.asarray( query_label ), axis = 1 )
		Val_data = Val_set
		Val_label = np.expand_dims( (Val_set_label), axis = 1 )

		output = {
				'Support_data' : Support_data,
				'Support_label': Support_label,
				'Query_data'   : Query_data,
				'Query_label'  : Query_label,
				'Val_data'     : Val_data,
				'Val_label'    : Val_label,
				'record'       :record
				}
		return output
class WidarDataloader():
	def __init__(self,isMultiDomain:bool = False,config=None):
		super().__init__()
		self.config = config
		self.filename = os.listdir( config.train_dir )
		self.csiAmplitude = np.array( [ ] )
		self.labels = np.asarray( [ ] )
		self.gesture_class = {}
		
		x = []
		for name in self.filename:
			x.append(int(re.findall( r'\d+\b', name )[1]))
		self.num_gesture_types = np.max(x)
		self._mapFilenameToClass( )
		self._getInputShape( )

		
		self.selection = config.domain_selection

		if isMultiDomain:
			print(f'Using the data from domain: {self.selection}')
			self.multiRx_data = self._getMultiOrientationData( self.selection )
		else:
			self.selected_gesture_samples_data = self._mapClassToDataNLabels(
					selected_gesture_samples_path = self._selectPositions( selection = self.selection )
					)
	def _mapFilenameToClass(self):
		date_filename = re.findall( r'\b\d+\b', self.config.train_dir )[0]
		sim_gesture_6ges = ['20181115','20181109','Pre_16','20181121','20181211','20181127']
		draw_gesture_10ges = [ '20181112', '20181116','Pre_16' ]
		if date_filename in sim_gesture_6ges:
			keys = [ 'Push&Pull',
					 'Sweep',
					 'Clap',
					 'Draw-O(Vertical)',
					 'Draw-Zigzag(Vertical)',
					 'Draw-N(Vertical)']
			for g_type in keys:
				recordGesture = [ ]
				for currentFileName in self.filename:
					if int( re.findall(r'\d+\b',currentFileName)[ 1 ] ) == int( keys.index( g_type ) ) + 1:
						filePath = os.path.join( self.config.train_dir, currentFileName )
						recordGesture.append( filePath )
				self.gesture_class[ g_type ] = recordGesture
		if date_filename in draw_gesture_10ges:
			keys = [("Draw-"+ str(i)) for i in range(1,self.num_gesture_types+1)]
			for g_type in keys:
				recordGesture = []
				for currentFileName in self.filename:
					if int(re.findall(r'\d+\b',currentFileName)[ 1 ]) == int(keys.index(g_type)) + 1:
						filePath = os.path.join(self.config.train_dir,currentFileName)
						recordGesture.append(filePath)
				self.gesture_class[g_type] = recordGesture
	def _getInputShape(self):
		data = sio.loadmat( os.path.join( self.config.train_dir, self.filename[ 0 ] ) )[ 'csiAmplitude' ]
		self.InputShape = list(data.shape)
		self.num_subcarriers = self.InputShape[0]
		self.len_signals = self.InputShape[1]
	def _getMultiOrientationData(self,selection):
		# _, _, Rx = selection
		selected_multiorientation_gesture_samples_data = {}

		for location in [2]:
			for orientation in [2]:
				for receiver in selection:
					domain = (location,orientation,receiver)
					path = self._selectPositions( domain )
					data = self._mapClassToDataNLabels(path)
					selected_multiorientation_gesture_samples_data[ f'{domain}' ] = data
		return selected_multiorientation_gesture_samples_data
	def _selectPositions(self,selection : tuple):
		location, orientation, Rx = selection
		selected_gesture_samples_path = {}
		for currentGesture in self.gesture_class:
			all_path = self.gesture_class[currentGesture]
			selected_path = []
			for currentFileName in all_path:
				'''
				0: date
				1: user ID
				2: gesture type
				3: location
				4: orientation
				5: repetition
				6: Rx ID
				'''
				# location
				if int( re.findall(r'\d+\b',currentFileName)[ -4 ] ) == location:
					if int( re.findall( r'\d+\b', currentFileName )[-3 ] ) == orientation:
						if int( re.findall( r'\d+\b', currentFileName )[ -1 ] ) == Rx:
							selected_path.append(currentFileName)
			selected_gesture_samples_path[ currentGesture ] = selected_path
		return selected_gesture_samples_path
	# def _getZscoreData( self, x):
	# 	x = stats.zscore( x, axis = 0, ddof = 0 )
	# 	return x
	def _mapClassToDataNLabels( self,selected_gesture_samples_path):
		gesture = {}
		x_all = []
		y_all = []
		for currentGesture in selected_gesture_samples_path:
			all_path = selected_gesture_samples_path[ currentGesture ]
			data = []
			# labels = []
			for currentPath in all_path:
				data_amp = sio.loadmat( currentPath )[ 'csiAmplitude' ]
				data_phase = sio.loadmat( currentPath )[ 'csiPhase' ]
				# data_phase = sanitisePhases(sio.loadmat(currentPath)['csiPhase'])
				# data_amp, data_phase = self.preprocessers.csiRatio( isWidar = True, data_amp = data_amp, data_phase = \
				#     data_phase)
				data.append(np.concatenate( (data_amp, data_phase), axis = 1 ))
				x_all.append(np.concatenate( (data_amp, data_phase), axis = 1 ))
				# labels.append(int( re.findall(r'\d+\b',currentPath)[ 2 ] ) - 1)
				y_all.append(int( re.findall(r'\d+\b',currentPath)[ 2 ] ) - 1)
			gesture[currentGesture] = np.asarray(data)
			self.gesture_type = list( gesture.keys( ) )
		return gesture
	def getSQDataForTest( self,nshots: int,mode:str, isTest:bool=False,Best = None,num_sample_per_gesture = 20):
		gesture_type = list( self.selected_gesture_samples_data.keys( ) )
		support_set = [ ]
		query_set = [ ]
		support_label = [ ]
		query_label = [ ]
		num_sample_per_gesture = num_sample_per_gesture
		num_val = num_sample_per_gesture-nshots
		Val_set = np.zeros( (6 * num_val, 200, 60, 3) )
		Val_set_label = [ ]
		record = []

		if mode == 'fix':
			for count, gesture in enumerate(gesture_type):
				if not isTest:
					idx_list = np.arange( 0, num_sample_per_gesture )
					shots_idx = np.random.choice( idx_list, nshots, replace = False )
					for i in shots_idx:
						idx_list = np.delete( idx_list, np.where( idx_list == i ) )
						support_set.append( self.selected_gesture_samples_data[ gesture ][ i ] )
						support_label.append(count)
					sample_idx = np.random.choice( idx_list, 1, replace = False )[0]
					query_set.append( self.selected_gesture_samples_data[ gesture ][ sample_idx ] )
					query_label.append(count)
					Val_set[count*num_val:count*num_val+num_val,:,:,:] = self.selected_gesture_samples_data[ gesture ][idx_list]
					[Val_set_label.append(count) for i in range(num_val)]
					record.append(shots_idx )
				else:
					idx_list = np.arange( 0, num_sample_per_gesture )
					shots_idx =  Best[ count ]
					for i in shots_idx:
						idx_list = np.delete( idx_list, np.where( idx_list == i ) )
						support_set.append( self.selected_gesture_samples_data[ gesture ][ i ] )
						support_label.append( count )
					sample_idx = np.random.choice( idx_list, 1, replace = False )[ 0 ]
					query_set.append( self.selected_gesture_samples_data[ gesture ][ sample_idx ] )
					query_label.append( count )
					Val_set[ count * num_val:count * num_val + num_val, :, :, : ] = self.selected_gesture_samples_data[ gesture ][ idx_list ]
					[ Val_set_label.append( count ) for i in range( num_val ) ]
					record.append( shots_idx )

			Support_data = np.asarray(support_set)
			Support_label = np.expand_dims(np.asarray(support_label),axis=1)
			Query_data = np.asarray(query_set)
			Query_label = np.expand_dims(np.asarray(query_label),axis=1)
			Val_data = Val_set
			Val_label = np.expand_dims((Val_set_label),axis=1)

			output = {  'Support_data':Support_data,
						'Support_label':Support_label,
						'Query_data':Query_data,
						'Query_label':Query_label,
						'Val_data':Val_data,
						'Val_label':Val_label,
						'record':record
					}
			return output
		if mode == 'multiRx':
			for count, gesture in enumerate( gesture_type ):
				if not isTest:
					idx_list = np.arange( 0, num_sample_per_gesture )
					shots_idx = np.random.choice( idx_list, nshots, replace = False )
					for i in shots_idx:
						idx_list = np.delete( idx_list, np.where( idx_list == i ) )
						support_set.append( self.selected_gesture_samples_data[ gesture ][ i ] )
						support_label.append(count)
					sample_idx = np.random.choice( idx_list, 1, replace = False )[0]
					query_set.append( self.selected_gesture_samples_data[ gesture ][ sample_idx ] )
					query_label.append(count)
					Val_set[count*num_val:count*num_val+num_val,:,:,:] = self.selected_gesture_samples_data[ gesture ][idx_list]
					[Val_set_label.append(count) for i in range(num_val)]
					record.append(shots_idx )
	def getMultiDomainSQDataForTest( self,nshots_per_domain,isTest:bool,Best = None ):
		def _delete_idx( idx_list,shots_idx, nshots_per_domain):
			for n in range( nshots_per_domain ):
				idx_list = np.delete( idx_list, list( idx_list ).index( list( shots_idx )[ n ] ) )
			return idx_list
		gesture_type = self.gesture_type
		Support_set = {  'Push&Pull': [],
						 'Sweep': [],
						 'Clap': [],
						 'Draw-O(Vertical)': [],
						 'Draw-Zigzag(Vertical)': [],
						 'Draw-N(Vertical)': []}
		query_set = {    'Push&Pull': [],
						 'Sweep': [],
						 'Clap': [],
						 'Draw-O(Vertical)': [],
						 'Draw-Zigzag(Vertical)': [],
						 'Draw-N(Vertical)': []}
		support_label = [ ]
		query_label = [ ]
		n_samples_perCls = 10
		num_val = n_samples_perCls - nshots_per_domain
		Val_set_multi_domain = []
		Val_set_label_multi_domain = [ ]
		record = []
		multiRx_data = self.multiRx_data

		for count, gesture in enumerate( gesture_type ):
			all_domain = list( multiRx_data )
			if not isTest:
				Current_record = [ ]
				for i in range( len( multiRx_data ) ):
					idx_list = np.arange( 0, n_samples_perCls )
					########
					shots_idx = np.random.choice( idx_list, nshots_per_domain, replace = False )
					#########
					# randIdx = np.random.choice( np.arange( 0, len(all_domain) ), 1, replace = False )[0]
					randIdx = 0
					current_domain = all_domain[ randIdx ]
					all_domain.pop(randIdx)
					idx_list = _delete_idx(idx_list,shots_idx,nshots_per_domain)

					Support_set[ gesture ].append(multiRx_data[ current_domain ][gesture ][shots_idx ])
					# sample_idx = np.random.choice( idx_list, num_val, replace = False )[ 0 ]
					query_set[ gesture ].append(multiRx_data[ current_domain ][ gesture ][idx_list ])
					[support_label.append( count ) for n in range( nshots_per_domain )]
					[ query_label.append( count ) for n in range( len(idx_list) ) ]
					Val_set_multi_domain.append(multiRx_data[ current_domain ][ gesture ][idx_list ])
					[ Val_set_label_multi_domain.append( count ) for m in range( len(idx_list) ) ]
					Current_record.append( shots_idx )
				record.append(np.asarray(Current_record).reshape(len( multiRx_data ), nshots_per_domain ) )
			else:
				for i in range( len( multiRx_data ) ):
					idx_list = np.arange( 0, n_samples_perCls )
					################
					record = Best
					shots_idx = record[count][i][0]
					#############
					randIdx = 0
					current_domain = all_domain[ randIdx ]
					all_domain.pop( randIdx )
					idx_list = _delete_idx( idx_list, shots_idx, nshots_per_domain )
					Support_set[ gesture ].append( multiRx_data[ current_domain ][ gesture ][ shots_idx ] )
					# sample_idx = np.random.choice( idx_list, num_val, replace = False )[ 0 ]
					query_set[ gesture ].append( multiRx_data[ current_domain ][ gesture ][ idx_list ] )
					[ support_label.append( count ) for n in range( nshots_per_domain ) ]
					[ query_label.append( count ) for n in range( len( idx_list ) ) ]
					Val_set_multi_domain.append( multiRx_data[ current_domain ][ gesture ][ idx_list ] )
					[ Val_set_label_multi_domain.append( count ) for m in range( len( idx_list ) ) ]
		else:
			# Support_set = np.concatenate( Support_set, axis = 0 )
			# Support_data = np.asarray( Support_set )
			Support_label = np.expand_dims( np.asarray( support_label ), axis = 1 )
			# Query_data = np.asarray( query_set )
			Query_label = np.expand_dims( np.asarray( query_label ), axis = 1 )
			Val_data = np.concatenate(Val_set_multi_domain,axis = 0 )
			Val_label = np.expand_dims( (Val_set_label_multi_domain), axis = 1 )
			record = record
		output = {
				'Support_data' : Support_set,
				'Support_label': Support_label,
				'Query_data'   : query_set,
				'Query_label'  : Query_label,
				'Val_data'     : Val_data,
				'Val_label'    : Val_label,
				'record'       : record
				}
		return output
class signDataLoader:


	''':returns
		filename: [0] home-276 -> user 5, 2760 samples,csid_home and csiu_home
		filename: [1] lab-150 -> user 1 to 5, 1500 samples/user
		filename: [2] lab-276 -> user 5, 5520 samples,downlink*
		filename: [3] lab-276 -> user 5, 5520 samples,uplink*
	'''
	
	def __init__( self,config = None ):

		self.config = config
		self.dataDir = config.train_dir
		self.data = []
		self.data, self.filename = self.loadData()
	def loadData( self,  ):

		def reformat( ori_data ):
			reformatData = np.zeros((ori_data.shape[3],ori_data.shape[0],ori_data.shape[1],ori_data.shape[2]),dtype='complex_')
			for i in range(ori_data.shape[-1]):
				reformatData[i,:,:,:] = ori_data[:,:,:,i]
			return reformatData
		print("Loading data................")
		fileName = os.listdir( self.dataDir )
		for name in fileName:
			path = os.path.join(self.dataDir,name)
			buf = sio.loadmat(path)
			buf.pop( '__header__', None )
			buf.pop('__version__',None)
			buf.pop( '__globals__', None )
			for i in range(len(buf)):
				if 'label' in list( buf.keys( ) )[ i ]:
					continue
				buf[list( buf.keys( ) )[ i ]] = reformat(buf[list( buf.keys( ) )[ i ]])
			self.data.append( buf )
		return [self.data,fileName]
	def getFormatedData(self,source:str='lab',):
		def getSplitData(x_all,y_all,n_samples_per_user:int,shuffle=True):
			n_base_classes = self.config.N_base_classes
			n_test_classes = 276 - n_base_classes
			n_train_samples = n_base_classes * n_samples_per_user
			n_test_samples = (276 - n_base_classes) * n_samples_per_user
			train_data = np.zeros( (n_train_samples, 200, 60, 3) )
			train_labels = np.zeros( (n_train_samples, 1) ,dtype = int)
			test_data = np.zeros( (n_test_samples, 200, 60, 3) )
			test_labels = np.zeros( (n_test_samples, 1) ,dtype = int)
			idx = np.where( y_all == 1 )[0]
			tra_count = 0
			tes_count = 0
			for i in idx:
				train_data[tra_count:tra_count + n_base_classes, :, :, : ] = x_all[ i:i + n_base_classes, :, :, : ]
				train_labels[tra_count:tra_count + n_base_classes, : ] = y_all[ i:i + n_base_classes, : ]
				test_data[tes_count:tes_count+n_test_classes,:,:,:] = x_all[ i + n_base_classes:i + 276, :, :, : ]
				test_labels[tes_count:tes_count+n_test_classes,:] = y_all[ i + n_base_classes:i + 276, : ]
				tra_count += n_base_classes
				tes_count += n_test_classes
			if shuffle:
				idx = np.random.permutation( len( train_labels ) )
				train_data = train_data[idx]
				train_labels = train_labels[idx]
			return [train_data, train_labels, test_data, test_labels]
		if source == 'lab':
			print( 'lab environment user 5, 276 classes,5520 samples,downlink*' )
			for idx, dic in enumerate(self.data):
				if 'csid_lab' in dic.keys():
					break
			x = self.data[ idx ][ 'csid_lab' ]
			x_amp = np.abs( x )
			x_phase = np.angle( x )
			x_all = np.concatenate( (x_amp, x_phase), axis=2 )
			y_all = self.data[ idx ][ 'label_lab' ]
			train_data, train_labels, test_data, test_labels = getSplitData(x_all=x_all,y_all=y_all,
					n_samples_per_user=20,shuffle=True)
			return [ train_data, train_labels, test_data, test_labels ]
		elif source == 'home':
			for idx, dic in enumerate(self.data):
				if 'csid_home' in dic.keys():
					break
			print('home environment user 5, 276 classes, 2760 samples')
			x = self.data[ idx ][ 'csid_home' ]
			x_amp = np.abs( x )
			x_phase = np.angle( x )

			x_all = np.concatenate( (x_amp, x_phase), axis=2 )
			y_all = self.data[ idx ][ 'label_home' ]
			train_data, train_labels, test_data, test_labels = getSplitData(
					x_all = x_all, y_all = y_all,
					n_samples_per_user = 10,shuffle=False
					)
			return [ train_data, train_labels, test_data, test_labels ]
		elif type(source) == list:
			def _getConcatenated(  x ):
				x_amp = np.abs( x )
				x_phase = np.angle( x )
				x_all = np.concatenate( (x_amp, x_phase), axis=2 )
				return x_all
			for idx, dic in enumerate(self.data):
				if 'csi1' in dic.keys():
					break
			print(f'Training Set: User{source[0]}-{source[1]}-{source[2]}-{source[3]}, Testing Set: User {source[4]}')
			source = [source[0]-1, source[1]-1, source[2]-1, source[3]-1, source[4]-1]
			x_1 = _getConcatenated( self.data[ idx ][ 'csi1' ],  )
			x_2 = _getConcatenated( self.data[ idx ][ 'csi2' ],  )
			x_3 = _getConcatenated( self.data[ idx ][ 'csi3' ],  )
			x_4 = _getConcatenated( self.data[ idx ][ 'csi4' ],  )
			x_5 = _getConcatenated( self.data[ idx ][ 'csi5' ],  )
			y_1 = self.data[ idx ][ 'label' ][ 0:1500 ]
			y_2 = self.data[ idx ][ 'label' ][ 1500:3000 ]
			y_3 = self.data[ idx ][ 'label' ][ 3000:4500 ]
			y_4 = self.data[ idx ][ 'label' ][ 4500:6000 ]
			y_5 = self.data[ idx ][ 'label' ][ 6000:7500 ]
			x = [x_1,x_2,x_3,x_4,x_5]
			y = [y_1,y_2,y_3,y_4,y_5]
			x_train = np.concatenate( (x[source[0]],x[source[1]],x[source[2]],x[source[3]]),axis = 0)
			y_train = np.concatenate(
					(y[ source[ 0 ] ], y[ source[ 1 ] ], y[ source[ 2 ] ], y[ source[ 3 ] ]), axis = 0
					)
			x_test = x[source[4]]
			y_test = y[source[4]]
			return [x_train,y_train,x_test,y_test]
	def getTrainTestSplit(self, data, labels, N_train_classes: int , N_samples_per_class: int  ):
		if N_train_classes == 276:
			train_data = data
			train_labels = labels
			test_data = None
			test_labels = None
			return [ train_data, train_labels, test_data, test_labels ]
		N_samples = len( labels )
		N_classes = int( N_samples / N_samples_per_class )
		N_train_samples = N_train_classes * N_samples_per_class
		N_test_samples = N_samples - N_train_samples
		N_test_classes = int( N_test_samples / N_samples_per_class )

		train_data = np.zeros( (N_train_samples, 200, 60, 3) )
		train_labels = np.zeros( (N_train_samples, 1) )
		test_data = np.zeros( (N_test_samples, 200, 60, 3) )
		test_labels = np.zeros( (N_test_samples, 1) )
		count_tra = 0
		count_tes = 0
		for i in list( np.arange( 0, N_samples, N_classes ) ):
			train_data[ count_tra:count_tra + N_train_classes, :, :, : ] = data[ i:i + N_train_classes, :, :, : ]
			train_labels[ count_tra:count_tra + N_train_classes, : ] = labels[ i:i + N_train_classes, : ]
			test_data[ count_tes:count_tes + N_test_classes, :, :, : ] = data[ i + N_train_classes:i + N_classes, :, :,
																		 : ]
			test_labels[ count_tes:count_tes + N_test_classes, : ] = labels[ i + N_train_classes:i + N_classes, : ]
			count_tra += N_train_classes
			count_tes += N_test_classes
		idx = np.random.permutation( len( train_labels ) )
		train_data = train_data[ idx, :, :, : ]
		train_labels = train_labels[ idx, : ]
		return [ train_data, train_labels, test_data, test_labels ]
if __name__ == '__main__':
	dataPth = "/media/b218/HOME/Code_ds/SensingDataset/"
	#%% SignFi
	config = getConfig( )
	config.train_dir =  dataPth + "SignFi/Dataset"
	config.N_base_classes = 250
	SDL = signDataLoader(config = config)
	X_train, Y_train, X_test, Y_test = SDL.getFormatedData( source = 'home' )
	#%% Widar
	config.domain_selection = [1,2,3,4,5]
	config.train_dir = dataPth + "Widar/20181109/User1"
	fineTuneModelEvalObj = WidarDataloader( config = config, isMultiDomain = True,)
	data = fineTuneModelEvalObj.getMultiDomainSQDataForTest(1,False,)
	#%% Wiar
	config = getConfig( )
	config.nshots = 1
	wiar = WiARdataLoader(config,data_path =  dataPth + 'WiAR/volunteer_2')
	data = wiar.getSQDataForTest()
	# %%
