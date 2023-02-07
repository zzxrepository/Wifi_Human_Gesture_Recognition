import tensorflow as tf
class getConfig:
    def __init__(self,):
        self.epoch =  150
        self.train_dir = None
        self.eval_dir = None
        self.lr = 1e-3
        self.batch_size = 32
        # To split the dataset into training and testing
        self.N_base_classes = None
        # To define the fine tuning model
        self.N_novel_classes = None
        self.source = None
        self.input_shape = [200,60,3]
        self.pretrainedfeatureExtractor_path = None
        self.tunedModel_path = None
        self.record = None
        self.nshots = None
        self.nshots_per_domain = None
        self.domain_selection = None
        self.nways = None
        self.matPath = None
        self.test_Domain = (2,3,3)
        self.learning_Domain = [(2,1,3), (2,2,3), (2,3,3), (2,4,3), (2,5,3)]
        self.history = None
        self.n_ft_cls = None
