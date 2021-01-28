class NeuralnetModel:
    '''
    Neural network model in Keras
    '''
    def __init__(self, args, method, par_learning):
        self.model_type = args.model_type
        self.n_side_pixl = args.n_side_pixl
        self.method = method
        self.par_learning = par_learning
        self.keras_model = None


    def define_model(self, vocab_length, X_extra_train, y_train, emb_matrix, paths):
        from keras.layers import Input, Dense, Reshape, Embedding, concatenate, Lambda, LSTM, Bidirectional, Dropout
        from keras.models import Model
        from keras.optimizers import RMSprop
        from keras.backend import mean
        from keras.layers.merge import add
        from keras_bert import load_trained_model_from_checkpoint
        from keras_bert import AdamWarmup, calc_train_steps
        import keras
        
        '''
        # branch 1 (subjects)
        left_seq_input = Input(shape=(1,), dtype='int32')
        if self.par_learning['emb_initialize'] == 'emb':
            left_embedding = Embedding(emb_matrix['subj'].shape[0], emb_matrix['subj'].shape[1], weights=[emb_matrix['subj']],
                                       input_length=1, trainable=False)
        elif self.par_learning['emb_initialize'] == 'uniform':
            left_embedding = Embedding(vocab_length['subj'], emb_matrix['subj'].shape[1], input_length=1,
                                       trainable=False)
        left_branch = left_embedding(left_seq_input)
        left_branch = Reshape((-1,))(left_branch)
        
        # branch 2 (relations)
        middle_seq_input = Input(shape=(1,), dtype='int32')
        if self.par_learning['emb_initialize'] == 'emb':
            mid_embedding = Embedding(emb_matrix['pred'].shape[0], emb_matrix['pred'].shape[1], weights=[emb_matrix['pred']],
                                      input_length=1, trainable=False)
        elif self.par_learning['emb_initialize'] == 'uniform':
            mid_embedding = Embedding(vocab_length['pred'], emb_matrix['pred'].shape[1], input_length=1,
                                      trainable=False)
        middle_branch = mid_embedding(middle_seq_input)
        middle_branch = Reshape((-1,))(middle_branch)
        
        # branch 3 (objects)
        right_seq_input = Input(shape=(1,), dtype='int32')
        if self.par_learning['emb_initialize'] == 'emb':
            right_embedding = Embedding(emb_matrix['obj'].shape[0], emb_matrix['obj'].shape[1], weights=[emb_matrix['obj']],
                                        input_length=1, trainable=False)
        elif self.par_learning['emb_initialize'] == 'uniform':
            right_embedding = Embedding(vocab_length['obj'], emb_matrix['obj'].shape[1], input_length=1,
                                        trainable=False)
        right_branch = right_embedding(right_seq_input)
        right_branch = Reshape((-1,))(right_branch)
        '''
        
        # branch 4 (Caption)
        #all_seq_input = Input(shape=(100,), dtype='int32')

        #print('Length Vocabulary Caption: ' + str(vocab_length['cap']))
        #print('Embedding Matrix Caption: ' + str(emb_matrix['cap'].shape))
        '''
        if self.par_learning['emb_initialize'] == 'emb':
            cap_embedding = Embedding(emb_matrix['cap'].shape[0], emb_matrix['cap'].shape[1], weights=[emb_matrix['cap']],
                                        input_length=100, trainable=False)
        elif self.par_learning['emb_initialize'] == 'uniform':
            right_embedding = Embedding(vocab_length['obj'], emb_matrix['obj'].shape[1], input_length=1,
                                        trainable=False)
        '''
        #cap_branch = cap_embedding(all_seq_input)
        #cap_branch = Bidirectional(LSTM(units=512, return_sequences=True,
        #                       recurrent_dropout=0.2, dropout=0.2))(cap_branch)
        #x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
        #                           recurrent_dropout=0.2, dropout=0.2))(x)
        #cap_representation = add([x, x_rnn])  # residual connection to the first biLSTM 
        
        #BiLSTM bidez lortutako errepresentazioaren batazbestekoa shape (None, 100, 1024) ==> (None, 1024)
        #cap_lambda = Lambda(lambda x: mean(x, axis=1))
        #cap_representation = cap_lambda(cap_branch)

        #BERT
        model_bert = load_trained_model_from_checkpoint(
                paths.config,
                paths.checkpoint,
                training=True,
                trainable=True,
                seq_len=100,
            )

        sequence_output  = model_bert.layers[-3].output
        #Atentzio mekanismoa jarri??
        pool_output = Dense(300, activation='sigmoid',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),name = 'real_output')(sequence_output)
        
        cap_representation = Reshape((-1,))(pool_output)

        # branch of X_extra
        extra_input = Input(shape=(X_extra_train.shape[1],))
        extra_branch = Reshape((X_extra_train.shape[1],), input_shape=(X_extra_train.shape[1],))(extra_input)

        # Concatenate
        #conc = concatenate([left_branch, right_branch, cap_representation], axis=1) #, middle_branch
        #conc_output = Dense(1024, activation=self.par_learning['activation'])(conc)
        conc = concatenate([cap_representation, extra_branch])

        #input_model = [left_seq_input, right_seq_input, model_bert.inputs[0], model_bert.inputs[1], extra_input] #, middle_seq_input
        input_model = [model_bert.inputs[0], model_bert.inputs[1], extra_input]
        output = Reshape((-1,))(conc)

        # Add hidden layers
        for jj in range(self.par_learning['n_layers']):
            output = Dense(self.par_learning['n_hidden'], activation=self.par_learning['activation'])(output)
        output = Dense(y_train.shape[1], activation=self.par_learning['output_acti'])(output)

        decay_steps, warmup_steps = calc_train_steps(
            y_train.shape[0],
            epochs=self.par_learning['n_epochs'],
            batch_size=self.par_learning['batch_size']
        )
        #print(input_model)
        # Compile
        self.keras_model = Model(input_model, output)
        #opti = RMSprop(lr=self.par_learning['LR'], rho=0.9, epsilon=1e-08, decay=0)
        self.keras_model.compile(AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=self.par_learning['LR']), loss=self.par_learning['loss'], metrics=self.par_learning['metrics'])
        #print(self.keras_model.summary())

    def method_learn(self, X_train, X_extra_train, y_train, y_pixl_train, EMBEDDINGS, paths):
        '''
        :param model_type: PIX, REG
        :param EMBEDDINGS: a dictionary with embedding matrix (for initialization), word dictionary, etc.
        '''

        if self.method == 'ctrl': # this method is the same for every model_type (only depends on y_train)
            self.keras_model = None

        elif self.model_type == 'REG':
            self.learn_model(X_train, X_extra_train, y_train, EMBEDDINGS, paths)

        elif self.model_type == 'PIX':
            self.learn_model(X_train, X_extra_train, y_pixl_train, EMBEDDINGS)


    def learn_model(self, X_train, X_extra_train, y_train, EMBEDDINGS, paths):
        '''
        INPUT: EMBEDDINGS is a dictionary that contains the embeddings to initialize the embedding layer (and a random version of it)
        OUTPUT: a learned model
        '''
        vocab_length, emb_matrix, emb_length = {},{},{}

        # vocabulary lengths
        vocab_length['subj'], vocab_length['pred'], vocab_length['obj'], vocab_length['cap'] = len(EMBEDDINGS['subj_list']), len(EMBEDDINGS['pred_list']), len(EMBEDDINGS['obj_list']), len(EMBEDDINGS['cap_list'])

        # -- Decide what embeddings to use -- #
        if self.method == 'emb':
            emb_matrix['subj'], emb_matrix['pred'], emb_matrix['obj'] = EMBEDDINGS['subj_EMB'], EMBEDDINGS['pred_EMB'], EMBEDDINGS['obj_EMB']#, EMBEDDINGS['cap_EMB']
        elif self.method == 'rnd':
            emb_matrix['subj'], emb_matrix['pred'], emb_matrix['obj'] = EMBEDDINGS['subj_EMB_rnd'], EMBEDDINGS['pred_EMB_rnd'], EMBEDDINGS['obj_EMB_rnd']
        elif self.method == 'onehot':
            emb_matrix['subj'], emb_matrix['pred'], emb_matrix['obj'] = EMBEDDINGS['subj_EMB_onehot'], EMBEDDINGS['pred_EMB_onehot'], EMBEDDINGS['obj_EMB_onehot']

        # -- get TRAINING DATA -- #
        #X_tr = [X_train['subj'], X_train['obj'], X_train['cap'], X_train['seg']] #, X_train['pred']
        X_tr = [X_train['cap'], X_train['seg']]
        X_tr.extend([X_extra_train])
        
        #print('Vocab Length: ' + str(vocab_length['subj']))
        #print('Emb Length: ' + str(emb_matrix['subj'].shape[0]))
        # -- DEFINE model -- #
        self.define_model(vocab_length, X_extra_train, y_train, emb_matrix, paths)
        
        # -- FIT model -- #
        self.keras_model.fit(X_tr, y_train, verbose=2, epochs=self.par_learning['n_epochs'], batch_size=self.par_learning['batch_size'])


    def model_predict(self, X, X_extra, y_train):
        '''
        :param X: new input to be predicted
        :param y_train: the ctrl method uses it to get statistics of mean and sd
        '''
        if self.method is not 'ctrl': # any of the: 'emb-nolearn_init' or 'emb-learn_init' or 'emb-nolearn_rnd' or 'emb-learn_rnd'
            #y_pred = self.keras_model.predict([X['subj'], X['obj'], X['cap'],X['seg'], X_extra])#, X['pred']
            y_pred = self.keras_model.predict([X['cap'], X['seg'], X_extra])#, X['pred']

        elif self.method == 'ctrl': # Random prediction method
            import numpy as np

            n_inst = X['subj'].shape[0]

            # get predictions:
            if self.model_type == 'REG':
                #get descriptive statistics of y to generate random predictions
                mu, sigma = np.mean(y_train, axis=0), np.std(y_train, axis=0) #mu and sigma are now vectors
                y_pred = []
                for i in range(y_train.shape[1]):
                    y_pred.append( np.random.normal(mu[i], sigma[i], n_inst) )

                y_pred = np.array(y_pred).T

            elif self.model_type == 'PIX':
                y_pred = np.random.uniform(0, 1, n_inst * self.n_side_pixl * self.n_side_pixl).reshape((n_inst, self.n_side_pixl* self.n_side_pixl))

        return y_pred



    def clear_model(self):
        del self.keras_model

