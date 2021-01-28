import numpy as np
import string
from keras_bert import load_trained_model_from_checkpoint, Tokenizer

def getCaptionWords(Train):
    caption_words = set()
    table = str.maketrans('', '', string.punctuation)
    captions = list(Train)
    for caption in captions:
        i = 0
        for word in caption.split(" "):
            word = word.lower()
            word = word.translate(table)
            caption_words.add(word)
            i+=1
        j = 100-i
        for x in range(0,j):
            caption_words.add('<pad>')
    return caption_words

def getCaptionIndices(caption, cap_list):
    caption_words = []
    table = str.maketrans('', '', string.punctuation)
    i = 0
    for word in caption.split(" "):
        word = word.lower()
        word = word.translate(table)
        caption_words.append(int(cap_list.index(word)))
        i += 1
    j = 100 - i
    
    for x in range(0,j):
        caption_words.append(int(cap_list.index('<pad>')))
        
    return caption_words

def get_BERT_Embeddings(token_dict, TRAIN_relevant):
    EMBEDDINGS = {}
    EMBEDDINGS['cap_list'] = []
    #EMBEDDINGS['cap_EMB'] = np.random.uniform(0, size=(len(token_dict),768))
    EMBEDDINGS['cap_indices'] = np.random.uniform(0, size=(len(TRAIN_relevant['subj']),100))

    tokenizer = Tokenizer(token_dict)
    for i in range(len(TRAIN_relevant['subj'])):
        tokenized_text = tokenizer.tokenize(TRAIN_relevant['cap'][i])
        indices, segments = tokenizer.encode(first=TRAIN_relevant['cap'][i], max_len=100)
        TRAIN_relevant['cap'][i] = tokenized_text
        #predicts = model.predict([np.array([indices]), np.array([segments])])[0]
        for j, index in enumerate(indices):
            #EMBEDDINGS['cap_EMB'][index] = predicts[j] 
            EMBEDDINGS['cap_list'].append(index)
        EMBEDDINGS['cap_indices'][i] = indices
    EMBEDDINGS['cap_segments'] = np.zeros((EMBEDDINGS['cap_indices'].shape[0],100))
    return EMBEDDINGS, TRAIN_relevant

def get_data(model_type, TRAIN, words, EMB, enforce_gen, n_side_pixl, token_dict, paths):
    import numpy as np
    EMBEDDINGS, OBJ_ctr_sd_enf_gen = {}, []
    
    # Get BERT model to predict embeddings
    #model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, seq_len=100)
    #model.summary(line_length=120)

    # 0. Get dictionary of ALL our embedding words
    EMB_dict = build_emb_dict(words, EMB)

    # 1. Get the RELEVANT training instances (filtering for 'predicates' and 'complete_only' variables)
    OBJ_ctr_sd, rel_ids, TRAIN_relevant = get_TRAIN_relevant(TRAIN, words)

    # Get contextualized embeddings
    EMB, TRAIN_relevant = get_BERT_Embeddings(token_dict, TRAIN_relevant)

    # 2. get dictionaries WORDLISTS (INDICES for the embedding layer!)
    EMBEDDINGS['obj_list'] = list(set(TRAIN_relevant['obj']))
    EMBEDDINGS['subj_list'] = list(set(TRAIN_relevant['subj']))
    EMBEDDINGS['pred_list'] = list(set(TRAIN_relevant['rel']))
    EMBEDDINGS['cap_list'] = list(EMB['cap_list'])

    allwords = np.concatenate((EMBEDDINGS['subj_list'], EMBEDDINGS['pred_list'], EMBEDDINGS['obj_list'], EMBEDDINGS['cap_list']), axis=0)
    EMBEDDINGS['allwords_list'] = list(
        set(allwords))  # IMPORTANT: The order of this list is what prevails later on as index for embeddings

    # 3. Get INITIALIZATION embeddings
    EMBEDDINGS['subj_EMB'] = wordlist2emb_matrix(EMBEDDINGS['subj_list'], EMB_dict)
    EMBEDDINGS['pred_EMB'] = wordlist2emb_matrix(EMBEDDINGS['pred_list'], EMB_dict)
    EMBEDDINGS['obj_EMB'] = wordlist2emb_matrix(EMBEDDINGS['obj_list'], EMB_dict)
    #EMBEDDINGS['cap_EMB'] = EMB['cap_EMB']
    EMBEDDINGS['allwords_EMB'] = wordlist2emb_matrix(EMBEDDINGS['allwords_list'],EMB_dict)
    # 3.1. Get RANDOM embeddings (of the size of allwords_EMB)
    EMBEDDINGS['allwords_EMB_rnd'] = get_random_EMB(EMBEDDINGS['allwords_EMB'])
    EMBEDDINGS['subj_EMB_rnd'] = get_random_EMB(EMBEDDINGS['subj_EMB'])
    EMBEDDINGS['pred_EMB_rnd'] = get_random_EMB(EMBEDDINGS['pred_EMB'])
    EMBEDDINGS['obj_EMB_rnd'] = get_random_EMB(EMBEDDINGS['obj_EMB'])
    #EMBEDDINGS['cap_EMB_rnd'] = get_random_EMB(EMBEDDINGS['cap_EMB'])
    # 3.2. get ONE-HOT embeddings:
    EMBEDDINGS['subj_EMB_onehot'] = np.identity(len(EMBEDDINGS['subj_list']))
    EMBEDDINGS['pred_EMB_onehot'] = np.identity(len(EMBEDDINGS['pred_list']))
    EMBEDDINGS['obj_EMB_onehot'] = np.identity(len(EMBEDDINGS['obj_list']))
    #EMBEDDINGS['cap_EMB_onehot'] = np.identity(len(EMBEDDINGS['cap_list']))
    #EMBEDDINGS['allwords_EMB_onehot'] = np.identity(len(EMBEDDINGS['allwords_list']))

    EMBEDDINGS['cap_indices'] = EMB['cap_indices']
    EMBEDDINGS['cap_segments'] = EMB['cap_segments']
    
    # 4. Get X data (i.e., get the SEQUENCES of INDICES for the embedding layer)
    X, X_extra, y, y_pixl, X_extra_enf_gen, X_enf_gen, y_enf_gen, y_enf_gen_pixl, \
    idx_IN_X_and_y, idx_enf_gen = relevant_instances2X_and_y(model_type, TRAIN_relevant, EMBEDDINGS, enforce_gen,
                                                             n_side_pixl)

    # 5. Get the OBJ_ctr_sd_enf_gen that we need for some performance measures!
    if enforce_gen['eval'] is not None:
        OBJ_ctr_sd_enf_gen = OBJ_ctr_sd[idx_enf_gen]

    # 6. Finally, if we have REDUCED the X and y data by ENFORCING generalization (excluding instances) we have to reduce OBJ_ctr_sd and TRAIN_relevant accordingly
    if enforce_gen['eval'] is not None:
        for key in TRAIN_relevant:
            TRAIN_relevant[key] = np.array(TRAIN_relevant[key])
            TRAIN_relevant[key] = TRAIN_relevant[key][idx_IN_X_and_y]
        OBJ_ctr_sd = OBJ_ctr_sd[idx_IN_X_and_y]
        rel_ids = np.array(rel_ids)
        rel_ids = rel_ids[idx_IN_X_and_y]

    return X, X_extra, y, y_pixl, X_extra_enf_gen, X_enf_gen, y_enf_gen, y_enf_gen_pixl, rel_ids, OBJ_ctr_sd, OBJ_ctr_sd_enf_gen, EMBEDDINGS, TRAIN_relevant



def relevant_instances2X_and_y(model_type, TRAIN_relevant, EMBEDDINGS, enforce_gen, n_side_pixl):
    # OUTPUT: the X and y data, gotten by converting each word into its corresponding index
    print('Getting X and y data')

    X_vars = ['subj_ctr_x', 'subj_ctr_y', 'subj_sd_x', 'subj_sd_y']
    y_vars = ['obj_sd_x', 'obj_sd_y', 'obj_ctr_x', 'obj_ctr_y']

    subj_list, pred_list, obj_list, cap_list, allwords_list = EMBEDDINGS['subj_list'], EMBEDDINGS['pred_list'], EMBEDDINGS[
        'obj_list'], EMBEDDINGS['cap_list'], EMBEDDINGS['allwords_list']
    cap_indices = EMBEDDINGS['cap_indices']
    cap_segments = EMBEDDINGS['cap_segments']
    
    # get X:
    X, X_enf_gen = {}, {}
    X['subj'], X['pred'], X['obj'], X['cap'], X['seg'] = [], [], [], [], []
    X_enf_gen['subj'], X_enf_gen['pred'], X_enf_gen['obj'], X_enf_gen['cap'], X_enf_gen['seg'] = [], [], [], [], []
    for i in range(len(TRAIN_relevant['subj'])):
        triplet = (TRAIN_relevant['subj'][i], TRAIN_relevant['rel'][i], TRAIN_relevant['obj'][i])

        # append to the GENERALIZED set
        if (enforce_gen['eval'] == 'triplets') and (triplet in enforce_gen['triplets']):
            X_enf_gen['subj'].append(subj_list.index(TRAIN_relevant['subj'][i]))
            X_enf_gen['pred'].append(pred_list.index(TRAIN_relevant['rel'][i]))
            X_enf_gen['obj'].append(obj_list.index(TRAIN_relevant['obj'][i]))
            #caption_words = []
            #caption_words = getCaptionIndices(TRAIN_relevant['cap'][i], cap_list)
            #X_enf_gen['cap'].append(caption_words)
            X_enf_gen['cap'].append(cap_indices[i])
            X_enf_gen['seg'].append(cap_segments[i])
        elif (enforce_gen['eval'] == 'words') and any(word in enforce_gen['words'] for word in triplet):
            X_enf_gen['subj'].append(subj_list.index(TRAIN_relevant['subj'][i]))
            X_enf_gen['pred'].append(pred_list.index(TRAIN_relevant['rel'][i]))
            X_enf_gen['obj'].append(obj_list.index(TRAIN_relevant['obj'][i]))
            #caption_words = []
            #caption_words = getCaptionIndices(TRAIN_relevant['cap'][i], cap_list)
            #X_enf_gen['cap'].append(caption_words)
            X_enf_gen['cap'].append(cap_indices[i])
            X_enf_gen['seg'].append(cap_segments[i])
            
        else:  # if either the triplet/word is not generalized or we aren't enforcing generalization
            X['subj'].append(subj_list.index(TRAIN_relevant['subj'][i]))
            X['pred'].append(pred_list.index(TRAIN_relevant['rel'][i]))
            X['obj'].append(obj_list.index(TRAIN_relevant['obj'][i]))
            #X['cap'].append(getCaptionIndices(TRAIN_relevant['cap'][i], cap_list))
            X['cap'].append(cap_indices[i])
            X['seg'].append(cap_segments[i])
            
    #X['cap'] = [xi+[0]*(100-len(xi)) for xi in X['cap']]
    #print(X['cap'][0])
    # Reshape
    X['subj'] = np.array(X['subj']).reshape((-1, 1))
    X['pred'] = np.array(X['pred']).reshape((-1, 1))
    X['obj'] = np.array(X['obj']).reshape((-1, 1))
    X['cap'] = np.array(X['cap']).reshape((-1, 100))
    X['seg'] = np.array(X['seg']).reshape((-1, 100))
    
    # FORMAT: if we have gotten some zero shot instances
    if X_enf_gen['subj'] != []:
        X_enf_gen['subj'] = np.array(X_enf_gen['subj']).reshape(
            (-1, 1))  # get them in the right FORMAT for the merged (SEP) model!
        X_enf_gen['pred'] = np.array(X_enf_gen['pred']).reshape((-1, 1))
        X_enf_gen['obj'] = np.array(X_enf_gen['obj']).reshape((-1, 1))
        X_enf_gen['cap'] = np.array(X_enf_gen['cap']).reshape((-1, 1))
        X_enf_gen['seg'] = np.array(X_enf_gen['seg']).reshape((-1, 1))
    else:
        X_enf_gen['subj'], X_enf_gen['pred'], X_enf_gen['obj'], X_enf_gen['cap'], X_enf_gen['seg'] = None, None, None, None, None

    # Get Y (if model_type = PIX we output the regular y besides y_pixl!)
    y, y_pixl, y_enf_gen, idx_IN_X_and_y, idx_enf_gen, y_enf_gen_pixl = [], [], [], [], [], []
    for i in range(len(TRAIN_relevant['subj'])):
        y_new_row = []

        for k in range(len(y_vars)):
            y_new_row.extend([float(TRAIN_relevant[y_vars[k]][i])])  # IMPORTANT: We assume that the variables are NUMERIC

        if model_type == 'PIX':
            obj_sd_x, obj_sd_y = float(TRAIN_relevant['obj_sd_x'][i]), float(TRAIN_relevant['obj_sd_y'][i])
            obj_ctr_x, obj_ctr_y = float(TRAIN_relevant['obj_ctr_x'][i]), float(TRAIN_relevant['obj_ctr_y'][i])
            y_pixl_new_row = coord2pixel_indiv(obj_sd_x, obj_sd_y, obj_ctr_x, obj_ctr_y, n_side_pixl)

        # get stuff for the generalzed setting:
        triplet = (TRAIN_relevant['subj'][i], TRAIN_relevant['rel'][i], TRAIN_relevant['obj'][i])

        if (enforce_gen['eval'] == 'triplets') and (triplet in enforce_gen['triplets']):
            y_enf_gen.append(y_new_row)
            if model_type == 'PIX':
                y_enf_gen_pixl.append(y_pixl_new_row)
            idx_enf_gen.append(i)

        elif (enforce_gen['eval'] == 'words') and any(word in enforce_gen['words'] for word in triplet):
            y_enf_gen.append(y_new_row)
            if model_type == 'PIX':
                y_enf_gen_pixl.append(y_pixl_new_row)
            idx_enf_gen.append(i)

        else:  # NON GENERALIZED
            y.append(y_new_row)
            if model_type == 'PIX':
                y_pixl.append(y_pixl_new_row)
            idx_IN_X_and_y.append(i)

    y = np.array(y)
    y_enf_gen = np.array(y_enf_gen) if y_enf_gen != [] else None

    if model_type == 'PIX':
        y_pixl = np.array(y_pixl)
        y_enf_gen_pixl = np.array(y_enf_gen_pixl) if y_enf_gen_pixl != [] else None
    else:
        y_pixl = [[[]]]  # necessary because we get the index 0 of y_pixl (if model_type != 'PIX') to save memory in learn_and_evaluate()

    print('We have gotten ' + str(len(idx_IN_X_and_y)) + ' instances (for both, train & test)')

    # Get X_extra
    X_extra, X_extra_enf_gen = [], []
    if X_vars != []:
        for i in range(len(TRAIN_relevant['subj'])):
            X_extra_new_row = []
            for k in range(len(X_vars)):  # we already ASSUME that we have at least one y-variable
                X_extra_new_row.extend(
                    [float(TRAIN_relevant[X_vars[k]][i])])  # IMPORTANT: We assume that the variables are NUMERIC
            # get stuff for the generalized:
            triplet = (TRAIN_relevant['subj'][i], TRAIN_relevant['rel'][i], TRAIN_relevant['obj'][i])
            if (enforce_gen['eval'] == 'triplets') and (triplet in enforce_gen['triplets']):
                X_extra_enf_gen.append(X_extra_new_row)
            elif (enforce_gen['eval'] == 'words') and any(word in enforce_gen['words'] for word in triplet):
                X_extra_enf_gen.append(X_extra_new_row)
            else:
                X_extra.append(X_extra_new_row)
    X_extra = np.array(X_extra) if X_extra != [] else None # IMPORTANT: we only make it a numpy array if we have something, because we use == [] as condition in models_learn
    X_extra_enf_gen = np.array(X_extra_enf_gen) if X_extra_enf_gen != [] else None

    return X, X_extra, y, y_pixl, X_extra_enf_gen, X_enf_gen, y_enf_gen, y_enf_gen_pixl, idx_IN_X_and_y, idx_enf_gen



def get_TRAIN_relevant(TRAIN, words):
    # IMPORTANT: we preserve the ORDER of TRAIN (so that we can recover information afterwards)

    TRAIN_relevant, rel_ids, OBJ_ctr_sd = {}, [], []
    print('Getting *relevant* instances, from a total of: ' + str(len(TRAIN['subj'])))

    var_names = [key for key in TRAIN]
    # INITIALIZE TRAIN_relavant
    for varname in var_names:
        TRAIN_relevant[varname] = []

    for i in range(len( TRAIN['subj'] )): # Samples loop

        we_have_it = True if ((TRAIN['subj'][i] in words) and (TRAIN['rel'][i] in words) and (TRAIN['obj'][i] in words)) else False # if we have the complete triplet

        if we_have_it == True:
            for varname in var_names:
                TRAIN_relevant[varname].append(TRAIN[varname][i])
            rel_ids.append(TRAIN['rel_id'][i])
            OBJ_ctr_sd.append([TRAIN['img_idx'][i], TRAIN['rel_id'][i], TRAIN['subj'][i], TRAIN['rel'][i],
                               TRAIN['obj'][i], TRAIN['subj_sd_x'][i], TRAIN['subj_sd_y'][i],
                               TRAIN['subj_ctr_x'][i], TRAIN['subj_ctr_y'][i], TRAIN['obj_sd_x'][i],
                               TRAIN['obj_sd_y'][i], TRAIN['obj_ctr_x'][i], TRAIN['obj_ctr_y'][i]])
            
            
    OBJ_ctr_sd = np.array(OBJ_ctr_sd)
    print('We have gotten ' + str(len(TRAIN_relevant['subj'])) + ' RELEVANT instances')
    return OBJ_ctr_sd, rel_ids, TRAIN_relevant


def get_random_EMB(actual_EMB):
    # Returns embedding matrix of the original shape with random normal vectors (dimension-wise)
    mu, sigma, vec_size = np.mean(actual_EMB), np.mean(np.std(actual_EMB, axis=0)), len(actual_EMB[0, :])
    rand_EMB = []

    for i in range(actual_EMB.shape[0]):  # build a dictionary of random vectors
        rand_EMB.append(np.random.normal(mu, sigma, vec_size))
    rand_EMB = np.array(rand_EMB)

    return rand_EMB


def coord2pixel_indiv(obj_sd_x, obj_sd_y, obj_ctr_x, obj_ctr_y, n_side_pixl):
    '''
    This function works with an individual example (extending it to many examples, where e.g., obj_sd_x is a vector, is easy)
    :param obj_sd_x (and the rest): real number (not vectors!)
    :param n_side_pixl: number of pixels as output (hyperparameter)
    :return y_pixl: matrix of pixels, i.e., a 2D tensor (n_side_pixl, n_side_pixl)
    '''

    # continuous bounding box corners (prevent problems of predictions outside [0,1])
    A_left_x, A_right_x = max((obj_ctr_x - obj_sd_x), 0), min((obj_ctr_x + obj_sd_x), 1)
    A_low_y, A_top_y = min((obj_ctr_y + obj_sd_y), 1), max((obj_ctr_y - obj_sd_y), 0)

    # translate continuous bounding box corners into indices in a n_side_pixl x n_side_pixl matrix
    i_left, i_right = np.rint( (n_side_pixl - 1)*A_left_x).astype(np.int), np.rint((n_side_pixl - 1)*A_right_x).astype(np.int)
    j_low, j_top = np.rint((n_side_pixl - 1)*A_low_y).astype(np.int), np.rint((n_side_pixl - 1)*A_top_y).astype(np.int)
    pixl_matr = np.zeros( (n_side_pixl, n_side_pixl) )

    # add ones inside of the bounding box
    i_range = range( i_left, i_right )
    i_range = [i_left] if ((i_left == i_right) or (i_range == [])) else i_range # AVOID THE CASE where width is 0 AND i_range=[] (as upper bound < lower bound)
    j_range = range( j_top, j_low )
    j_range = [j_low] if ((j_low == j_top) or (j_range == [])) else j_range # AVOID THE CASE where height is 0 AND i_range=[] (as upper bound < lower bound)
    pixl_matr[ np.array(i_range)[:, None], np.array(j_range)] = 1 # (IMPORTANT: indices must be np.arrays) put a 1 everywhere inside of the bounding box
    pixl_matr = pixl_matr.reshape((-1))

    return pixl_matr



def pixl_idx2coord_all_examples(y_pixl):
    '''
    Transforms the whole set of predicted matrices y_pixl into their continuous CENTER coordinates (Obj_ctr)
    :param y_pixl: array of MATRICES with predicted heatmaps (pixels). Each matrix = 1 example
    :return: PRED_obj_ctr_x, PRED_obj_ctr_y: arrays of length = number of examples
    '''

    PRED_obj_ctr_x, PRED_obj_ctr_y = [], []
    n_side_pixl = y_pixl.shape[1] #get automatically the number of pixels from the pixel matrix side

    for i in range( y_pixl.shape[0] ): # loop on number of examples
        idx_maximums = get_maximums_idx(y_pixl[i]) # get indices of maximum (allow for multiple of them)
        ctr_x, ctr_y = pixl_idx2coord_indiv(idx_maximums, n_side_pixl) # transform pixel indices into continuous coordinates
        PRED_obj_ctr_x.append(ctr_x)
        PRED_obj_ctr_y.append(ctr_y)

    PRED_obj_ctr_x, PRED_obj_ctr_y = np.array(PRED_obj_ctr_x), np.array(PRED_obj_ctr_y)

    return PRED_obj_ctr_x, PRED_obj_ctr_y


def get_maximums_idx( heat_matrix ):
    # Given a matrix of activations, it outputs the indices corresponding to its maximum values
    # INPUT: heat_matrix: matrix of continuous activations (within [0,1]) of size n_side_pixl x n_side_pixl
    # OUTPUT: maximums: indices corresponding to where the activations are maximum (accounts for multiple maximums)
    #maximums = np.unravel_index(np.argmax(heat_matrix), heat_matrix.shape)  # gives the index of the FIRST largest element. Doesn't account for multiple maximums!
    maximums = np.where(heat_matrix == heat_matrix.max()) # This one accounts for multiple maximums!
    return np.array(maximums)


def pixl_idx2coord_indiv(idx_maximums, n_side_pixl):
    '''
    This function receives input from get_maximums_indices()
    Given discrete pixels indices (i,j) where i,j = 0,...,n_side_pixl (where activations are maximal),
    it transforms them to (continuous) coordinates in [0,1]
    IMPORTANT: It only computes the CENTER of the Obj (not sd's). So it's useful for measures that only use Obj_ctr
    :param idx_maximums: index of maximums from get_maximums_idx()
    :param n_side_pixl: side of the activation matrix (necessary to transform indices to coordinates)
    :return pred_obj_ctr_x, pred_obj_ctr_y: predicted (continuous) coordinates in [0,1] (Obj_ctr)
    '''

    coord = np.mean(idx_maximums, axis = 1)

    PRED_coord = coord.astype(np.float)/float(n_side_pixl - 1) # Transform pixel indices to (continuous) coordinates
    pred_obj_ctr_x, pred_obj_ctr_y = PRED_coord[0], PRED_coord[1]

    return pred_obj_ctr_x, pred_obj_ctr_y



def get_folds(n_samples, n_folds):
    indices = np.random.permutation(np.arange(n_samples))
    n_test = int(np.floor(n_samples / n_folds))
    kf = [(np.delete(indices, np.arange(i * n_test, (i + 1) * n_test)), # train
           indices[i * n_test:(i + 1) * n_test]) for i in range(n_folds)] # test
    return kf



def mirror_x(subj_ctr_x, obj_ctr_x):
    # Computes the absolute value of the obj_ctr_x variable (to make it symmetric)

    aux_obj_ctr_x = [ (1 - float(obj_ctr_x[i])) if float(obj_ctr_x[i]) <= float(subj_ctr_x[i]) else float(obj_ctr_x[i]) for i in range(len(obj_ctr_x)) ]
    aux_subj_ctr_x = [ (1 - float(subj_ctr_x[i])) if float(obj_ctr_x[i]) <= float(subj_ctr_x[i]) else float(subj_ctr_x[i]) for i in range(len(obj_ctr_x)) ]
    subj_ctr_x, obj_ctr_x = aux_subj_ctr_x, aux_obj_ctr_x

    return subj_ctr_x, obj_ctr_x


def build_emb_dict(words, EMB):
    #Input: words= word list, EMB= embeddings in a np.array format
    #Output: Dictionary of embeddings
    EMB_dict = {}
    for i in range(len(words)):
        EMB_dict[words[i]] = EMB[i,:]
    return EMB_dict


def wordlist2emb_matrix(words_to_get, EMB_dict):
    # Input: words_to_get = word list from the EMB matrix, EMB_dict= dictionary of embeddings
    # Output: MATRIX of embeddings
    # IMPORTANT: it preserves the order of the words_to_get list

    EMB_matrix = []
    EMB_matrix.append( np.random.uniform(0,1, size=(300,)) )

    for i in range(len(words_to_get)):
        try:
            EMB_matrix.append( EMB_dict[words_to_get[i]] )
        except KeyError:
            #print('WARNING! word ' + words_to_get[i] + ' not found in our embeddings! (and it should be!!)')
            EMB_matrix.append( np.random.uniform(0,1, size=(300,)) )
            pass
            
    EMB_matrix = np.array(EMB_matrix)

    return EMB_matrix


def get_GEN(TRAIN_relevant, train_idx, test_idx, subj, obj, pred):
    # Gives the generalized triplets (combination) and the generalized words
    # IMPORTANT: The outputted indices (idx_gen_tuples) are over the instances of the TEST SET!
    print ('Getting generalized instances')

    train_tuples = [(TRAIN_relevant[subj][train_idx[ii]],TRAIN_relevant[pred][train_idx[ii]], TRAIN_relevant[obj][ train_idx[ii] ]) for ii in range(len(train_idx)) ]
    test_tuples = [(TRAIN_relevant[subj][test_idx[ii]],TRAIN_relevant[pred][test_idx[ii]], TRAIN_relevant[obj][test_idx[ii]]) for ii in range(len(test_idx)) ]

    # 2. get GENERALIZED TUPLES (its index)
    #train_tuples, test_tuples = tuples[train_idx], tuples[test_idx]
    idx_gen_tuples = [i for i, x in enumerate(test_tuples) if x not in train_tuples]#IMPORTANT!!! This index (idx_gen_tuples) is over the instances of the TEST SET!

    # 3.get GENARALIZED WORDS (its index)
    subjs, preds, objs = np.array(TRAIN_relevant[subj]), np.array(TRAIN_relevant[pred]), np.array(TRAIN_relevant[obj])

    # get unique wordlist in test_tuples[idx_gen_tuples] --> TRICK! if there are gen-words, they MUST be within gen-tuples) and in train tuples
    allwords_train = np.concatenate((subjs[train_idx], preds[train_idx],objs[train_idx]), axis=0)
    allwords_test = np.concatenate((subjs[test_idx], preds[test_idx],objs[test_idx]), axis=0)

    unique_words_train = list(set(allwords_train))
    unique_words_test = list(set(allwords_test))

    # get generalized words
    gen_words = [ unique_words_test[j] for j in range(len(unique_words_test)) if unique_words_test[j] not in unique_words_train] # intersect unique_words_train and unique_words_test and get the complementary

    # find INDICES of the tuples in the test set that contain any of the gen-words
    idx_gen_words = [ i for i,x in enumerate(test_tuples) if any(word in gen_words for word in x) ] # x=tuple, word= word within the tuple, i=index of the tuple x

    return idx_gen_tuples, idx_gen_words, gen_words


def get_CLEAN_train_test_idx(TRAIN_relevant, train_idx, test_idx, clean_eval):
    # Gives the indices of the triplets that we want to get as a clean test set
    # clean_eval contains the triplets, etc. of our clean selection of instances
    # IMPORTANT: idx_clean_train and idx_clean_test are over the instances of the TRAIN and TEST SETS respectively!
    print ('Getting *clean* train and test instances')

    #0. get either tuples or triplets from our train AND test data
    if (clean_eval['eval'] == 'triplets') or (clean_eval['eval'] == 'words'): #if we want clean words, we search them among the whole triplet
        train_tuples = [(TRAIN_relevant['subj'][train_idx[ii]],TRAIN_relevant['rel'][train_idx[ii]], TRAIN_relevant['obj'][ train_idx[ii] ]) for ii in range(len(train_idx)) ]
        test_tuples = [(TRAIN_relevant['subj'][test_idx[ii]],TRAIN_relevant['rel'][test_idx[ii]], TRAIN_relevant['obj'][test_idx[ii]]) for ii in range(len(test_idx)) ]

    #1. Decide what clean triplets/words to use
    if clean_eval['eval'] == 'triplets':
        clean_tuples = clean_eval['triplets']
    if clean_eval['eval'] == 'words':
        clean_words = clean_eval['words']

    # 2. get INDICES of our clean instances (triplets...) in both, TRAIN and TEST data
    if clean_eval['eval'] == 'triplets':
        idx_clean_test = [ i for i,x in enumerate(test_tuples) if x in clean_tuples ]
        idx_clean_train = [ i for i,x in enumerate(train_tuples) if x in clean_tuples ]
    if clean_eval['eval'] == 'words':
        idx_clean_test = [ i for i,x in enumerate(test_tuples) if any(word in clean_words for word in x) ]
        idx_clean_train = [i for i, x in enumerate(train_tuples) if any(word in clean_words for word in x)]

    return idx_clean_train, idx_clean_test



def get_enforce_gen(to_get):
    # This function can also get the clean_test instances, as there's nothing specific about generalized here
    import read_data as rd

    gen_triplets, gen_words = [],[]

    if to_get == 'triplets':

        # Read CSV
        GEN_INST = rd.load_training_data('../data/TRIPLETS_random.csv')

        # Get all triplets
        for i in range(len(GEN_INST['subj'])):
            gen_triplets.append( ( GEN_INST['subj'][i], GEN_INST['rel'][i], GEN_INST['obj'][i] ) )

        gen_triplets = list(set(gen_triplets))

    elif to_get == 'words':
        gen_words = rd.readWordlist('../data/WORDS_random.csv')
        gen_words = list(set(gen_words))

    return gen_triplets, gen_words




def aux_get_train_test_splits(X, X_extra, y, OBJ_ctr_sd, train_idx, test_idx):
    # This is an auxiliary function that gives back the train and test splits
    # Is not very elegant, but we don't create y_pixl_train and test splits because it takes too much memory

    # get X
    X_train, X_test = {},{}
    X_train['subj'], X_test['subj'] = X['subj'][train_idx], X['subj'][test_idx]
    X_train['pred'], X_test['pred'] = X['pred'][train_idx], X['pred'][test_idx]
    X_train['obj'], X_test['obj'] = X['obj'][train_idx], X['obj'][test_idx]
    X_train['cap'], X_test['cap'] = X['cap'][train_idx], X['cap'][test_idx]
    X_train['seg'], X_test['seg'] = X['seg'][train_idx], X['seg'][test_idx]

    # get y
    y_train, y_test = y[train_idx], y[test_idx] #not a dictionary!!! (a matrix array)
    OBJ_ctr_sd_train, OBJ_ctr_sd_test = OBJ_ctr_sd[train_idx], OBJ_ctr_sd[test_idx]
    X_extra_train, X_extra_test = X_extra[train_idx], X_extra[test_idx]

    return X_train, X_test, X_extra_train, X_extra_test, y_train, y_test, OBJ_ctr_sd_train, OBJ_ctr_sd_test



def compute_centers(x_subj, y_subj, width_subj, height_subj, x_obj, y_obj, width_obj, height_obj):
    '''
    Notice: the (0,0) coordinates of the image correspond to the TOP left corner (not bottom left)
    INPUT: absolute positions of:
           x_obj (x of top left corner of bounding box), y_obj (y of top left...),
           width_obj (width bounding box), height_obj (height bounding box),
           width_img (width img), height_img (height img)
    OUTPUT: centered positions (center of bounding box and standard dev. of bounding box):
           obj_ctr_x, obj_ctr_y, obj_sd_x, obj_sd_y
    '''
    # OBJECT:
    obj_ctr_x = float(x_obj) + (float(width_obj)/2)
    obj_ctr_y = float(y_obj) + (float(height_obj)/2)
    obj_sd_x = (float(width_obj)/2) # after simplifying in the formula of SD, it gives this
    obj_sd_y = (float(height_obj)/2) # after simplifying in the formula of SD, it gives this
    # SUBJECT:
    subj_ctr_x = float(x_subj) + (float(width_subj)/2)
    subj_ctr_y = float(y_subj) + (float(height_subj)/2)
    subj_sd_x = (float(width_subj)/2) #after simplifying in the formula of SD, it gives this
    subj_sd_y = (float(height_subj)/2) # after simplifying in the formula of SD, it gives this

    return subj_ctr_x, subj_ctr_y, subj_sd_x, subj_sd_y, obj_ctr_x, obj_ctr_y, obj_sd_x, obj_sd_y
