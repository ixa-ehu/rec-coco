import argparse
import param_tools as pt
import write_data as wd
import read_data as rd
import data_tools as dt
import models
import eval_tools as et
from param_tools import bool_str
from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer, get_checkpoint_paths
from keras_bert.datasets import get_pretrained, PretrainedList


def main():

    parser = argparse.ArgumentParser()

    # General arguments:
    parser.add_argument('--relations', default='textual', help='Relationship type. OPTIONS: textual, concept')
    parser.add_argument('--model_type', default='REG', help='OPTIONS: REG, PIX')
    parser.add_argument('--n_side_pixl', default=15, type=int, help='Number of pixels as output of PIX')
    parser.add_argument('--method_compare', default=['emb'],
                        help='Methods to compare. OPTIONS: init, rnd, onehot, ctrl')
    parser.add_argument('--n_folds', default=10, type=int, help='Number of cross-validation folds')
    parser.add_argument('--eval_generalized_set', default= None,
                        help='Whether we evaluate in a generalized set or not. If so, instances are left out for training. '
                             'OPTIONS: None, triplets, words')
    parser.add_argument('--eval_clean_set', default= None,
                        help='Whether we evaluate in a clean set (equal to the generalized ones, but without keeping the model from'
                             'seeing these words/triplets during training. BE CAREFUL! Do not use the same list as in generalized above'
                             '(because you will not find any tuple for the clean set if you have removed them first!) '
                             'OPTIONS: None, triplets, words')
    parser.add_argument('--save_indiv_predictions', default=True, type=bool_str, help='To store model predictions (they can be heavy, especially in PIX). '
                                                                        'Useful to visualize them afterwards.')
    parser.add_argument('--save_model', default=False, type=bool_str, help='To store the models (e.g., to explore weights afterwards)')

    args = parser.parse_args()

    if args.model_type == 'REG':
        perf_measures = ['R2', 'acc_y', 'F1_y', 'Pear_x', 'Pear_y', 'IoU_t']
    if args.model_type == 'PIX':
        perf_measures = ['acc_y', 'F1_y', 'Pear_x', 'Pear_y', 'max_acc_px']

    # Create folder for results
    saveFolder = '../results/results_REF-COCO_capt_BERT_ft_ctxt3' #wd.get_folder_name(args)

    # Get default params
    par_learning = pt.get_default_params(args.model_type)

    # --- Read data --- #
    TRAIN = rd.load_training_data('../training_data/Training_data_ref-coco_'+ args.relations +'.csv')
    TRAIN['subj_ctr_x'], TRAIN['obj_ctr_x'] = dt.mirror_x(TRAIN['subj_ctr_x'], TRAIN['obj_ctr_x'])
    words, EMB = rd.readDATA( '../embeddings/glove.6B.300d.clean.csv')

    print('Loading BERT...')
    # --- Load Pretrained BERT paths --- #
    model_path = get_pretrained(PretrainedList.multi_cased_base)
    paths = get_checkpoint_paths(model_path)
    
    # --- Initialize BERT Tokenizer  --- #
    token_dict = load_vocabulary(paths.vocab)

    # --- GENERALIZED and CLEAN triplets or words --- #
    enforce_gen, clean_eval = {}, {}
    enforce_gen['eval'], clean_eval['eval'] = args.eval_generalized_set, args.eval_clean_set
    enforce_gen['triplets'], enforce_gen['words'] = dt.get_enforce_gen(enforce_gen['eval'])
    clean_eval['triplets'], clean_eval['words'] = dt.get_enforce_gen(clean_eval['eval'])

    print('Getting training data...')
    X, X_extra, y, y_pixl, X_extra_enf_gen, X_enf_gen, y_enf_gen, y_enf_gen_pixl, rel_ids, OBJ_ctr_sd, OBJ_ctr_sd_enf_gen, \
    EMBEDDINGS, TRAIN_relevant = dt.get_data(args.model_type, TRAIN, words, EMB, enforce_gen, args.n_side_pixl, token_dict, paths)

    # Get folds
    kf = dt.get_folds(X['subj'].shape[0], args.n_folds)

    # --- INITIALIZE performance measures --- #
    PERF, PERF_clean, PERF_enf_gen = {},{},{}
    PERF['train'], PERF['test'], PERF_clean['train'], PERF_clean['test'] = {},{},{},{}
    for method_full in args.method_compare:
        PERF['train'][method_full], PERF['test'][method_full] = {},{}
        PERF_clean['train'][method_full], PERF_clean['test'][method_full] = {},{}
        PERF_enf_gen[method_full] = {}
        for meas in perf_measures:
            PERF['train'][method_full][meas], PERF['test'][method_full][meas] = [],[]
            PERF_clean['train'][method_full][meas], PERF_clean['test'][method_full][meas] = [],[]
            PERF_enf_gen[method_full][meas] = []

    idx_clean_train, idx_clean_test = [],[]
    for fold_count, (train_idx, test_idx) in enumerate(kf): # FOLDS loop
        
        # --- TRAIN and TEST data (splits) --- #
        # This aux function isn't elegant, but we don't want to triplicate y_pixl with train and test splits. Takes too much memory
        X_train, X_test, X_extra_train, X_extra_test, y_train, y_test, OBJ_ctr_sd_train, \
        OBJ_ctr_sd_test = dt.aux_get_train_test_splits(X, X_extra, y, OBJ_ctr_sd, train_idx, test_idx)
        aux_train_idx = train_idx if args.model_type == 'PIX' else 0
        aux_test_idx = test_idx if args.model_type == 'PIX' else 0
            
        # --- get CLEAN_train and CLEAN_test INDICES --- #
        if clean_eval['eval'] is not None:
            idx_clean_train, idx_clean_test = dt.get_CLEAN_train_test_idx(TRAIN_relevant, train_idx, test_idx, clean_eval)

        for method in args.method_compare: # METHODS LOOP

            print('=========================================')
            print('=======>> ' + method + ' <<=======')
            print('=========================================')

            # Initialize model object
            model = models.NeuralnetModel(args, method, par_learning)
            
            # --- LEARN the model --- #
            model.method_learn(X_train, X_extra_train, y_train, y_pixl[aux_train_idx], EMBEDDINGS, paths)
        
            '''print('Save model into json')
            import json
            model_json = model.keras_model.to_json()
            with open(saveFolder + '/MODEL_' + method_full + '_fld_' + str(fold_count + 1) + '.json', "w") as json_file:
                json_file.write(model_json)
            '''
            
            # --- PREDICT --- #
            y_pred_train = model.model_predict(X_train, X_extra_train, y_train)
            y_pred_test = model.model_predict(X_test, X_extra_test, y_train)

            #model.clear_model()
            
            if enforce_gen['eval'] is not None:
                y_pred_enf_gen = model.model_predict(X_enf_gen, X_extra_enf_gen, y_train)
                
            # --- EVALUATE performance --- #
            PERF_DICT_test = et.evaluate_perf(y_test, y_pixl[aux_test_idx], y_pred_test, OBJ_ctr_sd_test, perf_measures, args.model_type)
            PERF_DICT_train = et.evaluate_perf(y_train, y_pixl[aux_train_idx], y_pred_train, OBJ_ctr_sd_train, perf_measures, args.model_type)
            

            if (clean_eval['eval'] is not None) and (idx_clean_test != []) and (idx_clean_train != []):
                aux_idx_clean_train = idx_clean_train if args.model_type == 'PIX' else 0
                aux_idx_clean_test = idx_clean_test if args.model_type == 'PIX' else 0
                PERF_DICT_clean_train = et.evaluate_perf(y_train[idx_clean_train], y_pixl[aux_train_idx][aux_idx_clean_train], y_pred_train[idx_clean_train], OBJ_ctr_sd_train[idx_clean_train], perf_measures, args.model_type)
                PERF_DICT_clean_test = et.evaluate_perf(y_test[idx_clean_test], y_pixl[aux_test_idx][aux_idx_clean_test], y_pred_test[idx_clean_test], OBJ_ctr_sd_test[idx_clean_test], perf_measures, args.model_type)
            if enforce_gen['eval'] is not None:
                PERF_DICT_enf_gen = et.evaluate_perf(y_enf_gen, y_enf_gen_pixl, y_pred_enf_gen, OBJ_ctr_sd_enf_gen, perf_measures, args.model_type)

            # --- append --- #
            for meas in perf_measures:
                PERF['train'][method_full][meas].append(PERF_DICT_train[meas])
                PERF['test'][method_full][meas].append(PERF_DICT_test[meas])
                if (clean_eval['eval'] is not None) and (idx_clean_test != []) and (idx_clean_train != []):
                    PERF_clean['train'][method_full][meas].append(PERF_DICT_clean_train[meas])
                    PERF_clean['test'][method_full][meas].append(PERF_DICT_clean_test[meas])
                    print ('method==> ' + method + ' || ' + meas + '_CLEAN_ts= ' + str(PERF_DICT_clean_test[meas]) + ' | ' + meas + '_CLEAN_tr= ' + str(PERF_DICT_clean_train[meas]))
                if enforce_gen['eval'] is not None:
                    PERF_enf_gen[method_full][meas].append(PERF_DICT_enf_gen[meas])
                    print ('method==> ' + method + ' || ' + meas + '_GEN= ' + str(PERF_DICT_enf_gen[meas]))
                print ('method==> ' + method + ' || ' + meas + '_test= ' + str(PERF_DICT_test[meas]) + ' | ' + meas + '_train= ' + str(PERF_DICT_train[meas]))

            # -- write individual predictions -- #
            if args.save_indiv_predictions == True:
                indiv_predDir = saveFolder + '/INDIV_' + method_full + '_fld_' + str(fold_count + 1) + '.csv'
                wd.write_indiv_predictions(y_pred_test, OBJ_ctr_sd_test, args.model_type, 0.1, indiv_predDir)
                if enforce_gen['eval'] is not None:
                    wd.write_indiv_predictions(y_pred_enf_gen, OBJ_ctr_sd_enf_gen, args.model_type, 0.1, indiv_predDir.replace('INDIV', 'INDIV-GEN'))
                if (clean_eval['eval'] is not None) and (idx_clean_test != []) and (idx_clean_train != []):
                    wd.write_indiv_predictions(y_pred_test[idx_clean_test], OBJ_ctr_sd_test[idx_clean_test], args.model_type, 0.1, indiv_predDir.replace('INDIV', 'INDIV-CLEAN_TST'))

            # -- store model weights -- #
            if args.save_model == True:
                import h5py
                model.keras_model.save_weights(saveFolder + '/MODEL_' + method_full + '_fld_' + str(fold_count + 1) + '.h5py')

            # --- write results --- #
            wd.write_results_all(PERF, args.method_compare, perf_measures, saveFolder + '/TRAIN-TEST.csv')
            if enforce_gen['eval'] is not None:
                wd.write_results_enf_gen(PERF_enf_gen, args.method_compare, perf_measures, saveFolder + '/GEN.csv')
            if (clean_eval['eval'] is not None) and (idx_clean_test != []) and (idx_clean_train != []):
                wd.write_results_all(PERF_clean, args.method_compare, perf_measures, saveFolder + '/CLEAN.csv')


if __name__ == "__main__":
     main()
