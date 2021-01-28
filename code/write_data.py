import numpy as np

def writeCSV(words, matrix, saveDir):
    # Writes a column of words and a matrix into a CSV
        import csv
        matrix = np.array(matrix)
        # Build the "joint" matrix
        MATRIX = []
        for i in range(len(words)):
            vec = matrix[i,]
            currentWord = [ words[i] ]
            currentWord.extend(vec)
            MATRIX.append(currentWord)
        # Write the CSV file
        with open(saveDir, "wb") as f:
            writer = csv.writer(f)
            writer.writerows(MATRIX)


def matrix2csv(MATRIX, saveDir):
    '''
    Assumes MATRIX is already in the format [word, vector] (or whatever,
    as far as each row has THE SAME number of columns)
    '''
    import csv as csv
    with open(saveDir, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(MATRIX)



def write_results_all(PERF, method_compare, perf_measures, saveDir):
    # Write a table with results

    MATRIX = []
    first_row = ['METHOD']

    for measure in perf_measures:
        first_row.extend([ measure + '_train' , measure + '_test' ])
    MATRIX.append(first_row)

    for method in method_compare:
        new_row = [ method ]
        for measure in perf_measures:
            if PERF['train'][method][measure] != [] and PERF['test'][method][measure] != []:
                new_row.extend([round(np.mean(PERF['train'][method][measure]), 4), round(np.mean(PERF['test'][method][measure]), 4) ])
            else:
                new_row.extend([ PERF['train'][method][measure], PERF['test'][method][measure] ])
        MATRIX.append(new_row)

    matrix2csv(MATRIX, saveDir)



def write_results_enf_gen(PERF_enf_gen, method_compare, perf_measures, saveDir):
    # Write a table with results

    MATRIX = []
    first_row = ['METHOD']

    for measure in perf_measures:
        first_row.extend([ measure ])
    MATRIX.append(first_row)

    for method in method_compare:
        new_row = [ method ]
        for measure in perf_measures:
            if PERF_enf_gen[method][measure] != []:
                new_row.extend([round(np.mean(PERF_enf_gen[method][measure]),4) ])
            else:
                new_row.extend([ PERF_enf_gen[method][measure] ])
        MATRIX.append(new_row)

    matrix2csv(MATRIX, saveDir)



def write_indiv_predictions( y_pred, OBJ_ctr_sd_test, model_type, undersample_frac, saveDir):
    # Writes the predictions AND the ground truth
    # INPUT: OBJ_ctr_sd_test: is in the format (order of variables) specified in first_row
    # y_pred: is the matrix of predictions (if model_type='REG') or a 3D tensor of matrices of PIXELS (if model_type='PIX')
    # undersample_fract: fraction of examples that are kept

    undersample = False # (only applies to PIX) OPTIONS: True or False (if we want to undersample or not)
    y_pred = np.array(y_pred)
    first_row = ['img_idx','rel_id','subj','rel','obj','subj_sd_x','subj_sd_y', 'subj_ctr_x','subj_ctr_y','obj_sd_x','obj_sd_y','obj_ctr_x','obj_ctr_y' ]

    if model_type == 'REG':
        y_vars = ['obj_sd_x', 'obj_sd_y', 'obj_ctr_x', 'obj_ctr_y']
        first_row.extend(['PRED_' + el for el in y_vars])

    elif model_type == 'PIX':
        # UNDERSAMPLE first (pixel-wise predictions take a lot of space):
        import random
        if undersample == True:
            n_to_get = int(np.floor(undersample_frac * len(y_pred)))
            idx_to_get = random.sample(range(0, len(y_pred)), n_to_get)
            y_pred, OBJ_ctr_sd_test = y_pred[idx_to_get], OBJ_ctr_sd_test[idx_to_get] # undersample both: y_pred and OBJ_ctr_sd_test
        # FLATTEN the matrices (of pixels) into 1D vectors extend the first row with the pixels
        first_row.extend(['v_' + str(el) for el in range(y_pred.shape[1])])

    MATRIX = np.concatenate((OBJ_ctr_sd_test, y_pred), axis=1)
    MATRIX = np.vstack((first_row, MATRIX))
    matrix2csv(MATRIX, saveDir)



def get_folder_name(args):
    import os

    pixl_side = '_px-' + str(args.n_side_pixl) if args.model_type == 'PIX' else ''
    gen = '_generalized-' + str(args.eval_generalized_set) if args.eval_generalized_set is not None else ''
    clean = '_clean-' + str(args.eval_clean_set) if args.eval_clean_set is not None else ''

    # FOLDER (to store diverse results files):
    saveFolder = '../results/' + 'Results#' + args.model_type + str(pixl_side) + '_' + args.relations + gen + clean
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    return saveFolder

