import data_tools
import numpy as np

def evaluate_perf(y_true, y_pixl_true, y_pred, OBJ_ctr_sd, perf_measures, model_type):
    '''
    Evaluates model performance based on model_type
    '''
    coord_measures = ['R2', 'acc_y', 'F1_y', 'Pear_x', 'Pear_y', 'IoU_m', 'IoU_t' ]
    pixl_measures = ['max_F1_px', 'max_acc_px']

    if model_type == 'REG':
        PERF_DICT = evaluate_perf_COORD(y_true, y_pred, OBJ_ctr_sd, perf_measures, model_type)

    elif model_type == 'PIX':
        PERF_DICT_coord, PERF_DICT_px = {},{}
        if any(el in perf_measures for el in pixl_measures):
            # measures such as acc, F1, etc are not possible with pixels (only pixel-specific measures)
            PERF_DICT_px = evaluate_perf_PIXL(y_pixl_true, y_pred, perf_measures)
        if any(el in perf_measures for el in coord_measures):
            #transform the discrete into continuous (and take maximum activation)
            n_side_pixl = int(np.sqrt(y_pred.shape[1]))
            y_pred = y_pred.reshape( (y_pred.shape[0], n_side_pixl, n_side_pixl ) )
            PRED_obj_ctr_x, PRED_obj_ctr_y = data_tools.pixl_idx2coord_all_examples(y_pred)
            PERF_DICT_coord = evaluate_perf_COORD(y_true, [PRED_obj_ctr_x, PRED_obj_ctr_y], OBJ_ctr_sd, perf_measures, model_type)
        PERF_DICT = merge_two_dicts(PERF_DICT_coord, PERF_DICT_px)

    return PERF_DICT


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


def evaluate_perf_PIXL(y_pixl_true, y_pixl_pred, perf_measures):
    '''
    ONLY allows metrics that use the pixels
    :param y_pixl_true: 2D tensor (n_samples, n_side_pixl*n_side_pixl) (flattened pixel {0,1} mask matrices)
    :param y_pixl_pred: 2D tensor (n_samples, n_side_pixl*n_side_pixl) (flattened pixel activation matrices)
    :return: PERF_DICT
    '''
    print('Evaluating performance at *pixels*...')
    PERF_DICT = {}

    # CLASSIFICATION (VARYING THRESHOLD): (IMPORTANT: now classification is NOT above/below, but pixel accuracy)
    if ('max_acc_px' in perf_measures) or ('max_F1_px' in perf_measures):
        th_max_acc, th_max_F1, max_acc, max_F1 = find_threshold(y_pixl_true, y_pixl_pred)
        if 'max_acc_px' in perf_measures:
            PERF_DICT['max_acc_px'] = max_acc
        if 'max_F1_px' in perf_measures:
            PERF_DICT['max_F1_px'] = max_F1

    return PERF_DICT



def evaluate_perf_COORD(y_true, y_pred, OBJ_ctr_sd, perf_measures, model_type):
    #IMPORTANT: it assumes the order in y_vars is the same order of columns as y_pred (which models_data.py already does)
    import copy

    y_vars = ['obj_sd_x', 'obj_sd_y', 'obj_ctr_x', 'obj_ctr_y']

    PERF_DICT = {}

    #get some abbreviated variables
    obj_ctr_x, obj_ctr_y = y_true[:, y_vars.index('obj_ctr_x')], y_true[:, y_vars.index('obj_ctr_y')]
    obj_sd_x, obj_sd_y = OBJ_ctr_sd[:, 9].astype(np.float), OBJ_ctr_sd[:, 10].astype(np.float)
    subj_ctr_y = OBJ_ctr_sd[:, 8].astype(np.float)

    if model_type == 'REG':
        PRED_obj_ctr_x, PRED_obj_ctr_y = y_pred[:, y_vars.index('obj_ctr_x')], y_pred[:, y_vars.index('obj_ctr_y')]
        PRED_obj_sd_x, PRED_obj_sd_y = y_pred[:, y_vars.index('obj_sd_x')], y_pred[:, y_vars.index('obj_sd_y')]
    elif model_type == 'PIX':
        PRED_obj_ctr_x, PRED_obj_ctr_y = y_pred[0], y_pred[1]
        y_true = np.array([ obj_ctr_x, obj_ctr_y ])

    # Start measures
    if 'R2' in perf_measures:
        import sklearn.metrics
        PERF_DICT['R2'] = sklearn.metrics.r2_score(y_true, y_pred)

    # IoU
    if ('IoU_t' in perf_measures) or ('IoU_m' in perf_measures):
            PERF_DICT['IoU_m'], PERF_DICT['IoU_t'] = bb_intersection_over_union_arrays(obj_ctr_x, obj_ctr_y, obj_sd_x, obj_sd_y, PRED_obj_ctr_x, PRED_obj_ctr_y, PRED_obj_sd_x, PRED_obj_sd_y )

    # CORRELATIONS:
    if ('Pear_x' in perf_measures) or ('Pear_y' in perf_measures):
        Sp_x, Sp_y, Pear_x, Pear_y = correlations(obj_ctr_x, obj_ctr_y, PRED_obj_ctr_x, PRED_obj_ctr_y)
        if 'Pear_x' in perf_measures:
            PERF_DICT['Pear_x'] = Pear_x
        if 'Pear_y' in perf_measures:
            PERF_DICT['Pear_y'] = Pear_y

    # CLASSIFICATION:
    if (('acc_y'in perf_measures) or ('F1_y' in perf_measures)):
        obj_ctr_y_centered = obj_ctr_y - subj_ctr_y
        PRED_obj_ctr_y_centered = PRED_obj_ctr_y - subj_ctr_y

        # Discrecize to {0,1} values
        PRED_obj_ctr_y_centered_01, obj_ctr_y_centered_01 = copy.deepcopy( PRED_obj_ctr_y_centered ), copy.deepcopy(obj_ctr_y_centered)  # IMPORTANT: if you don't create a copy, python just copies the reference
        PRED_obj_ctr_y_centered_01[PRED_obj_ctr_y_centered_01 <= 0] = 0 # threshold of 0.5 (if we've learned with balanced class costs)
        PRED_obj_ctr_y_centered_01[PRED_obj_ctr_y_centered_01 > 0] = 1
        obj_ctr_y_centered_01[obj_ctr_y_centered_01 <= 0] = 0 # threshold of 0.5 (if we've learned with balanced class costs)
        obj_ctr_y_centered_01[obj_ctr_y_centered_01 > 0] = 1
        macro_acc_y, F1_y = classification_f1_acc(obj_ctr_y_centered_01, PRED_obj_ctr_y_centered_01)
        if 'acc_y' in perf_measures:
            PERF_DICT['acc_y'] = macro_acc_y
        if 'F1_y' in perf_measures:
            PERF_DICT['F1_y'] = F1_y

    return PERF_DICT



def correlations( ctr_x, ctr_y, PRED_ctr_x, PRED_ctr_y):
    # Computes the Spearman and Pearson correlations between predicted centers and actual centers
    import scipy.stats
    import os

    Sp_x = scipy.stats.spearmanr(ctr_x, PRED_ctr_x)[0]
    Sp_y = scipy.stats.spearmanr(ctr_y, PRED_ctr_y)[0]
    Pear_x = scipy.stats.pearsonr(ctr_x, PRED_ctr_x)[0]
    Pear_y = scipy.stats.pearsonr(ctr_y, PRED_ctr_y)[0]
    
   
    if os.path.isfile("ctr_x.csv"):
        os.remove("ctr_x.csv")
        os.remove("pred_x.csv")
        os.remove("ctr_y.csv")
        os.remove("pred_y.csv")

    np.savetxt("ctr_x.csv", ctr_x, delimiter=",")
    np.savetxt("pred_x.csv", PRED_ctr_x, delimiter=",")
    np.savetxt("ctr_y.csv", ctr_y, delimiter=",")
    np.savetxt("pred_y.csv", PRED_ctr_y, delimiter=",")
    
    #plt.scatter(ctr_y, PRED_ctr_y)
    #plt.scatter(ctr_x, PRED_ctr_x)
   

    return Sp_x, Sp_y, Pear_x, Pear_y



def bb_intersection_over_union_arrays(obj_ctr_x, obj_ctr_y, obj_sd_x, obj_sd_y, PRED_obj_ctr_x, PRED_obj_ctr_y, PRED_obj_sd_x, PRED_obj_sd_y ):
    # IMPORTANT: top_y coordinate is smaller than low_y (but left_x and right_x otherwise!!!)
    # NOTATION: A=true, B=PRED bounding boxes
    # Note: This function is optimized for time rather than memory

    A_left_x, A_right_x = (obj_ctr_x - obj_sd_x), (obj_ctr_x + obj_sd_x)
    A_low_y, A_top_y = (obj_ctr_y + obj_sd_y), (obj_ctr_y - obj_sd_y)
    B_left_x, B_right_x = (PRED_obj_ctr_x - PRED_obj_sd_x), (PRED_obj_ctr_x + PRED_obj_sd_x)
    B_low_y, B_top_y = (PRED_obj_ctr_y + PRED_obj_sd_y), (PRED_obj_ctr_y - PRED_obj_sd_y)

    # INTERSECTION rectangle:
    INT_left_x = np.nanmax( np.concatenate((np.array([A_left_x]), np.array([B_left_x])), axis=0) , axis=0)
    INT_right_x = np.nanmin( np.concatenate((np.array([A_right_x]),np.array([ B_right_x])), axis=0) , axis=0)
    INT_top_y = np.nanmax( np.concatenate((np.array([A_top_y]), np.array([B_top_y])), axis=0) , axis=0) # top_y coordinate is smaller than low_y
    INT_low_y = np.nanmin( np.concatenate((np.array([A_low_y]),np.array([ B_low_y ])), axis=0) , axis=0) # top_y coordinate is smaller than low_y

    # compute AREA intersection rectangle
    interArea = (INT_right_x - INT_left_x) * (INT_low_y - INT_top_y)

    # UNION: area of both the prediction and ground-truth rectangles
    boxAArea = ( A_right_x - A_left_x) * (A_low_y - A_top_y)
    boxBArea = ( B_right_x - B_left_x) * (B_low_y - B_top_y)

    # compute INTERSECTION OVER UNION
    iou = interArea.astype(np.float) / (boxAArea.astype(np.float) + boxBArea.astype(np.float) - interArea.astype(np.float))
    iou[iou < 0] = 0
    mean_iou = np.mean(iou)
    iou[iou >= 0.5] = 1 # count only as correct detections of IoU>50% (following PASCAL VOC measure)
    iou[iou < 0.5] = 0
    thres_iou = np.mean(iou) # proportion of correct predictions

    return mean_iou, thres_iou



def classification_f1_acc(y_true, y_pred):
    '''
    This function works for both, classification of above/below and pixel performance
    IMPORTANT: y_pred are flattened matrices (if we evaluate pixels) AND already thresholded to {0,1} values
    :param y_true: 1D tensor (n_samples * n_side_pixl*n_side_pixl)
    :param y_pred: 1D tensor (n_samples * n_side_pixl*n_side_pixl)
    :return: Returns macro-averaged F1 and accuracy
    '''
    from sklearn import metrics

    result = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
    acc, F1 = np.mean(result[1]), np.mean(result[2]) # macro average across classes! ([1] is recall, [2] is F1)

    return acc, F1



def find_threshold(y_pixl_true, y_pixl_pred):
    print ('Finding pixel F1 and macro_acc across thresholds...')
    import copy

    thresholds = np.arange(0, 0.4, 0.01)
    acc_px, F1_px = [],[]

    for i in range(len(thresholds)): # thresholds loop
        y_pixl_pred_th = copy.deepcopy(y_pixl_pred)
        y_pixl_pred_th[y_pixl_pred_th < thresholds[i]] = 0
        y_pixl_pred_th[y_pixl_pred_th >= thresholds[i]] = 1
        acc_px_th, F1_px_th = classification_f1_acc( y_pixl_true.reshape(-1), y_pixl_pred_th.reshape(-1) )
        acc_px.append(round(acc_px_th, 4))
        F1_px.append(round(F1_px_th, 4))

    max_acc, max_F1 = np.array(acc_px).max(), np.array(F1_px).max()
    th_max_acc, th_max_F1 = thresholds[np.where( np.array(acc_px) == max_acc )], thresholds[np.where( np.array(F1_px) == max_F1 )]

    return th_max_acc, th_max_F1, max_acc, max_F1
