import numpy as np

# get overlap taking into account a certain don\'t care percentage
def _overlap(pred, gt, DC):
    '''
    (Internal function) See overlap(s1, s2, params=None).
    :param pred:
    :param gt:
    :param DC:
    :return:
    '''
 
    pred = np.where(pred>0, 1, 0)
    gt = np.where(gt>0, 1, 0)
   
    transitions = np.abs(np.diff(gt)) #will get a 1 in the transition points
    if gt[0] == 1: transitions[0] = 1
    if gt[-1] == 1: transitions[-1] = 1
 
    transitions = np.where(transitions==1)[0].tolist() #filter them
 
    if DC > 0: #Apply don\'t care
        for t in transitions:
            if t-DC > 0:
                if t+DC <= gt.shape[0]:
                    pred[t-DC:t+DC] = 0
                    gt[t-DC:t+DC] = 0
                else:
                    pred[t-DC:] = 0
                    gt[t-DC:] = 0
            elif t-DC < 0:
                if t+DC <= gt.shape[0]:
                    pred[:t+DC] = 0
                    gt[:t+DC] = 0
                else:
                    pred[:] = 0
                    gt[:] = 0
 
 
    I = np.sum((pred*gt))
 
    U = np.sum(np.where((pred+gt) > 0, 1, 0))
 
    if U == 0 and I == 0: return 1
 
    return I/U

def overlap(s1, s2, params=None):
    '''
    Computes the overlap of two matrices per column
    :param s1: first matrix
    :param s2: second matrix
    :param params:  dict with a dict mapping labels to columns (e.g. {'labels':{'label1':0, 'labeln':n}})
                    None for using indices as names.
    :return: Overlaps
    '''
    if params is None or 'labels' not in params.keys():
        overlaps = dict([(i,0) for i in range(s1.shape[1])])
        labels = dict([(i,i) for i in range(s1.shape[1])])
    else:
        labels = params['labels']
        overlaps = dict([(label,0) for label in labels])

    if params is None or 'DC' not in params.keys():
        DC = 0
    else:
        DC = params['DC']
        
    for label in labels:
        overlaps[label] = _overlap(s1, s2, DC)
        # I = np.sum(s1[:,labels[label]]*s2[:,labels[label]])
        # U = np.sum(np.where((s1[:,labels[label]]+s2[:,labels[label]])>=1, 1, 0))
        # if U == 0: overlaps[label] = np.nan
        # else: overlaps[label] = I/U
    return overlaps

def f1_array(s1, s2, params = None):

    '''
    computes the F1 score of two matrices per column
    :param s1: first matrix
    :param s2: second matrix
    :param params:  Dict with:
                       - Label 'labels':
                            A dict mapping labels to columns (e.g. {'labels':{'label1':0, 'labeln':n}})
                            Do not include the key 'labels' for using indices as names.
                       - Label 'min_overlap':
                            The minimum overlap required for considering an activation a TP
                            Do not include the key 'labels' for using min_overlap=0.2.

    :return: F1 scores
    '''
#     if params is None or 'labels' not in params.keys():
#         F1 = dict([(i,0) for i in range(s1.shape[1])])
#         labels = dict([(i,i) for i in range(s1.shape[1])])
#     else:
#         labels = params['labels']
#         F1 = dict([(label,0) for label in labels])

    if params is None or 'min_overlap' not in params.keys():
        min_overlap = 0.2
    else:
        min_overlap = params['min_overlap']
        

    gt = s2
    pred = s1
    gt_diff = np.diff(gt)
    gt_activations = []
    inits = np.where(gt_diff==1)[0] # inits
    ends = np.where(gt_diff==-1)[0] # ends
    if len(inits) != len(ends):
        print(inits)
        print(ends)
        return np.nan

    for i in range(len(inits)): gt_activations.append((inits[i]+1, ends[i]))

    actual_act = len(inits)

    pred_diff = np.diff(pred)
    pred_activations = []
    inits = np.where(pred_diff==1)[0] # inits
    ends = np.where(pred_diff==-1)[0] # ends

    if inits.shape[0] > ends.shape[0]: ends = np.concatenate((ends,np.array([pred.shape[0]])))
    if inits.shape[0] < ends.shape[0]: inits = np.concatenate((np.array([0]),inits))

    for i in range(len(inits)): pred_activations.append((inits[i], ends[i]))

    closest_interval = {}

    num_act = len(inits)
    TP = 0
    FN = 0
    FP = 0
    for p in pred_activations:
        tmp_pred = np.zeros(pred.shape[0])
        tmp_pred[p[0]:p[1]] = 1
        max_intersection = float('-inf')
        best_act = -1
        match = 0
        for act in gt_activations:
            tmp_gt = np.zeros(gt.shape[0])
            tmp_gt[act[0]:act[1]] = 1
            I = np.sum(tmp_pred*tmp_gt)
            if I > 0:
                match = 1
            U = float((np.where((tmp_pred+tmp_gt) > 0)[0]).shape[0])
            if I/U > max_intersection and I/U > min_overlap:
                max_intersection = I/U
                best_act = act

        if best_act != -1:
            try:
                closest_interval[best_act].append((p,max_intersection))
            except:
                closest_interval[best_act] = [(p,max_intersection)]
        else:
            FP += 1

    if len(closest_interval.keys()) == 0: TP = 0
    else:
        for activation in closest_interval: TP += 1


    FN = actual_act-TP

    if TP == 0 and FP == 0 and FN == 0:
        F1 = 1
    elif TP == 0 and (FP != 0 or FN != 0):
        F1 = 0
    else:
        if TP+FP == 0: precision = 1
        else: precision = TP / float(TP+FP)
        if TP+FN == 0: recall = 1
        else: recall = TP / float(TP+FN)
        F1 = (2*precision*recall) / float(precision+recall)
        
    return F1

def f1(s1, s2, params = None):
    '''
    computes the F1 score of two matrices per column
    :param s1: first matrix
    :param s2: second matrix
    :param params:  Dict with:
                       - Label 'labels':
                            A dict mapping labels to columns (e.g. {'labels':{'label1':0, 'labeln':n}})
                            Do not include the key 'labels' for using indices as names.
                       - Label 'min_overlap':
                            The minimum overlap required for considering an activation a TP
                            Do not include the key 'labels' for using min_overlap=0.2.

    :return: F1 scores
    '''
    if params is None or 'labels' not in params.keys():
        F1 = dict([(i,0) for i in range(s1.shape[1])])
        labels = dict([(i,i) for i in range(s1.shape[1])])
    else:
        labels = params['labels']
        F1 = dict([(label,0) for label in labels])

    if params is None or 'min_overlap' not in params.keys():
        min_overlap = 0.2
    else:
        min_overlap = params['min_overlap']
        
    print(labels)
    for label in labels:
        gt = s1[:, labels[label]]
        pred = s2[:, labels[label]]
        gt_diff = np.diff(gt)
        gt_activations = []
        inits = np.where(gt_diff==1)[0] # inits
        ends = np.where(gt_diff==-1)[0] # ends

        for i in range(len(inits)): gt_activations.append((inits[i]+1, ends[i]))

        actual_act = len(inits)

        pred_diff = np.diff(pred)
        pred_activations = []
        inits = np.where(pred_diff==1)[0] # inits
        ends = np.where(pred_diff==-1)[0] # ends

        if inits.shape[0] > ends.shape[0]: ends = np.concatenate((ends,np.array([pred.shape[0]])))
        if inits.shape[0] < ends.shape[0]: inits = np.concatenate((np.array([0]),inits))

        for i in range(len(inits)): pred_activations.append((inits[i], ends[i]))

        closest_interval = {}

        num_act = len(inits)
        TP = 0
        FN = 0
        FP = 0
        print(pred_activations)
        for p in pred_activations:
            tmp_pred = np.zeros(pred.shape[0])
            tmp_pred[p[0]:p[1]] = 1
            max_intersection = float('-inf')
            best_act = -1
            match = 0
            for act in gt_activations:
                tmp_gt = np.zeros(gt.shape[0])
                tmp_gt[act[0]:act[1]] = 1
                I = np.sum(tmp_pred*tmp_gt)
                if I > 0:
                    match = 1
                U = float((np.where((tmp_pred+tmp_gt) > 0)[0]).shape[0])
                if I/U > max_intersection and I/U > min_overlap:
                    max_intersection = I/U
                    best_act = act

            if best_act != -1:
                try:
                    closest_interval[best_act].append((p,max_intersection))
                except:
                    closest_interval[best_act] = [(p,max_intersection)]
            else:
                FP += 1

        if len(closest_interval.keys()) == 0: TP = 0
        else:
            for activation in closest_interval: TP += 1


        FN = actual_act-TP
        
        if TP == 0 and FP == 0 and FN == 0:
            F1[label] = 1
        elif TP == 0 and (FP != 0 or FN != 0):
            F1[label] = 0
        else:
            if TP+FP == 0: precision = 1
            else: precision = TP / float(TP+FP)
            if TP+FN == 0: recall = 1
            else: recall = TP / float(TP+FN)
            F1[label] = (2*precision*recall) / float(precision+recall)
    return F1


if __name__ == "__main__":

    #
    # TODO (aclapes@gmail.com): impelement some illustrative testing code here
    #

    quit()