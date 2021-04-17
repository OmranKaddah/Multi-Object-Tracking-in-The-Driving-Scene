
from __future__ import print_function
import pandas 
import matplotlib; matplotlib.use('Agg')
import sys, os, copy, math, numpy as np, matplotlib.pyplot as plt
from tabulate import tabulate
from munkres import Munkres
from collections import defaultdict
try:
    from ordereddict import OrderedDict # can be installed using pip
except:
    from collections import OrderedDict # only included from python 2.7 on

import mailpy
from box_util import boxoverlap, box3doverlap
from evaluate_kitti3dmot_model import *


def run(*argv):
    """
    Parameters:
        argv = [signture, dir ,"3D/2D","Baseline","Your model*", subfolder]
            signture:
            
            3D/2D:

            Baseline: Name of basline
                must match the folder where the results are stored.
                tracked obejects are not in different
                subfolders
            
            Your model/*: name of your model
                must match the folder where the results are stored.
                Add * at the end if tracked obejects are not in different
                subfolders
            
            subfolder: (optional)
                to store in a subfoler
    """
    num_sample_pts = 41.0
    # check for correct number of arguments. if user_sha and email are not supplied,
    # no notification email is sent (this option is used for auto-updates)
    if  len(argv)<5:
      print("Usage: python eval_kitti3dmot.py result_sha ?D(e.g. 2D or 3D)")
      sys.exit(1);

    # get unique sha key of submitted results
    result_sha = argv[0]
    obj_tracked = result_sha.split("_")[0]
    dir = argv[1]
    dt_typ= result_sha.split("_")[3]
    baseline_name = argv[3]
    mail = mailpy.Mail("")
    D = argv[2]
    # 

    if argv[2] == '2D':
        eval_3diou, eval_2diou = False, True      # eval 2d
    elif argv[2] == '3D':
        eval_3diou, eval_2diou = True, False        # eval 3d
    else:
        print("Usage: python eval_kitti3dmot.py result_sha ?D(e.g. 2D or 3D)")
        sys.exit(1);            


    # evaluate results


    if len(argv) ==6:
        table_name = 'results/{}/{}/results_{}_{}_table_{}.csv'.format(dir,argv[5],obj_tracked,dt_typ, D)
    else:
        table_name = 'results/{}/results_{}_{}_table_{}.csv'.format(dir,obj_tracked,dt_typ, D)

    if os.path.exists(table_name):
        df  = pandas.read_csv(table_name)
        if not (df["Model"] == baseline_name).any():
            print("Evaluating baseline")
            success, baseline, base_avgs = evaluate(result_sha, dir,baseline_name, mail,eval_3diou,eval_2diou)
            df.loc[len(df.index)] = [baseline_name,baseline.sMOTA, baseline.MOTA, baseline.MOTP, baseline.MT, baseline.ML, baseline.id_switches, baseline.fragments, 
                baseline.F1, baseline.precision, baseline.recall, baseline.FAR, baseline.tp, baseline.fp, baseline.fn,
                base_avgs[0], base_avgs[1], base_avgs[2]]
    else:
        print("Evaluating baseline :")
        success, baseline, base_avgs = evaluate(result_sha, dir, baseline_name, mail,eval_3diou,eval_2diou)
        # basline_data = [[baseline_name],[baseline.sMOTA], [baseline.MOTA], [baseline.MOTP], [baseline.MT], [baseline.ML], [baseline.id_switches], [baseline.fragments], 
        #     [baseline.F1], [baseline.precision], [baseline.recall], [baseline.FAR], [baseline.tp], [baseline.fp], [baseline.fn],
        #     [base_avgs[0]], [base_avgs[1]], [base_avgs[2]]]
        cols = ["Model","sMOTA", "MOTA", "MOTP", "MT", "ML", "IDS",  "FRAG"  ,"F1" ,"Prec",  "Recall",  "FAR",  "TP", "FP","FN", "sAMOTA", "AMOTA", "AMOTP"]

        df = pandas.DataFrame( columns =cols)
        df.loc[len(df.index)] = [baseline_name,baseline.sMOTA, baseline.MOTA, baseline.MOTP, baseline.MT, baseline.ML, baseline.id_switches, baseline.fragments, 
            baseline.F1, baseline.precision, baseline.recall, baseline.FAR, baseline.tp, baseline.fp, baseline.fn,
            base_avgs[0], base_avgs[1], base_avgs[2]]



        

    other_name = argv[4]
    mail = mailpy.Mail("")
    print("Evaluating "+other_name[:-1] +" :")
    success, other_model, om_avgs = evaluate(result_sha, dir, other_name, mail,eval_3diou,eval_2diou)

    new_row = [other_name[:-1],other_model.sMOTA, other_model.MOTA, other_model.MOTP, other_model.MT, other_model.ML, other_model.id_switches, other_model.fragments, \
            other_model.F1, other_model.precision, other_model.recall, other_model.FAR, other_model.tp, other_model.fp, other_model.fn,\
            om_avgs[0], om_avgs[1], om_avgs[2]]
    df.loc[len(df.index)] = new_row
    print(df.loc[(df['Model'] == baseline_name) | (df['Model'] == other_name[:-1])])
    string_format = df.to_latex(index=False)

    if len(argv) ==6:
        table_name = '.results/{}/{}/comparison_{}_{}_latex_{}.txt'.format(dir,argv[5],obj_tracked,dt_typ, D)
    else:
        table_name = 'results/{}/comparison_{}_{}_latex_{}.txt'.format(dir,obj_tracked,dt_typ, D)


    print(string_format,  file=open(table_name, 'w'))

    if len(argv) ==6:
        table_name = 'results/{}/{}/results_{}_{}_table_{}.csv'.format(dir,argv[5],obj_tracked,dt_typ, D)

    else:
        table_name = 'results/{}/results_{}_{}_table_{}.csv'.format(dir,obj_tracked,dt_typ, D)

    file_ = open(table_name,"w") 
    df.to_csv(file_, index = False, header=True)
    return other_model.MOTA, other_model.MOTP

if __name__ == "__main__":
    run()