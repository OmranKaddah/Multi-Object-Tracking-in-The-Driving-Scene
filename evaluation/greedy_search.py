import os
import subprocess
from evaluation.comp_rec_v2 import run
import datetime
import shutil
import random
import numpy as np
import pandas
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='test', help="name of the expirments.")
parser.add_argument('--tracking_output', type=str, default='test', help="directory of tracking outputs. such visual results, ")
parser.add_argument('--scale', type=float, default=1, help="scale variance of uniform distribution for hyperparameters")
parser.add_argument('--init_from', type=str, default='',help="if provided, it will initalize the parameters according to the specified .txt file")
parser.add_argument('--debug_level', type=float, default=0, help="  ------------- \n \
                    Debug Levels: \n \
                    ------------- \n \
                    0 - Outputs basically nothing, except relevant error messages and tracking files. \n \
                    1 - Console output, logging. \n \
                    2 - Quantitative evaluation. \n \
                    3 - Most relevant visual results (per-frame, eg. segmentation, tracking results, ...). \n \
                    4 - Point clouds (per-frame), less relevant visual results. \n \
                    5 - Additional possibly relevant frame output (segmentation 3D data, integrated models, ...). \n \
                    >=6 - All possible/necessary debug stuff. Should make everything really really really slow.")
parser.add_argument('--lr', type=float, default=0.93 ,nargs='+',help="Objects that are going to be tracked during the expriments.")
parser.add_argument('--TRACKED', type=str, default=["car","ped"],nargs='+',help="Objects that are going to be tracked during the expriments.")
parser.add_argument('--dimensions', type=str, default=["3D"],nargs='+',help="List the dimensions for which tracking evalaution is performed.")
parser.add_argument('--script_dir', type=str, default="/usr/stud/kaddah/ciwt_gd", help="Directory of the the app.")
parser.add_argument('--dataset', type=str, default="kitti", help="The name of the dataset")
parser.add_argument('--dataset_dir', type=str, default="/usr/stud/kaddah/storage/datasets/", help="directory of dataset")
parser.add_argument('--baseline', type=str, default="AB3DMOT", help="baseline name")
parser.add_argument('--sequences', type=str, default="training", help="baseline name")
paraser_args = parser.parse_args()

PHASE= paraser_args.sequences
TRACKED = paraser_args.TRACKED
DIMENSION = paraser_args.dimensions
SCRIPT_DIR= paraser_args.script_dir
DATASET_NAME= paraser_args.dataset
DATASET_DIR= paraser_args.dataset_dir
baseline = paraser_args.baseline
scale = paraser_args.scale


lr = paraser_args.lr
hyp = {}
sign_dic = {}
change = {}
MUST_BE_INT = ["tracking_exp_decay", "max_hole_size", "not_selected_tolerance",
             "min_observations_needed_to_init_hypothesis", "tracking_temporal_window_size"]
MUST_BE_0_1 = ["association_appearance_model_weight", "IoU3D_model_weight"]


options_by_type= [
    # association options
    [i  for i in range(8)],
    # trackinging options
    [i  for i in range(8,18)]

]
def read_param(dir,hyp_dic, int_list):
    params = open(dir,"r")
    for line in params:
        key, value = line.split(": ")
        if key in int_list:
            value = int(value)
        else:
            value = float(value)
        hyp_dic[key] = value
    return hyp_dic
def init(hyp,first_time = False):
    sign = lambda : random.choices([-1,1])
    # basline_dir="/usr/stud/kaddah/",


    change["association_appearance_model_weight"]=0.1
    change["association_weight_distance_from_camera_param"]=0.05
    change["IoU3D_model_weight"] = 0.1
    change["gaiting_appearance_threshold"]=0.2
    change["gaiting_IOU_threshold"]=0.2
    change["gaiting_mh_distance_threshold"]=1.0
    change["gaiting_min_association_score"]=0.2
    change["gaiting_size_2D"]=0.2
    change["tracking_e1"]=0.8
    change["tracking_e3"]=0.8
    change["tracking_e4"]=3.0
    change["id_handling_overlap_threshold"]=0.1
    change["tracking_exp_decay"]=2
    change["max_hole_size"]= 2
    change["tracking_temporal_window_size"]=1
    change["hole_penalty_decay_parameter"]=0.3
    change["min_observations_needed_to_init_hypothesis"]=1
    change["tracking_single_detection_hypo_threshold"]=0.05


    change["kf_2d_observation_noise"]=0.25
    change["kf_2d_initial_variance"]=5.0
    change["kf_2d_system_noise"]=0.25
    change["kf_3d_observation_noise"]=0.25
    change["kf_3d_initial_variance"]=5.0
    change["kf_3d_system_noise"]=0.25
    change["kf_system_pos_variance_car_x"]=0.25
    change["kf_system_pos_variance_car_z"]=0.25
    change["kf_system_velocity_variance_car_x"]=0.25
    change["kf_system_velocity_variance_car_z"]=0.25
    change["kf_orientation_obs_noise"]=0.05
    change["not_selected_tolerance"]=1


    hyp["kf_2d_observation_noise"]=1.0
    hyp["kf_2d_initial_variance"]=10.0
    hyp["kf_2d_system_noise"]=1.0
    hyp["kf_3d_observation_noise"]=1.0
    hyp["kf_3d_initial_variance"]=10.0
    hyp["kf_3d_system_noise"]=1.0
    hyp["kf_system_pos_variance_car_x"]=1.0
    hyp["kf_system_pos_variance_car_z"]=1.0
    hyp["kf_system_velocity_variance_car_x"]=1.0
    hyp["kf_system_velocity_variance_car_z"]=1.0
    hyp["kf_orientation_obs_noise"]=0.05
    hyp["not_selected_tolerance"]=10

    hyp["association_appearance_model_weight"]=0.4
    hyp["association_weight_distance_from_camera_param"]=0.07
    hyp["IoU3D_model_weight"] = 0.8
    hyp["gaiting_appearance_threshold"]=0.525
    hyp["gaiting_IOU_threshold"]=0.354
    hyp["gaiting_mh_distance_threshold"]=6.52
    hyp["gaiting_min_association_score"]=0.428
    hyp["gaiting_size_2D"]=0.776
    hyp["tracking_e1"]=1.41
    hyp["tracking_e3"]=1.044
    hyp["tracking_e4"]=36.51
    hyp["id_handling_overlap_threshold"]=0.5
    hyp["tracking_exp_decay"]=33
    hyp["max_hole_size"]= 9
    hyp["tracking_temporal_window_size"]=3
    hyp["hole_penalty_decay_parameter"]=1.044
    hyp["min_observations_needed_to_init_hypothesis"]=2
    hyp["tracking_single_detection_hypo_threshold"]=0.75

    if paraser_args.init_from != '':
        hyp  = read_param("./results/"+paraser_args.init_from+"/"+paraser_args.init_from+".txt",hyp, MUST_BE_INT)

    if not first_time: 
        for key in hyp.keys():
            sign_dic[key] = sign()[0]
            hyp[key] = hyp[key] + change[key] * sign_dic[key]
            if key in MUST_BE_INT:
                hyp[key] = int(hyp[key])


def update(selected, toggle = 1):
    for ix, key in enumerate(hyp.keys()):
        if ix in selected:
            change[key] = change[key] * lr
            hyp[key] =  hyp[key] + toggle * change[key] * sign_dic[key]
            if key in MUST_BE_INT:
                hyp[key] = int(hyp[key])
            if hyp[key] <0:
                hyp[key] = 0.01
            if key in MUST_BE_0_1 and hyp[key]>1:
                hyp[key] = 0.94

def exp(Experiment_name, hyp_param, df):
    
    Basline_Name= baseline
    if PHASE == 'training' or PHASE =='validation':
        section = 'training'
    else:
        import sys
        print("this is not a valid sequence name!")
        sys.exit()
    debug_level= paraser_args.debug_level
    DATASET=DATASET_DIR+DATASET_NAME+"/{}".format(section)
    PREPROC=DATASET_DIR+DATASET_NAME+"/{}/preproc".format(section)
    OUT_DIR=DATASET_DIR+DATASET_NAME+"/{}/{}".format(section,paraser_args.tracking_output)
    # Possible modes: 'detection', 'detection_shape'
    # The latter is considered obsolete; this option is still here for eval purposes
    MODE="detection" # detection_shape
    CIWT="/usr/stud/kaddah/ciwt_gd/build/apps/CIWTApp"

    # Specify detectors (should correspond to detector dir names and config names)
    # DETS=['det_02_regionlets', 'det_02_3DOP', 'pointrcnn']
    DETS=['pointrcnn']

    if DATASET_NAME=="kitti":
        end_frames=["153", "446", "232", "143", "313", "296", "269", "799", "389", "802", "293", "372", "77", "339", "105", "375", "208", "144", "338", "1058", "836"]
        if PHASE =="training":
            SEQUENCES = [2, 6, 7, 8, 10, 13, 14, 16, 18]

        else :
            SEQUENCES = [0,1,3,4,5,9,11,12,15,17,19,20]
    else:
        import sys
        print("This is not a valid dataset")
        sys.exit()


    # Exec tracker for each detection set
    for DET in DETS:
        # Exec tracker for each sequence
        for SEQ in SEQUENCES:
            SEQNAME="%04d"%(SEQ) 
            print("Process: {}".format(SEQNAME))

            IM_LEFT="{}/image_02/{}/%06d.png".format(DATASET,SEQNAME)
            IM_RIGHT="{}/image_03/{}/%06d.png".format(DATASET,SEQNAME)
            DISP="{}/disparity/disp_{}/%06d_left.png".format(PREPROC,SEQNAME)
            END_FRAME=end_frames[SEQ]
            CALIB="{}/calib/{}.txt".format(DATASET,SEQNAME)
            PROP="{}/cached_proposals/{}/%06d.bin".format(PREPROC,SEQNAME)
            OUT_DIR_DET="{}/{}".format(OUT_DIR,DET)
            DETPATH="{}/detection/{}/{}/%06d.txt".format(PREPROC,DET,SEQNAME)
            DETCFG="{}/cfg/{}.cfg".format(SCRIPT_DIR,DET)

            if os.path.exists(SCRIPT_DIR+"/results/"+Experiment_name):
                shutil.rmtree(SCRIPT_DIR+"/results/"+Experiment_name)
            shutil.copytree(SCRIPT_DIR+"/results/ready_results_baseline",SCRIPT_DIR+"/results/"+Experiment_name)

            args = ["--config", str(DETCFG),
                                "--left_image_path", str(IM_LEFT) ,
                                "--end_frame", str(END_FRAME) ,
                                "--right_image_path", str(IM_RIGHT),
                                "--left_disparity_path", str(DISP) ,
                                "--calib_path", str(CALIB) ,
                                "--object_proposal_path", str(PROP) ,
                                "--detections_path", str(DETPATH),
                                "--output_dir", str(OUT_DIR_DET),
                                "--tracking_mode", str(MODE),
                                "--sequence_name", str(SEQNAME),
                                "--debug_level", str(debug_level) ,
                                "--is_oriented", str(1) ,
                                "--enable_coupling", str(1) ,
                                "--check_exit_to2D_projeciton", str(0) ,
                                "--association_score_is_3DIoU", str(1) 
                                , 
                                "--association_appearance_model_weight", str(hyp_param["association_appearance_model_weight"]),
                                "--IoU3D_model_weight", str(hyp_param["IoU3D_model_weight"]),
                                "--association_weight_distance_from_camera_param", str(hyp_param["association_weight_distance_from_camera_param"]) ,
                                "--gaiting_appearance_threshold", str(hyp_param["gaiting_appearance_threshold"]) ,
                                "--gaiting_IOU_threshold", str(hyp_param["gaiting_IOU_threshold"]) , 
                                "--gaiting_mh_distance_threshold", str(hyp_param["gaiting_mh_distance_threshold"]) ,
                                "--gaiting_min_association_score", str(hyp_param["gaiting_min_association_score"]) , 
                                "--gaiting_size_2D", str(hyp_param["gaiting_size_2D"]) ,
                                "--tracking_e1", str(hyp_param["tracking_e1"]) , 
                                "--tracking_e3", str(hyp_param["tracking_e3"]) , 
                                "--tracking_e4", str(hyp_param["tracking_e4"]) , 
                                "--id_handling_overlap_threshold", str(hyp_param["id_handling_overlap_threshold"]) ,
                                "--tracking_exp_decay", str(hyp_param["tracking_exp_decay"]) ,

                                "--max_hole_size", str(hyp_param["max_hole_size"]) , 
                                "--tracking_temporal_window_size", str(hyp_param["tracking_temporal_window_size"]) ,
                                "--hole_penalty_decay_parameter", str(hyp_param["hole_penalty_decay_parameter"]) , 
                                "--min_observations_needed_to_init_hypothesis", str(hyp_param["min_observations_needed_to_init_hypothesis"]) ,
                                "--tracking_single_detection_hypo_threshold", str(hyp_param["tracking_single_detection_hypo_threshold"])
                                # ,

                                # "--kf_2d_observation_noise", str(hyp["kf_2d_observation_noise"]) ,
                                # "--kf_2d_initial_variance", str(hyp["kf_2d_initial_variance"]) ,
                                # "--kf_2d_system_noise", str(hyp["kf_2d_system_noise"]) ,
                                # "--kf_3d_observation_noise", str(hyp["kf_3d_observation_noise"]) ,
                                # "--kf_3d_initial_variance",str(hyp["kf_3d_initial_variance"]) ,
                                # "--kf_3d_system_noise",str(hyp["kf_3d_system_noise"]) ,
                                # "--kf_system_pos_variance_car_x",str(hyp["kf_system_pos_variance_car_x"]) ,
                                # "--kf_system_pos_variance_car_z",str(hyp["kf_system_pos_variance_car_z"]) ,
                                # "--kf_system_velocity_variance_car_x",str(hyp["kf_system_velocity_variance_car_x"]) ,
                                # "--kf_system_velocity_variance_car_z",str(hyp["kf_system_velocity_variance_car_z"]) ,
                                # "--kf_orientation_obs_noise",str(hyp["kf_orientation_obs_noise"])
                                ]
            subprocess.run([CIWT] + args)
    for DET in DETS:
        shutil.move(OUT_DIR+"/"+DET+"/data",SCRIPT_DIR+"/results"+"/"+Experiment_name+"/CIWT_Pointrcnn/val/")
    
    # Determine sequences here
    if PHASE =="training":
        SEQUENCES = [2, 6, 7, 8, 10, 13, 14, 16, 18]

    else:
        SEQUENCES = [0,1,3,4,5,9,11,12,15,17,19,20]


    seqs_frames = [SEQUENCES, [ int(end_frames[s])+1 for s in SEQUENCES]]
    mota = 0
    motp = 0
    for dim in DIMENSION:
        for obj in  TRACKED:
            argv = []
            argv.append("{}_3d_det_val".format(obj))
            argv.append(Experiment_name)
            argv.append(dim)
            argv.append(df[dim][obj])
            # remove '*' if the trajectories are for each objects are separated in different subfolders
            argv.append("CIWT_Pointrcnn"+'*')
            argv.append(seqs_frames)
            # argv.append(PHASE)
            print("start evaluation")
            
            temp_mota, temp_motp, df[dim][obj] = run(*argv)
            mota += temp_mota
            motp += temp_motp

    if os.path.exists(SCRIPT_DIR+"/results/"+Experiment_name):
        shutil.rmtree(SCRIPT_DIR+"/results/"+Experiment_name)
    mota = mota / len(TRACKED)
    motp = motp / len(TRACKED)
    return mota, motp, df


if not os.path.exists("./results"):
    print("./results folder is not found!")
    import sys
    sys.exit()
    
outputdir="./results"

# loading data frame
tvd = {}
for dim in DIMENSION:
    
    tvd[dim]  ={}
    for obj in TRACKED:
        if PHASE=='training':
            table_name = "results/ready_results_baseline/results_{}_tr_table_{}.csv".format(obj,dim)
        elif PHASE=='validation':
            table_name = "results/ready_results_baseline_val/results_{}_val_table_{}.csv".format(obj,dim)
        else:
            table_name = "results/ready_results_baseline_test/results_{}_test_table_{}.csv".format(obj,dim)

        tvd[dim][obj]  = pandas.read_csv(table_name)

# data frame for linear combination results of tracked object 
lincomb_tvd = {}
for dim in DIMENSION:
    u = None
    # here it does not matter if it is car or ped, because the colmuns are the same
    lincomb_tvd[dim] = pandas.DataFrame(columns=tvd[dim][TRACKED[0]].columns) 
    for obj in TRACKED:
        if u is None:
            u  = tvd[dim][obj].loc[tvd[dim][obj]["Model"]==baseline].iloc[0].values
        else:
            u[1:]  += tvd[dim][obj].loc[tvd[dim][obj]["Model"]==baseline].iloc[0].values[1:]
    u[1:] = u[1:]/len(TRACKED)
    lincomb_tvd[dim].loc[len(lincomb_tvd[dim].index)]  = u


experiment_name_init = paraser_args.name



if not os.path.exists("./results/"+experiment_name_init +"_f"):
    os.mkdir("./results/"+experiment_name_init+ "_f" )

opt_typ = random.choice([0,1])
# selected = len(options_by_type[opt_typ])

init(hyp,True)
pre_mota, pre_motp, tvd = exp(experiment_name_init, hyp, tvd)
# pre_mota = 0.0
# pre_motp = 0.0
lists ={}
best_expriment = 0
permut = None
# init()
# update(options_by_type[opt_typ])
exp_count = 0

for key in hyp.keys():
    print('Now changing the hyperparameter {}'.format(key))
    pre_hyp = 0
    imporved = 0
    pre_hyp = hyp[key]
    sign = 1
    hyp[key] += sign * change[key]
    if key in MUST_BE_INT:
        hyp[key] = int(hyp[key])
    if hyp[key] <0:
        hyp[key] = 0.01
    if key in MUST_BE_0_1 and hyp[key]>1:
        hyp[key] = 0.94
    experiment_name = experiment_name_init+"_coup_{}".format(exp_count)
    mota, motp, tvd = exp(experiment_name, hyp, tvd)
    while True:
        # experiment_name = experiment_name_init+"_coup_{}".format(exp_count)

        file1 = open(SCRIPT_DIR+"/results"+"/"+experiment_name_init+ "_f"+"/"+experiment_name +".txt","w")
        settings = ""
        copy_hyp = hyp
        for key_copy in copy_hyp.keys():
            settings += str(key_copy) + ": {} \n".format(hyp[key_copy])
        file1.write(settings)
        file1.close()
        for dim in DIMENSION:
            u = None
            for obj in TRACKED:
                if u is None:
                    u  = tvd[dim][obj].loc[tvd[dim][obj]["Model"]==experiment_name].iloc[0].values
                else:
                    u[1:]  += tvd[dim][obj].loc[tvd[dim][obj]["Model"]==experiment_name].iloc[0].values[1:]

                string_format = tvd[dim][obj].to_latex(index=False)

                print(string_format,  file=open("results/"+experiment_name_init +"_f/" +"{}_results_{}_val_table_{}.txt".format(paraser_args.name,obj,dim), 'w'))
                file_ = open("results/"+experiment_name_init +"_f/" +"{}_results_{}_val_table_{}.csv".format(paraser_args.name,obj,dim),"w") 
                tvd[dim][obj] = tvd[dim][obj].sort_values(by=['MOTA'],ascending=False)
                tvd[dim][obj].to_csv(file_, index = False, header=True)
                file_.close()
            u[1:] =  u[1:]/len(TRACKED)
            lincomb_tvd[dim].loc[len(lincomb_tvd[dim].index)]  = u
            print(string_format,  file=open("results/"+experiment_name_init +"_f/" +"{}_results_{}_fused_table_{}.txt".format(paraser_args.name,obj,dim), 'w'))
            file_ = open("results/"+experiment_name_init +"_f/" +"{}_results_{}_fused_table_{}.csv".format(paraser_args.name,obj,dim),"w") 
            lincomb_tvd[dim] = lincomb_tvd[dim].sort_values(by=['MOTA'], ascending=False)
            lincomb_tvd[dim].to_csv(file_, index = False, header=True)
            file_.close()
        if mota > pre_mota:
            

            pre_mota = mota 
            pre_motp = motp
            imporved = 1

        else:
            # if it has imporved the results before, then stop searching
            # for this hyperparameter.
            hyp[key] = pre_hyp
            if imporved ==1:
                exp_count += 1
                break
            imporved -=1
            # this means discovering both direction did not end with good results
            if imporved <= -2:
                exp_count += 1
                break
            sign *=-1
        hyp[key] += sign * scale * change[key]
        if key in MUST_BE_INT:
            hyp[key] = int(hyp[key])
        if hyp[key] <0:
            hyp[key] = 0.01
        if key in MUST_BE_0_1 and hyp[key]>1:
            hyp[key] = 0.94
        exp_count += 1
        experiment_name = experiment_name_init+"_coup_{}".format(exp_count)
        mota, motp, tvd = exp(experiment_name, hyp, tvd)
    
        