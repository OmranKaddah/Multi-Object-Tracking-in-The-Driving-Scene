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
parser.add_argument('--scale', type=float, default=2, help="scale variance of uniform distribution for hyperparameters")
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

parser.add_argument('--TRACKED', type=str, default=["car","ped"],nargs='+',help="Objects that are going to be tracked during the expriments.")
parser.add_argument('--dimensions', type=str, default=["3D"],nargs='+',help="List the dimensions for which tracking evalaution is performed.")
parser.add_argument('--script_dir', type=str, default="/usr/stud/kaddah/ciwt_gd", help="Directory of the the app.")
parser.add_argument('--dataset', type=str, default="kitti", help="The name of the dataset")
parser.add_argument('--dataset_dir', type=str, default="/usr/stud/kaddah/storage/datasets/", help="directory of dataset")
parser.add_argument('--baseline', type=str, default="AB3DMOT", help="baseline name")

paraser_args = parser.parse_args()
TRACKED = paraser_args.TRACKED
DIMENSION = paraser_args.dimensions
SCRIPT_DIR= paraser_args.script_dir
DATASET_NAME= paraser_args.dataset
DATASET_DIR= paraser_args.dataset_dir
baseline = paraser_args.baseline
scale = paraser_args.scale

lr = 0.93
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
def init(first_time = False):

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


    # change["kf_2d_observation_noise"]=1.0
    # change["kf_2d_initial_variance"]=10.0
    # change["kf_2d_system_noise"]= 1.0
    # change["kf_3d_observation_noise"]= 1.0
    # change["kf_3d_initial_variance"]= 10.0
    # change["kf_3d_system_noise"]=1.0
    # change["kf_system_pos_variance_car_x"]= 1.0
    # change["kf_system_pos_variance_car_z"]= 1.0
    # change["kf_system_velocity_variance_car_x"]= 1.0
    # change["kf_system_velocity_variance_car_z"]=1.0
    # change["kf_orientation_obs_noise"]=0.025
    # change["not_selected_tolerance"]=5

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

    # hyp["kf_2d_observation_noise"]=1.0
    # hyp["kf_2d_initial_variance"]=10.0
    # hyp["kf_2d_system_noise"]= 1.0
    # hyp["kf_3d_observation_noise"]= 1.0
    # hyp["kf_3d_initial_variance"]= 10.0
    # hyp["kf_3d_system_noise"]=1.0
    # hyp["kf_system_pos_variance_car_x"]= 1.0
    # hyp["kf_system_pos_variance_car_z"]= 1.0
    # hyp["kf_system_velocity_variance_car_x"]= 1.0
    # hyp["kf_system_velocity_variance_car_z"]=1.0
    # hyp["kf_orientation_obs_noise"]=0.025
    # hyp["not_selected_tolerance"]=5
    if not first_time: 
        for key in hyp.keys():
            hyp[key] = hyp[key] + random.uniform(-scale *change[key], scale*change[key])
            if key in MUST_BE_INT:
                hyp[key] = int(hyp[key])





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

    # Path to your binary

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
    
    if PHASE =="training":
        SEQUENCES = [2, 6, 7, 8, 10, 13, 14, 16, 18]

    else :
        SEQUENCES = [0,1,3,4,5,9,11,12,15,17,19,20]


    # settings = open(SCRIPT_DIR+"/evaluation/evaluate_tracking.seqmap.val",'w')
    # for SEQ in SEQUENCES:
    #     settings.write("%04d empty 000000 %06d \n"%(SEQ,int(end_frames[SEQ])))
    # settings.close()
    seqs_frames = [SEQUENCES, [ int(end_frames[s])+1 for s in SEQUENCES]]
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
            
            mota, motp, df[dim][obj] = run(*argv)

    if os.path.exists(SCRIPT_DIR+"/results/"+Experiment_name):
        shutil.rmtree(SCRIPT_DIR+"/results/"+Experiment_name)
    return mota, motp, df


if not os.path.exists("./results"):
    print("COiuld not find ./results folder Exiting the program..")
    exit()
outputdir="./results"

# training data frame
tvd = {}
for dim in DIMENSION:
    
    tvd[dim]  ={}
    for obj in TRACKED:

        table_name = "results/ready_results_baseline_val/results_{}_val_table_{}.csv".format(obj,dim)
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


for exp_count in range(100):
    init()
    experiment_name = experiment_name_init+"_coup_{}".format(exp_count)
    _, _, tvd = exp(experiment_name, hyp, tvd)


    
    file1 = open(SCRIPT_DIR+"/results"+"/"+experiment_name_init+ "_f"+"/"+experiment_name +".txt","w")
    settings = ""

    for key in hyp.keys():
        settings += str(key) + ": {} \n".format(hyp[key])

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

# tvd = tvd.sort_values(by=['MOTA'])
# # 
# for dim in DIMENSION:
    
#     for obj in TRACKED:
#         shutil.move("results/{}_results_{}_val_table_{}.txt".format(paraser_args.name,obj,dim),"./results/"+experiment_name_init +"_f/" )
#         shutil.move("results/{}_results_{}_val_table_{}.csv".format(paraser_args.name,obj,dim), "./results/"+experiment_name_init +"_f/")
#         shutil.move("results/{}_results_{}_fused_table_{}.txt".format(paraser_args.name,obj,dim),"./results/"+experiment_name_init +"_f/" )
#         shutil.move("results/{}_results_{}_fused_table_{}.csv".format(paraser_args.name,obj,dim), "./results/"+experiment_name_init +"_f/")
