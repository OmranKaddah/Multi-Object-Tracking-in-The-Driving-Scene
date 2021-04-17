import os
from sys import exit
import subprocess
from evaluation.comp_rec_v2 import run
import datetime
import shutil
import random
import numpy as np
import pandas
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default=['56_4_63_5_grd_coup_6_coup_55'],nargs='+', help="Pass name of the expirments to be replicated,\
                    the experiments should have their own folder with .txt paramters file that holds the same name of the folder.")
parser.add_argument('--tracking_output', type=str, default='test', help="directory of tracking outputs. such visual results, ")
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
parser.add_argument('--dimensions', type=str, default=["3D","2D"],nargs='+',help="List the dimensions for which tracking evalaution is performed.")
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

MUST_BE_INT = ["tracking_exp_decay", "max_hole_size", "not_selected_tolerance",
             "min_observations_needed_to_init_hypothesis", "tracking_temporal_window_size"]

hyp = {}

hyp["association_appearance_model_weight"]=0
hyp["association_weight_distance_from_camera_param"]=0
hyp["IoU3D_model_weight"] = 0
hyp["gaiting_appearance_threshold"]=0
hyp["gaiting_IOU_threshold"]=0
hyp["gaiting_mh_distance_threshold"]=0
hyp["gaiting_min_association_score"]=0
hyp["gaiting_size_2D"]=0
hyp["tracking_e1"]=0
hyp["tracking_e3"]=0
hyp["tracking_e4"]=0
hyp["id_handling_overlap_threshold"]=0
hyp["tracking_exp_decay"]=0
hyp["max_hole_size"]= 0
hyp["tracking_temporal_window_size"]=0
hyp["hole_penalty_decay_parameter"]=0
hyp["min_observations_needed_to_init_hypothesis"]=0
hyp["tracking_single_detection_hypo_threshold"]=0



def exp(Experiment_name, hyp_param, df, PHASE, index =-1):
    Basline_Name="AB3DMOT"
    DATASET_NAME="kitti"
    debug_level=3

    DATASET=DATASET_DIR+DATASET_NAME+"/{}".format("training")
    PREPROC=DATASET_DIR+DATASET_NAME+"/{}/preproc".format("training")
    OUT_DIR=DATASET_DIR+DATASET_NAME+"/{}/{}".format("training",Experiment_name)
    # Possible modes: 'detection', 'detection_shape'
    # The latter is considered obsolete; this option is still here for eval purposes
    MODE="detection" # detection_shape
    CIWT="/usr/stud/kaddah/ciwt_gd/build/apps/CIWTApp"

    # Specify detectors (should correspond to detector dir names and config names)
    # DETS=['det_02_regionlets', 'det_02_3DOP', 'pointrcnn']
    DETS=['pointrcnn']
    end_frames=["153", "446", "232", "143", "313", "296", "269", "799", "389", "802", "293", "372", "77", "339", "105", "375", "208", "144", "338", "1058", "836"]


    if DATASET_NAME=="kitti":
        if PHASE=="training":
            SEQUENCES = [2, 6, 7, 8, 10, 13, 14, 16, 18]

        elif PHASE=="validation":
            SEQUENCES = [0,1,3,4,5,9,11,12,15,17,19,20]

    if not os.path.exists(SCRIPT_DIR+"/results/"+Experiment_name):
        if PHASE == "validation":
            shutil.copytree(SCRIPT_DIR+"/results/ready_results_baseline_val",SCRIPT_DIR+"/results/"+Experiment_name)
        else:
            shutil.copytree(SCRIPT_DIR+"/results/ready_results_baseline",SCRIPT_DIR+"/results/"+Experiment_name)
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

            # if os.path.exists(SCRIPT_DIR+"/results/"+Experiment_name):
            #     shutil.rmtree(SCRIPT_DIR+"/results/"+Experiment_name, ignore_errors=True)
            # shutil.copytree(SCRIPT_DIR+"/results/ready_results_baseline/",SCRIPT_DIR+"/results/"+Experiment_name+"/")

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
                                "--association_appearance_model_weight", str(hyp_param["association_appearance_model_weight"]) ,
                                "--IoU3D_model_weight", str(hyp_param["IoU3D_model_weight"]) ,
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
                                # "--not_selected_tolerance", str(hyp_param["not_selected_tolerance"]) ,
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
        shutil.move(OUT_DIR+"/"+DET+"/data",SCRIPT_DIR+"/results/"+Experiment_name+"/CIWT_Pointrcnn/val/data")
    

    # Determine sequences here

    if PHASE =="training":
        SEQUENCES = [2, 6, 7, 8, 10, 13, 14, 16, 18]

    else:
        SEQUENCES = [0,1,3,4,5,9,11,12,15,17,19,20]


    seqs_frames = [SEQUENCES, [ int(end_frames[s])+1 for s in SEQUENCES]]
    
    tracked = ['car', 'ped']
    DIMENSION = ['3D', '2D']
    for dim in DIMENSION:
        for obj in  tracked:
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
    return  df


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

tracked = ['car','ped']
DIMENSION =  ['3D', '2D']
if not os.path.exists("./results"):
    print("Enter the paramters files in reevaluated folder. Exiting the program..")
    
outputdir="./results"

df = {}
vdf = {}
for dim in DIMENSION:
    df[dim] = {}
    vdf[dim]  ={}
    for obj in tracked:
        table_name = "results/ready_results_baseline/results_{}_tr_table_{}.csv".format(obj,dim)
        df[dim][obj] =  pandas.read_csv(table_name)
        table_name = "results/ready_results_baseline_val/results_{}_val_table_{}.csv".format(obj,dim)
        vdf[dim][obj]  = pandas.read_csv(table_name)

# "48_coup_20",
# expirements = [ "62_coup_17", "62_coup_2", "56_coup_59", "45_coup_34", "45_coup_35"]
# expirements = ["45_4_dir_coup_92", "49_4_dir_coup_44"]
# expirements = ["48_3_dir_coup_57","63_4_dir_coup_19"]
expirements = paraser_args.name
for expirement in expirements:
    num_exps = 1
    experiment_name_init = expirement
    experiment_name_init_val = expirement + "_val"

    experiment_name = experiment_name_init 
    hyp  = read_param(outputdir+"/"+expirement+"/"+expirement+".txt",hyp, MUST_BE_INT)
    df = exp(experiment_name, hyp, df, "training")
    file1 = open(outputdir+"/"+experiment_name+"/"+experiment_name +".txt","w")



    experiment_name_val = experiment_name_init_val
    try:
        os.mkdir(outputdir+"/"+experiment_name_init_val)
    except:
        pass
    vdf = exp(experiment_name_val, hyp, vdf, "validation")

    settings = ""

    for key in hyp.keys():
        settings += str(key) + ": {} \n".format(hyp[key])


    file1.write(settings)
    file1.close()

    for dim in DIMENSION:

        for obj in tracked:

            string_format = df[dim][obj].to_latex(index=False)

            print(string_format,  file=open(outputdir+"/"+experiment_name+"/results_{}_tr_table_{}.txt".format(obj,dim), 'w'))
            file_ = open(outputdir+"/"+experiment_name+"/results_{}_tr_table_{}.csv".format(obj,dim),"w") 
            df[dim][obj].to_csv(file_, index = False, header=True)
            file_.close()
            string_format = vdf[dim][obj].to_latex(index=False)

            print(string_format,  file=open(outputdir+"/"+experiment_name_val+"/results_{}_val_table_{}.txt".format(obj,dim), 'w'))
            file_ = open(outputdir+"/"+experiment_name_val+"/results_{}_val_table_{}.csv".format(obj,dim),"w") 
            vdf[dim][obj].to_csv(file_, index = False, header=True)
            file_.close()






    # if not os.path.exists(outputdir+experiment_name_init ):
    #     os.mkdir(outputdir+experiment_name_init  )


    # if not os.path.exists(outputdir+experiment_name_init_val ):
    #     os.mkdir(outputdir+experiment_name_init_val )


    # for dim in DIMENSION:
    #     for obj in tracked:
    #         df[dim][obj] = df[dim][obj].sort_values(by=['MOTA'])
    #         if os.path.exists("./results/"+experiment_name_init +"/results_{}_tr_table_{}.txt".format(obj,dim) ):
    #             os.remove("./results/"+experiment_name_init +"/results_{}_tr_table_{}.txt".format(obj,dim))
    #         if os.path.exists("./results/"+experiment_name_init+"/results_{}_tr_table_{}.csv".format(obj,dim) ):
    #             os.remove("./results/"+experiment_name_init +"/results_{}_tr_table_{}.csv".format(obj,dim))
    #         shutil.move(outputdir+"/results_{}_tr_table_{}.txt".format(obj,dim),"./results/"+experiment_name_init +"/" )
    #         shutil.move(outputdir+"/results_{}_tr_table_{}.csv".format(obj,dim), "./results/"+experiment_name_init +"/")

    #         vdf[dim][obj] = vdf[dim][obj].sort_values(by=['MOTA'])
    #         if os.path.exists("./results/"+experiment_name_init+"/results_{}_val_table_{}.txt".format(obj,dim) ):
    #             os.remove("./results/"+experiment_name_init +"/results_{}_val_table_{}.txt".format(obj,dim))
    #         if os.path.exists("./results/"+experiment_name_init +"/results_{}_val_table_{}.csv".format(obj,dim) ):
    #             os.remove("./results/"+experiment_name_init +"/results_{}_val_table_{}.csv".format(obj,dim))
    #         shutil.move(outputdir+"/results_{}_val_table_{}.txt".format(obj,dim),"./results/"+experiment_name_init_val +"/" )
    #         shutil.move(outputdir+"/results_{}_val_table_{}.csv".format(obj,dim), "./results/"+experiment_name_init_val +"/")



    # experiment_name = experiment_name_init 
    # shutil.move(outputdir+"/"+experiment_name,"./results"+"/"+experiment_name_init+"/")


    # experiment_name = experiment_name_init_val 
    # shutil.move(outputdir+"/"+experiment_name,"./results"+"/"+experiment_name_init_val+"/")
