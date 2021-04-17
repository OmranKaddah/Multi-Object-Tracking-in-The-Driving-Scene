import os
import subprocess
from evaluation.comp_rec_v2 import run
import datetime
import shutil
import random
import numpy as np
import pandas



hyp = {}

hyp["association_appearance_model_weight"]=[ 0.39300000000000007]
hyp["association_weight_distance_from_camera_param"]=[ 0.0665 ]
hyp["IoU3D_model_weight"] = [ 0.807]
hyp["gaiting_appearance_threshold"]=[ 0.539 ]
hyp["gaiting_IOU_threshold"]=[0.368]
hyp["gaiting_mh_distance_threshold"]=[ 6.449999999999999  ]
hyp["gaiting_min_association_score"]=[0.44199999999999995 ]
hyp["gaiting_size_2D"]=[0.7620000000000001]
hyp["tracking_e1"]=[2.21 ]
hyp["tracking_e3"]=[ 1.844]
hyp["tracking_e4"]=[ 33.51  ]
hyp["id_handling_overlap_threshold"]=[0.4 ]
hyp["tracking_exp_decay"]=[31 ]
hyp["max_hole_size"]= [7]
hyp["tracking_temporal_window_size"]=[ 2]
hyp["hole_penalty_decay_parameter"]=[ 1.344]
hyp["min_observations_needed_to_init_hypothesis"]=[1]
hyp["tracking_single_detection_hypo_threshold"]=[  0.7 ]

hyp["kf_2d_observation_noise"]=[0.5]
hyp["kf_2d_initial_variance"]=[10.0]
hyp["kf_2d_system_noise"]=[1.0]
hyp["kf_3d_observation_noise"]=[ 0.5]
hyp["kf_3d_initial_variance"]=[10.0]
hyp["kf_3d_system_noise"]=[ 1.0]
hyp["kf_system_pos_variance_car_x"]=[1.0]
hyp["kf_system_pos_variance_car_z"]=[ 1.0]
hyp["kf_system_velocity_variance_car_x"]=[  1.0]
hyp["kf_system_velocity_variance_car_z"]=[ 1.0]
hyp["kf_orientation_obs_noise"]=[0.025]
hyp["not_selected_tolerance"]=[ 10]
hyp["enable_coupling"]=[1]
# hyp["kf_2d_observation_noise"]=[1.0]
# hyp["kf_2d_initial_variance"]=[10.0]
# hyp["kf_2d_system_noise"]=[1.0]
# hyp["kf_3d_observation_noise"]=[ 1.0]
# hyp["kf_3d_initial_variance"]=[10.0]
# hyp["kf_3d_system_noise"]=[ 1.0]
# hyp["kf_system_pos_variance_car_x"]=[1.0]
# hyp["kf_system_pos_variance_car_z"]=[ 1.0]
# hyp["kf_system_velocity_variance_car_x"]=[  1.0]
# hyp["kf_system_velocity_variance_car_z"]=[ 1.0]
# hyp["kf_orientation_obs_noise"]=[0.05]
# hyp["not_selected_tolerance"]=[ 10]
# hyp["enable_coupling"]=[0]
num_exps = len(hyp["not_selected_tolerance"])

def exp(Experiment_name, hyp_param, df, index, PHASE):
    Basline_Name="AB3DMOT"
    DATASET_NAME="kitti"
    debug_level=3
    SCRIPT_DIR="/usr/stud/kaddah/ciwt_gd"
    DATASET_DIR= "/usr/stud/kaddah/storage/datasets/"
    DATASET=DATASET_DIR+DATASET_NAME+"/{}".format("training")
    PREPROC=DATASET_DIR+DATASET_NAME+"/{}/preproc".format("training")
    OUT_DIR=DATASET_DIR+DATASET_NAME+"/{}/tracking_output".format("training")
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
        else:
            end_frames=["464", "146", "242", "256", "420", "808", "113", "214", "164", "348", "1175", "773", "693", "151", 
                "849", "700", "509", "304", "179", "403", "172", "202", "435", "429", "315", "175", "169", "84", "174"]
            SEQUENCES=range(29)
    # Path to your binary
    

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
            # if PHASE == "validation":
            #     shutil.copytree(SCRIPT_DIR+"/results/ready_results_baseline_val",SCRIPT_DIR+"/results/"+Experiment_name)
            # else:
            #     shutil.copytree(SCRIPT_DIR+"/results/ready_results_baseline",SCRIPT_DIR+"/results/"+Experiment_name)


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
                                "--enable_coupling", str(hyp_param["enable_coupling"][index]) ,
                                "--check_exit_to2D_projeciton", str(0) ,
                                "--association_score_is_3DIoU", str(1) 
                                , 
                                "--association_appearance_model_weight", str(hyp_param["association_appearance_model_weight"][index]) ,
                                "--IoU3D_model_weight", str(hyp_param["IoU3D_model_weight"][index]) ,
                                "--association_weight_distance_from_camera_param", str(hyp_param["association_weight_distance_from_camera_param"][index]) ,
                                "--gaiting_appearance_threshold", str(hyp_param["gaiting_appearance_threshold"][index]) ,
                                "--gaiting_IOU_threshold", str(hyp_param["gaiting_IOU_threshold"][index]) , 
                                "--gaiting_mh_distance_threshold", str(hyp_param["gaiting_mh_distance_threshold"][index]) ,
                                "--gaiting_min_association_score", str(hyp_param["gaiting_min_association_score"][index]) , 
                                "--gaiting_size_2D", str(hyp_param["gaiting_size_2D"][index]) ,
                                "--tracking_e1", str(hyp_param["tracking_e1"][index]) , 
                                "--tracking_e3", str(hyp_param["tracking_e3"][index]) , 
                                "--tracking_e4", str(hyp_param["tracking_e4"][index]) , 
                                "--id_handling_overlap_threshold", str(hyp_param["id_handling_overlap_threshold"][index]) ,
                                "--tracking_exp_decay", str(hyp_param["tracking_exp_decay"][index]) ,
                                "--not_selected_tolerance", str(hyp_param["not_selected_tolerance"][index]) ,
                                "--max_hole_size", str(hyp_param["max_hole_size"][index]) , 
                                "--tracking_temporal_window_size", str(hyp_param["tracking_temporal_window_size"][index]) ,
                                "--hole_penalty_decay_parameter", str(hyp_param["hole_penalty_decay_parameter"][index]) , 
                                "--min_observations_needed_to_init_hypothesis", str(hyp_param["min_observations_needed_to_init_hypothesis"][index]) ,
                                "--tracking_single_detection_hypo_threshold", str(hyp_param["tracking_single_detection_hypo_threshold"][index]) ,

                                "--kf_2d_observation_noise", str(hyp_param["kf_2d_observation_noise"][index]) ,
                                "--kf_2d_initial_variance", str(hyp_param["kf_2d_initial_variance"][index]) ,
                                "--kf_2d_system_noise", str(hyp_param["kf_2d_system_noise"][index]) ,
                                "--kf_3d_observation_noise", str(hyp_param["kf_3d_observation_noise"][index]) ,
                                "--kf_3d_initial_variance",str(hyp_param["kf_3d_initial_variance"][index]) ,
                                "--kf_3d_system_noise",str(hyp_param["kf_3d_system_noise"][index]) ,
                                "--kf_system_pos_variance_car_x",str(hyp_param["kf_system_pos_variance_car_x"][index]) ,
                                "--kf_system_pos_variance_car_z",str(hyp_param["kf_system_pos_variance_car_z"][index]) ,
                                "--kf_system_velocity_variance_car_x",str(hyp_param["kf_system_velocity_variance_car_x"][index]) ,
                                "--kf_system_velocity_variance_car_z",str(hyp_param["kf_system_velocity_variance_car_z"][index]) ,
                                "--kf_orientation_obs_noise",str(hyp_param["kf_orientation_obs_noise"][index])
                                ]
            subprocess.run([CIWT] + args)
    for DET in DETS:
        shutil.move(OUT_DIR+"/"+DET+"/data",SCRIPT_DIR+"/results/"+Experiment_name+"/CIWT_Pointrcnn/val/data")
    

    # Determine sequences here



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



tracked = ['car','ped']
DIMENSION =  ['3D', '2D']
SCRIPT_DIR="/usr/stud/kaddah/ciwt_gd"
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


# experiment_name_init = "48_3_dir_man_coup"
experiment_name_init = "63_4_dir_man_coup_19"
experiment_name_init_val = experiment_name_init + "_val"

outputdir = "./results"



for exp_count in range(num_exps):
    experiment_name = experiment_name_init + "_{}".format(exp_count)

    if not os.path.exists("./results/params_ped_{}/".format(experiment_name)):
        os.mkdir("./results/params_ped_{}/".format(experiment_name))
    if not os.path.exists("./results/"+experiment_name ):
        os.mkdir("./results/"+experiment_name )

    if not os.path.exists("./results/params_ped_{}_val".format(experiment_name)):
        os.mkdir("./results/params_ped_{}_val".format(experiment_name))
    if not os.path.exists("./results/"+experiment_name ):
        os.mkdir("./results/"+experiment_name )

    df = exp(experiment_name, hyp, df, exp_count, "training")
    file1 = open(SCRIPT_DIR+"/results"+"/"+experiment_name+"/"+experiment_name +".txt","w")



    experiment_name_val = experiment_name_init_val + "_{}".format(exp_count)

    vdf = exp(experiment_name_val, hyp, vdf, exp_count, "validation")


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






# for dim in DIMENSION:
#     for obj in tracked:
#         df[dim][obj] = df[dim][obj].sort_values(by=['MOTA'])
#         if not os.path.exists("./results/"+experiment_name_init):
#             os.mkdir("./results/"+experiment_name_init)
#         shutil.move("results/results_{}_tr_table_{}.txt".format(obj,dim),"./results/"+experiment_name_init+"_f" +"/" )
#         shutil.move("results/results_{}_tr_table_{}.csv".format(obj,dim), "./results/"+experiment_name_init +"_f"+"/")
#         vdf[dim][obj] = vdf[dim][obj].sort_values(by=['MOTA'])
#         if not os.path.exists("./results/"+experiment_name_init_val):
#             os.mkdir("./results/"+experiment_name_init_val)
#         shutil.move("results/results_{}_val_table_{}.txt".format(obj,dim),"./results/"+experiment_name_init_val +"_f"+"/" )
#         shutil.move("results/results_{}_val_table_{}.csv".format(obj,dim), "./results/"+experiment_name_init_val +"_f"+"/")


# for exp_count in range(num_exps):
#     experiment_name = experiment_name_init + "_{}".format(exp_count)
#     shutil.move(SCRIPT_DIR+"./results"+"/"+experiment_name,"./results"+"/"+experiment_name_init+"_f"+"/")
# shutil.move("./results/params_ped_{}".format(experiment_name_init), "./results/"+experiment_name_init+"_f" +"/")


# for exp_count in range(num_exps):
#     experiment_name = experiment_name_init_val + "_{}".format(exp_count)
#     shutil.move(SCRIPT_DIR+"./results"+"/"+experiment_name,"./results"+"/"+experiment_name_init_val+"_f"+"/")
# shutil.move("./results/params_ped_{}".format(experiment_name_init_val), "./results/"+experiment_name_init_val+"_f" +"/")
