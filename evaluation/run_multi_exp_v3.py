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
parser.add_argument('--name', type=str, default='test', help="")
parser.add_argument('--tracking_output', type=str, default='test', help="")
parser.add_argument('--scale', type=float, default=2, help="")
parser.add_argument('--debug_level', type=float, default=0, help="")
paraser_args = parser.parse_args()

lr = 0.93
hpy = {}
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

    hpy["association_appearance_model_weight"]=0.4
    hpy["association_weight_distance_from_camera_param"]=0.07
    hpy["IoU3D_model_weight"] = 0.8
    hpy["gaiting_appearance_threshold"]=0.525
    hpy["gaiting_IOU_threshold"]=0.354
    hpy["gaiting_mh_distance_threshold"]=6.52
    hpy["gaiting_min_association_score"]=0.428
    hpy["gaiting_size_2D"]=0.776
    hpy["tracking_e1"]=1.41
    hpy["tracking_e3"]=1.044
    hpy["tracking_e4"]=36.51
    hpy["id_handling_overlap_threshold"]=0.5
    hpy["tracking_exp_decay"]=33
    hpy["max_hole_size"]= 9
    hpy["tracking_temporal_window_size"]=3
    hpy["hole_penalty_decay_parameter"]=1.044
    hpy["min_observations_needed_to_init_hypothesis"]=2
    hpy["tracking_single_detection_hypo_threshold"]=0.75

    # hpy["kf_2d_observation_noise"]=1.0
    # hpy["kf_2d_initial_variance"]=10.0
    # hpy["kf_2d_system_noise"]= 1.0
    # hpy["kf_3d_observation_noise"]= 1.0
    # hpy["kf_3d_initial_variance"]= 10.0
    # hpy["kf_3d_system_noise"]=1.0
    # hpy["kf_system_pos_variance_car_x"]= 1.0
    # hpy["kf_system_pos_variance_car_z"]= 1.0
    # hpy["kf_system_velocity_variance_car_x"]= 1.0
    # hpy["kf_system_velocity_variance_car_z"]=1.0
    # hpy["kf_orientation_obs_noise"]=0.025
    # hpy["not_selected_tolerance"]=5
    if not first_time: 
        for key in hpy.keys():
            hpy[key] = hpy[key] + random.uniform(-paraser_args.scale *change[key], paraser_args.scale*change[key])
            if key in MUST_BE_INT:
                hpy[key] = int(hpy[key])


def update(selected, toggle = 1):
    for ix, key in enumerate(hpy.keys()):
        if ix in selected:
            change[key] = change[key] * lr
            hpy[key] =  hpy[key] + toggle * change[key] * sign_dic[key]
            if hpy[key] <0:
                hpy[key] = 0.01
            if key in MUST_BE_0_1 and hpy[key]>1:
                hpy[key] = 0.94
            if key in MUST_BE_INT:
                hpy[key] = int(hpy[key])




def exp(Experiment_name, hyp_param, df):
    PHASE="training"
    Basline_Name="AB3DMOT"
    DATASET_NAME="kitti"
    debug_level= paraser_args.debug_level
    SCRIPT_DIR="/usr/stud/kaddah/ciwt_gd"
    DATASET_DIR= "/usr/stud/kaddah/storage/datasets/"
    DATASET=DATASET_DIR+DATASET_NAME+"/{}".format(PHASE)
    PREPROC=DATASET_DIR+DATASET_NAME+"/{}/preproc".format(PHASE)
    OUT_DIR=DATASET_DIR+DATASET_NAME+"/{}/{}".format(PHASE,paraser_args.tracking_output)
    # Possible modes: 'detection', 'detection_shape'
    # The latter is considered obsolete; this option is still here for eval purposes
    MODE="detection" # detection_shape
    CIWT="/usr/stud/kaddah/ciwt_gd/build/apps/CIWTApp"

    # Specify detectors (should correspond to detector dir names and config names)
    # DETS=['det_02_regionlets', 'det_02_3DOP', 'pointrcnn']
    DETS=['pointrcnn']

    if DATASET_NAME=="kitti":
        if PHASE=="training" or PHASE=="validation":
            end_frames=["153", "446", "232", "143", "313", "296", "269", "799", "389", "802", "293", "372", "77", "339", "105", "375", "208", "144", "338", "1058", "836"]
            SEQUENCES = [2, 6, 7, 8, 10, 13, 14, 16, 18]

        else:
            end_frames=["464", "146", "242", "256", "420", "808", "113", "214", "164", "348", "1175", "773", "693", "151", 
                "849", "700", "509", "304", "179", "403", "172", "202", "435", "429", "315", "175", "169", "84", "174"]
            SEQUENCES=range(29)
    # Path to your binary
    # e.sequence_name
    # ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']
    # e.n_frames
    # [233, 270, 800, 390, 294, 340, 106, 209, 339]

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

                                # "--kf_2d_observation_noise", str(hpy["kf_2d_observation_noise"]) ,
                                # "--kf_2d_initial_variance", str(hpy["kf_2d_initial_variance"]) ,
                                # "--kf_2d_system_noise", str(hpy["kf_2d_system_noise"]) ,
                                # "--kf_3d_observation_noise", str(hpy["kf_3d_observation_noise"]) ,
                                # "--kf_3d_initial_variance",str(hpy["kf_3d_initial_variance"]) ,
                                # "--kf_3d_system_noise",str(hpy["kf_3d_system_noise"]) ,
                                # "--kf_system_pos_variance_car_x",str(hpy["kf_system_pos_variance_car_x"]) ,
                                # "--kf_system_pos_variance_car_z",str(hpy["kf_system_pos_variance_car_z"]) ,
                                # "--kf_system_velocity_variance_car_x",str(hpy["kf_system_velocity_variance_car_x"]) ,
                                # "--kf_system_velocity_variance_car_z",str(hpy["kf_system_velocity_variance_car_z"]) ,
                                # "--kf_orientation_obs_noise",str(hpy["kf_orientation_obs_noise"])
                                ]
            subprocess.run([CIWT] + args)
    for DET in DETS:
        shutil.move(OUT_DIR+"/"+DET+"/data",SCRIPT_DIR+"/results"+"/"+Experiment_name+"/CIWT_Pointrcnn/val/")
    
    PHASE="training"
    # Determine sequences here
    if PHASE =="training":
    #     SEQUENCES = [0,1,3,4,5,9,11,12,15,17,19,20]
    # elif PHASE == "validation":
        SEQUENCES = [2, 6, 7, 8, 10, 13, 14, 16, 18]
    else:
        SEQUENCES=range(29)
    # settings = open(SCRIPT_DIR+"/evaluation/evaluate_tracking.seqmap.val",'w')
    # for SEQ in SEQUENCES:
    #     settings.write("%04d empty 000000 %06d \n"%(SEQ,int(end_frames[SEQ])))
    # settings.close()
    seqs_frames = [SEQUENCES, [ int(end_frames[s])+1 for s in SEQUENCES]]
    argv = []
    tracked = 'car'
    argv.append("{}_3d_det_val".format(tracked))
    argv.append(Experiment_name)
    argv.append('3D')
    argv.append(df)
    # remove '*' if the trajectories are for each objects are separated in different subfolders
    argv.append("CIWT_Pointrcnn"+'*')
    argv.append(seqs_frames)
    # argv.append(PHASE)
    print("start evaluation")
    mota, motp, df = run(*argv)
    return mota, motp, df



tracked = 'car'
DIMENSION =  '3D' 
SCRIPT_DIR="/usr/stud/kaddah/ciwt_gd"
table_name = "results/ready_results_baseline_val/results_{}_val_table_{}.csv".format(tracked,DIMENSION)
df  = pandas.read_csv(table_name)
init(True)
experiment_name_init = paraser_args.name
experiment_name = experiment_name_init
pre_mota, pre_motp, df = exp(experiment_name, hpy, df)
# pre_mota, pre_motp = 7740098026228639, 0.6546862221413912
bestA = pre_mota
bestP = pre_motp
# selected = len(options_by_type[opt_typ])
updated = False
failed_before= False
list_mota = []
list_motp = []
lists ={}
best_expriment = 0
permut = None

for k in hpy.keys():
    lists[k] = []
for exp_count in range(100):
    init()
    experiment_name = experiment_name_init+"_coup_{}".format(exp_count)
    mota, motp, df = exp(experiment_name, hpy, df)


    if mota > bestA :
        bestA = mota 
        bestP = motp
        best_expriment = exp_count
    pre_mota = mota 
    pre_motp = motp


    
    file1 = open(SCRIPT_DIR+"/results"+"/"+experiment_name+"/"+experiment_name +".txt","w")
    settings = ""

    for key in hpy.keys():
        settings += str(key) + ": {} \n".format(hpy[key])

        lists[k].append(hpy[key])
    list_mota.append(mota)
    list_motp.append(motp)
    file1.write(settings)
    file1.close()
    string_format = df.to_latex(index=False)

    print(string_format,  file=open("results/{}_results_{}_val_table_{}.txt".format(paraser_args.name,tracked,DIMENSION), 'w'))
    file_ = open("results/{}_results_{}_val_table_{}.csv".format(paraser_args.name,tracked,DIMENSION),"w") 
    df.to_csv(file_, index = False, header=True)

# if not os.path.exists("./results/params_ped/"):
#     os.mkdir("./results/params_ped/")
if not os.path.exists("./results/"+experiment_name_init +"_f"):
    os.mkdir("./results/"+experiment_name_init+ "_f" )
# for key in hpy.keys():
    
#     np.save("./results/params_ped/{}.npy".format(key),np.asarray(lists[k]))

# np.save("./results/params_ped/mota_{}.npy".format(key),np.asarray(mota))
# np.save("./results/params_ped/motp_{}.npy".format(key),np.asarray(motp))

df = df.sort_values(by=['MOTA'])
# 
# shutil.move("results/{}_results_{}_val_table_{}.txt".format(paraser_args.name,tracked,DIMENSION),"./results/"+experiment_name_init +"_f/" )
shutil.move("results/{}_results_{}_val_table_{}.csv".format(paraser_args.name,tracked,DIMENSION), "./results/"+experiment_name_init +"_f/")
for exp_count in range(100):
    experiment_name = experiment_name_init+"_{}".format(exp_count)
    shutil.move(SCRIPT_DIR+"/results"+"/"+experiment_name,"/results"+"/"+experiment_name_init+"_f/")
# shutil.move("./results/params_ped", "./results/"+experiment_name_init +"_f/")
print("BEST EXPRIMENT IS : ", best_expriment)