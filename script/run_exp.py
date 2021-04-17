import os
import subprocess
from sacred import Experiment
import evaluation.comp_rec as evaluator
import datetime
import shutil
ex = Experiment()

@ex.config
def add_config():
    Experiment_name= str(datetime.datetime.today().d)[:10],

    # basline_dir="/usr/stud/kaddah/",
    Experiment_name="default"
    Basline_Name="AB3DMOT"
    DATASET_NAME="kitti",
    debug_level=3
    PHASE="training"



@ex.automain
def main(Experiment_name, DATASET_NAME, Basline_Name,debug_level, PHASE):
    SCRIPT_DIR="/usr/stud/kaddah/ciwt_gd",
    DATASET_DIR= "/usr/stud/kaddah/storage/datasets/",
    DATASET=DATASET_DIR+DATASET_NAME+"/{}".format(PHASE),
    PREPROC=DATASET_DIR+DATASET_NAME+"/{}/preproc".format(PHASE),
    OUT_DIR=DATASET_DIR+DATASET_NAME+"/{}/tracking_output".format(PHASE),
    # Possible modes: 'detection', 'detection_shape'
    # The latter is considered obsolete; this option is still here for eval purposes
    MODE="detection", # detection_shape
    CIWT="/usr/stud/kaddah/ciwt_gd/build/apps/CIWTApp",

    # Specify detectors (should correspond to detector dir names and config names)
    # DETS=['det_02_regionlets', 'det_02_3DOP', 'pointrcnn']
    DETS=['pointrcnn']

    if DATASET_NAME=="kitti":
        if PHASE=="training" or PHASE=="validation":
            end_frames=["153", "446", "232", "143", "313", "296", "269", "799", "389", "802", "293", "372", "77", "339", "105", "375", "208", "144", "338", "1058", "836"]
            SEQUENCES=range(20)
        else:
            end_frames=["464", "146", "242", "256", "420", "808", "113", "214", "164", "348", "1175", "773", "693", "151", 
                "849", "700", "509", "304", "179", "403", "172", "202", "435", "429", "315", "175", "169", "84", "174"]
            SEQUENCES=range(29)
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
            END_FRAME="%d"%(end_frames[SEQ])
            CALIB="{}/calib/{}.txt".format(DATASET,SEQNAME)
            PROP="{}/cached_proposals/{}/%06d.bin".format(PREPROC,SEQNAME)
            OUT_DIR_DET="{}/{}".format(OUT_DIR,DET)
            DETPATH="{}/detection/{}/{}/%06d.txt".format(PREPROC,DET,SEQNAME)
            DETCFG="{}/cfg/{}.cfg".format(SCRIPT_DIR,DET)
            subprocess.run([CIWT+"--config {}".format(DETCFG)+
                                "--left_image_path {}".format(IM_LEFT) +
                                "--end_frame {}".format(END_FRAME) +
                                "--right_image_path {}".format(IM_RIGHT)+
                                "--left_disparity_path {}".format(DISP) +
                                "--calib_path {}".format(CALIB) +
                                "--object_proposal_path {}".format(PROP) +
                                "--detections_path {}".format(DETPATH)+
                                "--output_dir {}".format(OUT_DIR_DET)+
                                "--tracking_mode {}".format(MODE)+
                                "--sequence_name {}".format(SEQNAME)+
                                "--DATASET_NAME  {}".format(DATASET_NAME) +
                                "--debug_level {}".format(debug_level)
                            ])
    for DET in DETS:
        shutil.move(OUT_DIR+"/"+DET+"/data",SCRIPT_DIR+"/results"+"/"Experiment_name)
    PHASE="training"
    # Determine sequences here
    if PHASE =="training":
        SEQUENCES = [0,1,3,4,5,9,11,12,15,17,19,20]
    elif PHASE == "validation":
        SEQUENCES = [2, 6, 7, 8, 10, 13, 14, 16, 18]
    else:
        SEQUENCES=range(29)
    settings = open(basline_dir+"evalutaion/evaluate_tracking.seqmap.val",'w')
    for SEQ in SEQUENCES:
        settings.write("%04d empty 000000 %06d \n"%(SEQ,end_frames[SEQ]))
    argv = []
    for det in DETS:
        for tracked in ['ped','car','cyc']:
            argv.append("{}_3d_det_val".format(tracked))
            argv.append(Experiment_name)
            argv.append('3D')
            argv.append(Basline_Name)
            # remove '*' if the trajectories are for each objects are separated in different subfolders
            argv.append(det+'*')
            argv.append(PHASE)
    evaluator.run(*argv)
    argv = []
    for det in DETS:
        for tracked in ['ped','car','cyc']:
            argv.append("{}_3d_det_val".format(tracked))
            argv.append(Experiment_name)
            argv.append('2D')
            argv.append(Basline_Name)
            # remove '*' if the trajectories are for each objects are separated in different subfolders
            argv.append(det+'*')
            argv.append(PHASE)
    evaluator.run(*argv)


    PHASE="validation"
    # Determine sequences here
    if PHASE =="training":
        SEQUENCES = [0,1,3,4,5,9,11,12,15,17,19,20]
    elif PHASE == "validation":
        SEQUENCES = [2, 6, 7, 8, 10, 13, 14, 16, 18]
    else:
        SEQUENCES=range(29)
    settings = open(basline_dir+"evalutaion/evaluate_tracking.seqmap.val",'w')
    for SEQ in SEQUENCES:
        settings.write("%04d empty 000000 %06d \n"%(SEQ,end_frames[SEQ]))

    argv = []
    for det in DETS:
        for tracked in ['ped','car','cyc']:
            argv.append("{}_3d_det_val".format(tracked))
            argv.append(Experiment_name)
            argv.append('3D')
            argv.append(Basline_Name)
            # remove '*' if the trajectories are for each objects are separated in different subfolders
            argv.append(det+'*')
            argv.append(PHASE)
    evaluator.run(*argv)

    argv = []
    for det in DETS:
        for tracked in ['ped','car','cyc']:
            argv.append("{}_3d_det_val".format(tracked))
            argv.append(Experiment_name)
            argv.append('2D')
            argv.append(Basline_Name)
            # remove '*' if the trajectories are for each objects are separated in different subfolders
            argv.append(det+'*')
            argv.append(PHASE)
    evaluator.run(*argv)