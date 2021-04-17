#!/bin/bash


SCRIPT_DIR=/usr/stud/kaddah/ciwt_gd
#echo ${SCRIPT_DIR}
#exit
PHASE=training
DATASET=/usr/stud/kaddah/storage/datasets/kitti/$PHASE
PREPROC=/usr/stud/kaddah/storage/datasets/kitti/$PHASE/preproc
OUT_DIR=/usr/stud/kaddah/storage/datasets/kitti/$PHASE/tracking_output

# Possible modes: 'detection', 'detection_shape'
# The latter is considered obsolete; this option is still here for eval purposes
MODE=detection # detection_shape

ORIENTED=0
#enable 2D-3D coupling
COUPLED=1
#terminates tracks with projection to image if the all points are outside the image frame
EXIST3D2D=1
#use 3D IoU for association
IoU3D=0
# Determine sequences here
SEQUENCES=`seq 0 20`

# Specify detectors (should correspond to detector dir names and config names)
# DETS=(det_02_regionlets det_02_3DOP)
DETS=(pointrcnn)

# DETS=(pointrcnn)
end_frames=("153" "446" "232" "143" "313" "296" "269" "799" "389" "802" "293" "372" "77" "339" "105" "375" "208" "144" "338" "1058" "836")
# if PHASE=testing
# then
# 	end_frames=(466 148 244 258 422 810 115 216 166 350 1177 775 695 153 851 702 511 306 181 405 174 204 437 431 317 177 171 86 176)
# fi
# end_frames=("464" "146" "242" "256" "420" "808" "113" "214" "164" "348" "1175" "773" "693" "151" "849" "700" "509" "304" "179" "403" "172" "202" "435" "429" "315" "175" "169" "84" "174")
# Path to your binary
CIWT=/usr/stud/kaddah/ciwt_gd/build/apps/CIWTApp

# Exec tracker for each detection set
for DET in ${DETS[@]}; do
	# Exec tracker for each sequence
	for SEQ in ${SEQUENCES[@]}; do
		SEQNAME=`printf "%04d" $SEQ`
		echo "Process: $SEQNAME"

		IM_LEFT=$DATASET/image_02/$SEQNAME/%06d.png
		IM_RIGHT=$DATASET/image_03/$SEQNAME/%06d.png
		DISP=$PREPROC/disparity/disp_$SEQNAME/%06d_left.png
		END_FRAME=`printf "%d" ${end_frames[$SEQ]}`
		CALIB=$DATASET/calib/$SEQNAME.txt
		PROP=$PREPROC/cached_proposals/$SEQNAME/%06d.bin
		OUT_DIR_DET=$OUT_DIR/$DET
		DETPATH=$PREPROC/detection/$DET/$SEQNAME/%06d.txt
		DETCFG=$SCRIPT_DIR/cfg/$DET.cfg
		$CIWT --config $DETCFG --left_image_path ${IM_LEFT} --end_frame $END_FRAME --right_image_path ${IM_RIGHT} --left_disparity_path ${DISP} --calib_path $CALIB --object_proposal_path $PROP --detections_path $DETPATH --output_dir ${OUT_DIR_DET} --tracking_mode $MODE  --sequence_name $SEQNAME --dataset_name kitti --debug_level 3 --is_oriented $ORIENTED --enable_coupling $COUPLED --check_exit_to2D_projeciton $EXIST3D2D --association_score_is_3DIoU $IoU3D --report_extrapolated 1 --corners_left 0
	done
done
