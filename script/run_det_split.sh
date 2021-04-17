#!/bin/bash

# Place your det dirs here
DETDIRS=(/usr/stud/kaddah/storage/datasets/kitti/training/preproc/detection/merged_pointrcnn)

DESTDIR=/usr/stud/kaddah/storage/datasets/kitti/training/preproc/detection/pointrcnn


for DETDIR in ${DETDIRS[@]}; do
	NAME=`basename $DETDIR`
	DIROUT=$DESTDIR/$NAME
	echo $DIROUT
	mkdir -p $DIROUT

	for SEQ in $DETDIR/*; do
		python3 ./script/split_detections_sequence_to_frame.py --sequence_path $SEQ --output_dir $DIROUT
	done
done
# replace  /usr/stud/kaddah/storage/datasets/ with your own directory
rm -r /usr/stud/kaddah/storage/datasets/kitti/training/preproc/detection/merged_pointrcnn
mv /usr/stud/kaddah/storage/datasets/kitti/training/preproc/detection/pointrcnn /usr/stud/kaddah/storage/datasets/kitti/training/preproc/detection/pointrcnn_s
mv /usr/stud/kaddah/storage/datasets/kitti/training/preproc/detection/pointrcnn_s/merged_pointrcnn /usr/stud/kaddah/storage/datasets/kitti/training/preproc/detection/
mv /usr/stud/kaddah/storage/datasets/kitti/training/preproc/detection/merged_pointrcnn /usr/stud/kaddah/storage/datasets/kitti/training/preproc/detection/pointrcnn
rm -r /usr/stud/kaddah/storage/datasets/kitti/training/preproc/detection/pointrcnn_s
