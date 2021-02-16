from accumulation import *
import argparse
import os 

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--kitti_dir", help="path to kitti360 dataset")
parser.add_argument("-o", "--output_dir", help="path to output_dir")
args = parser.parse_args()
root_dir = args.kitti_dir
output_dir = args.output_dir

for sequence in os.listdir(os.path.join(root_dir,"data_3d_raw")):
    all_spcds = os.listdir(os.path.join(os.path.join(os.path.join(root_dir,"data_3d_semantics"),sequence),"static"))
    all_spcds.sort()
    for i in range(len(all_spcds)):
        spcd = all_spcds[i]
        if i == 0:
            spcd_prev = None 
        else:
            spcd_prev = all_spcds[i-1]
        if i == len(all_spcds)-1:
            spcd_next = None 
        else:
            spcd_next = all_spcds[i+1]
        partial_name = os.path.splitext(spcd)[0].split('_')
        first_frame = int(partial_name[0])
        last_frame = int(partial_name[1])
        PA = PointAccumulation(root_dir, output_dir, sequence, first_frame, last_frame, 20, 1, 0.02, True, True)
        PA.createOutputDir()
        PA.loadTransformation()
        PA.getInterestedWindow()
        PA.loadTimestamps()
        PA.addVelodynePoints()
        PA.getPointsInRange()
        PA.recoverLabel(spcd,spcd_prev,spcd_next,0.5)
        PA.writeToFiles()
