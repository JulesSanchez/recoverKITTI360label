from accumulation import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--kitti_dir", help="path to kitti360 dataset")
parser.add_argument("-o", "--output_dir", help="path to output_dir")
parser.add_argument("-s", "--sequence", help="sequence name")
parser.add_argument("-f", "--first_frame", type=int)
parser.add_argument("-l", "--last_frame", type=int)
parser.add_argument("-d", "--data_source", help="1:velodyne scans", default=1, type=int)
args = parser.parse_args()

root_dir = args.kitti_dir
sequence = args.sequence
output_dir = args.output_dir
first_frame = args.first_frame
last_frame = args.last_frame
source = args.data_source
travel_padding = 20
min_dist_dense = 0.02

PA = PointAccumulation(root_dir, output_dir, sequence, first_frame, last_frame, travel_padding, source, min_dist_dense, True, False)

print('Initialization Done!')

if not PA.createOutputDir():
    print('Error: Unable to create the output directory!')

if not PA.loadTransformation():
    print('Error: Unable to load the calibrations!')

if not PA.getInterestedWindow():
    print('Error: Invalid window of interested!')

print("Loaded " + str(len(PA.Tr_pose_world)) + " poses")

PA.loadTimestamps()

print("Loaded " + str(len(PA.veloTimestamps)) + " velo timestamps")

PA.addVelodynePoints()
PA.getPointsInRange()
PA.writeToFiles()