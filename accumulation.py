from utils import *
from commons import *
import numpy as np 
from sklearn.neighbors import KDTree
import cv2 
import open3d

class PointAccumulation:

    def __init__(self, root_dir, output_dir, sequence_, first_frame, last_frame, travel_padding, source_, min_dist_dense=0.02, verbose_=1, compute_labels = False):
        self.rootDir = root_dir#root directory of the laser data
        self.outputDir = output_dir
        self.sequenceName = sequence_ #sequence name (the laser data will be saved in root_dir/sequence)
        self.firstFrame = first_frame #starting frame number
        self.lastFrame = last_frame #ending frame number
        self.travelPadding = travel_padding #padding distance (in meters)
        self.sourceType = source_ #source of data (0: sick only, 1: velodyne only, 2: both)
        self.minDistDense = min_dist_dense #point cloud resolution in meter, optional default = 0.02
        self.verbose = verbose_ #boolean number to indicate if display the msg, optional default = 1
        self.computeLabels = compute_labels

        self.outDistDense = 0.2 #max distance for point to any neighbor to be inlier
        self.maxPointDist = 1000 #max distance for 3d points to any of the poses

        self.firstFrameWindow = 0
        self.lastFrameWindow = 0

        self.baseDir = self.rootDir + "/data_3d_raw/" + self.sequenceName
        self.poseDir = self.rootDir + "/data_poses/" + self.sequenceName
        self.calibDir = self.rootDir + "/calibration"
        self.superpcDir = self.rootDir + "/data_3d_semantics/" + self.sequenceName + "/static/"

        self.Tr_cam_pose = [] #cam0x -> pose
        self.Tr_cam_velo = np.empty((4,4)) #cam00 -> velo
        self.Tr_sick_velo = np.empty((4,4)) #sick -> velo
        self.Tr_pose_world = [] #global poses of the vehicle
        self.Tr_pose_index = [] #index of the global poses
        self.Tr_velo_pose = np.empty((4,4)) #velo -> pose

        self.sickTimestamps = []
        self.veloTimestamps = []

        self.Md = np.empty((3,0)) #Accumulated point cloud
        self.Ll = np.empty((3,0))  #Accumulated sensor locations 
        self.Md_prev = np.empty((3,0))  
        self.Ll_prev = np.empty((3,0))  
        self.Tr_velo_window = [] #global poses of the velodyne within [firstFrame, lastFrame]
        self.Vp = [] #vehicle transitions 
        self.Ts = [] #timestamp of accumulated point cloud 
        self.Fidx = [] #vector of valid frames 
        self.Ap = [] #vector of dense point cloud at valid frames
        self.localIdx = [] #ADDED, vector of indices in the original pointcloud in a single pointcloud
        self.globalIdx = [] #ADDED, vector of indices in the original pointcloud in the accumulated pointcloud
        self.numPts = {} #ADDED, dictionnary of number of points in each frame

        self.outputPath = "%s/%s_%06d_%06d" % (self.outputDir, self.sequenceName, self.firstFrame, self.lastFrame)
        self.outputPathLabel = "%s/%s/labels" % (self.outputDir, self.sequenceName)

        if self.sourceType == 2:
            self.output_file = "%s/lidar_points_all.dat" % self.outputPath
            self.output_file_timestamps = "%s/lidar_timestamp_all.dat" % self.outputPath
        
        elif self.sourceType == 1:
            self.output_file = "%s/lidar_points_velodyne.txt" % self.outputPath
            self.output_file_timestamps = "%s/lidar_timestamp_velodyne.dat" % self.outputPath         

        else :
            self.output_file = "%s/lidar_points_sick.dat" % self.outputPath
            self.output_file_timestamps = "%s/lidar_timestamp_sick.dat" % self.outputPath         

        self.output_file_loc = "%s/lidar_loc.dat" % self.outputPath
        self.output_file_pose = "%s/lidar_pose.dat" % self.outputPath

    def createOutputDir(self):
        print("Output direction : " + self.outputPath)
        return mkdir(self.outputPath)

    def loadTransformation(self):
        cam2poseName = self.calibDir + "/calib_cam_to_pose.txt"
        self.Tr_cam_pose, success = loadCamPose(cam2poseName)
        if not success:
            return False

        extrinsicName = self.calibDir + "/calib_cam_to_velo.txt"
        self.Tr_cam_velo, success = loadTransform(extrinsicName)
        if not success:
            return False

        sick2veloName = self.calibDir + "/calib_sick_to_velo.txt"
        self.Tr_sick_velo, success = loadTransform(sick2veloName)
        if not success:
            return False

        poseName = self.poseDir + "/poses.txt"    
        self.Tr_pose_world, self.Tr_pose_index, success = loadPoses(poseName)
        if not success:
            return False

        self.Tr_velo_pose = self.Tr_cam_pose[0] @ np.linalg.inv(self.Tr_cam_velo)          

        return success

    '''
     * @brief GetInterestedWindow Get the first and last frame for clipping points
     * @return true if succeed.
    '''

    def getInterestedWindow(self):
        travelDist = 0. 
        lastPose =  np.zeros(3) 
        lastPoseUpdated = False 
        for firstFrameWindow in range(self.firstFrame, self.lastFrame):
            frameIndex = self.getFrameIndex(firstFrameWindow-1)
            if frameIndex >= 0:
                if lastPoseUpdated:
                    currPose = self.Tr_pose_world[frameIndex][:3,3]
                    currPose = currPose - lastPose 
                    travelDist = travelDist + np.linalg.norm(currPose)
                    if travelDist > self.travelPadding:
                        break
            lastPose = self.Tr_pose_world[frameIndex][:3,3]
            lastPoseUpdated = True

        travelDist = 0. 
        lastPose =  np.zeros(3) 
        lastPoseUpdated = False 
        for lastFrameWindow in range(self.lastFrame,-1,-1):
            frameIndex = self.getFrameIndex(lastFrameWindow-1)
            if frameIndex >= 0:
                if lastPoseUpdated:
                    currPose = self.Tr_pose_world[frameIndex][:3,3]
                    currPose = currPose - lastPose 
                    travelDist = travelDist + np.linalg.norm(currPose)
                    if travelDist > self.travelPadding:
                        break
            lastPose = self.Tr_pose_world[frameIndex][:3,3]
            lastPoseUpdated = True

        if self.verbose:
            print("Window of interest within frame %010d and %010d\n" % (firstFrameWindow, lastFrameWindow))

        if lastFrameWindow < firstFrameWindow:
            return False 

        self.Tr_velo_window = []
        self.Fidx = []
        self.Vp = []

        for frame in range(self.firstFrame, self.lastFrame+1):
            if frame <1:
                continue
            frameIndex = self.getFrameIndex(frame)
            if frameIndex <0:
                continue
            self.Tr_velo_window.append(self.Tr_pose_world[frameIndex]@self.Tr_velo_pose)
            self.Fidx.append(frame)
            vehiclePose = self.Tr_pose_world[frameIndex][:3,3]
            if frame >= firstFrameWindow and frame <= lastFrameWindow:
                self.Vp.append(vehiclePose)

        return True

    '''
     * @brief LoadTimestamps Load timestamps for sick and velodyne data
     * @return true if succeed. (success)
    '''

    def loadTimestamps(self):
        veloTsName = self.baseDir + "/velodyne_points/timestamps.txt"
        self.veloTimestamps, success = LoadTimestamp(veloTsName)
        if not success:
            return False 
        sickTsName = self.baseDir + "/sick_points/timestamps.txt"
        self.sickTimestamps, success = LoadTimestamp(sickTsName)
        if not success:
            return False 
        return success 

    '''
     * @brief LoadSickData Load one frame of sick data
     * @param frame Frame number to load

     * @out data Loaded sick data
     * @return true if succeed. (success)
    '''

    def loadSickData(self, frame):
        frameName = "%s/sick_points/data/%010d.bin" % (self.baseDir, frame)

        cols = 2
        tmp, success = readMatrixCol(frameName, cols)
        if not success:
            return False 

        data = np.zeros(tmp.shape[0], cols+1)
        data[0:,1] = -tmp[:,:1]
        data[0:,2] = -tmp[:,1:2]

        return data, success

    '''
     * @brief LoadVelodyneData Load one frame of velodyne data
     * @param frame Frame number to load
     * @param blind_splot_angle Angle for blind region

     * @out data Loaded velodyne data
     * @return true if succeed
    '''    

    def loadVelodyneData(self, frame, blind_splot_angle):
        frameName = "%s/velodyne_points/data/%010d.bin" % (self.baseDir, frame)

        cols = 4
        tmp, success = readMatrixCol(frameName, cols)
        self.numPts[int(frame)] = len(tmp)
        if not success:
            return False 

        data, self.localIdx = removeBlindSpots(tmp, blind_splot_angle)

        return data[:,:3], success

    '''
     * @brief CurlParametersFromPoses Get curl parameters at a frame
     * @param frame Frame number
     * @param Tr_pose_curr Pose at the given frame

     * @out r Output rotation matrix for curl
     * @out t Output translation matrix for curl
    '''

    def curlParametersFromPoses(self, frame, Tr_pose_curr):
        Tr_pose_pose = np.eye(4)
        if frame == 0:
            indexNext = self.getFrameIndex(frame+1)
            if indexNext >= 0:
                Tr_pose_pose = np.linalg.inv(self.Tr_pose_world[indexNext])@Tr_pose_curr
        else:
            indexPrev = self.getFrameIndex(frame-1)
            if indexPrev>0:
                Tr_pose_pose = np.linalg.inv(Tr_pose_curr)@self.Tr_pose_world[indexPrev]
        
        Tr_delta = np.linalg.inv(self.Tr_velo_pose)@Tr_pose_pose@self.Tr_velo_pose
        Tr_delta_r = Tr_delta[:3,:3]

        r, _ = cv2.Rodrigues(Tr_delta_r)
        t = Tr_delta[:,3]

        return r, t


    '''
     * @brief GetFrameIndex Search the pose index of a given frame
     * @param frame Frame number
     * @return -1 if frame has no valid pose
     *         index of the pose if there is a pose for the frame
    '''

    def getFrameIndex(self, frame):
        try:
            idx = self.Tr_pose_index.index(frame)
            return idx 
        except ValueError:
            return -1

    '''
     * @brief AddQdToMd Merge point cloud Qd to the accumulated Md
     * @param Qd Point cloud at a single frame
     * @param loc Sensor locations
     * @param frame T
    '''

    def addQdToMd(self, Qd, loc, frame):
        self.Md = np.hstack((self.Md_prev, Qd))
        self.Ll = np.hstack((self.Ll_prev, loc))

        #Not sparsifying at the moment
        #idx_sparse, idx_size = sparsifyData(self.Md,len(self.Md),self.minDistDense,self.outDistDense,self.Md_prev.shape[1])
        #Md_sparse = extractCols(self.Md,idx_sparse)
        #Ll_sparse = extractCols(self.Ll,idx_sparse)

        numAdded = self.Md.shape[1] - self.Md_prev.shape[1]
        for _ in range(numAdded):
            self.Ts.append(frame)

        self.Md_prev = self.Md 
        self.Ll_prev = self.Ll


    '''
     * @brief AddSickPoints Accumulate sick points
    '''
    #NOT IMPLEMENTED
    def addSickPoints(self):
        return

    '''
     * @brief AddVelodynePoints Accumulate velodyne points
    '''

    def addVelodynePoints(self):
        Vd_poses = []
        Vd = []
        Vd_frames = []

        for frame in range(self.firstFrame,self.lastFrame+1):
            if frame <1:
                continue 
            frameIndex = self.getFrameIndex(frame)
            if frameIndex<0:
                continue 

            veloData, _ = self.loadVelodyneData(frame,np.pi/8)
            r, t = self.curlParametersFromPoses(frame, self.Tr_pose_world[frameIndex])
            veloDataCropped, cropIdx = cropVelodyneData(veloData,3,80)
            intermediateIdx = [self.localIdx[i] for i in cropIdx]
            self.localIdx = intermediateIdx
            cropIdx = cropVelodyneDataIndices(veloDataCropped,3,self.maxPointDist)
            veloDataCurled = curlVelodyneData(veloDataCropped, r, t)

            Tr_curr = self.Tr_pose_world[frameIndex]@self.Tr_velo_pose

            Tr_curr_r = Tr_curr[:3,:3]
            Tr_curr_t = Tr_curr[:3,3]

            veloDataFull = Tr_curr_r@veloDataCurled.T 
            veloDataFull = veloDataFull.T  + Tr_curr_t

            self.Ap.append(veloDataFull)

            veloDataCropped = extractRows(veloDataCurled, veloDataCropped, cropIdx)
            self.localIdx = [self.localIdx[i] for i in cropIdx]

            Vd.append(veloDataCropped.T)
            Vd_poses.append(self.Tr_pose_world[frameIndex])
            Vd_frames.append(frame)
            self.globalIdx = self.globalIdx + self.localIdx

            if self.verbose:
                print("Processed frame (Velodyne loading pass): %010d \n" % frame)

        if self.minDistDense >= 0.02:
            for i in range(len(Vd)):
                Pd = Vd[i]#getHalfPoints(Vd[i],0)
                Tr_curr = Vd_poses[i]@self.Tr_velo_pose
                Tr_curr_r = Tr_curr[:3,:3]
                Tr_curr_t = Tr_curr[:3,3]
                Qd = Tr_curr_r@Pd
                Qd = (Qd.T + Tr_curr_t).T 
                loc = np.zeros_like(Qd)
                loc = (loc.T + Tr_curr_t).T 
                self.addQdToMd(Qd,loc,Vd_frames[i])

                if self.verbose:
                    print("Processed frame (Velodyne forward pass) %010d with %d points\n" % (Vd_frames[i], Qd.shape[1]))


    '''
     * @brief GetPointsInRange Clip points to lie within max_point_dist from any of the poses
    '''

    def getPointsInRange(self):
        Ts_out = []
        Idx_out = []
        maxDist2 = self.maxPointDist**2 
        k = 0
        numPts = self.Md.shape[1]
        numPos = len(self.Vp)
        inRange = np.zeros(numPts)
        for p in range(numPos):
            pos = self.Vp[p]
            localMd = self.Md.T - pos.reshape(3)
            localMd = np.sum(localMd**2,axis=1) < maxDist2
            inRange = np.logical_or(inRange,localMd)
        self.Md = self.Md.T[inRange,:]
        self.Ll = self.Ll.T[inRange,:]
        for k in range(numPts):
            if inRange[k]:
                Ts_out.append(self.Ts[k])
                Idx_out.append(self.globalIdx[k])
        self.Ts = Ts_out
        self.globalIdx = Idx_out
        print("Loaded : " + str(np.sum(inRange)) + " points after range checking")

    '''
     * @brief WriteToFiles Write results to file
    '''

    def writeToFiles(self):
        if not self.computeLabels:
            np.save('ts.npy', np.array(self.Ts))
            np.save('indices.npy', np.array(self.globalIdx))
            if (self.verbose):
                print("writing points to file %s ...\n" % self.output_file)
            writeMatrixToFile(self.output_file, self.Md)

            if (self.sourceType==2):
                if (self.verbose):
                    print("writing location to file %s ...\n" % self.output_file_loc)
                writeMatrixToFile(self.output_file_loc, self.Ll)

            if (self.verbose):
                print("writing pose to file %s ...\n" % self.output_file_pose)
            writePoseToFile(self.output_file_pose, self.Fidx, self.Tr_velo_window)

            if (self.verbose):
                print("write timestamp to file %s ...\n" % self.output_file_timestamps)
            writeTimestampToFile(self.output_file_timestamps, self.Ts)
        else:
            label_folder = self.outputPathLabel
            mkdir(label_folder)
            if (self.verbose):
                print("write labels to folder %s ...\n" % label_folder)
            writeLabelsToFolder(label_folder, self.labels, self.Ts, self.globalIdx, self.numPts)


    def recoverLabel(self,superPointCloud,superPointCloudPrev,superPointCloudnext,rangeS=0.1):

        self.labels = np.zeros(len(self.Ts))

        if not superPointCloudPrev is None :
            superpcPath = self.superpcDir + superPointCloudPrev
            superpcd = open3d.io.read_point_cloud(superpcPath)
            n_pts = np.asarray(superpcd.points).shape[0]
            superpcd = readBinaryPly(superpcPath,n_pts)

            tree = KDTree(superpcd[:,:3])
            dist, ind = tree.query(self.Md)
            mask = dist[:,0]<rangeS
            self.labels[mask] = superpcd[ind[:,0][mask],6]

        if not superPointCloudnext is None :
            superpcPath = self.superpcDir + superPointCloudnext
            superpcd = open3d.io.read_point_cloud(superpcPath)
            n_pts = np.asarray(superpcd.points).shape[0]
            superpcd = readBinaryPly(superpcPath,n_pts)

            tree = KDTree(superpcd[:,:3])
            dist, ind = tree.query(self.Md)
            mask = dist[:,0]<rangeS
            self.labels[mask] = superpcd[ind[:,0][mask],6]

        superpcPath = self.superpcDir + superPointCloud
        superpcd = open3d.io.read_point_cloud(superpcPath)
        n_pts = np.asarray(superpcd.points).shape[0]
        superpcd = readBinaryPly(superpcPath,n_pts)

        tree = KDTree(superpcd[:,:3])
        dist, ind = tree.query(self.Md)
        mask = dist[:,0]<rangeS
        self.labels[mask] = superpcd[ind[:,0][mask],6]
