import os 
import numpy as np
import struct 

KOGMO_TIMESTAMP_TICKSPERSECOND =  1000000000.0

'''
 * @brief LoadTransform Load a single transform from the file
 * @param filename

 * @out transform 4*4 matrix from file
 * @return True if loading is successful. (success)
'''

def loadTransform(filename):

    transform = np.eye(4)

    try:
        infile = open(filename).readline().rstrip("\n").split(' ')
    except:
        print("Failed to open transforms " + filename)
        return transform, False 
    
    for i in range(12):
        xi = i//4
        yi = i%4
        transform[xi,yi] = float(infile[i])

    return transform, True

'''
 * @brief LoadCamPose Load poses from file.
 * @param filename

 * @out poses A vector of 4x4 matrix as the poses.
 * @return True if loading is successful. (success)
'''

def loadCamPose(filename):
    poses = [None for _ in range(4)]

    try:
        infile = open(filename)
    except:
        print("Failed to open camera poses " + filename)
        return poses, False 

    for line in infile:
        lineProcessed = line.rstrip("\n").split(' ')
        if any("image_0" in x for x in lineProcessed):
            transform = np.eye(4)
            index = int(lineProcessed[0][7])
            for i in range(12):
                xi = i//4
                yi = i%4
                transform[xi,yi] = float(lineProcessed[i+1])
            poses[index] = transform

    infile.close()
    return poses, True

'''
 * @brief LoadCamPose Load poses from file.
 * @param filename

 * @out poses A vector of 4x4 matrix as the poses.
 * @out indices Valid indices of the poses
 * @return True if loading is successful. (success)
'''

def loadPoses(filename):
    poses = []
    indices = []

    try:
        infile = open(filename)
    except:
        print("Failed to open poses " + filename)
        return poses, False 

    for line in infile:
        lineProcessed = line.rstrip("\n").split(' ')
        transform = np.eye(4)
        index = int(lineProcessed[0])
        for i in range(12):
            xi = i//4
            yi = i%4
            transform[xi,yi] = float(lineProcessed[i+1])
        poses.append(transform)
        indices.append(index)

    infile.close()
    return poses, indices, True

'''
 * @brief LoadTimestamp Load timestamp from file.
 * @param filename

 * @out timestamps Values of the timestamps
 * @return True if loading is successful. (success)
'''

def LoadTimestamp(filename):
    timestamps = []
    try:
        infile = open(filename)
    except:
        print("Failed to open timestamps " + filename)
        return timestamps, False 
    
    for line in infile:
        lineProcessed = line.rstrip("\n")
        ts = string2Timestamp(lineProcessed)
        if ts == 0:
            print("Invalid timestamp at line " + lineProcessed)
            return timestamps, False
        timestamps.append(ts)
    infile.close()
    return timestamps, True

'''
 * @brief String2Timestamp Convert timestamp in string to a double value.
 * @param time_str Timestamp in string format.

 * @out timestamps in double value.
'''

def string2Timestamp(time_str):
    try:
        date, time = time_str.split()
    except:
        return 0
    
    hour, minute, second = time.split(':')
    year, month, day = date.split('-')

    hour = float(hour)
    minute = float(minute)
    second = float(second)
    #Change range of some values
    year = int(year) - 1900
    month = int(month) - 1
    day = int(day)

    secs = second + 60*(minute + 60*(hour + 24*(day + 30*(month + 12*(year)))))

    return secs * KOGMO_TIMESTAMP_TICKSPERSECOND

'''
 * @brief ReadMatrixCol Read a matrix from file with specified number of columns.
 * @param filename Filename of the matrix
 * @param cols Number of columns

 * @out matrix Eigen matrix with loaded value
 * @return True if loading is successful. (success)
'''

def readMatrixCol(filename, cols):

    try:
        matrix = np.fromfile(filename, dtype=np.float32, count=-1).reshape([-1,cols])
    except:
        print("Failed to open file " + filename)
        return np.zeros(1), False 

    return matrix, True

'''
 * @brief WriteMatrixToFile Write Eigen matrix to file
 * @param name Filename
 * @param mat Eigen matrix
'''

def writeMatrixToFile(name, matrix):
    outfile = open(name, 'w')
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    for i in range(rows):
        for j in range(cols):
            outfile.write(str(matrix[i,j]) + " ")
        outfile.write("\n")
    outfile.close()

'''
 * @brief WritePoseToFile Write pose to file
 * @param name Filename
 * @param idx Vector of frame numbers for all poses
 * @param poses Vector of poses
'''

def writePoseToFile(name, idx, poses):
    outfile = open(name, 'w')
    num = len(idx)
    for i in range(num):
        outfile.write(str(i) + " ")
        for l in range(16):
            c = l//4
            r = l %4
            outfile.write(str(poses[i][c,r]) + " ")
        outfile.write("\n")
    outfile.close()

'''
 * @brief WriteTimestampToFile Write timestamp to file
 * @param name Filename
 * @param timestamp Vector of timestamps
'''

def writeTimestampToFile(name, timestamp):
    outfile = open(name, 'w')
    num = len(timestamp)
    for i in range(num):
        outfile.write(str(timestamp[i]) + "\n")


def writeLabelsToFolder(path, labels, Ts, indices, numPts):
    array_Ts = np.array(Ts).reshape(-1)
    array_indices = np.array(indices).reshape(-1)
    values = np.unique(array_Ts)
    for val in values:
        name_file = path + "/%010d.bin" % val
        nPts = numPts[val]
        lookedAt = array_Ts==val 
        labelSaved = np.ones(nPts,dtype=np.int16)*(-1) 
        labelSaved[array_indices[lookedAt]] = labels[lookedAt].reshape(-1)
        labelSaved.tofile(name_file)


def writePointCloudsToFolder(path, Md, labels, Ts, indices, numPts):
    array_Ts = np.array(Ts).reshape(-1)
    array_indices = np.array(indices).reshape(-1)
    values = np.unique(array_Ts)
    for val in values:
        name_file = path + "/%010d.bin" % val
        nPts = numPts[val]
        lookedAt = array_Ts==val 
        pc = Md[lookedAt]
        labels_local = labels[lookedAt]
        og_indices = array_indices[lookedAt]
        np.hstack((pc,labels_local.reshape(-1,1),og_indices.reshape(-1,1))).tofile(name_file)

'''
 * @brief _mkdir Create directories recursively
 * @param dir Directory name to be created
 * @return True if succeed or if dir already exists (success)
'''

def mkdir(dir):
    try:
        os.makedirs(dir, exist_ok=True)
        return True
    except:
        return False

def readBinaryPly(pcdFile, n_pts, static=True):

    if static:
        fmt = '=fffBBBiiBf'
    else:
        fmt = '=fffBBBiiBif'
    fmt_len = struct.calcsize(fmt)
    
    with open(pcdFile, 'rb') as f:
        plyData = f.readlines()

    headLine = plyData.index(b'end_header\n')+1
    plyData = plyData[headLine:]
    plyData = b"".join(plyData)

    n_pts_loaded = len(plyData)/fmt_len
    assert(n_pts_loaded==n_pts)
    n_pts_loaded = int(n_pts_loaded)

    data = []
    for i in range(n_pts_loaded):
        pts=struct.unpack(fmt , plyData[i*fmt_len:(i+1)*fmt_len])
        data.append(pts)
    data=np.asarray(data)
    return data
