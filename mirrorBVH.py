import numpy as np

from transformationUtil import *
from BVHFile import BVHFile
from createBVH import createBVH


def swapLeftRight(name):
    if name.startswith("Right"):
        return name.replace("Right", "Left", 1)
    elif name.startswith("Left"):
        return name.replace("Left", "Right", 1)
    return name


def getMirroredJoint(file: BVHFile, jointIdx):
    name = file.jointNames[jointIdx]
    if name.startswith("Right") or name.startswith("Left"):
        return file.jointNames.index(swapLeftRight(name))
    elif name == "Site":
        parentName = file.jointNames[jointIdx - 1]
        return file.jointNames.index(swapLeftRight(parentName)) + 1
    else:
        return jointIdx


def getMirroredPosition(position):
    return np.array([position[0], position[1], -1 * position[2]])


def getMirroredData(file: BVHFile, i: int):
    translationData = file.translationDatas[i]
    eulerData = file.eulerDatas[i]
    jointsPosition = file.calculateJointsPositionFromEulerData(
        translationData, eulerData
    )
    mirroredTranslationData = getMirroredPosition(
        toCartesian(jointsPosition[0])
    ) - toCartesian(file.jointOffsets[0])

    mirroredEulerData = []
    index = 0
    for jointIdx in range(file.numJoints):
        if file.jointNames[jointIdx] == "Site":
            continue
        eulerAngle = eulerData[index * 3 : index * 3 + 3]
        index += 1
        rotation = eulerToMat(eulerAngle, order=file.eulerOrder)
        rotation[0, 2] = -1 * rotation[0, 2]
        rotation[1, 2] = -1 * rotation[1, 2]
        rotation[2, 0] = -1 * rotation[2, 0]
        rotation[2, 1] = -1 * rotation[2, 1]
        # rotation matrix to euler

    return mirroredTranslationData, np.array(mirroredEulerData)


def mirrorBVH(filePath, newFilePath):
    file = BVHFile(filePath)

    mirroredDatas = []
    for i in range(file.numFrames):
        mirroredDatas.append(getMirroredData(file, i))


if __name__ == "__main__":
    mirrorBVH("./lafanWalkingData/left8.bvh", "./right8.bvh")
