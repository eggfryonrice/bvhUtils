import numpy as np

from transformationUtil import *
from BVHFile import BVHFile
from createBVH import createBVH
from pygameScene import pygameScene


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

    mirroredEulerData = np.zeros_like(eulerData)
    index = 0
    for jointIdx in range(file.numJoints):
        eulerAngle = eulerData[index]
        index += 1
        rotation = eulerToMat(eulerAngle, order=file.eulerOrder)
        rotation[0, 2] = -1 * rotation[0, 2]
        rotation[1, 2] = -1 * rotation[1, 2]
        rotation[2, 0] = -1 * rotation[2, 0]
        rotation[2, 1] = -1 * rotation[2, 1]
        mirroredEulerAngle = matToEuler(rotation, order=file.eulerOrder)
        mirroredJointIdx = getMirroredJoint(file, jointIdx)
        mirroredEulerData[mirroredJointIdx] = mirroredEulerAngle

    return mirroredTranslationData, mirroredEulerData


def mirrorBVH(filePath, newFilePath):
    file = BVHFile(filePath)
    nonEndSiteIdx = [i for i, name in enumerate(file.jointNames) if name != "Site"]
    datas = np.zeros((file.numFrames, len(nonEndSiteIdx) * 3 + 3))
    for i in range(file.numFrames):
        translationData, eulerData = getMirroredData(file, i)
        datas[i, :3] = translationData
        datas[i, 3:] = (eulerData[nonEndSiteIdx, :] * 180 / math.pi).reshape(
            (len(nonEndSiteIdx) * 3)
        )
    createBVH(filePath, newFilePath, np.array(datas))


if __name__ == "__main__":
    originalFilePath = "example.bvh"
    mirroredFilePath = "example_cut.bvh"
    mirrorBVH(originalFilePath, mirroredFilePath)
    originalFile = BVHFile(originalFilePath)
    mirroredFile = BVHFile(mirroredFilePath)
    scene = pygameScene(mirroredFile.frameTime)
    frame = 0
    originalStartRootPosition = originalFile.calculateJointPositionFromFrame(0, 0)
    mirroredStartRootPosition = mirroredFile.calculateJointPositionFromFrame(0, 0)
    transformation = translationMat(
        toCartesian(mirroredStartRootPosition) - toCartesian(originalStartRootPosition)
    )
    while scene.running:
        originalJointsPosition, originalLinks = (
            originalFile.calculateJointsPositionAndLinksFromFrame(frame, transformation)
        )
        mirroredJointsPosition, mirroredLinks = (
            mirroredFile.calculateJointsPositionAndLinksFromFrame(frame)
        )
        scene.updateScene(
            (
                frame,
                [
                    (mirroredJointsPosition, (1.0, 0.5, 0.5)),
                    (originalJointsPosition, (0.5, 0.0, 0.0)),
                ],
                [(mirroredLinks, (0.5, 0.5, 1.0)), (originalLinks, (0.5, 0.0, 0.0))],
            )
        )
        frame = (frame + 1) % mirroredFile.numFrames
