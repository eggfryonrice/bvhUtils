import numpy as np

from transformationUtil import *
from createBVH import createBVH


def swapLeftRight(name):
    if name.startswith("Right"):
        return name.replace("Right", "Left", 1)
    elif name.startswith("Left"):
        return name.replace("Left", "Right", 1)
    return name


def getMirroredJoint(reader, joint):
    name = joint.name
    if name.startswith("Right") or name.startswith("Left"):
        return reader.nameToJoint[swapLeftRight(name)]
    elif name == "Site":
        return reader.nameToJoint[swapLeftRight(joint.parent.name)].child[0]
    else:
        return joint


def fixData(data, channels, channelName, value):
    for name, i in channels:
        if name == channelName:
            data[i] = value


def fixDataByEulerAngles(data, channels, eulerAngles):
    fixData(data, channels, "Xrotation", eulerAngles[0])
    fixData(data, channels, "Yrotation", eulerAngles[1])
    fixData(data, channels, "Zrotation", eulerAngles[2])


def getMirroredPosition(position):
    return np.array([position[0], position[1], -1 * position[2]])


def getMirroredData(reader, data):
    mirroredData = [0 for _ in range(len(reader.motionData.data[0]))]
    reader.calculateJointsPositionFromData(data)
    rootJoint = reader.joints[0]
    mirroredPosition = getMirroredPosition(rootJoint.position)
    fixData(
        mirroredData,
        rootJoint.channels,
        "Xposition",
        mirroredPosition[0] - rootJoint.offset[0],
    )
    fixData(
        mirroredData,
        rootJoint.channels,
        "Yposition",
        mirroredPosition[1] - rootJoint.offset[1],
    )
    fixData(
        mirroredData,
        rootJoint.channels,
        "Zposition",
        mirroredPosition[2] - rootJoint.offset[2],
    )
    # getMirroredDataHelper(rootJoint, mirroredData, reader, data, np.eye(4))

    for joint in reader.joints:
        if joint.name == "Site":
            continue
        eulerAngles = [0, 0, 0]
        if joint == rootJoint:
            v1 = getMirroredJoint(reader, joint.child[0]).offset[0:3]
            v2 = getMirroredJoint(reader, joint.child[1]).offset[0:3]
            v3 = getMirroredJoint(reader, joint.child[2]).offset[0:3]
            Rv1 = getMirroredPosition(joint.child[0].position) - getMirroredPosition(
                joint.position
            )
            Rv2 = getMirroredPosition(joint.child[1].position) - getMirroredPosition(
                joint.position
            )
            Rv3 = getMirroredPosition(joint.child[2].position) - getMirroredPosition(
                joint.position
            )
            V = np.column_stack([v1, v2, v3])
            RV = np.column_stack([Rv1, Rv2, Rv3])
            rotation = np.dot(RV, np.linalg.inv(V))
        else:
            for name, i in joint.channels:
                if name == "Xrotation":
                    eulerAngles[0] = data[i]
                if name == "Yrotation":
                    eulerAngles[1] = data[i]
                if name == "Zrotation":
                    eulerAngles[2] = data[i]
            rotation = Rotation.from_euler("xyz", eulerAngles, degrees=True).as_matrix()
            rotation[0, 2] = -1 * rotation[0, 2]
            rotation[1, 2] = -1 * rotation[1, 2]
            rotation[2, 0] = -1 * rotation[2, 0]
            rotation[2, 1] = -1 * rotation[2, 1]
        eulerAngles = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
        fixDataByEulerAngles(
            mirroredData,
            getMirroredJoint(reader, joint.child[0]).parent.channels,
            eulerAngles,
        )

    return mirroredData


def mirrorBVH(filePath, newFilePath):
    file = BVHFile(filePath)
    translationDatas = file.translationDatas
    eulerDatas = file.eulerDatas

    mirroredDatas = []
    for data in datas:
        mirroredDatas.append(getMirroredData(reader, data))
    createBVH(filePath, newFilePath, mirroredDatas)


if __name__ == "__main__":
    mirrorBVH("./lafanWalkingData/left8.bvh", "./right8.bvh")
