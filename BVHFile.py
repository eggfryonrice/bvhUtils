import numpy as np
from typing import Callable
import math

from pygameScene import pygameScene, sceneInput
from transformationUtil import *


class BVHFile:
    def __init__(self, fileName: str):
        self.numJoints: int = 0
        self.jointNames: list[str] = []
        jointOffsets: list[list[float]] = []
        self.childToParentDict: dict[int, int] = {}

        translationDatas: list[list[float]] = []
        eulerDatas: list[list[list[float]]] = []
        # here, order means order of rotation channels in bvh file
        self.eulerOrder: str = ""
        self.numFrames: int = 0
        self.frameTime: float = 1.0

        self.currentFrame: int = 0

        # parse bvh file by the first element of each line
        with open(fileName, "r") as f:
            jointStack: list[int] = [-1]

            # parse the upper part of bvh file
            for l in f:
                tokens: list[str] = l.strip().split()
                if not tokens:
                    continue

                if tokens[0] in ["ROOT", "JOINT", "End"]:
                    parent = jointStack[-1]
                    self.numJoints += 1
                    self.jointNames.append(tokens[1])
                    self.childToParentDict[self.numJoints - 1] = parent
                    jointStack.append(len(self.jointNames) - 1)

                elif tokens[0] == "OFFSET":
                    jointOffsets.append(list(map(float, tokens[1:4])) + [1])

                elif tokens[0] == "CHANNELS":
                    if self.eulerOrder == "":
                        for name in tokens:
                            if name == "Xrotation":
                                self.eulerOrder += "x"
                            elif name == "Yrotation":
                                self.eulerOrder += "y"
                            elif name == "Zrotation":
                                self.eulerOrder += "z"

                elif tokens[0] == "}":
                    jointStack.pop()

                elif tokens[0] == "MOTION":
                    break

            # parse the bottom part of bvh file
            for l in f:
                tokens = l.strip().split()

                if tokens[0] == "Frames:":
                    self.numFrames = int(tokens[-1])

                elif tokens[0] == "Frame":
                    self.frameTime = float(tokens[-1])

                else:
                    # we asume translation data is first three data in each line,
                    # and others are all rotation datas,
                    # where offsets doesn't have rotaton data
                    frameData = list(map(float, tokens))
                    translationDatas.append(frameData[:3])
                    eulerDatas.append(
                        [frameData[i : i + 3] for i in range(3, len(frameData), 3)]
                    )

        self.jointOffsets: np.ndarray = np.array(jointOffsets)
        self.jointOffsetShifts: np.ndarray = translationMats(self.jointOffsets[:, :3])

        self.translationDatas: np.ndarray = np.array(translationDatas)
        self.eulerDatas: np.ndarray = np.zeros((self.numFrames, self.numJoints, 3))
        nonEndSiteIdx = [i for i, name in enumerate(self.jointNames) if name != "Site"]
        self.eulerDatas[:, nonEndSiteIdx, :] = eulerDatas
        self.eulerDatas = self.eulerDatas * math.pi / 180

    def getLinks(self, jointsPosition: np.ndarray) -> list[list[np.ndarray]]:
        links: list[list[np.ndarray]] = []
        for jointIdx in range(self.numJoints):
            parentIdx = self.childToParentDict[jointIdx]
            if parentIdx >= 0:
                links.append([jointsPosition[jointIdx], jointsPosition[parentIdx]])
        return links

    def calculateJointsPositionFromQuaternionData(
        self,
        translationData: np.ndarray,
        quaternionData: np.ndarray,
        transformation: np.ndarray = np.eye(4),
    ) -> np.ndarray:
        rotations = quatsToMats(quaternionData)

        jointsTransformation = np.zeros((self.numJoints, 4, 4))
        jointsTransformation[0] = transformation @ translationMat(translationData)

        for jointIdx in range(1, self.numJoints):
            parentIdx = self.childToParentDict[jointIdx]
            jointsTransformation[jointIdx] = (
                jointsTransformation[parentIdx]
                @ self.jointOffsetShifts[parentIdx]
                @ rotations[parentIdx]
            )
        jointsPosition = np.einsum(
            "ijk,ik->ij", jointsTransformation, self.jointOffsets
        )

        return jointsPosition

    def calculateJointPositionFromQuaternionData(
        self,
        jointIdx: int,
        translationData: np.ndarray,
        quaternionData: np.ndarray,
        transformation: np.ndarray = np.eye(4),
    ) -> np.ndarray:
        rootToJoint: list[int] = [jointIdx]
        while rootToJoint[-1] != 0:
            rootToJoint.append(self.childToParentDict[rootToJoint[-1]])
        rootToJoint.reverse()

        rotations = quatsToMats(quaternionData[rootToJoint])

        jointsTransformation = np.zeros((len(rootToJoint), 4, 4))
        jointsTransformation[0] = transformation @ translationMat(translationData)

        for i in range(1, len(rootToJoint)):
            jointsTransformation[i] = (
                jointsTransformation[i - 1]
                @ self.jointOffsetShifts[self.childToParentDict[rootToJoint[i]]]
                @ rotations[i - 1]
            )
        jointPosition = jointsTransformation[-1] @ self.jointOffsets[jointIdx]

        return jointPosition

    # calcualte position of all joints by given data
    def calculateJointsPositionFromEulerData(
        self,
        translationData: np.ndarray,
        eulerData: np.ndarray,
        transformation: np.ndarray = np.eye(4),
    ) -> np.ndarray:
        quaternionData = eulersToQuats(eulerData, self.eulerOrder)
        return self.calculateJointsPositionFromQuaternionData(
            translationData, quaternionData, transformation
        )

    # calculate position of given joint given datas
    def calculateJointPositionFromEulerData(
        self,
        jointIdx: int,
        translationData: np.ndarray,
        eulerData: np.ndarray,
        transformation: np.ndarray = np.eye(4),
    ) -> np.ndarray:
        quaternionData = eulersToQuats(eulerData, self.eulerOrder)
        return self.calculateJointPositionFromQuaternionData(
            jointIdx, translationData, quaternionData, transformation
        )

    # calcualte position of all joints using data of given frame
    def calculateJointsPositionFromFrame(
        self, frame: int, transformation: np.ndarray = np.eye(4)
    ) -> np.ndarray:
        translationData = self.translationDatas[frame]
        eulerData = self.eulerDatas[frame]
        return self.calculateJointsPositionFromEulerData(
            translationData, eulerData, transformation
        )

    def calculateJointPositionFromFrame(
        self, jointIdx: int, frame: int, transformation: np.ndarray = np.eye(4)
    ) -> np.ndarray:
        translationData = self.translationDatas[frame]
        eulerData = self.eulerDatas[frame]
        return self.calculateJointPositionFromEulerData(
            jointIdx, translationData, eulerData, transformation
        )

    def calculateJointsPositionAndLinksFromQuaternionData(
        self,
        translationData: np.ndarray,
        quaternionData: np.ndarray,
        transformation: np.ndarray = np.eye(4),
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        rotations = quatsToMats(quaternionData)

        jointsTransformation = np.zeros((self.numJoints, 4, 4))
        jointsTransformation[0] = transformation @ translationMat(translationData)

        for jointIdx in range(1, self.numJoints):
            parentIdx = self.childToParentDict[jointIdx]
            jointsTransformation[jointIdx] = (
                jointsTransformation[parentIdx]
                @ self.jointOffsetShifts[parentIdx]
                @ rotations[parentIdx]
            )
        jointsPosition = toCartesian(
            np.einsum("ijk,ik->ij", jointsTransformation, self.jointOffsets)
        )

        links = []
        for jointIdx in range(1, self.numJoints):
            parentIdx = self.childToParentDict[jointIdx]
            if np.linalg.norm(toCartesian(self.jointOffsets[jointIdx])) <= 1e-3:
                rotation = np.array([1, 0, 0, 0])
            else:
                # original link is standing parrallel to y axis
                # rotate it directly to offset position
                # and then apply rotation
                rotation = multQuat(
                    matToQuat(jointsTransformation[jointIdx]),
                    vecToVecQuat(
                        np.array([0, 1, 0]), toCartesian(self.jointOffsets[jointIdx])
                    ),
                )
            links.append(
                (jointsPosition[parentIdx], jointsPosition[jointIdx], rotation)
            )

        return jointsPosition, links

    def calculateJointsPositionAndLinksFromEulerData(
        self,
        translationData: np.ndarray,
        eulerData: np.ndarray,
        transformation: np.ndarray = np.eye(4),
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        quaternionData = eulersToQuats(eulerData, self.eulerOrder)
        return self.calculateJointsPositionAndLinksFromQuaternionData(
            translationData, quaternionData, transformation
        )

    def calculateJointsPositionAndLinksFromFrame(
        self, frame: int, transformation: np.ndarray = np.eye(4)
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        translationData = self.translationDatas[frame]
        eulerData = self.eulerDatas[frame]
        return self.calculateJointsPositionAndLinksFromEulerData(
            translationData, eulerData, transformation
        )

    def getJointVelocity(self, idx: int, frame: int) -> float:
        currJointPosition = self.calculateJointPositionFromEulerData(
            idx, self.translationDatas[frame], self.eulerDatas[frame]
        )
        if frame == 0:
            compJointPosition = self.calculateJointPositionFromEulerData(
                idx, self.translationDatas[1], self.eulerDatas[1]
            )
        else:
            compJointPosition = self.calculateJointPositionFromEulerData(
                idx, self.translationDatas[frame - 1], self.eulerDatas[frame - 1]
            )
        return float(
            np.linalg.norm(
                toCartesian(compJointPosition) - toCartesian(currJointPosition)
            )
            / self.frameTime
        )


if __name__ == "__main__":
    filePath = "example.bvh"
    file = BVHFile(filePath)
    scene = pygameScene(frameTime=file.frameTime)

    frame = 0
    while scene.running:
        jointsPosition, links = file.calculateJointsPositionAndLinksFromFrame(frame)
        scene.updateScene(
            (frame, [(jointsPosition, (1.0, 0.5, 0.5))], [(links, (0.5, 0.5, 1.0))])
        )
        frame = (frame + 1) % file.numFrames
