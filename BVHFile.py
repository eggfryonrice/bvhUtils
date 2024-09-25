import numpy as np
from numpy.typing import NDArray
import math

from pygameScene import pygameScene
from transformationUtil import *


class BVHFile:
    def __init__(self, fileName: str):
        self.numJoints: int = 0
        self.jointNames: list[str] = []
        jointOffsets: list[list[float]] = []
        self.childToParentDict: dict[int, int] = {}

        translationDatas: list[list[float]] = []
        eulerDatas: list[list[list[float]]] = []
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

        self.jointOffsets: NDArray[np.float64] = np.array(jointOffsets)
        self.jointOffsetShifts: NDArray[np.float64] = defineShifts(
            self.jointOffsets[:, :3]
        )

        self.translationDatas: NDArray[np.float64] = np.array(translationDatas)
        self.eulerDatas: NDArray[np.float64] = np.zeros(
            (self.numFrames, self.numJoints, 3)
        )
        nonEndSiteIdx = [i for i, name in enumerate(self.jointNames) if name != "Site"]
        self.eulerDatas[:, nonEndSiteIdx, :] = eulerDatas
        self.eulerDatas = self.eulerDatas * math.pi / 180

        self.jointsPositions: NDArray[np.float64] = np.zeros(
            (self.numFrames, self.numJoints, 4)
        )
        self.jointsPositionCalculated: list[bool] = [
            False for _ in range(self.numFrames)
        ]

    def getLinks(
        self, jointsPosition: NDArray[np.float64]
    ) -> list[list[NDArray[np.float64]]]:
        links: list[list[NDArray[np.float64]]] = []
        for jointIdx in range(self.numJoints):
            parentIdx = self.childToParentDict[jointIdx]
            if parentIdx >= 0:
                links.append([jointsPosition[jointIdx], jointsPosition[parentIdx]])
        return links

    # calcualte position of all joints using data of given frame
    def calculateJointsPositionByFrame(
        self, frame: int, transformation: NDArray[np.float64] = np.eye(4)
    ) -> NDArray[np.float64]:
        if self.jointsPositionCalculated[frame]:
            return self.jointsPositions[frame] @ transformation.T

        translationData = self.translationDatas[frame]
        eulerData = self.eulerDatas[frame]

        jointsPosition = self.calculateJointsPositionFromData(
            translationData, eulerData, np.eye(4)
        )

        self.jointsPositions[frame] = jointsPosition
        self.jointsPositionCalculated[frame] = True

        return jointsPosition @ transformation.T

    # calcualte position of all joints by given data
    def calculateJointsPositionFromData(
        self,
        translationData: NDArray[np.float64],
        eulerData: NDArray[np.float64],
        transformation: NDArray[np.float64] = np.eye(4),
    ) -> NDArray[np.float64]:
        rotations = defineRotationsFromEulers(eulerData, self.eulerOrder)

        jointsTransformation = np.zeros((self.numJoints, 4, 4))
        jointsTransformation[0] = defineShift(translationData)

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

        return jointsPosition @ transformation.T

    # calculate position of given joint given datas
    def calculateJointPositionFromData(
        self,
        jointIdx: int,
        translationData: NDArray[np.float64],
        eulerData: NDArray[np.float64],
        transformation: NDArray[np.float64] = np.eye(4),
    ) -> NDArray[np.float64]:
        rootToJoint: list[int] = [jointIdx]
        while rootToJoint[-1] != 0:
            rootToJoint.append(self.childToParentDict[rootToJoint[-1]])
        rootToJoint.reverse()

        rotations = defineRotationsFromEulers(eulerData[rootToJoint], self.eulerOrder)

        jointsTransformation = np.zeros((len(rootToJoint), 4, 4))
        jointsTransformation[0] = defineShift(translationData)

        for i in range(1, len(rootToJoint)):
            jointsTransformation[i] = (
                jointsTransformation[i - 1]
                @ self.jointOffsetShifts[self.childToParentDict[rootToJoint[i]]]
                @ rotations[i - 1]
            )
        jointPosition = jointsTransformation[-1] @ self.jointOffsets[jointIdx]

        return transformation @ jointPosition

    # return frame, joint, link information
    def updateSceneWithNextFrame(
        self,
    ) -> tuple[int, NDArray[np.float64], list[list[NDArray[np.float64]]]]:
        jointsPosition = self.calculateJointsPositionByFrame(self.currentFrame)
        links = self.getLinks(jointsPosition)
        currentData = (self.currentFrame, jointsPosition, links)
        self.currentFrame = (self.currentFrame + 1) % self.numFrames
        return currentData


if __name__ == "__main__":
    filePath = "example.bvh"
    file = BVHFile(filePath)
    file.calculateJointsPositionByFrame(0)
    scene = pygameScene(filePath, frameTime=file.frameTime)
    scene.run(file.updateSceneWithNextFrame)
