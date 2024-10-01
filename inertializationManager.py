import numpy as np
from typing import Callable

from BVHFile import BVHFile
from pygameScene import pygameScene
from contactManager import contactManager
from transformationUtil import *

# frame, translationData, eulerData, transformation, discontinuity flag
managerInput = tuple[int, np.ndarray, np.ndarray, np.ndarray, bool]


class inertializationManager:
    def __init__(
        self,
        file: BVHFile,
        dataFtn: Callable[[], managerInput],
        halfLife: float = 0.35,
        handleContact=True,
        contactJointNames=["LeftToe", "RightToe"],
        unlockRadius: float = 20,
        footHeight: float = 2,
        contactVelocityThreshold: float = 1,
    ):
        self.file: BVHFile = file

        self.dataFtn = dataFtn

        self.halfLife = halfLife

        self.currentData: managerInput = self.dataFtn()
        self.nextData: managerInput = self.dataFtn()
        self.previousData: managerInput = self.currentData

        self.translationOffset = np.array([0, 0, 0])
        self.translationVelocityOffset = np.array([0, 0, 0])

        self.jointsQuatOffset = np.array(
            [np.array([1, 0, 0, 0]) for _ in range(self.file.numJoints)]
        )
        self.jointsQuatVelocityOffset = np.array(
            [np.array([0, 0, 0]) for _ in range(self.file.numJoints)]
        )

        self.handleContact = handleContact
        self.previousJointsPositionExists = False
        self.contactManager = contactManager(
            self.file,
            contactJointNames=contactJointNames,
            unlockRadius=unlockRadius,
            footHeight=footHeight,
            halfLife=self.halfLife,
        )
        self.contactVelocityThreshold = contactVelocityThreshold

    def dampOffsets(self):
        y = 2 * 0.6931 / self.halfLife
        eydt = np.exp(-y * self.file.frameTime)

        j1 = self.translationVelocityOffset + self.translationOffset * y

        self.translationOffset = eydt * (
            self.translationOffset + j1 * self.file.frameTime
        )
        self.translationVelocityOffset = eydt * (
            self.translationVelocityOffset - j1 * y * self.file.frameTime
        )

        j0 = quatsToScaledAngleAxises(self.jointsQuatOffset)
        j1 = self.jointsQuatVelocityOffset + j0 * y
        self.jointsQuatOffset = scaledAngleAxisesToQuats(
            eydt * (j0 + j1 * self.file.frameTime)
        )
        self.jointsQuatVelocityOffset = eydt * (
            self.jointsQuatVelocityOffset - j1 * y * self.file.frameTime
        )

    def getQuaternionData(
        self, eulerData: np.ndarray, transformation: np.ndarray
    ) -> np.ndarray:
        quaternionData = eulersToQuats(eulerData)
        quaternionData[0] = multQuat(matToQuat(transformation), quaternionData[0])
        return quaternionData

    # discontinuity signs is true on last frame before discontinuity
    # data is given as following. when discontinuity happens at frame 3, data order is
    # 1, 2, 3, 3, 4, 5, ....
    # where first 3 is motion before discontinuity at frame 3,
    # and second 3 is motion after discontinuity at frame 3
    def adjustJointsPosition(self) -> tuple[int, np.ndarray]:
        frame, currTranslationData, currEulerData, currTransformation, discontinuity = (
            self.currentData
        )

        self.dampOffsets()
        translationData = currTranslationData + self.translationOffset
        quaternionData = multQuats(self.jointsQuatOffset, eulersToQuats(currEulerData))
        jointsPosition = self.file.calculateJointsPositionFromQuaternionData(
            translationData, quaternionData, currTransformation
        )

        # in normal case, return offset considered joint position
        if not discontinuity:
            self.previousData = self.currentData
            self.currentData = self.nextData
            self.nextData = self.dataFtn()
            return frame, jointsPosition

        # on discontinuity, we need information for two frames prior to discontinuity,
        # two frames after discontinuity
        _, prevTranslationData, prevEulerData, prevTransformation, _ = self.previousData
        frame, nextTranslationData, nextEulerData, nextTransformation, _ = self.nextData
        nnextData = self.dataFtn()
        _, nnextTranslationData, nnextEulerData, nnextTransformation, _ = nnextData

        # calculate current source root Position and velocity,
        # joints quat and quatVelocity
        prevRootPosition = toCartesian(
            self.file.calculateJointPositionFromData(
                0, prevTranslationData, prevEulerData, prevTransformation
            )
        )
        currRootPosition = toCartesian(
            self.file.calculateJointPositionFromData(
                0, currTranslationData, currEulerData, currTransformation
            )
        )
        currRootVelocity = (currRootPosition - prevRootPosition) / self.file.frameTime

        prevQuaternionData = self.getQuaternionData(prevEulerData, prevTransformation)
        currQuaternionData = self.getQuaternionData(currEulerData, currTransformation)
        currQuatVelocity = (
            quatsToScaledAngleAxises(
                multQuats(currQuaternionData, invQuats(prevQuaternionData))
            )
            / self.file.frameTime
        )

        # calculate next source root Position and velocity,
        # joints quat and quatVelocity
        nextRootPosition = toCartesian(
            self.file.calculateJointPositionFromData(
                0, nextTranslationData, nextEulerData, nextTransformation
            )
        )
        nnextRootPosition = toCartesian(
            self.file.calculateJointPositionFromData(
                0, nnextTranslationData, nnextEulerData, nnextTransformation
            )
        )

        nextRootVelocity = (nnextRootPosition - nextRootPosition) / self.file.frameTime

        nextQuaternionData = self.getQuaternionData(nextEulerData, nextTransformation)
        nnextQuaternionData = self.getQuaternionData(
            nnextEulerData, nnextTransformation
        )
        nextQuatVelocity = (
            quatsToScaledAngleAxises(
                multQuats(nnextQuaternionData, invQuats(nextQuaternionData))
            )
            / self.file.frameTime
        )

        # update offset
        self.translationOffset = (
            self.translationOffset + currRootPosition - nextRootPosition
        )
        self.translationVelocityOffset = (
            self.translationVelocityOffset + currRootVelocity - nextRootVelocity
        )
        self.jointsQuatOffset = absQuats(
            multQuats(
                multQuats(self.jointsQuatOffset, currQuaternionData),
                invQuats(nextQuaternionData),
            )
        )

        self.jointsQuatVelocityOffset = (
            self.jointsQuatVelocityOffset + currQuatVelocity - nextQuatVelocity
        )

        self.previousData = self.nextData
        self.currentData = nnextData
        self.nextData = self.dataFtn()
        return frame, jointsPosition

    def updateScene(self) -> tuple[
        int,
        np.ndarray,
        list[list[np.ndarray]],
        list[tuple[np.ndarray, tuple[int, int, int]]],
    ]:
        frame, jointsPosition = self.adjustJointsPosition()

        if self.handleContact:
            if not self.previousJointsPositionExists:
                self.previousJointsPosition = jointsPosition
                self.previousJointsPositionExists = True
            contactJointVelocity = (
                np.linalg.norm(
                    jointsPosition[self.contactManager.contactJoints]
                    - self.previousJointsPosition[self.contactManager.contactJoints],
                    axis=1,
                )
                / self.file.frameTime
            )
            self.previousJointsPosition = jointsPosition
            jointsPosition = self.contactManager.adjustJointsPosition(
                jointsPosition, contactJointVelocity < self.contactVelocityThreshold
            )

        links = self.file.getLinks(jointsPosition)
        return frame, jointsPosition, links, []


class exampleDataFtn1:
    def __init__(self):
        file = BVHFile("example.bvh")
        self.file: BVHFile = file
        self.translation = np.array([0, 0, 0])

        startRootPosition = toCartesian(
            self.file.calculateJointsPositionFromFrame(0)[0]
        )
        endRootPosition = toCartesian(
            self.file.calculateJointsPositionFromFrame(self.file.numFrames - 1)[0]
        )
        self.startEndTranslation = (endRootPosition - startRootPosition) * np.array(
            [1, 0, 1]
        )

    def ftn(self) -> managerInput:
        self.file.currentFrame += 1
        if self.file.currentFrame >= self.file.numFrames:
            self.file.currentFrame = 0
            self.translation = self.translation + self.startEndTranslation

        return (
            self.file.currentFrame,
            (self.file.translationDatas[self.file.currentFrame] + self.translation),
            self.file.eulerDatas[self.file.currentFrame],
            np.eye(4),
            ((self.file.currentFrame == self.file.numFrames - 1)),
        )


class exampleDataFtn2:
    def __init__(self):
        file = BVHFile("example.bvh")
        self.file: BVHFile = file

    def ftn(self) -> managerInput:
        self.file.currentFrame += 1
        if self.file.currentFrame >= self.file.numFrames:
            self.file.currentFrame = 0
        return (
            self.file.currentFrame,
            (
                self.file.translationDatas[self.file.currentFrame]
                + (self.file.currentFrame > 45) * np.array([0, -30, 0])
            ),
            self.file.eulerDatas[self.file.currentFrame],
            np.eye(4),
            (self.file.currentFrame == 45),
        )


class exampleDataFtn3:
    def __init__(self):
        file1 = BVHFile("walking.bvh")
        file2 = BVHFile("dancing.bvh")
        self.file1: BVHFile = file1
        self.file2: BVHFile = file2
        self.file1Frame: int = 227
        self.file2Frame: int = 99
        self.currentFrame = 0
        self.beforeDiscontinuity = True

        jointsPosition1 = self.file1.calculateJointsPositionFromFrame(self.file1Frame)
        jointsPosition2 = self.file2.calculateJointsPositionFromFrame(self.file2Frame)
        self.afterDiscontinuityTransformation = computeTransformationFromPointsPair(
            jointsPosition1, jointsPosition2
        )

    def ftn(self) -> managerInput:
        if self.beforeDiscontinuity:
            if self.currentFrame == 60:
                self.beforeDiscontinuity = False
                return (
                    self.currentFrame,
                    self.file1.translationDatas[
                        self.file1Frame - 60 + self.currentFrame
                    ],
                    self.file1.eulerDatas[self.file1Frame - 60 + self.currentFrame],
                    np.eye(4),
                    True,
                )
            data = (
                self.currentFrame,
                self.file1.translationDatas[self.file1Frame - 60 + self.currentFrame],
                self.file1.eulerDatas[self.file1Frame - 60 + self.currentFrame],
                np.eye(4),
                False,
            )
            self.currentFrame += 1
            return data
        else:
            data = (
                self.currentFrame,
                self.file2.translationDatas[self.file2Frame - 60 + self.currentFrame],
                self.file2.eulerDatas[self.file2Frame - 60 + self.currentFrame],
                self.afterDiscontinuityTransformation,
                False,
            )
            self.currentFrame += 1
            if self.currentFrame > 120:
                self.beforeDiscontinuity = True
                self.currentFrame = 0
            return data


if __name__ == "__main__":
    filePath = "example.bvh"
    file = BVHFile(filePath)
    dataFtn = exampleDataFtn3()
    manager = inertializationManager(file, dataFtn.ftn, handleContact=False)
    scene = pygameScene(
        filePath, frameTime=3 * file.frameTime, cameraRotation=np.array([0, math.pi, 0])
    )
    scene.run(manager.updateScene)
