import numpy as np
from typing import Callable

from BVHFile import BVHFile
from pygameScene import pygameScene
from contactManager import contactManager
from transformationUtil import *


class inertializationManager:
    def __init__(
        self,
        file: BVHFile,
        dataFtn: Callable[[], tuple[int, np.ndarray, np.ndarray, bool]],
        halfLife: float = 0.15,
        handleContact=True,
        contactJointNames=["LeftToe", "RightToe"],
        unlockRadius: float = 20,
        footHeight: float = 2,
        contactVelocityThreshold: float = 1,
    ):
        self.file: BVHFile = file

        self.dataFtn = dataFtn

        self.halfLife = halfLife

        self.transformation = np.eye(4)

        self.currentData: tuple[int, np.ndarray, np.ndarray, bool] = self.dataFtn()
        self.nextData: tuple[int, np.ndarray, np.ndarray, bool] = self.dataFtn()
        self.previousData: tuple[int, np.ndarray, np.ndarray, bool] = self.currentData

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

    # discontinuity signs is true on last frame before discontinuity
    # data is given as following. when discontinuity happens at frame 3, data order is
    # 1, 2, 3, 3, 4, 5, ....
    # where first 3 is motion before discontinuity at frame 3,
    # and second 3 is motion after discontinuity at frame 3
    def adjustJointsPosition(self) -> tuple[int, np.ndarray]:
        frame, currTranslationData, currEulerData, discontinuity = self.currentData

        self.dampOffsets()
        translationData = currTranslationData + self.translationOffset
        quaternionData = multQuats(self.jointsQuatOffset, eulersToQuats(currEulerData))
        jointsPosition = self.file.calculateJointsPositionFromQuaternionData(
            translationData, quaternionData, self.transformation
        )

        # in normal case, return offset considered joint position
        if not discontinuity:
            self.previousData = self.currentData
            self.currentData = self.nextData
            self.nextData = self.dataFtn()
            return frame, jointsPosition

        # on discontinuity, we need information for two frames prior to discontinuity,
        # two frames after discontinuity
        _, prevTranslationData, prevEulerData, _ = self.previousData
        frame, nextTranslationData, nextEulerData, _ = self.nextData
        nnextData = self.dataFtn()
        _, nnextTranslationData, nnextEulerData, _ = nnextData

        # calculate current source root Position and velocity,
        # joints quat and quatVelocity
        prevRootPosition = toCartesian(
            self.file.calculateJointPositionFromData(
                0, prevTranslationData, prevEulerData, self.transformation
            )
        )
        currRootPosition = toCartesian(
            self.file.calculateJointPositionFromData(
                0, currTranslationData, currEulerData, self.transformation
            )
        )
        currRootVelocity = (currRootPosition - prevRootPosition) / self.file.frameTime

        prevQuaternionData = eulersToQuats(prevEulerData)
        currQuaternionData = eulersToQuats(currEulerData)
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
                0, nextTranslationData, nextEulerData, self.transformation
            )
        )
        nnextRootPosition = toCartesian(
            self.file.calculateJointPositionFromData(
                0, nnextTranslationData, nnextEulerData, self.transformation
            )
        )
        nextRootVelocity = (nnextRootPosition - nextRootPosition) / self.file.frameTime

        nextQuaternionData = eulersToQuats(nextEulerData)
        nnextQuaternionData = eulersToQuats(nnextEulerData)
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


class exampleDataFtn:
    def __init__(self, file: BVHFile):
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

    def ftn(self) -> tuple[int, np.ndarray, np.ndarray, bool]:
        self.file.currentFrame += 1
        if self.file.currentFrame >= self.file.numFrames:
            self.file.currentFrame = 0
            self.translation = self.translation + self.startEndTranslation

        return (
            self.file.currentFrame,
            self.file.translationDatas[self.file.currentFrame] + self.translation,
            self.file.eulerDatas[self.file.currentFrame],
            self.file.currentFrame == self.file.numFrames - 1,
        )


if __name__ == "__main__":
    filePath = "example.bvh"
    file = BVHFile(filePath)
    dataFtn = exampleDataFtn(file)
    manager = inertializationManager(file, dataFtn.ftn, handleContact=True)
    scene = pygameScene(
        filePath, frameTime=file.frameTime, cameraRotation=np.array([0, math.pi, 0])
    )
    scene.run(manager.updateScene)
