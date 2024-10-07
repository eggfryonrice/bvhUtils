import numpy as np
from typing import Callable

from BVHFile import BVHFile
from pygameScene import pygameScene, sceneInput
from contactManager import contactManager
from transformationUtil import *

# frame, translationData, eulerData, contactState, discontinuity flag
inertializationManagerInput = tuple[int, np.ndarray, np.ndarray, np.ndarray, bool]


class inertializationManager:
    def __init__(
        self,
        file: BVHFile,
        dataFtn: Callable[[], inertializationManagerInput],
        halfLife: float = 0.15,
        compare: bool = False,
        handleContact: bool = True,
        contactJointNames=["LeftToe", "RightToe"],
        contactHalfLife: float = 0.15,
        unlockRadius: float = 20,
        footHeight: float = 2,
    ):
        self.file: BVHFile = file

        self.dataFtn = dataFtn

        self.halfLife = halfLife

        self.currentData: inertializationManagerInput = self.dataFtn()
        self.nextData: inertializationManagerInput = self.dataFtn()
        self.previousData: inertializationManagerInput = self.currentData

        self.translationOffset = np.array([0, 0, 0])
        self.translationVelocityOffset = np.array([0, 0, 0])

        self.jointsQuatOffset = np.array(
            [np.array([1, 0, 0, 0]) for _ in range(self.file.numJoints)]
        )
        self.jointsQuatVelocityOffset = np.array(
            [np.array([0, 0, 0]) for _ in range(self.file.numJoints)]
        )

        self.compare = compare

        self.handleContact = handleContact
        self.contactManager = contactManager(
            self.file,
            contactJointNames=contactJointNames,
            unlockRadius=unlockRadius,
            footHeight=footHeight,
            halfLife=contactHalfLife,
        )

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
    def manageInertialization(self) -> tuple[int, np.ndarray, np.ndarray]:
        frame, currTranslationData, currQuatData, _, discontinuity = self.currentData

        self.dampOffsets()
        translationData = currTranslationData + self.translationOffset
        quatData = multQuats(self.jointsQuatOffset, currQuatData)

        # in normal case, return offset considered joint position
        if not discontinuity:
            self.previousData = self.currentData
            self.currentData = self.nextData
            self.nextData = self.dataFtn()
            return frame, translationData, quatData

        # on discontinuity, we need information for two frames prior to discontinuity,
        # two frames after discontinuity
        _, prevTranslationData, prevQuatData, _, _ = self.previousData
        frame, nextTranslationData, nextQuatData, _, _ = self.nextData
        nnextData = self.dataFtn()
        _, nnextTranslationData, nnextQuatData, _, _ = nnextData

        # calculate current source root Position and velocity,
        # joints quat and quatVelocity
        prevRootPosition = toCartesian(
            self.file.calculateJointPositionFromQuaternionData(
                0, prevTranslationData, prevQuatData
            )
        )
        currRootPosition = toCartesian(
            self.file.calculateJointPositionFromQuaternionData(
                0, currTranslationData, currQuatData
            )
        )
        currRootVelocity = (currRootPosition - prevRootPosition) / self.file.frameTime

        currQuatVelocity = (
            quatsToScaledAngleAxises(multQuats(currQuatData, invQuats(prevQuatData)))
            / self.file.frameTime
        )

        # calculate next source root Position and velocity,
        # joints quat and quatVelocity
        nextRootPosition = toCartesian(
            self.file.calculateJointPositionFromEulerData(
                0, nextTranslationData, nextQuatData
            )
        )
        nnextRootPosition = toCartesian(
            self.file.calculateJointPositionFromEulerData(
                0, nnextTranslationData, nnextQuatData
            )
        )

        nextRootVelocity = (nnextRootPosition - nextRootPosition) / self.file.frameTime

        nextQuatVelocity = (
            quatsToScaledAngleAxises(multQuats(nnextQuatData, invQuats(nextQuatData)))
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
                multQuats(self.jointsQuatOffset, currQuatData),
                invQuats(nextQuatData),
            )
        )

        self.jointsQuatVelocityOffset = (
            self.jointsQuatVelocityOffset + currQuatVelocity - nextQuatVelocity
        )

        self.previousData = self.nextData
        self.currentData = nnextData
        self.nextData = self.dataFtn()
        return frame, translationData, quatData

    def getNextSceneInput(self) -> sceneInput:
        jointsPositions = []
        linkss = []
        _, currTranslationData, currQuatData, contact, _ = self.currentData
        frame, translationData, quatData = self.manageInertialization()

        if self.handleContact:
            adjustedTranslation, adjustedQuat = self.contactManager.manageContact(
                (translationData, quatData, contact)
            )

        adjustedJointsPosition, adjustedLinks = (
            self.file.calculateJointsPositionAndLinksFromQuaternionData(
                adjustedTranslation, adjustedQuat
            )
        )
        jointsPositions.append((adjustedJointsPosition, (1.0, 0.5, 0.5)))
        linkss.append((adjustedLinks, (0.5, 0.5, 1.0)))

        if self.compare:
            originalTranslation = currTranslationData + self.contactManager.translation
            originalQuat = currQuatData
            originalJointsPosition, originalLinks = (
                self.file.calculateJointsPositionAndLinksFromQuaternionData(
                    originalTranslation, originalQuat
                )
            )
            jointsPositions.append((originalJointsPosition, (1.0, 0.0, 0.0)))
            linkss.append((originalLinks, (1.0, 0.0, 0.0)))

        return (
            frame,
            jointsPositions,
            linkss,
        )


class exampleDataFtn1:
    def __init__(self, filePath):
        file = BVHFile(filePath)
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

    def ftn(self) -> inertializationManagerInput:
        self.file.currentFrame += 1
        if self.file.currentFrame >= self.file.numFrames:
            self.file.currentFrame = 0
            self.translation = self.translation + self.startEndTranslation

        c1 = (
            self.file.getJointVelocity(
                self.file.jointNames.index("LeftToe"), self.file.currentFrame
            )
            < 30
        )
        c2 = (
            self.file.getJointVelocity(
                self.file.jointNames.index("RightToe"), self.file.currentFrame
            )
            < 30
        )

        return (
            self.file.currentFrame,
            (self.file.translationDatas[self.file.currentFrame] + self.translation),
            eulersToQuats(self.file.eulerDatas[self.file.currentFrame]),
            np.array([c1, c2]),
            ((self.file.currentFrame == self.file.numFrames - 1)),
        )


class exampleDataFtn2:
    def __init__(self, filePath):
        file = BVHFile(filePath)
        self.file: BVHFile = file

    def ftn(self) -> inertializationManagerInput:
        self.file.currentFrame += 1
        if self.file.currentFrame >= self.file.numFrames:
            self.file.currentFrame = 0

        c1 = (
            self.file.getJointVelocity(
                self.file.jointNames.index("LeftToe"), self.file.currentFrame
            )
            < 30
        )
        c2 = (
            self.file.getJointVelocity(
                self.file.jointNames.index("RightToe"), self.file.currentFrame
            )
            < 30
        )
        return (
            self.file.currentFrame,
            (
                self.file.translationDatas[self.file.currentFrame]
                + (self.file.currentFrame > 45) * np.array([0, -30, 0])
            ),
            eulersToQuats(self.file.eulerDatas[self.file.currentFrame]),
            np.array([c1, c2]),
            (self.file.currentFrame == 45),
        )


class exampleDataFtn3:
    def __init__(self, filePath1, fileFrame1, filePath2, fileFrame2):
        file1 = BVHFile(filePath1)
        file2 = BVHFile(filePath2)
        self.file1: BVHFile = file1
        self.file2: BVHFile = file2
        self.file1Frame: int = fileFrame1
        self.file2Frame: int = fileFrame2
        self.currentFrame = 0
        self.beforeDiscontinuity = True

        jointsPosition1 = self.file1.calculateJointsPositionFromFrame(self.file1Frame)
        jointsPosition2 = self.file2.calculateJointsPositionFromFrame(self.file2Frame)

        self.afterDiscontinuityTransformation = computeTransformationFromPointsPair(
            jointsPosition1, jointsPosition2
        )

        self.prepTime = 120

    def ftn(self) -> inertializationManagerInput:
        frame = self.currentFrame
        if self.beforeDiscontinuity:
            if self.file1Frame - self.prepTime + self.currentFrame < 0:
                self.currentFrame = self.prepTime - self.file1Frame
                frame = self.currentFrame
            translationData = self.file1.translationDatas[
                self.file1Frame - self.prepTime + frame
            ]
            eulerData = self.file1.eulerDatas[self.file1Frame - self.prepTime + frame]
            quatData = eulersToQuats(eulerData)

            c1 = (
                self.file1.getJointVelocity(
                    self.file1.jointNames.index("LeftToe"),
                    self.file1Frame - self.prepTime + frame,
                )
                < 30
            )
            c2 = (
                self.file1.getJointVelocity(
                    self.file1.jointNames.index("RightToe"),
                    self.file1Frame - self.prepTime + frame,
                )
                < 30
            )
            contact = np.array([c1, c2])

            if frame == self.prepTime:
                self.beforeDiscontinuity = False
                return (
                    frame,
                    translationData,
                    quatData,
                    contact,
                    True,
                )
            self.currentFrame += 1
            return (frame, translationData, quatData, contact, False)
        else:
            if (self.currentFrame > self.prepTime * 2) or (
                self.file2Frame - self.prepTime + self.currentFrame
                >= self.file2.numFrames
            ):
                self.beforeDiscontinuity = True
                self.currentFrame = 0
            translationData = self.file2.translationDatas[
                self.file2Frame - self.prepTime + self.currentFrame
            ]
            eulerData = self.file2.eulerDatas[
                self.file2Frame - self.prepTime + self.currentFrame
            ]
            translationData = toCartesian(
                self.afterDiscontinuityTransformation
                @ toProjective(
                    translationData + toCartesian(self.file2.jointOffsets[0])
                )
            ) - toCartesian(self.file2.jointOffsets[0])
            quatData = eulersToQuats(eulerData)
            quatData[0] = multQuat(
                matToQuat(self.afterDiscontinuityTransformation), quatData[0]
            )

            c1 = (
                self.file2.getJointVelocity(
                    self.file2.jointNames.index("LeftToe"),
                    self.file2Frame - self.prepTime + self.currentFrame,
                )
                < 30
            )
            c2 = (
                self.file2.getJointVelocity(
                    self.file2.jointNames.index("RightToe"),
                    self.file2Frame - self.prepTime + self.currentFrame,
                )
                < 30
            )

            contact = np.array([c1, c2])

            self.currentFrame += 1
            return (frame, translationData, quatData, contact, False)


if __name__ == "__main__":
    dataFtn = exampleDataFtn1("example.bvh")
    dataFtn = exampleDataFtn2("example.bvh")
    dataFtn = exampleDataFtn3("walking.bvh", 241, "dancing.bvh", 99)

    filePath = "example.bvh"
    file = BVHFile(filePath)
    manager = inertializationManager(
        file,
        dataFtn.ftn,
        halfLife=0.15,
        handleContact=True,
        compare=False,
    )
    scene = pygameScene(
        frameTime=file.frameTime,
    )
    while scene.running:
        scene.updateScene(manager.getNextSceneInput())
