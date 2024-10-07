import numpy as np
from typing import Callable

from BVHFile import BVHFile
from pygameScene import pygameScene, sceneInput
from transformationUtil import *

# translationData, quaternionData, contact
contactManagerInput = tuple[np.ndarray, np.ndarray, np.ndarray]


class contactJointHandler:
    def __init__(
        self,
        initialPosition: np.ndarray,
        frameTime: float,
        unlockRadius: float,
        footHeight: float,
        halfLife: float,
    ):
        self.frameTime = frameTime

        self.unlockRadius = unlockRadius
        self.footHeight = footHeight
        self.halfLife = halfLife

        self.priorContactState: bool = False
        self.lock: bool = False

        # position that will be displayed
        self.position: np.ndarray = toCartesian(initialPosition)
        self.positionOffset: np.ndarray = np.array([0, 0, 0])
        # moving velocity
        self.velocity: np.ndarray = np.array([0, 0, 0])
        self.velocityOffset: np.ndarray = np.array([0, 0, 0])
        # position of contact
        self.contactPoint: np.ndarray = np.array([0, 0, 0])
        # prior position of input
        self.priorInputPosition: np.ndarray = self.position

    def dampOffset(self):
        y = 2 * 0.6931 / self.halfLife
        j1 = self.velocityOffset + self.positionOffset * y
        eydt = np.exp(-y * self.frameTime)

        self.positionOffset = eydt * (self.positionOffset + j1 * self.frameTime)
        self.velocityOffset = eydt * (self.velocityOffset - j1 * y * self.frameTime)

    def handleContact(self, inputPosition: np.ndarray, contactState: bool):
        inputPosition = inputPosition.copy()
        inputPosition[1] = max([self.footHeight, inputPosition[1]])
        inputVelocity = (inputPosition - self.priorInputPosition) / self.frameTime
        self.priorInputPosition = inputPosition

        if (not self.priorContactState) and contactState:
            self.lock = True
            self.contactPoint = self.position
            self.contactPoint[1] = self.footHeight
            self.positionOffset = self.position - self.contactPoint
            self.velocityOffset = self.velocity - np.array([0, 0, 0])

        if self.lock and (
            (np.linalg.norm(inputPosition - self.contactPoint) > self.unlockRadius)
            or (self.priorContactState and (not contactState))
        ):
            self.lock = False
            self.positionOffset = self.position - inputPosition
            self.velocityOffset = self.velocity - inputVelocity

        self.priorContactState = contactState

        # detect sudden discontinuity
        if np.linalg.norm(self.position - inputPosition) > self.unlockRadius * 3:
            self.lock = False
            self.positionOffset = np.array([0, 0, 0])
            self.velocityOffset = np.array([0, 0, 0])

        if self.lock:
            targetPosition = self.contactPoint
            targetVelocity = np.array([0, 0, 0])
        else:
            targetPosition = inputPosition
            targetVelocity = inputVelocity

        self.dampOffset()

        self.position = targetPosition + self.positionOffset
        self.velocity = targetVelocity + self.velocityOffset

        self.position[1] = max([self.footHeight, self.position[1]])

        return self.position


class contactManager:
    def __init__(
        self,
        file: BVHFile,
        contactJointNames=["LeftToe", "RightToe"],
        unlockRadius: float = 20,
        footHeight: float = 0,
        halfLife: float = 0.15,
    ):
        self.file: BVHFile = file

        self.contactJoints = list(map(file.jointNames.index, contactJointNames))
        self.contactJointsP1 = [
            self.file.childToParentDict[jointIdx] for jointIdx in self.contactJoints
        ]
        self.contactJointsP2 = [
            self.file.childToParentDict[jointIdx] for jointIdx in self.contactJointsP1
        ]
        self.contactJointsP3 = [
            self.file.childToParentDict[jointIdx] for jointIdx in self.contactJointsP2
        ]
        self.contactJointsEndSite = [jointIdx + 1 for jointIdx in self.contactJoints]

        self.translation = np.array([0, 0, 0])

        self.unlockRadius = unlockRadius
        self.footHeight = footHeight
        self.halfLife = halfLife

        self.initializedByFirstData = False

    # given position of joints, find joint with lowest y value
    # then transform the animation so that lowest joint globally has footHeight as height
    def adjustHeight(self, jointsPosition: np.ndarray):
        jointsHeight = jointsPosition[:, 1] / jointsPosition[:, 3]
        lowestJointHeight = np.min(jointsHeight)
        self.translation = (
            np.array([0, self.footHeight - lowestJointHeight, 0]) + self.translation
        )

    def initializeHandlers(self, jointsPosition: np.ndarray):
        self.contactHandlers: list[contactJointHandler] = []
        for jointIdx in self.contactJoints:
            handler = contactJointHandler(
                jointsPosition[jointIdx],
                self.file.frameTime,
                unlockRadius=self.unlockRadius,
                footHeight=self.footHeight,
                halfLife=self.halfLife,
            )
            self.contactHandlers.append(handler)

    def manageContact(self, inputData: contactManagerInput):
        translationData, quaternionData, contact = inputData
        jointsPosition = self.file.calculateJointsPositionFromQuaternionData(
            translationData + self.translation, quaternionData
        )

        if not self.initializedByFirstData:
            # move character so that feet is on the ground
            self.adjustHeight(jointsPosition)
            # initialize handler for each contact joint
            self.initializeHandlers(jointsPosition)
            # marked that class is now initialized
            self.initializedByFirstData = True
            # initial position is character moved to the ground
            adjustedTranslationData = translationData + self.translation
            return adjustedTranslationData, quaternionData

        adjustedTranslationData = translationData + self.translation
        adjustedQuaternionData = quaternionData.copy()

        # calculate where contact joint should move by contact handler
        for i in range(len(self.contactJoints)):
            p0Idx = self.contactJoints[i]
            p1Idx = self.contactJointsP1[i]
            p2Idx = self.contactJointsP2[i]
            p3Idx = self.contactJointsP3[i]
            # get toe position by contact handler
            p0 = toCartesian(jointsPosition[p0Idx])
            p0H = self.contactHandlers[i].handleContact(p0, contact[i])

            p1 = toCartesian(jointsPosition[p1Idx])
            p1H = p1 + p0H - p0
            p2 = toCartesian(jointsPosition[p2Idx])
            p3 = toCartesian(jointsPosition[p3Idx])

            d12 = np.linalg.norm(p2 - p1)
            d23 = np.linalg.norm(p3 - p2)
            d13 = np.linalg.norm(p3 - p1H)
            # handle when contact point is further then leg lenth
            eps = 1e-2
            if d13 >= (d12 + d23) * (1 - eps):
                p1H = p3 + (p1H - p3) / d13 * (d12 + d23) * (1 - eps)
                p0H = p1H + p0 - p1
                d13 = (d12 + d23) * (1 - eps)

            # for resulting p1H, p2H, p3,
            # when we lay foot of perpendicular from p2H to p1H-p3,
            # then distance from that point from p1H will be saved as d
            d = (d13**2 - d23**2 + d12**2) / (2 * d13)
            # put p2 into the plane with plane vector p3-p1 and p2Dirction
            # we first move to appropriate point over p3-p1H
            # then we move p2H along the plane
            p2H = p1H + normalize(p3 - p1H) * d
            # we take p2 Direciton this way:
            p2Direction1 = orthogonalComponent(p3 - p1H, p2 - p3)
            p2Direction2 = orthogonalComponent(p3 - p1H, p0 - p1)
            p2Direction = p2Direction1 * 2 + p2Direction2
            n = np.cross((p3 - p1H), p2Direction)
            n = np.cross(n, (p3 - p1H))
            n = normalize(n)
            p2H = p2H + n * ((d12**2 - d**2) ** 0.5)
            # ensure knee doesn't go back
            originalCross = np.cross(p3 - p2, p2 - p1)
            newCross = np.cross(p3 - p2H, p2H - p1H)
            if np.dot(originalCross, newCross) < 0:
                p2H = p2H - 2 * n * ((d12**2 - d**2) ** 0.5)

            # now we adjust rotation to be adjusted with the new position of p0, p1, p2
            # p3's parent's global rotation
            p3ParentGlobalRotation = np.array([1, 0, 0, 0])
            Idx = p3Idx
            while self.file.childToParentDict[Idx] >= 0:
                Idx = self.file.childToParentDict[Idx]
                p3ParentGlobalRotation = multQuat(
                    quaternionData[Idx], p3ParentGlobalRotation
                )
            # adjust rotation of p3 to adjust p2 location
            adjustedQuaternionData[p3Idx] = multQuat(
                multQuat(
                    invQuat(p3ParentGlobalRotation), vecToVecQuat(p2 - p3, p2H - p3)
                ),
                multQuat(p3ParentGlobalRotation, quaternionData[p3Idx]),
            )
            # p2's parent's global rotation after p3 adjustment
            p2ParentGlobalRotation = multQuat(
                p3ParentGlobalRotation, adjustedQuaternionData[p3Idx]
            )
            # calculate new position of p1 after adjusting p2
            p1AfterP2Adjust = toCartesian(
                self.file.calculateJointPositionFromQuaternionData(
                    p1Idx, adjustedTranslationData, adjustedQuaternionData
                )
            )
            # adjust rotation of p2 by new position of p1
            adjustedQuaternionData[p2Idx] = multQuat(
                multQuat(
                    invQuat(p2ParentGlobalRotation),
                    vecToVecQuat(p1AfterP2Adjust - p2H, p1H - p2H),
                ),
                multQuat(p2ParentGlobalRotation, quaternionData[p2Idx]),
            )
            # p1's parent's global rotation after p2 adjustment
            p1ParentGlobalRotation = multQuat(
                p2ParentGlobalRotation, adjustedQuaternionData[p2Idx]
            )
            # calculate new position of p0 after adjusting p1
            p0AfterP1Adjust = toCartesian(
                self.file.calculateJointPositionFromQuaternionData(
                    p0Idx, adjustedTranslationData, adjustedQuaternionData
                )
            )
            # adjust rotation of p1 by new position of p0
            adjustedQuaternionData[p1Idx] = multQuat(
                multQuat(
                    invQuat(p1ParentGlobalRotation),
                    vecToVecQuat(p0AfterP1Adjust - p1H, p0H - p1H),
                ),
                multQuat(p1ParentGlobalRotation, quaternionData[p1Idx]),
            )

            # we should adjust rotation of p0 in the future when offset lengfth is not zero
            # maybe next time....
        return adjustedTranslationData, adjustedQuaternionData


class exampleDataFtn:
    def __init__(self, file: BVHFile, contactVelocityThreshold: float = 30):
        self.file: BVHFile = file
        self.contactVelocityThreshold = contactVelocityThreshold

    def ftn(self) -> tuple[int, contactManagerInput]:
        prevFrame = self.file.currentFrame
        self.file.currentFrame = (self.file.currentFrame + 1) % self.file.numFrames
        prevPosition = toCartesian(
            self.file.calculateJointsPositionFromFrame(prevFrame)[[4, 9]]
        )
        currentPosition = toCartesian(
            self.file.calculateJointsPositionFromFrame(self.file.currentFrame)[[4, 9]]
        )
        velocity = (currentPosition - prevPosition) / self.file.frameTime
        speed = np.linalg.norm(velocity, axis=1)

        translationData = (
            self.file.translationDatas[self.file.currentFrame]
            + np.array([self.file.currentFrame * 0, 0, 0])
            + (self.file.currentFrame > 45) * np.array([0, -30, 0])
        )
        quaternionData = eulersToQuats(self.file.eulerDatas[self.file.currentFrame])

        return (
            self.file.currentFrame,
            (translationData, quaternionData, speed < self.contactVelocityThreshold),
        )


if __name__ == "__main__":
    filePath = "example.bvh"
    file = BVHFile(filePath)
    scene = pygameScene(
        frameTime=1 * file.frameTime,
    )
    dataFtn = exampleDataFtn(file)
    manager = contactManager(file)

    while scene.running:
        jointsPositions, linkss = [], []
        frame, data = dataFtn.ftn()
        translationData, quaternionData = manager.manageContact(data)
        jointsPosition, links = file.calculateJointsPositionAndLinksFromQuaternionData(
            translationData, quaternionData
        )

        jointsPositions.append((jointsPosition, (1.0, 0.5, 0.5)))
        linkss.append((links, (0.5, 0.5, 1.0)))

        originalJointsPosition, originalLinks = (
            file.calculateJointsPositionAndLinksFromQuaternionData(
                data[0] + manager.translation, data[1]
            )
        )
        jointsPositions.append((originalJointsPosition, (1.0, 0.0, 0.0)))
        linkss.append((originalLinks, (1.0, 0.0, 0.0)))

        scene.updateScene((frame, jointsPositions, linkss))
