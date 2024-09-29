import numpy as np
from numpy.typing import NDArray
from typing import Callable

from BVHFile import BVHFile
from pygameScene import pygameScene
from transformationUtil import *


class contactJointHandler:
    def __init__(
        self,
        initialPosition: NDArray[np.float64],
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
        self.position: NDArray[np.float64] = toCartesian(initialPosition)
        self.positionOffset: NDArray[np.float64] = np.array([0, 0, 0])
        # moving velocity
        self.velocity: NDArray[np.float64] = np.array([0, 0, 0])
        self.velocityOffset: NDArray[np.float64] = np.array([0, 0, 0])
        # position of contact
        self.contactPoint: NDArray[np.float64] = np.array([0, 0, 0])
        # prior position of input
        self.priorInputPosition: NDArray[np.float64] = self.position

    def dampOffset(
        self,
    ):

        y = 2 * 0.6931 / self.halfLife
        j1 = self.velocityOffset + self.positionOffset * y
        eydt = np.exp(-y * self.frameTime)

        self.positionOffset = eydt * (self.positionOffset + j1 * self.frameTime)
        self.velocityOffset = eydt * (self.velocityOffset - j1 * y * self.frameTime)

    def handleContact(self, inputPosition: NDArray[np.float64], contactState: bool):
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
        dataFtn: Callable[
            [], tuple[int, NDArray[np.float64], NDArray[np.float64], NDArray[np.bool]]
        ],
        contactJointNames=["LeftToe", "RightToe"],
        unlockRadius: float = 20,
        footHeight: float = 2,
        halfLife: float = 0.03,
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

        self.transformation = np.eye(4)

        self.dataFtn = dataFtn

        self.unlockRadius = unlockRadius
        self.footHeight = footHeight
        self.halfLife = halfLife

        self.initializedByFirstData = False

    # given position of joints, find joint with lowest y value
    # then transform the animation so that lowest joint globally has footHeight as height
    def adjustHeight(self, jointsPosition: NDArray[np.float64]):
        jointsHeight = jointsPosition[:, 1] / jointsPosition[:, 3]
        lowestJointHeight = np.min(jointsHeight)
        self.transformation = (
            translationMat(np.array([0, self.footHeight - lowestJointHeight, 0]))
            @ self.transformation
        )

    def initializeHandlers(self, jointsPosition: NDArray[np.float64]):
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

    def adjustJointsPosition(
        self, jointsPosition: NDArray[np.float64], contact: NDArray[np.bool]
    ):

        if not self.initializedByFirstData:
            # move character so that feet is on the ground
            self.adjustHeight(jointsPosition)
            # initialize handler for each contact joint
            self.initializeHandlers(jointsPosition)
            # marked that class is now initialized
            self.initializedByFirstData = True
            # initial position is character moved to the ground
            adjustedJointsPosition = jointsPosition @ self.transformation.T
        else:
            # calculate where contact joint should move by contact handler
            handledContactJointsPosition = np.zeros((len(self.contactJoints), 4))
            handledContactJointsPositionP1 = np.zeros((len(self.contactJoints), 4))
            handledContactJointsPositionP2 = np.zeros((len(self.contactJoints), 4))
            for i in range(len(self.contactJoints)):
                # get toe position by contact handler
                p0 = toCartesian(jointsPosition[self.contactJoints[i]])
                p0H = self.contactHandlers[i].handleContact(p0, contact[i])

                p1 = toCartesian(jointsPosition[self.contactJointsP1[i]])
                p1H = p1 + p0H - p0
                p2 = toCartesian(jointsPosition[self.contactJointsP2[i]])
                p3 = toCartesian(jointsPosition[self.contactJointsP3[i]])

                d12 = np.linalg.norm(p2 - p1)
                d23 = np.linalg.norm(p3 - p2)
                d13 = np.linalg.norm(p3 - p1H)
                # handle when contact point is further then leg lenth
                eps = 1e-3
                if d13 >= (d12 + d23) * (1 - eps):
                    p1H = p3 + (p1H - p3) / d13 * (d12 + d23) * (1 - eps)
                    p0H = p1H + p0 - p1
                    d13 = (d12 + d23) * (1 - eps)

                # for resulting p1H, p2H, p3,
                # when we lay foot of perpendicular from`` p2H to p1H-p3,
                # then distance from that point from p1H will be saved as d
                d = (d13**2 - d23**2 + d12**2) / (2 * d13)
                # put p2 into the plane made by p1H, p2, and p3
                # we first move to appropriate point over p3-p1H
                # then we move p2H along the plane
                p2H = p1H + (p3 - p1H) / np.linalg.norm(p3 - p1H) * d
                n = np.cross((p3 - p1H), (p0 - p1))
                n = np.cross(n, (p3 - p1H))
                n = n / np.linalg.norm(n)
                p2H = p2H + n * ((d12**2 - d**2) ** 0.5)
                # ensure knee doesn't go back
                originalCross = np.cross(p3 - p2, p2 - p1)
                newCross = np.cross(p3 - p2H, p2H - p1H)
                if np.dot(originalCross, newCross) < 0:
                    p2H = p2H - 2 * n * ((d12**2 - d**2) ** 0.5)

                handledContactJointsPosition[i, :] = toProjective(p0H)
                handledContactJointsPositionP1[i, :] = toProjective(p1H)
                handledContactJointsPositionP2[i, :] = toProjective(p2H)

            adjustedJointsPosition = jointsPosition
            adjustedJointsPosition[self.contactJoints] = handledContactJointsPosition
            adjustedJointsPosition[self.contactJointsP1] = (
                handledContactJointsPositionP1
            )
            adjustedJointsPosition[self.contactJointsP2] = (
                handledContactJointsPositionP2
            )
            adjustedJointsPosition[self.contactJointsEndSite] = (
                handledContactJointsPosition
            )
        return adjustedJointsPosition

    def updateScene(
        self,
    ) -> tuple[
        int,
        NDArray[np.float64],
        list[list[NDArray[np.float64]]],
        list[tuple[NDArray[np.float64], tuple[int, int, int]]],
    ]:
        frame, translationData, eulerData, contact = self.dataFtn()
        jointsPosition = self.file.calculateJointsPositionFromData(
            translationData, eulerData, self.transformation
        )
        adjustedJointsPosition = self.adjustJointsPosition(jointsPosition, contact)

        highlight = []
        for i in range(len(self.contactJoints)):
            if contact[i]:
                highlight.append(
                    (adjustedJointsPosition[self.contactJoints[i]], (255, 0, 0))
                )

        links = self.file.getLinks(adjustedJointsPosition)
        return (frame, adjustedJointsPosition, links, highlight)

    # this functionw as made just to show how we pick contact position
    def updateSceneWithContactTrajectory(
        self,
    ) -> tuple[
        int,
        NDArray[np.float64],
        list[list[NDArray[np.float64]]],
        list[tuple[NDArray[np.float64], tuple[int, int, int]]],
    ]:
        frame, translationData, eulerData, contact = self.dataFtn()
        jointsPosition = self.file.calculateJointsPositionFromData(
            translationData, eulerData, self.transformation
        )
        adjustedJointsPosition = self.adjustJointsPosition(jointsPosition, contact)

        highlight = []
        for i in range(len(self.contactJoints)):
            highlight.append(
                (adjustedJointsPosition[self.contactJoints[i]], (255, 0, 0))
            )

        jointsPosition = self.file.calculateJointsPositionFromData(
            translationData, eulerData, self.transformation
        )
        links = self.file.getLinks(jointsPosition)
        return (frame, jointsPosition, links, highlight)


class exampleDataFtn:
    def __init__(self, file: BVHFile, contactVelocityThreshold: float = 1):
        self.file: BVHFile = file
        self.contactVelocityThreshold = contactVelocityThreshold

    def ftn(
        self,
    ) -> tuple[int, NDArray[np.float64], NDArray[np.float64], NDArray[np.bool]]:
        prevFrame = self.file.currentFrame
        self.file.currentFrame = (self.file.currentFrame + 1) % self.file.numFrames
        prevPosition = toCartesian(
            self.file.calculateJointsPositionFromFrame(prevFrame)[[4, 9]]
        )
        currentPosition = toCartesian(
            self.file.calculateJointsPositionFromFrame(self.file.currentFrame)[[4, 9]]
        )
        velocity = currentPosition - prevPosition
        speed = np.linalg.norm(velocity, axis=1)

        return (
            self.file.currentFrame,
            self.file.translationDatas[self.file.currentFrame]
            + np.array([self.file.currentFrame * 0, 0, 0])
            + (self.file.currentFrame > 45) * np.array([0, -10, 0]),
            self.file.eulerDatas[self.file.currentFrame],
            speed < self.contactVelocityThreshold,
        )


if __name__ == "__main__":
    filePath = "example.bvh"
    file = BVHFile(filePath)
    dataFtn = exampleDataFtn(file)
    animator = contactManager(file, dataFtn.ftn)
    scene = pygameScene(
        filePath, frameTime=1 * file.frameTime, cameraRotation=np.array([0, math.pi, 0])
    )
    scene.run(animator.updateScene)

    dataFtn.file.currentFrame = 0
    scene.run(animator.updateSceneWithContactTrajectory)