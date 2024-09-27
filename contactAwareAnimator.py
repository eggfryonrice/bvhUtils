import numpy as np
from numpy.typing import NDArray
from typing import Callable

from BVHFile import BVHFile
from pygameScene import pygameScene
from transformationUtil import *


class contactHandler:
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

        self.contactState: bool = False
        self.lock: bool = False

        # position that will be displayed
        self.position: NDArray[np.float64] = toCartesian(initialPosition)
        self.positionOffset = np.array([0, 0, 0])
        # moving velocity
        self.velocity: NDArray[np.float64] = np.array([0, 0, 0])
        self.velocityOffset = np.array([0, 0, 0])
        # position of contact
        self.contactPoint: NDArray[np.float64] = np.array([0, 0, 0])
        # position of the data
        self.dataPosition: NDArray[np.float64] = np.array([0, 0, 0])

    def damping(
        self,
        currentPosition: NDArray[np.float64],
        targetPosition: NDArray[np.float64],
        currentVelocity: NDArray[np.float64],
        targetVelocity: NDArray[np.float64],
    ):
        positionDiff = targetPosition - currentPosition
        velocityDiff = targetVelocity - currentVelocity

        y = 2 * 0.6931 / self.halfLife
        j1 = velocityDiff + positionDiff * y
        eydt = np.exp(-y * self.frameTime)

        position = currentPosition + eydt * (positionDiff + j1 * self.frameTime)
        velocity = currentVelocity + eydt * (velocityDiff - j1 * y * self.frameTime)

        return position, velocity

    def handleContact(self, inputPosition: NDArray[np.float64], contactState: bool):
        inputPosition[1] = max([self.footHeight, inputPosition[1]])
        targetVelocity = (inputPosition - self.dataPosition) / self.frameTime
        self.dataPosition = inputPosition

        if (not self.contactState) and contactState:
            self.lock = True
            self.contactPoint = self.position
            self.contactPoint[1] = self.footHeight

        if self.lock and (
            np.linalg.norm(self.dataPosition - self.contactPoint) > self.unlockRadius
        ):
            self.lock = False

        if self.lock and self.contactState and (not contactState):
            self.lock = False

        if self.lock:
            targetPosition = self.contactPoint
            targetVelocity = np.array([0, 0, 0])
        else:
            targetPosition = inputPosition

        self.position, self.velocity = self.damping(
            self.position, targetPosition, self.velocity, targetVelocity
        )

        return self.position


class contactAwareAnimator:
    def __init__(
        self,
        file: BVHFile,
        dataFtn: Callable[
            [], tuple[int, NDArray[np.float64], NDArray[np.float64], NDArray[np.bool]]
        ],
        contactJointNames=["LeftToe", "RightToe"],
        unlockRadius: float = 10,
        footHeight: float = 2,
        halfLife: float = 0.1,
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
        self.contactJointsOffset = [jointIdx + 1 for jointIdx in self.contactJoints]

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
            defineShift(np.array([0, self.footHeight - lowestJointHeight, 0]))
            @ self.transformation
        )

    def initializeHandlers(self, jointsPosition: NDArray[np.float64]):
        self.contactHandlers: list[contactHandler] = []
        for jointIdx in self.contactJoints:
            handler = contactHandler(
                jointsPosition[jointIdx],
                self.file.frameTime,
                unlockRadius=self.unlockRadius,
                footHeight=self.footHeight,
                halfLife=self.halfLife,
            )
            self.contactHandlers.append(handler)

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
        highlight = []

        if not self.initializedByFirstData:
            # move character so that feet is on the ground
            self.adjustHeight(jointsPosition)
            # initialize handler for each contact joint
            self.initializeHandlers(jointsPosition)
            # marked that class is now initialized
            self.initializedByFirstData = True
            # initial position is character moved to the ground
            adjustedJointsPosition = self.file.calculateJointsPositionFromData(
                translationData, eulerData, self.transformation
            )
        else:
            # calculate where contact joint should move by contact handler
            handledContactJointsPosition = np.zeros((len(self.contactJoints), 4))
            handledContactJointsPositionP1 = np.zeros((len(self.contactJoints), 4))
            handledContactJointsPositionP2 = np.zeros((len(self.contactJoints), 4))
            for i in range(len(self.contactJoints)):
                # get toe position by contact handler
                p0 = toCartesian(jointsPosition[self.contactJoints[i]])
                p0H = self.contactHandlers[i].handleContact(p0, contact[i])
                if contact[i]:
                    highlight.append((toProjective(p0H), (255, 0, 0)))

                p1 = toCartesian(jointsPosition[self.contactJointsP1[i]])
                p1H = p1 + p0H - p0
                p2 = toCartesian(jointsPosition[self.contactJointsP2[i]])
                p3 = toCartesian(jointsPosition[self.contactJointsP3[i]])
                d12 = np.linalg.norm(p2 - p1)
                d23 = np.linalg.norm(p3 - p2)
                d13 = np.linalg.norm(p3 - p1H)
                # handle when contact point is further then leg lenth
                if d13 >= d23 + d12:
                    p2H = p3 + (p1H - p3) / np.linalg.norm(p1H - p3) * d23
                    p1H = p3 + (p1H - p3) / np.linalg.norm(p1H - p3) * (d12 + d23)
                    p0H = p1H + p0 - p1
                # IK for normal case
                else:
                    # for resulting p1H, p2H, p3,
                    # when we lay foot of perpendicular from p2H to p1H-p3,
                    # then distance from that point from p1H will be saved as d
                    d = (d13**2 - d23**2 + d12**2) / (2 * d13)
                    # put p2 into the plane made by p1H, p3 and
                    # p2 translated with same amount with p1
                    # we first move to appropriate point over p3-p1H
                    # then we move p2H along the plane
                    p2H = p1H + (p3 - p1H) / d13 * d
                    n = np.cross((p3 - p1H), (p2 - p1))
                    n = np.cross(n, (p3 - p1H))
                    n = n / np.linalg.norm(n)
                    print((d12**2 - d**2), d13, d23, d12)
                    p2H = p2H + n * ((d12**2 - d**2) ** 0.5)

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
            adjustedJointsPosition[self.contactJointsOffset] = (
                handledContactJointsPosition
            )

        links = self.file.getLinks(adjustedJointsPosition)
        return (frame, adjustedJointsPosition, links, highlight)


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
            self.file.calculateJointsPositionByFrame(prevFrame)[[4, 9]]
        )
        currentPosition = toCartesian(
            self.file.calculateJointsPositionByFrame(self.file.currentFrame)[[4, 9]]
        )
        velocity = currentPosition - prevPosition
        speed = np.linalg.norm(velocity, axis=1)

        return (
            self.file.currentFrame,
            self.file.translationDatas[self.file.currentFrame]
            + (self.file.currentFrame > 10) * np.array([0, -5, 0]),
            self.file.eulerDatas[self.file.currentFrame],
            speed < self.contactVelocityThreshold,
        )


if __name__ == "__main__":
    filePath = "example.bvh"
    file = BVHFile(filePath)
    dataFtn = exampleDataFtn(file)
    animator = contactAwareAnimator(file, dataFtn.ftn)
    scene = pygameScene(
        filePath, frameTime=1 * file.frameTime, cameraRotation=np.array([0, math.pi, 0])
    )
    scene.run(animator.updateScene)
