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
        unlockRadius: float = 10,
        footHeight: float = 2,
        toeHeight: float = 2,
        halfLife: float = 0.15,
    ):
        self.frameTime = frameTime

        self.unlockRadius = unlockRadius
        self.footHeight = footHeight
        self.toeHeight = toeHeight
        self.halfLife = halfLife

        self.contactState: bool = False
        self.lock: bool = False

        # position that will be displayed
        self.position: NDArray[np.float64] = toCartesian(initialPosition)
        # moving velocity
        self.velocity: NDArray[np.float64] = np.array([0, 0, 0])
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

    def handleContact(self, inputPosition4D: NDArray[np.float64], contactState: bool):
        inputPosition = toCartesian(inputPosition4D)
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

        return toProjective(self.position)


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
        toeHeight: float = 2,
        halfLife: float = 0.15,
    ):
        self.file: BVHFile = file

        self.contactJoints = list(map(file.jointNames.index, contactJointNames))

        self.transformation = np.eye(4)

        self.dataFtn = dataFtn

        self.unlockRadius = unlockRadius
        self.footHeight = footHeight
        self.toeHeight = toeHeight
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
            handler = contactHandler(jointsPosition[jointIdx], self.file.frameTime)
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
        hightlight = []

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
            for i in range(len(self.contactJoints)):
                handledPosition = self.contactHandlers[i].handleContact(
                    jointsPosition[self.contactJoints[i]], contact[i]
                )
                handledContactJointsPosition[i, :] = handledPosition
            # hightlight = [
            #     handledContactJointsPosition[i, :]
            #     for i in range(len(self.contactJoints))
            # ]
            hightlight = []
            for i in range(len(self.contactJoints)):
                if contact[i]:
                    hightlight.append((handledContactJointsPosition[i, :], (255, 0, 0)))

            # adjust joints by IK

            adjustedJointsPosition = jointsPosition

        links = self.file.getLinks(adjustedJointsPosition)
        return (frame, adjustedJointsPosition, links, hightlight)


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
            + (self.file.currentFrame > 10) * np.array([0, 5, 0]),
            self.file.eulerDatas[self.file.currentFrame],
            speed < self.contactVelocityThreshold,
        )


if __name__ == "__main__":
    filePath = "example.bvh"
    file = BVHFile(filePath)
    dataFtn = exampleDataFtn(file)
    animator = contactAwareAnimator(file, dataFtn.ftn)
    scene = pygameScene(
        filePath, frameTime=file.frameTime, cameraRotation=np.array([0, math.pi, 0])
    )
    scene.run(animator.updateScene)
