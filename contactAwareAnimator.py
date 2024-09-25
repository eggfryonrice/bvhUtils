import numpy as np
from numpy.typing import NDArray
from typing import Callable

from BVHFile import BVHFile
from pygameScene import pygameScene
from transformationUtil import *


class contactHandler:
    def __init__(
        self,
        unlockRadius: float = 10,
        footHeight: float = 2,
        toeHeight: float = 2,
        halfLife: float = 1,
    ):
        self.contactState: bool = False
        self.contactLock: bool = False
        self.contactPosition: NDArray[np.float64] = np.array[0, 0, 0]
        self.contactPoint: NDArray[np.float64] = np.array[0, 0, 0]

    def handleContact(contactPosition: NDArray[np.float64], contactState: bool):

        return


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
        halfLife: float = 1,
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

    def initializeHandlers(self):
        self.handlers = []
        for i in range(len(self.contactJoints)):
            handler = contactHandler()
            self.handlers.append(handler)

    def updateScene(
        self,
    ) -> tuple[int, list[NDArray[np.float64]], list[list[NDArray[np.float64]]]]:
        frame, translationData, eulerData, contact = self.dataFtn()

        if not self.initializedByFirstData:
            jointsPosition = self.file.calculateJointsPositionFromData(
                translationData, eulerData, self.transformation
            )
            self.adjustHeight(jointsPosition)
            self.initializeHandlers()
            self.initializedByFirstData = True
            adjustedJointsPosition = jointsPosition
        else:
            # calculate where contact joint should move by contact handler
            handledContactJointsPosition = []
            for i in range(len(self.contactJoints)):
                handledPosition = self.contactHandlers[i].handleContact()
                handledContactJointsPosition.append(handledPosition)

            # adjust joints by IK

            adjustedJointsPosition = self.file.calculateJointsPositionFromData(
                translationData, eulerData, self.transformation
            )

        links = self.file.getLinks(adjustedJointsPosition)
        return (frame, adjustedJointsPosition, links)


def exampleDataFtn(
    file: BVHFile, contactVelocityThreshold: float = 1
) -> tuple[int, NDArray[np.float64], NDArray[np.float64], NDArray[np.bool]]:
    prevFrame = file.currentFrame
    file.currentFrame = (file.currentFrame + 1) % file.numFrames
    prevPosition4D = file.calculateJointsPositionByFrame(prevFrame)[[4, 9]]
    currentPosition4D = file.calculateJointsPositionByFrame(file.currentFrame)[[4, 9]]
    prevPosition = prevPosition4D[:3] / prevPosition4D[3]
    currentPosition = currentPosition4D[:3] / currentPosition4D[3]
    velocity = currentPosition - prevPosition
    speed = np.linalg.norm(velocity, axis=1)

    return (
        file.currentFrame,
        file.translationDatas[file.currentFrame],
        file.eulerDatas[file.currentFrame],
        speed < contactVelocityThreshold,
    )


if __name__ == "__main__":
    filePath = "example.bvh"
    file = BVHFile(filePath)
    contactAwareAnimator(file, lambda: exampleDataFtn(file))
    scene = pygameScene(filePath, frameTime=file.frameTime)
