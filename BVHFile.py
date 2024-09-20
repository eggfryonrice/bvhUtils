import numpy as np
from numpy.typing import NDArray
import math
from typing import Optional

from pygameScene import pygameScene
from util import *


class Joint:
    def __init__(
        self,
        parent: Optional["Joint"],
        channels: list[tuple[str, int]],
        name: str,
    ):
        # np array of length 4
        self.offset: NDArray[np.float64] = np.array([0, 0, 0, 1])
        # parent joint
        self.parent: Optional[Joint] = parent
        # child joints
        self.child: list[Joint] = []
        # joint name
        self.name: str = name
        # (name, i)
        # name is whether it is X, Y, or Z rotation or position.
        # i is position of that channel in the list of motion data in each frames
        self.channels: list[tuple[str, int]] = channels
        # store transformation each frame
        self.transformation: NDArray[np.float64] = np.eye(4)
        # store position in 4d
        self.position: NDArray[np.float64] = np.array([0, 0, 0, 1])


class BVHFile:
    def __init__(self, fileName: str):
        self.joints: list[Joint] = []

        self.datas: list[list[float]] = []
        self.numFrames: int = 0
        self.frameTime: float = 1.0

        # current frame
        self.currentFrame: int = 0

        # dict from name to channel
        self.nameToJoint: dict[str, Joint] = {}

        # parse bvh file by the first element of each line
        with open(fileName, "r") as f:
            jointStack: list[Joint] = []
            channelsCounter = 0

            # parse the upper part of bvh file
            for l in f:
                tokens: list[str] = l.strip().split()
                if not tokens:
                    continue

                if tokens[0] in ["ROOT", "JOINT", "End"]:
                    parent = None if len(jointStack) == 0 else jointStack[-1]
                    joint = Joint(parent=parent, channels=[], name=tokens[1])
                    self.nameToJoint[tokens[1]] = joint
                    if joint.parent is not None:
                        joint.parent.child.append(joint)
                    self.joints.append(joint)
                    jointStack.append(joint)

                elif tokens[0] == "OFFSET":
                    jointStack[-1].offset = np.array(
                        list(map(float, tokens[1:4])) + [1]
                    )

                elif tokens[0] == "CHANNELS":
                    channelsN = int(tokens[1])
                    channels: list[tuple[str, int]] = [
                        (tokens[i + 2], channelsCounter + i) for i in range(channelsN)
                    ]
                    jointStack[-1].channels.extend(channels)
                    channelsCounter += channelsN

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
                    frameData = list(map(float, tokens))
                    self.datas.append(frameData)

    def getJointsLinksPosition(
        self,
    ) -> tuple[list[NDArray[np.float64]], list[list[NDArray[np.float64]]]]:
        joints: list[NDArray[np.float64]] = []
        links: list[list[NDArray[np.float64]]] = []
        for joint in self.joints:
            joints.append(joint.position)
            if joint.parent is not None:
                links.append([joint.position, joint.parent.position])
        return joints, links

    # calculate position of given joint (assumes that paren's joint position is already calcualted)
    def calculateJointPosition(
        self,
        joint: Joint,
        data: list[float],
        rootTransformation: NDArray[np.float64] = np.eye(4),
    ) -> NDArray[np.float64]:
        # if joint has XYZ position, shift will be calculated lastly
        transformation = np.eye(4)
        for name, i in joint.channels:
            if name == "Xposition":
                transformation = transformation @ defineShift(np.array([data[i], 0, 0]))
            elif name == "Yposition":
                transformation = transformation @ defineShift(np.array([0, data[i], 0]))
            elif name == "Zposition":
                transformation = transformation @ defineShift(np.array([0, 0, data[i]]))

        # if joint has parent joint, multiply these matrices on joint's offset:
        # parent's rotation
        # shift to parent's offset location
        # apply parent's transformation
        if joint.parent is not None:
            for name, i in joint.parent.channels:
                if name == "Xrotation":
                    transformation = transformation @ defineRotationX(
                        data[i] * math.pi / 180
                    )
                elif name == "Yrotation":
                    transformation = transformation @ defineRotationY(
                        data[i] * math.pi / 180
                    )
                elif name == "Zrotation":
                    transformation = transformation @ defineRotationZ(
                        data[i] * math.pi / 180
                    )

            transformation = (
                joint.parent.transformation
                @ defineShift(joint.parent.offset[0:3])
                @ transformation
            )
        else:
            transformation = rootTransformation @ transformation

        joint.transformation = transformation

        joint.position = joint.transformation @ joint.offset
        return joint.position

    # calcualte position of all joints using data of given frame
    def calculateJointsPositionByFrame(
        self, frame: int, rootTransformation: NDArray[np.float64] = np.eye(4)
    ) -> None:
        frame = frame % self.numFrames
        data = self.datas[frame]
        for joint in self.joints:
            self.calculateJointPosition(joint, data, rootTransformation)

    # calcualte position of all joints by given data
    def calculateJointsPositionFromData(
        self, data: list[float], rootTransformation: NDArray[np.float64] = np.eye(4)
    ) -> None:
        for joint in self.joints:
            self.calculateJointPosition(joint, data, rootTransformation)

    # return frame, joint, link information
    def updateSceneWithNextFrame(
        self,
    ) -> tuple[int, list[NDArray[np.float64]], list[list[NDArray[np.float64]]]]:
        self.calculateJointsPositionByFrame(self.currentFrame)
        joints, links = self.getJointsLinksPosition()
        currentData = (self.currentFrame, joints, links)
        self.currentFrame = (self.currentFrame + 1) % self.numFrames
        return currentData


if __name__ == "__main__":
    fileName = "example.bvh"
    file = BVHFile(fileName)
    scene = pygameScene(fileName, frameTime=file.frameTime)
    scene.run(file.updateSceneWithNextFrame)
