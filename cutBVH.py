import math
import numpy as np

from BVHFile import BVHFile
from pygameScene import pygameScene
from createBVH import createBVH


# for designated bvh file, split bvh file from time1 to time2, and write it in newFileName
def cutBVH(filePath: str, newFilePath: str, t1: float, t2: float):
    file: BVHFile = BVHFile(filePath)
    startFrame: int = int(t1 / file.frameTime)
    endFrame: int = min(file.numFrames - 1, int(t2 / file.frameTime))
    nonEndSiteIdx = [i for i, name in enumerate(file.jointNames) if name != "Site"]
    datas: np.ndarray = np.zeros((file.numFrames, len(nonEndSiteIdx) * 3 + 3))
    datas[:, :3] = file.translationDatas
    datas[:, 3:] = (file.eulerDatas[:, nonEndSiteIdx, :] * 180 / math.pi).reshape(
        (file.numFrames, len(nonEndSiteIdx) * 3)
    )
    createBVH(filePath, newFilePath, datas[startFrame : endFrame + 1])


if __name__ == "__main__":
    originalFilePath = "example.bvh"
    newFilePath = "example_cut.bvh"
    cutBVH(originalFilePath, newFilePath, 1, 2)
    file = BVHFile(newFilePath)
    scene = pygameScene(newFilePath)
    scene.run(file.updateSceneWithNextFrame)
