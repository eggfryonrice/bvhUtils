from BVHFile import BVHFile
from pygameScene import pygameScene
from createBVH import createBVH


# for designated bvh file, split bvh file from time1 to time2, and write it in newFileName
def cutBVH(filePath: str, newFilePath: str, t1: float, t2: float):
    file: BVHFile = BVHFile(filePath)
    startFrame: int = int(t1 / file.frameTime)
    endFrame: int = min(file.numFrames - 1, int(t2 / file.frameTime))
    createBVH(filePath, newFilePath, file.datas[startFrame : endFrame + 1])


if __name__ == "__main__":
    originalFilePath = "example.bvh"
    newFilePath = "example_cut.bvh"
    cutBVH(originalFilePath, newFilePath, 1, 3)
    file = BVHFile(newFilePath)
    scene = pygameScene(newFilePath)
    scene.run(file.updateSceneWithNextFrame)
