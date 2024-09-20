import pygame
import numpy as np
from numpy.typing import NDArray
import math
import time
from multiprocessing import Process
from queue import Queue
from typing import Callable

from util import *


class pygameScene:
    def __init__(
        self,
        caption: str = "",
        cameraCenter: NDArray[np.float64] = np.array([0, 0, 0]),
        frameTime: float = 1 / 60,
        width: int = 1920,
        height: int = 1080,
    ):
        self.running = multiprocessing.Value("i", True)

        self.caption: str = caption
        self.width: int = width
        self.height: int = height

        # get info from reader to get cameracenter and floor position
        self.cameraCenter: NDArray[np.float64] = cameraCenter

        # camera transformation info
        self.cameraDistance: int = 2000
        self.rotationAngle: NDArray[np.float64] = np.array(
            [-1 * math.pi / 4, math.pi, 0.0]
        )
        self.zoom: float = 2

        # input info for camera transformation
        self.mouseDragging: bool = False
        self.prevMousePosition: tuple[int, int] = (0, 0)
        self.prevRotationAngle: NDArray[np.float64] = np.array([0, 0, 0])

        self.frameTime: float = frameTime

    # setup non-serializable data (pygame attributes) later for multiprocessing
    def setupPygame(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.caption)
        self.clock = pygame.time.Clock()

    def updateCameraCenter(self, joints) -> None:
        if len(joints) == 0:
            return
        self.cameraCenter = joints[0][0:3] / joints[0][3]
        for joint in joints:
            if joint[1] / joint[3] < self.cameraCenter[1]:
                self.cameraCenter[1] = joint[1] / joint[3]

    # get projected location on the screen of 4d point
    def projection(self, point: NDArray[np.float64]) -> tuple[int, int]:

        cameraDistance = self.cameraDistance
        rotationAngle = self.rotationAngle
        zoom = self.zoom

        rotationX = defineRotationX3D(rotationAngle[0])
        rotationY = defineRotationY3D(rotationAngle[1])
        rotationZ = defineRotationZ3D(rotationAngle[2])

        rotatedPoint = (
            rotationX @ rotationY @ rotationZ @ (toCartesian(point) - self.cameraCenter)
        )

        factor = cameraDistance / (cameraDistance + rotatedPoint[2])
        x2d = rotatedPoint[0] * factor * zoom + self.screen.get_rect()[2] // 2
        y2d = -rotatedPoint[1] * factor * zoom + self.screen.get_rect()[3] // 2
        return int(x2d), int(y2d)

    # handle input such as pushing x button, mouse motion, left button, and scroll
    def handleInput(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                with self.running.get_lock():
                    self.running.value = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if not self.mouseDragging:
                        self.prevMousePosition = pygame.mouse.get_pos()
                        self.prevRotationAngle = self.rotationAngle.copy()
                    self.mouseDragging = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouseDragging = False

            elif event.type == pygame.MOUSEMOTION:
                # mouse motion  contribute to rotation of camera
                # if mouse move from one side from the other side of the screen,
                # camera rotates for pi in the oppposite direction
                if self.mouseDragging:
                    mouseX, mouseY = pygame.mouse.get_pos()
                    dx = mouseX - self.prevMousePosition[0]
                    dy = mouseY - self.prevMousePosition[1]
                    self.rotationAngle[0] = (
                        self.prevRotationAngle[0]
                        + -1 * dy * math.pi / self.screen.get_rect()[3]
                    )
                    self.rotationAngle[1] = (
                        self.prevRotationAngle[1]
                        + -1 * dx * math.pi / self.screen.get_rect()[2]
                    )

            # mouse wheeel contribute to zoom in and zoom out
            elif event.type == pygame.MOUSEWHEEL:
                self.zoom += event.y * 0.1
                if self.zoom < 0.1:
                    self.zoom = 0.1

    # draw homogeneous point
    def drawHomogeneousPoint(
        self,
        point: NDArray[np.float64],
        color: tuple[int, int, int] = (255, 255, 255),
        size: int = 4,
    ) -> None:
        pygame.draw.circle(self.screen, color, self.projection(point), size)

    # draw line through series of homogeneous points
    def drawLineFromHomogenousPoints(
        self,
        points: list[NDArray[np.float64]],
        color: tuple[int, int, int] = (255, 255, 255),
        width: int = 2,
    ) -> None:
        if len(points) <= 1:
            return
        fromPt = self.projection(points[0])
        for i in range(1, len(points)):
            toPt = self.projection(points[i])
            pygame.draw.line(self.screen, color, fromPt, toPt, width)
            fromPt = toPt

    # draw floor on the same height as cameracenter
    # grid is number of grid generated, gridDistance is distance between grids
    def drawFloor(
        self,
        color: tuple[int, int, int] = (100, 100, 100),
        grid: int = 21,
        gridDistance: int = 50,
    ) -> None:
        centerX = self.cameraCenter[0]
        centerZ = self.cameraCenter[2]

        floorPoints = []
        for i in range(-1 * (grid // 2), grid // 2 + grid % 2):
            for j in range(-1 * (grid // 2), grid // 2 + grid % 2):
                pt4D = np.array(
                    [
                        centerX + i * gridDistance,
                        self.cameraCenter[1],
                        centerZ + j * gridDistance,
                        1,
                    ]
                )
                floorPoints.append(self.projection(pt4D))

        floorLines = []
        for i in range(grid * grid - grid):
            floorLines.append((i, i + grid))
        for i in range(grid * grid):
            if i % grid != grid - 1:
                floorLines.append((i, i + 1))

        for i, j in floorLines:
            pygame.draw.aaline(self.screen, color, floorPoints[i], floorPoints[j])

    # draw elapsed time on top right
    def drawElapsedTimeAndFrame(self, frame: int) -> None:
        font = pygame.font.Font(None, 36)
        elapsedSurface = font.render(
            f"Time: {frame * self.frameTime:.2f}s, Frame: {frame}",
            True,
            (255, 255, 255),
        )
        elapsedRect = elapsedSurface.get_rect(
            topright=(self.screen.get_rect().width - 10, 10)
        )
        self.screen.blit(elapsedSurface, elapsedRect)

    def drawPendingIcon(self, radius: float = 50, frequency: float = 0.5) -> None:
        center = (self.screen.get_rect().width / 2, self.screen.get_rect().height / 2)
        pygame.draw.circle(self.screen, (255, 255, 255), center, radius, 2)
        angle = pygame.time.get_ticks() * frequency / 1000 * 360
        # Rotating line to simulate the "pending" effect
        x = center[0] + radius * math.cos(math.radians(angle))
        y = center[1] + radius * math.sin(math.radians(angle))
        pygame.draw.line(self.screen, (255, 255, 255), center, (x, y), 3)

    # display scene
    # element of Queue is tuple of int, list of points, and list of lines to be drawn
    def displayScene(
        self,
        queue: Queue[
            tuple[int, list[NDArray[np.float64]], list[list[NDArray[np.float64]]]]
        ],
    ) -> None:
        self.setupPygame()
        while self.running.value and queue.empty():
            self.clock.tick(1 / self.frameTime)
        frame, joints, links = queue.get()

        # adjust camera center with respect to first joints information
        self.updateCameraCenter(joints)

        while self.running.value:
            self.handleInput()
            self.screen.fill((0, 0, 0))
            self.drawFloor()

            for joint in joints:
                self.drawHomogeneousPoint(joint)
            for link in links:
                self.drawLineFromHomogenousPoints(link)

            self.drawElapsedTimeAndFrame(frame)

            if not queue.empty():
                frame, joints, links = queue.get()
            else:
                self.drawPendingIcon()

            pygame.display.flip()
            self.clock.tick(1 / self.frameTime)
        pygame.quit()

    def enqueueData(
        self,
        queue: MPQueue[
            tuple[int, list[NDArray[np.float64]], list[list[NDArray[np.float64]]]]
        ],
        f: Callable[
            [], tuple[int, list[NDArray[np.float64]], list[list[NDArray[np.float64]]]]
        ],
    ) -> None:
        while self.running.value:
            if queue.qsize() > 5 / self.frameTime:
                time.sleep(0.5)
            else:
                queue.put(f())

        queue.clear()

    def run(
        self,
        f: Callable[
            [], tuple[int, list[NDArray[np.float64]], list[list[NDArray[np.float64]]]]
        ],
    ):
        queue = MPQueue()

        p1 = Process(target=self.displayScene, args=(queue,))
        p1.start()

        p2 = Process(target=self.enqueueData, args=(queue, f))
        p2.start()

        p1.join()
        p2.join()


def exampleFunction() -> (
    tuple[int, list[NDArray[np.float64]], list[list[NDArray[np.float64]]]]
):
    return (0, [], [])


if __name__ == "__main__":
    scene = pygameScene("example scene")
    scene.run(exampleFunction)
