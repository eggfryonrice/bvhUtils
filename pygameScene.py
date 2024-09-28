import pygame
import numpy as np
import math
import time
import multiprocessing
from multiprocessing import Process
from queue import Queue
from typing import Callable, TypeVar, Generic

from transformationUtil import *

T = TypeVar("T")


class MPQueue(Generic[T]):
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.size = multiprocessing.Value("i", 0)

    def put(self, item: T) -> None:
        self.queue.put(item)
        with self.size.get_lock():
            self.size.value += 1

    def get(self) -> T:
        item = self.queue.get()
        with self.size.get_lock():
            self.size.value -= 1
        return item

    def qsize(self) -> int:
        return self.size.value

    def empty(self) -> bool:
        return self.qsize() == 0

    def clear(self) -> None:
        while self.qsize() > 0:
            self.get()


class pygameScene:
    def __init__(
        self,
        caption: str = "",
        frameTime: float = 1 / 60,
        cameraRotation: np.ndarray = np.array([math.pi / 4, math.pi, 0.0]),
        width: int = 1920,
        height: int = 1080,
    ):
        self.running = multiprocessing.Value("i", True)

        self.caption: str = caption
        self.width: int = width
        self.height: int = height

        # get info from reader to get cameracenter and floor position
        self.cameraCenter: np.ndarray = np.array([0, 0, 0])

        # camera transformation info
        self.cameraDistance: int = 2000
        self.cameraRotation: np.ndarray = cameraRotation
        self.zoom: float = 2

        # input info for camera transformation
        self.mouseDragging: bool = False
        self.prevMousePosition: tuple[int, int] = (0, 0)
        self.prevRotationAngle: np.ndarray = np.array([0, 0, 0])

        self.frameTime: float = frameTime

    # setup non-serializable data (pygame attributes) later for multiprocessing
    def setupPygame(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.caption)
        self.clock = pygame.time.Clock()

    def updateCameraCenter(self, jointsPosition: np.ndarray) -> None:
        if len(jointsPosition) == 0:
            return
        self.cameraCenter = jointsPosition[0, 0:3] / jointsPosition[0, 3]
        jointsHeight = jointsPosition[:, 1] / jointsPosition[:, 3]
        self.cameraCenter[1] = np.min(jointsHeight)

    # get projected location on the screen of 4d point
    def projection(self, points: np.ndarray) -> np.ndarray:
        cameraDistance = self.cameraDistance
        rotationAngle = self.cameraRotation
        zoom = self.zoom

        rotationX = rotationMatX3D(rotationAngle[0])
        rotationY = rotationMatY3D(rotationAngle[1])
        rotationZ = rotationMatZ3D(rotationAngle[2])

        singlePoint = False
        if points.ndim == 1 and points.shape == (4,):
            points = points.reshape(1, 4)
            singlePoint = True

        camera_center_broadcast = self.cameraCenter.reshape(1, 3)
        cartesian_points = toCartesian(points) - camera_center_broadcast

        rotated_points = np.einsum("ij,...j->...i", rotationX, cartesian_points)
        rotated_points = np.einsum("ij,...j->...i", rotationY, rotated_points)
        rotated_points = np.einsum("ij,...j->...i", rotationZ, rotated_points)

        factor = cameraDistance / (cameraDistance + rotated_points[..., 2])

        x2d = rotated_points[..., 0] * factor * zoom + self.screen.get_rect()[2] // 2
        y2d = -rotated_points[..., 1] * factor * zoom + self.screen.get_rect()[3] // 2

        result = np.stack((x2d, y2d), axis=-1)

        if singlePoint:
            return result[0]

        return result.astype(np.int32)

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
                        self.prevRotationAngle = self.cameraRotation.copy()
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
                    dy = self.prevMousePosition[1] - mouseY
                    self.cameraRotation[0] = (
                        self.prevRotationAngle[0]
                        + -1 * dy * math.pi / self.screen.get_rect()[3]
                    )
                    self.cameraRotation[1] = (
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
        point: np.ndarray,
        color: tuple[int, int, int] = (255, 255, 255),
        size: int = 4,
    ) -> None:
        pygame.draw.circle(self.screen, color, tuple(self.projection(point)), size)

    # draw homogeneous points, gets n * 4 np array of positions as input
    def drawHomogeneousPoints(
        self,
        pointsPosition: np.ndarray,
        color: tuple[int, int, int] = (255, 255, 255),
        size: int = 4,
    ) -> None:
        pointsPosition2D = self.projection(pointsPosition)
        for i in range(pointsPosition.shape[0]):
            pygame.draw.circle(self.screen, color, pointsPosition2D[i], size)

    # draw line through series of homogeneous points
    def drawLineFromHomogenousPoints(
        self,
        points: list[np.ndarray],
        color: tuple[int, int, int] = (255, 255, 255),
        width: int = 2,
    ) -> None:
        if len(points) <= 1:
            return
        fromPt = self.projection(points[0])
        for i in range(1, len(points)):
            toPt = self.projection(points[i])
            pygame.draw.line(self.screen, color, tuple(fromPt), tuple(toPt), width)
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
                floorPoints.append(pt4D)
        floorPoints2D = self.projection(np.array(floorPoints))

        floorLines = []
        for i in range(grid * grid - grid):
            floorLines.append((i, i + grid))
        for i in range(grid * grid):
            if i % grid != grid - 1:
                floorLines.append((i, i + 1))

        for i, j in floorLines:
            pygame.draw.aaline(
                self.screen, color, tuple(floorPoints2D[i]), tuple(floorPoints2D[j])
            )

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
        x = center[0] + radius * math.cos(math.radians(angle))
        y = center[1] + radius * math.sin(math.radians(angle))
        pygame.draw.line(self.screen, (255, 255, 255), center, (x, y), 3)

    # display scene
    # element of Queue is tuple of int, list of points, and list of lines to be drawn
    def displayScene(
        self,
        queue: Queue[
            tuple[
                int,
                np.ndarray,
                list[list[np.ndarray]],
                list[tuple[np.ndarray, tuple[int, int, int]]],
            ]
        ],
    ) -> None:
        self.setupPygame()
        while self.running.value and queue.empty():
            self.clock.tick(1 / self.frameTime)
        frame, jointsPosition, links, highlightPoints = queue.get()

        # adjust camera center with respect to first joints information
        self.updateCameraCenter(jointsPosition)

        while self.running.value:
            self.handleInput()
            self.screen.fill((0, 0, 0))
            self.drawFloor()

            self.drawHomogeneousPoints(jointsPosition)
            for link in links:
                self.drawLineFromHomogenousPoints(link)

            for point, color in highlightPoints:
                self.drawHomogeneousPoint(point, color=color)

            self.drawElapsedTimeAndFrame(frame)

            if not queue.empty():
                frame, jointsPosition, links, highlightPoints = queue.get()
            else:
                self.drawPendingIcon()

            pygame.display.flip()
            self.clock.tick(1 / self.frameTime)
        pygame.quit()

    def enqueueData(
        self,
        queue: MPQueue[
            tuple[
                int,
                np.ndarray,
                list[list[np.ndarray]],
                list[tuple[np.ndarray, tuple[int, int, int]]],
            ]
        ],
        f: Callable[
            [],
            tuple[
                int,
                np.ndarray,
                list[list[np.ndarray]],
                list[tuple[np.ndarray, tuple[int, int, int]]],
            ],
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
            [],
            tuple[
                int,
                np.ndarray,
                list[list[np.ndarray]],
                list[tuple[np.ndarray, tuple[int, int, int]]],
            ],
        ],
    ):
        queue = MPQueue()

        p1 = Process(target=self.displayScene, args=(queue,))
        p1.start()

        p2 = Process(target=self.enqueueData, args=(queue, f))
        p2.start()

        p1.join()
        p2.join()

        self.__init__(
            caption=self.caption,
            frameTime=self.frameTime,
            cameraRotation=self.cameraRotation,
            width=self.width,
            height=self.height,
        )


def exampleFunction() -> tuple[
    int,
    np.ndarray,
    list[list[np.ndarray]],
    list[tuple[np.ndarray, tuple[int, int, int]]],
]:
    return (0, np.array([[0, 0, 0, 1]]), [], [(np.array([100, 0, 0, 1]), (255, 0, 0))])


if __name__ == "__main__":
    scene = pygameScene("example scene")
    scene.run(exampleFunction)
