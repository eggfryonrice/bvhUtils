import pygame
from pygame.locals import *  # type: ignore
from OpenGL.GL import *  # type: ignore
from OpenGL.GLU import *  # type: ignore
from OpenGL.GLUT import *  # type: ignore
import numpy as np
import math

# list of (ndarray of 3d points and color)
jointsPositionsInput = list[tuple[np.ndarray, tuple[float, float, float]]]
# list of (list of (fromPoint, toPoint, link rotation quaternion) and color)
linkssInput = list[
    tuple[
        list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        tuple[float, float, float],
    ]
]

sceneInput = tuple[
    int,
    jointsPositionsInput,
    linkssInput,
]


class pygameScene:
    def __init__(
        self,
        frameTime: float = 0.033,
        cameraAngleX: float = math.pi / 4,
        cameraAngleY: float = -math.pi / 2,
        width: int = 1600,
        height: int = 900,
        sphereRadius: float = 3,
        cuboidWidth: float = 4,
    ):
        pygame.init()
        self.running = True

        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        self.width = width
        self.height = height

        self.clock = pygame.time.Clock()
        self.frameTime = frameTime

        glutInit()
        self.initOpengl()
        self.initLighting()
        self.lightPosition = [0, 1000, 0, 0]

        self.sphereRadius: float = sphereRadius
        self.cuboidWidth: float = cuboidWidth

        # Camera parameters
        self.cameraCenter: np.ndarray = np.array([0, 0, 0])
        # cameracenter will be initialized by first data
        self.cameraCenterInitializedByFirstData = False
        self.cameraAngleX = cameraAngleX
        self.cameraAngleY = cameraAngleY
        self.cameraDistance = 1000

        self.mouseDragging: bool = False
        self.prevMousePosition: tuple[int, int] = (0, 0)

    def initOpengl(self):
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.width / self.height), 0.1, 5000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def initLighting(self):
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        # Define light properties
        light_ambient = [0.2, 0.2, 0.2, 1.0]
        light_diffuse = [0.8, 0.8, 0.8, 1.0]
        light_specular = [1.0, 1.0, 1.0, 1.0]

        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)

        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        material_specular = [0.1, 0.1, 0.1, 1.0]
        material_shininess = [10.0]
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess)

    def initCameraCenter(self, jointsPositions: jointsPositionsInput):
        if len(jointsPositions) == 0:
            return
        self.cameraCenter = jointsPositions[0][0][0]
        for jointsPosition, _ in jointsPositions:
            for jointPosition in jointsPosition:
                if self.cameraCenter[1] > jointPosition[1]:
                    self.cameraCenter[1] = jointPosition[1]
        self.cameraCenterInitializedByFirstData = True

    def handleInput(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if not self.mouseDragging:
                        self.prevMousePosition = pygame.mouse.get_pos()
                    self.mouseDragging = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouseDragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self.mouseDragging:
                    mouseX, mouseY = pygame.mouse.get_pos()
                    xrot = (mouseY - self.prevMousePosition[1]) * math.pi / self.height
                    yrot = (mouseX - self.prevMousePosition[0]) * math.pi / self.width
                    self.cameraAngleX = max(
                        -math.pi / 2 + 1e-8,
                        min(math.pi / 2 - 1e-8, self.cameraAngleX + xrot),
                    )
                    self.cameraAngleY = self.cameraAngleY + yrot

                    self.prevMousePosition = (mouseX, mouseY)

            elif event.type == pygame.MOUSEWHEEL:
                self.cameraDistance += event.y * 30
                if self.cameraDistance < 20:
                    self.cameraDistance = 20

        glLoadIdentity()

        # rotate camera along x axis first, and then along y axis
        cameraY = self.cameraDistance * math.sin(self.cameraAngleX)
        cameraX = (
            self.cameraDistance
            * math.cos(self.cameraAngleX)
            * math.cos(self.cameraAngleY)
        )
        cameraZ = (
            self.cameraDistance
            * math.cos(self.cameraAngleX)
            * math.sin(self.cameraAngleY)
        )
        cx, cy, cz = self.cameraCenter
        gluLookAt(cx + cameraX, cy + cameraY, cz + cameraZ, cx, cy, cz, 0, 1, 0)

        glLightfv(GL_LIGHT0, GL_POSITION, self.lightPosition)

    def drawSphere(self, position, color=(0.5, 0.5, 1.0)):
        glPushMatrix()
        glTranslatef(position[0], position[1], position[2])
        glColor3f(color[0], color[1], color[2])
        glutSolidSphere(self.sphereRadius, 20, 20)
        glPopMatrix()

    def drawCuboid(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        rotationQuat: np.ndarray,
        color: tuple[float, float, float] = (1.0, 0.5, 0.5),
    ):
        glPushMatrix()

        mid_point = (start_pos + end_pos) / 2
        direction = end_pos - start_pos
        length = max(float(np.linalg.norm(direction) - 2 * self.sphereRadius), 0.0)

        glTranslatef(mid_point[0], mid_point[1], mid_point[2])

        angle = 2 * np.arccos(rotationQuat[0])
        axis = rotationQuat[1:]
        if np.linalg.norm(axis) > 1e-3:
            glRotatef(angle * 180 / math.pi, axis[0], axis[1], axis[2])

        glScalef(self.cuboidWidth, length, self.cuboidWidth)

        glColor3f(color[0], color[1], color[2])

        # Draw the cuboid with normals
        vertices = self.cube_faces()
        normals = self.cube_normals()

        glBegin(GL_QUADS)
        for i, face in enumerate(vertices):
            glNormal3fv(normals[i])  # Set the normal for each face
            for vertex in face:
                glVertex3fv(vertex)
        glEnd()

        glPopMatrix()

    def cube_faces(self):
        vertices = [
            [0.5, -0.5, -0.5],  # Front bottom right
            [0.5, 0.5, -0.5],  # Front top right
            [-0.5, 0.5, -0.5],  # Front top left
            [-0.5, -0.5, -0.5],  # Front bottom left
            [0.5, -0.5, 0.5],  # Back bottom right
            [0.5, 0.5, 0.5],  # Back top right
            [-0.5, -0.5, 0.5],  # Back bottom left
            [-0.5, 0.5, 0.5],  # Back top left
        ]

        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Front face
            [vertices[5], vertices[4], vertices[6], vertices[7]],  # Back face
            [vertices[3], vertices[2], vertices[7], vertices[6]],  # Left face
            [vertices[1], vertices[0], vertices[4], vertices[5]],  # Right face
            [vertices[2], vertices[1], vertices[5], vertices[7]],  # Top face
            [vertices[0], vertices[3], vertices[6], vertices[4]],  # Bottom face
        ]
        return faces

    def cube_normals(self):
        normals = [
            [0, 0, -1],  # Front face normal
            [0, 0, 1],  # Back face normal
            [-1, 0, 0],  # Left face normal
            [1, 0, 0],  # Right face normal
            [0, 1, 0],  # Top face normal
            [0, -1, 0],  # Bottom face normal
        ]
        return normals

    def drawChessBoard(self, numGrid: int = 14, blockSize: float = 50):
        # floor is located at cameracenterheight - joint radius
        height = self.cameraCenter[1] - self.sphereRadius

        glPushMatrix()

        halfSize = (numGrid * blockSize) / 2

        for i in range(numGrid):
            for j in range(numGrid):
                x = -halfSize + i * blockSize + self.cameraCenter[0]
                z = -halfSize + j * blockSize + self.cameraCenter[2]

                if (i + j) % 2 == 0:
                    glColor3f(0.9, 0.9, 0.9)
                else:
                    glColor3f(0.1, 0.1, 0.1)

                glBegin(GL_QUADS)
                glVertex3f(x, height, z)
                glVertex3f(x + blockSize, height, z)
                glVertex3f(x + blockSize, height, z + blockSize)
                glVertex3f(x, height, z + blockSize)
                glEnd()

        glPopMatrix()

    # draw elapsed time on top right
    def drawElapsedTimeAndFrame(self, frame: int) -> None:
        font = pygame.font.Font(None, 50)
        elapsedSurface = font.render(
            f"Time: {frame * self.frameTime:.2f}s, Frame: {frame}",
            True,
            (255, 255, 255),
            (0, 0, 0),
        )

        text_data = pygame.image.tostring(elapsedSurface, "RGBA", True)
        width, height = elapsedSurface.get_size()

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.width, 0, self.height)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glRasterPos2d(self.width - width - 10, self.height - 50)
        glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def updateScene(self, objects: sceneInput):
        frame, jointsPositions, linkss = objects

        if not self.cameraCenterInitializedByFirstData:
            self.initCameraCenter(jointsPositions)

        self.handleInput()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # type: ignore

        if not self.running:
            return

        self.drawChessBoard()

        for jointsPosition, color in jointsPositions:
            for jointPosition in jointsPosition:
                self.drawSphere(jointPosition, color=color)

        for links, color in linkss:
            for link in links:
                self.drawCuboid(link[0], link[1], link[2], color=color)

        self.drawElapsedTimeAndFrame(frame)

        pygame.display.flip()

        self.clock.tick(1 / self.frameTime)


if __name__ == "__main__":
    scene = pygameScene()
    while scene.running:
        scene.updateScene((0, [(np.array([[0, 0, 0], [0, 0, 50]]), (1, 0.5, 0.5))], []))
