from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *
from math import radians
import numpy as np

# Set up the vertices of the cube
"""
CUBE_VERTICES = [
    # Front face
    (-1.0, -1.0,  1.0),
    ( 1.0, -1.0,  1.0),
    ( 1.0,  1.0,  1.0),
    (-1.0,  1.0,  1.0),

    # Back face
    (-1.0, -1.0, -1.0),
    ( 1.0, -1.0, -1.0),
    ( 1.0,  1.0, -1.0),
    (-1.0,  1.0, -1.0)
]
"""
# キューブの頂点座標を定義
vertices = np.array([
    # 前面
    [-1.0, -1.0,  1.0],
    [ 1.0, -1.0,  1.0],
    [ 1.0,  1.0,  1.0],
    [-1.0,  1.0,  1.0],

    # 背面
    [-1.0, -1.0, -1.0],
    [ 1.0, -1.0, -1.0],
    [ 1.0,  1.0, -1.0],
    [-1.0,  1.0, -1.0],

    # 上面
    [-1.0,  1.0,  1.0],
    [ 1.0,  1.0,  1.0],
    [ 1.0,  1.0, -1.0],
    [-1.0,  1.0, -1.0],

    # 底面
    [-1.0, -1.0,  1.0],
    [ 1.0, -1.0,  1.0],
    [ 1.0, -1.0, -1.0],
    [-1.0, -1.0, -1.0],

    # 左面
    [-1.0, -1.0, -1.0],
    [-1.0, -1.0,  1.0],
    [-1.0,  1.0,  1.0],
    [-1.0,  1.0, -1.0],

    # 右面
    [ 1.0, -1.0, -1.0],
    [ 1.0, -1.0,  1.0],
    [ 1.0,  1.0,  1.0],
    [ 1.0,  1.0, -1.0]
], dtype=np.float32)
# Set up the edges of the cube
"""
CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]
"""
# キューブの面を定義
indices = np.array([
    0, 1, 2, 0, 2, 3,    # 前面
    4, 5, 6, 4, 6, 7,    # 背面
    8, 9, 10, 8, 10, 11, # 上面
    12, 13, 14, 12, 14, 15, # 底面
    16, 17, 18, 16, 18, 19, # 左面
    20, 21, 22, 20, 22, 23  # 右面
], dtype=np.uint32)

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    return shader

def create_shader_program(vertex_shader_source, fragment_shader_source):
    # Compile the vertex shader
    vertex_shader = compile_shader(vertex_shader_source, GL_VERTEX_SHADER)

    # Compile the fragment shader
    fragment_shader = compile_shader(fragment_shader_source, GL_FRAGMENT_SHADER)

    # Link the vertex and fragment shader into a shader program
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    # Check for errors
    if not glGetProgramiv(program, GL_LINK_STATUS):
        info = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Could not link program: {info}")

    # Delete the shaders as they are now linked to the program
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return program

def main():
    # Initialize Pygame
    pygame.init()
    pygame.display.set_mode((640, 480), DOUBLEBUF | OPENGL)

    # Create the shader program
    program = create_shader_program("""
        attribute vec3 position;
        uniform mat4 modelview;
        uniform mat4 projection;

        void main()
        {
            gl_Position = projection * modelview * vec4(position, 1.0);
        }
    """, """
        void main()
        {
            gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
    """)

    # Get the attribute and uniform locations
    position_location = glGetAttribLocation(program, "position")
    modelview_location = glGetUniformLocation(program, "modelview")
    projection_location = glGetUniformLocation(program, "projection")

    # Create the vertex buffer object (VBO)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    ##glBufferData(GL_ARRAY_BUFFER, len(CUBE_VERTICES) * 4 * 3, (GLfloat * len(CUBE_VERTICES) * 3)(*CUBE_VERTICES), GL_STATIC_DRAW)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    # Create the index buffer object (IBO)
    ibo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
    ##glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(CUBE_EDGES) * 2, (GLushort * len(CUBE_EDGES) * 2)(*sum(CUBE_EDGES, ()) ), GL_STATIC_DRAW)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # Enable the position attribute
    glEnableVertexAttribArray(position_location)
    glVertexAttribPointer(position_location, 3, GL_FLOAT, GL_FALSE, 0, None)

    # Enable depth testing
    glEnable(GL_DEPTH_TEST)

    # Set the clear color
    glClearColor(0.0, 0.0, 0.0, 0.0)

    # Set up the projection matrix
    aspect_ratio = 640 / 480
    ##projection = perspective(45.0, aspect_ratio, 0.1, 100.0)
    projection = gluPerspective(45.0, aspect_ratio, 0.1, 100.0)
    glUniformMatrix4fv(projection_location, 1, GL_FALSE, projection)

    # Set up the modelview matrix
    modelview = identity()
    modelview = translate(modelview, (0.0, 0.0, -5.0))
    glUniformMatrix4fv(modelview_location, 1, GL_FALSE, modelview)

    # Main loop
    clock = pygame.time.Clock()
    angle = 0
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return

        # Clear the screen and depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Rotate the modelview matrix
        angle += 1.0
        modelview = identity()
        modelview = translate(modelview, (0.0, 0.0, -5.0))
        modelview = rotate(modelview, radians(angle), (1.0, 1.0, 1.0))
        glUniformMatrix4fv(modelview_location, 1, GL_FALSE, modelview)

        # Draw the cube
        glDrawElements(GL_LINES, len(CUBE_EDGES) * 2, GL_UNSIGNED_SHORT, None)

        # Flip the display
        pygame.display.flip()

        # Limit the framerate
        clock.tick(60)

if __name__ == "__main__":
    main()
