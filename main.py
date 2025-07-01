import glfw
import moderngl
import numpy as np
from pyrr import Matrix44
import math
import random
import noise
import a_star

SCREEN_HEIGHT = 600
SCREEN_WIDTH = 800

vertex_shader_code = """ 
#version 330
in vec3 in_position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out float height;

void main() {
    vec4 world_position = model * vec4(in_position, 1.0);
    height = in_position.y;
    gl_Position = projection * view * world_position;
}
"""

fragment_shader_code = """
#version 330
in float height;
out vec4 fragColor;

void main() {
    // Normalize height to roughly match your terrain height range
    float normalized_height = clamp((height - 0.0) / 15.0, 0.0, 1.0);
    
    // Gradient from blue (low) to green (mid) to red (high)
    vec3 low_color = vec3(0.2, 0.3, 0.8);    // Blue
    vec3 mid_color = vec3(0.0, 0.9, 0.2);    // Green
    vec3 high_color = vec3(0.9, 0.2, 0.2);   // Red
    
    vec3 color;
    if (normalized_height < 0.5) {
        color = mix(low_color, mid_color, normalized_height * 2.0);
    } else {
        color = mix(mid_color, high_color, (normalized_height - 0.5) * 2.0);
    }
    
    fragColor = vec4(color, 1.0);
}
"""

class Terrain:
    def __init__(self, ctx):
        self.ctx = ctx

        self.WIDTH = 20 # x
        self.HEIGHT = 10 # y
        self.DEPTH = 40 # z

        self.HALF_WIDTH = self.WIDTH / 2
        self.HALF_HEIGHT = self.HEIGHT / 2
        self.HALF_DEPTH = self.DEPTH / 2

        self.vertex_amount = 10000

        self.rows, self.cols = calculate_rows_cols(self.WIDTH, self.DEPTH, self.vertex_amount) # gets the amount of cols and rows needed depending on the vertex amount

        self.top_vertices = self.generate_top_face_vertices()
        self.top_vertices = self.top_vertices.reshape(-1, 3)  # shape (N, 3)

        self.BASE_HEIGHT = 0

        self.bottom_vertices = np.array([
            [-self.HALF_WIDTH, self.BASE_HEIGHT,  self.HALF_DEPTH],  # front-bottom-left
             [self.HALF_WIDTH, self.BASE_HEIGHT,  self.HALF_DEPTH],  # front-bottom-right
            [-self.HALF_WIDTH, self.BASE_HEIGHT, -self.HALF_DEPTH],  # back-bottom-left
             [self.HALF_WIDTH, self.BASE_HEIGHT, -self.HALF_DEPTH],  # back-bottom-right
        ], dtype='f4') # 32-bit float

        # side vertices

        self.front_side_vertices = self.generate_side_face_vertices(1)  # front side
        self.front_side_vertices = self.front_side_vertices.reshape(-1, 3)  # shape (N, 3)

        self.back_side_vertices = self.generate_side_face_vertices(2)  # back side
        self.back_side_vertices = self.back_side_vertices.reshape(-1, 3)

        self.left_side_vertices = self.generate_side_face_vertices(3)  # left side
        self.left_side_vertices = self.left_side_vertices.reshape(-1, 3)

        self.right_side_vertices = self.generate_side_face_vertices(4)  # right side
        self.right_side_vertices = self.right_side_vertices.reshape(-1, 3)

        self.vertices = np.vstack([
            self.bottom_vertices, 
            self.front_side_vertices, 
            self.back_side_vertices, 
            self.left_side_vertices, 
            self.right_side_vertices, 
            self.top_vertices
            ]).astype('f4')
        
        # defines triangles using the vertex

        self.top_indices = self.generate_top_face_indices()
        self.front_indices = self.generate_side_face_indices(1)  # front side
        self.back_indices = self.generate_side_face_indices(2)  # back side
        self.left_indices = self.generate_side_face_indices(3)  # left side
        self.right_indices = self.generate_side_face_indices(4)  # right side
        self.bottom_indices = np.array([ 0, 1, 2, 1, 3, 2, ], dtype='i4') # 32-bit int | only the bottom face is a single quad only, 2 triangles is needed

        self.indices = np.concatenate([self.bottom_indices, self.top_indices, self.back_indices, self.front_indices, self.right_indices, self.left_indices]).astype('i4')

        self.program = self.create_shader_program()
        self.vao = self.create_vbo_ibo_vao(self.program)

    def generate_top_face_vertices(self):
        vertices = []
        for row in range(self.rows):
            z = -self.HALF_DEPTH + (row / (self.rows - 1)) * self.DEPTH # z from -20 to +20
            for col in range(self.cols):
                x = -self.HALF_WIDTH + (col / (self.cols - 1)) * self.WIDTH # x from -20 to +20
                y = self.HEIGHT + height_manager(x, z)
                vertices.extend([x, y, z])
        return np.array(vertices, dtype='f4')
    
    def generate_side_face_vertices(self, side):
        top_vertices = self.generate_top_face_vertices().reshape((self.rows, self.cols, 3))
        vertices = []

        if side == 1:
            for col in range(self.cols): # front side | makes vertical strips of vertices with y same as the current row/col based on which side is being calculated
                x = -self.HALF_WIDTH + (col / (self.cols - 1)) * self.WIDTH # starts with -20 and then has a factor based on the amount of cols and rows and adds that to the position and then scales that factor because the factor is a number between 0 and 1 and we need it to be between -20 and 20
                z = -self.HALF_DEPTH
                y = 0
                for row in range(self.rows):
                    y = self.BASE_HEIGHT + (row / (self.rows - 1)) * (top_vertices[row, col, 1] - self.BASE_HEIGHT)
                    vertices.extend([x, y, z])
            return np.array(vertices, dtype='f4')
        
        elif side == 2:
            for col in range(self.cols): # back side
                x = -self.HALF_WIDTH + (col / (self.cols - 1)) * self.WIDTH
                z = self.HALF_DEPTH
                y = 0
                for row in range(self.rows):
                    y = self.BASE_HEIGHT + (1 - row / (self.rows - 1)) * (top_vertices[row, col, 1] - self.BASE_HEIGHT)
                    vertices.extend([x, y, z])
            return np.array(vertices, dtype='f4')

        elif side == 3:
            for row in range(self.rows): # left side
                z = -self.HALF_DEPTH + (row / (self.rows - 1)) * self.DEPTH
                x = -self.HALF_WIDTH
                y = 0
                for col in range(self.cols):
                    y = self.BASE_HEIGHT + (1 - col / (self.cols - 1)) * (top_vertices[row, col, 1] - self.BASE_HEIGHT)
                    vertices.extend([x, y, z])
            return np.array(vertices, dtype='f4')

        elif side == 4:
            for row in range(self.rows): # right side
                z = -self.HALF_DEPTH + (row / (self.rows - 1)) * self.DEPTH
                x = self.HALF_WIDTH
                y = 0
                for col in range(self.cols):
                    y = self.BASE_HEIGHT + (col / (self.cols - 1)) * (top_vertices[row, col, 1] - self.BASE_HEIGHT)
                    vertices.extend([x, y, z])
            return np.array(vertices, dtype='f4')
    
    def generate_top_face_indices(self):
        indices = []
        offset = len(self.bottom_vertices) + len(self.front_side_vertices) + len(self.back_side_vertices) + len(self.left_side_vertices) + len(self.right_side_vertices)
        for row in range(self.rows - 1): # ifnore the last row because no triangle below it
            for col in range(self.cols - 1):
                bottom_left = row * self.cols + col + offset
                bottom_right = row * self.cols + (col + 1) + offset
                top_left = (row + 1) * self.cols + col + offset
                top_right = (row + 1) * self.cols + (col + 1) + offset

                indices.extend([bottom_left, bottom_right, top_left])  # First triangle
                indices.extend([bottom_right, top_right, top_left])    # Second triangle

        return np.array(indices, dtype='i4')
    
    def generate_side_face_indices(self, side): # makes the triangles for the side faces
        indices = []
        if side == 1:  # front side
            offset = len(self.bottom_vertices)
            for row in range(self.rows - 1):
                for col in range(self.cols - 1):
                    bottom_left = row * self.cols + col + offset
                    bottom_right = row * self.cols + (col + 1) + offset
                    top_left = (row + 1) * self.cols + col + offset
                    top_right = (row + 1) * self.cols + (col + 1) + offset

                    indices.extend([bottom_left, bottom_right, top_left]) # triangle 1
                    indices.extend([bottom_right, top_right, top_left]) # triangle 2
            return np.array(indices, dtype='i4')
        
        elif side == 2: # back side
            offset = len(self.bottom_vertices) + len(self.front_side_vertices)
            for row in range(self.rows - 1):
                for col in range(self.cols - 1):
                    bottom_left = row * self.cols + col + offset
                    bottom_right = row * self.cols + (col + 1) + offset
                    top_left = (row + 1) * self.cols + col + offset
                    top_right = (row + 1) * self.cols + (col + 1) + offset

                    indices.extend([bottom_left, bottom_right, top_left])
                    indices.extend([bottom_right, top_right, top_left])
            return np.array(indices, dtype='i4')
        
        elif side == 3: # left side
            offset = len(self.bottom_vertices) + len(self.front_side_vertices) + len(self.back_side_vertices)
            for row in range(self.rows - 1):
                for col in range(self.cols - 1):
                    bottom_left = row * self.cols + col + offset
                    bottom_right = row * self.cols + (col + 1) + offset
                    top_left = (row + 1) * self.cols + col + offset
                    top_right = (row + 1) * self.cols + (col + 1) + offset

                    indices.extend([bottom_left, bottom_right, top_left])
                    indices.extend([bottom_right, top_right, top_left])
            return np.array(indices, dtype='i4')
        
        elif side == 4: # right side
            offset = len(self.bottom_vertices) + len(self.front_side_vertices) + len(self.back_side_vertices) + len(self.left_side_vertices)
            for row in range(self.rows - 1):
                for col in range(self.cols - 1):
                    bottom_left = row * self.cols + col + offset
                    bottom_right = row * self.cols + (col + 1) + offset
                    top_left = (row + 1) * self.cols + col + offset
                    top_right = (row + 1) * self.cols + (col + 1) + offset

                    indices.extend([bottom_left, bottom_right, top_left])
                    indices.extend([bottom_right, top_right, top_left])
            return np.array(indices, dtype='i4')

    def create_vbo_ibo_vao(self, program): # vertices buffer object, index buffer object, vertex array object, (vbo, ibo translates the values into bytes)
        vbo = self.ctx.buffer(self.vertices.tobytes()) # (vbo, ibo translates the values into bytes)
        ibo = self.ctx.buffer(self.indices.tobytes()) # (vbo, ibo translates the values into bytes)
        return self.ctx.vertex_array(program, [(vbo, '3f', 'in_position')], index_buffer=ibo) # combines the buffers and adds the shader to it aswell
    
    def create_shader_program(self):
        program = self.ctx.program( # makes a shader program based on the shaders, should add into the terrain class
            vertex_shader = vertex_shader_code,
            fragment_shader = fragment_shader_code
        )
        return program
    
    def render(self, view, projection):
        # rotate the model
        angle = glfw.get_time() # the time used to rotate the object by adding to

        model = Matrix44.from_y_rotation(angle, dtype='f4') #as before model controls the objects position in space this time we use from_y_rotation instead of identity to rotate the object

        # sends matrices to GPU
        self.program['model'].write(model)
        self.program['view'].write(view)
        self.program['projection'].write(projection)

        self.vao.render()

def height_manager(x, z): # make my own height manager that uses perlin noise to generate a terrain
    WIDTH = 20.0
    DEPTH = 40.0
    
    # Normalize x, z from terrain coords to [0, 1]
    nx = (x + WIDTH / 2) / WIDTH
    nz = (z + DEPTH / 2) / DEPTH
    
    # Scale factors for noise frequency relative to normalized coords
    base_freq = 2.0   # How many hills across the terrain
    mountain_freq = 7.0  # More frequent ridges for mountains
    control_freq = 5.0   # Controls mountain chain shape
    
    # Base hills
    base_height = 0
    amplitude = 0.5
    octaves = 6
    persistence = 1
    lacunarity = 2.0
    
    for i in range(octaves):
        freq = base_freq * (lacunarity ** i)
        amp = amplitude * (persistence ** i)
        base_height += amp * noise.pnoise2(nx * freq, nz * freq)
    
    # Control noise for chaining mountains (normalized to 0..1)
    control = noise.pnoise2(nx * control_freq, nz * control_freq)
    control = (control + 1) / 2
    
    # Ridge noise for mountain ridges
    raw_mountain = noise.pnoise2(nx * mountain_freq, nz * mountain_freq)
    ridge = 0.5 - abs(raw_mountain)
    
    offset = 0.05  # small offset in normalized space (around 1 unit in x)
    sample_left = noise.pnoise2((nx - offset) * mountain_freq, nz * mountain_freq)
    sample_right = noise.pnoise2((nx + offset) * mountain_freq, nz * mountain_freq)
    
    ridge_left = 1.0 - abs(sample_left)
    ridge_right = 1.0 - abs(sample_right)
    
    ridge_smooth = (ridge + ridge_left + ridge_right) / 3.0
    ridge_smooth = ridge_smooth ** 1.5
    
    # Combine ridge with control to chain mountains
    mountain_amplitude = 10.0
    mountain_height = ridge_smooth * control * mountain_amplitude
    
    # Final height
    height = base_height + mountain_height
    
    # Clamp minimum height
    if height < 0:
        height = 0
    
    return height

def calculate_rows_cols(x, z, vertex_amount):
    rows = math.ceil(math.sqrt((vertex_amount * z / x)))
    cols = math.ceil((x / z) * rows)
    return rows, cols

def initialize_window():
    glfw.init()
    if not glfw.init():
        raise Exception("GLFW could not initialize!")
    
    window = glfw.create_window(SCREEN_WIDTH, SCREEN_HEIGHT, "Terrain", None, None)

    if not window:
        glfw.terminate() # removes the RAM that the window uses
        raise Exception("Window could not be created!")
    
    glfw.make_context_current(window) # connects the window to OpenGL

    glfw.swap_interval(1)

    ctx = moderngl.create_context() # modern OpenGL context
    ctx.enable(moderngl.DEPTH_TEST) # makes the closest pixel to the camera display

    return window, ctx

def main():
    window, ctx = initialize_window()

    terrain = Terrain(ctx)

    while not glfw.window_should_close(window):
        glfw.poll_events() # listens for keyboard, mouse or window events like closing the window
        ctx.clear(0.1, 0.1, 0.1, 1.0) # this clears the screen with the color gray before rendering the next frame

        view = Matrix44.look_at( # defines the camera position and where it is looking
            eye = (0.0, 30, -34.0), # the eye is a bit further in the z direction of the center of the scene
            target = (0.0, 0.0, 0.0), # target is what the camera will be looking at and right now its looking at the origin
            up = (0.0, 1.0, 0.0), # just for the camera to know which way up is
            dtype='f4' # float32
        )

        projection = Matrix44.perspective_projection( # this is a 3d perspective effect that makes objects appear smaller further away
            45.0, 800 / 600, 0.1, 100.0, dtype = 'f4'
        )

        terrain.render(view, projection)

        glfw.swap_buffers(window)

if __name__ == '__main__':
    main()