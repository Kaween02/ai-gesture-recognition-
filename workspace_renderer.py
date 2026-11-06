import math
from pathlib import Path

import moderngl
import numpy as np
import trimesh

from PySide6.QtCore import QTimer
from PySide6.QtOpenGLWidgets import QOpenGLWidget


class WorkspaceRenderer(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ctx = None
        self.program = None

        self.model_buffer = None
        self.model_vao = None

        self.grid_buffer = None
        self.grid_vao = None

        # Existing hand controls use these attributes.
        self.distance = 8.0
        self.hand_angle = 0.0

        self.show_wireframe = False
        self.model_name = "Demo robot"

        self.model_data = None
        self.model_dirty = True

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.update)
        self.model_visible = True
        self.set_mesh(self.create_demo_robot())
    def create_object(self, name):
        if name == "cube":
            mesh = trimesh.creation.box(
                extents=(2.0, 2.0, 2.0)
            )

        elif name == "sphere":
            mesh = trimesh.creation.icosphere(
                subdivisions=3,
                radius=1.0,
            )

        elif name == "robot":
            mesh = self.create_demo_robot()

        else:
            raise ValueError(f"Unsupported object: {name}")

        self.set_mesh(mesh)
        self.model_name = name.title()

    def clear_workspace(self):
        # Hide the object; the floor grid remains visible.
        self.model_visible = False
        self.model_name = "Empty workspace"
        self.update()
    def create_demo_robot(self):
        parts = []

        def box(size, position):
            mesh = trimesh.creation.box(extents=size)
            mesh.apply_translation(position)
            parts.append(mesh)

        # Body and head.
        box((1.3, 1.5, 0.7), (0, 0, 0))
        box((1.0, 0.75, 0.75), (0, 1.2, 0))

        # Arms.
        box((0.35, 1.3, 0.4), (-0.95, 0, 0))
        box((0.35, 1.3, 0.4), (0.95, 0, 0))

        # Legs.
        box((0.45, 1.1, 0.5), (-0.35, -1.3, 0))
        box((0.45, 1.1, 0.5), (0.35, -1.3, 0))

        # Feet.
        box((0.5, 0.25, 0.85), (-0.35, -1.9, 0.15))
        box((0.5, 0.25, 0.85), (0.35, -1.9, 0.15))

        # Eyes protrude slightly from the front.
        box((0.18, 0.15, 0.12), (-0.25, 1.25, 0.41))
        box((0.18, 0.15, 0.12), (0.25, 1.25, 0.41))
        

        return trimesh.util.concatenate(parts)

    def set_mesh(self, mesh):
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("The file does not contain a triangle mesh.")

        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("The model is empty.")

        # Keep this first viewer suitable for the laptop GPU.
        if len(mesh.faces) > 300_000:
            raise ValueError(
                "This model exceeds 300,000 triangles. "
                "Use a lower-detail model for this version."
            )

        vertices = np.asarray(mesh.vertices, dtype=np.float64)

        if not np.isfinite(vertices).all():
            raise ValueError("The model contains invalid coordinates.")

        minimum = vertices.min(axis=0)
        maximum = vertices.max(axis=0)
        center = (minimum + maximum) * 0.5
        size = float(np.max(maximum - minimum))

        if size <= 1e-10:
            raise ValueError("The model has no usable size.")

        # Normalize the largest dimension to 2.8 world units.
        normalized = (vertices - center) * (2.8 / size)

        positions = normalized[mesh.faces].reshape(-1, 3)

        # Flat normals give mechanical models clear surfaces.
        normals = np.repeat(
            np.asarray(mesh.face_normals),
            3,
            axis=0,
        )

        data = np.column_stack((positions, normals))
        data = np.ascontiguousarray(data, dtype="f4")

        if not np.isfinite(data).all():
            raise ValueError("The model contains invalid surface normals.")

        # Replace the current model only after validation.
        self.model_data = data
        self.model_visible = True
        self.model_dirty = True
        self.reset_view()
        self.update()

    def load_model(self, filename):
        path = Path(filename)

        if path.suffix.lower() not in {".glb", ".obj", ".stl"}:
            raise ValueError("Choose a GLB, OBJ or STL model.")

        scene = trimesh.load_scene(
            str(path),
            allow_remote=False,
            process=False,
        )

        mesh = scene.to_mesh()
        self.set_mesh(mesh)

        self.model_name = path.name
        return len(mesh.faces)

    def reset_view(self):
        self.distance = 8.0
        self.hand_angle = 0.0
        self.update()

    def set_wireframe(self, enabled):
        self.show_wireframe = bool(enabled)
        self.update()

    def initializeGL(self):
        self.ctx = moderngl.create_context(require=330)

        print("GPU:", self.ctx.info["GL_RENDERER"])
        print("OpenGL:", self.ctx.info["GL_VERSION"])

        self.context().aboutToBeDestroyed.connect(
            self.release_gpu
        )

        self.program = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec3 in_position;
                in vec3 in_normal;

                uniform float angle;
                uniform float aspect;
                uniform float distance;
                uniform float spin;

                out vec3 surface_normal;

                void main() {
                    float a = angle * spin;
                    float c = cos(a);
                    float s = sin(a);

                    vec3 p = vec3(
                        c * in_position.x + s * in_position.z,
                        in_position.y,
                        -s * in_position.x + c * in_position.z
                    );

                    surface_normal = vec3(
                        c * in_normal.x + s * in_normal.z,
                        in_normal.y,
                        -s * in_normal.x + c * in_normal.z
                    );

                    // Fixed camera tilt.
                    float tilt = 0.35;
                    float ct = cos(tilt);
                    float st = sin(tilt);

                    p = vec3(
                        p.x,
                        ct * p.y - st * p.z,
                        st * p.y + ct * p.z
                    );

                    p.z -= distance;

                    float near_plane = 0.1;
                    float far_plane = 100.0;
                    float f = 1.0 / tan(radians(50.0) * 0.5);

                    float depth_a = -(
                        far_plane + near_plane
                    ) / (far_plane - near_plane);

                    float depth_b = -(
                        2.0 * far_plane * near_plane
                    ) / (far_plane - near_plane);

                    gl_Position = vec4(
                        p.x * f / aspect,
                        p.y * f,
                        depth_a * p.z + depth_b,
                        -p.z
                    );
                }
            """,
            fragment_shader="""
                #version 330

                in vec3 surface_normal;

                uniform vec3 color;
                uniform float lit;

                out vec4 frag_color;

                void main() {
                    float brightness = 1.0;

                    if (lit > 0.5) {
                        vec3 normal = normalize(surface_normal);

                        if (!gl_FrontFacing) {
                            normal = -normal;
                        }

                        vec3 light = normalize(
                            vec3(0.5, 1.0, 0.8)
                        );

                        float diffuse = max(
                            dot(normal, light), 0.0
                        );

                        brightness = 0.28 + diffuse * 0.72;
                    }

                    frag_color = vec4(
                        color * brightness, 1.0
                    );
                }
            """,
        )

        grid = []

        for position in range(-6, 7):
            grid.extend([
                (position, -1.6, -6, 0, 1, 0),
                (position, -1.6,  6, 0, 1, 0),
                (-6, -1.6, position, 0, 1, 0),
                ( 6, -1.6, position, 0, 1, 0),
            ])

        grid_data = np.asarray(grid, dtype="f4")

        self.grid_buffer = self.ctx.buffer(
            grid_data.tobytes()
        )
        self.grid_vao = self.ctx.vertex_array(
            self.program,
            [
                (
                    self.grid_buffer,
                    "3f 3f",
                    "in_position",
                    "in_normal",
                )
            ],
        )

        self.model_dirty = True
        self.ctx.enable(moderngl.DEPTH_TEST)

    def upload_model(self):
        new_buffer = self.ctx.buffer(
            self.model_data.tobytes()
        )

        try:
            new_vao = self.ctx.vertex_array(
                self.program,
                [
                    (
                        new_buffer,
                        "3f 3f",
                        "in_position",
                        "in_normal",
                    )
                ],
            )
        except Exception:
            new_buffer.release()
            raise

        if self.model_vao is not None:
            self.model_vao.release()

        if self.model_buffer is not None:
            self.model_buffer.release()

        self.model_buffer = new_buffer
        self.model_vao = new_vao
        self.model_dirty = False

    def paintGL(self):
        if self.ctx is None or self.program is None:
            return

        framebuffer = self.ctx.detect_framebuffer(
            self.defaultFramebufferObject()
        )
        framebuffer.use()

        width, height = framebuffer.size

        if width <= 0 or height <= 0:
            return

        self.ctx.viewport = (0, 0, width, height)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.wireframe = False

        framebuffer.clear(
            0.025, 0.035, 0.055, 1.0, depth=1.0
        )

        if self.model_dirty:
            self.upload_model()

        self.program["aspect"].value = width / height
        self.program["distance"].value = self.distance
        self.program["angle"].value = self.hand_angle

        # Stationary floor grid.
        self.program["spin"].value = 0.0
        self.program["lit"].value = 0.0
        self.program["color"].value = (0.12, 0.22, 0.28)
        self.grid_vao.render(mode=moderngl.LINES)
        if not self.model_visible:
            return
        
        # Loaded model.
        self.program["spin"].value = 1.0
        self.program["lit"].value = (
            0.0 if self.show_wireframe else 1.0
        )
        self.program["color"].value = (1.0, 0.32, 0.10)

        self.ctx.wireframe = self.show_wireframe

        try:
            self.model_vao.render(mode=moderngl.TRIANGLES)
        finally:
            self.ctx.wireframe = False

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0

        self.distance = max(
            4.0,
            min(18.0, self.distance - steps * 0.6),
        )

        self.update()
        event.accept()

    def showEvent(self, event):
        super().showEvent(event)
        self.timer.start()

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def release_gpu(self):
        if self.ctx is None:
            return

        self.timer.stop()
        self.makeCurrent()

        try:
            for name in (
                "model_vao",
                "model_buffer",
                "grid_vao",
                "grid_buffer",
                "program",
            ):
                resource = getattr(self, name)

                if resource is not None:
                    resource.release()
                    setattr(self, name, None)

            self.ctx.release()
            self.ctx = None
            self.model_dirty = True
        finally:
            self.doneCurrent()