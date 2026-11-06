import time
from pathlib import Path

from voice_test import normalize_words


OBJECT_DIRECTORY = (
    Path(__file__).resolve().parent / "models" / "objects"
)

SUPPORTED_EXTENSIONS = {".glb", ".obj", ".stl"}


class VoiceActions:
    def __init__(self, controller):
        self.controller = controller
        self.pending_models = []
        self.selection_expires = 0.0

    @property
    def workspace(self):
        return self.controller.workspace

    @property
    def renderer(self):
        return self.workspace.renderer

    def say(self, text):
        self.controller.speech.say(text)

    def reset_hand_reference(self):
        self.workspace.previous_hand = None
        self.workspace.pinching = False
        self.workspace.opened_at = time.monotonic()

    def update_title(self):
        self.workspace.setWindowTitle(
            f"Ultron — {self.renderer.model_name}"
        )

    def clear_selection(self):
        self.pending_models = []
        self.selection_expires = 0.0

    def handle(self, command):
        try:
            if command.startswith("SELECT_MODEL:"):
                self.select_model(command.split(":", 1)[1])
                return

            # A new command cancels an older selection question.
            self.clear_selection()

            if command == "OPEN_WORKSPACE":
                self.controller.open_workspace()
                self.say("for you always sir.")

            elif command == "CLOSE_WORKSPACE":
                self.controller.close_workspace()
                self.say("have a good day, sir.")

            elif command.startswith("CREATE_"):
                shapes = {
                    "CREATE_CUBE": "cube",
                    "CREATE_SPHERE": "sphere",
                    "CREATE_ROBOT": "robot",
                }

                shape = shapes.get(command)

                if shape is None:
                    return

                self.renderer.create_object(shape)
                self.reset_hand_reference()
                self.controller.open_workspace()
                self.update_title()

                self.say(f"model rendered, sir. I created a {shape} for you.")

            elif command.startswith("LOAD_MODEL:"):
                name = command.split(":", 1)[1]
                self.find_model(name)

            elif command == "SHOW_WIREFRAME":
                self.renderer.set_wireframe(True)
                self.say("model wireframe enabled.")

            elif command == "SHOW_SOLID":
                self.renderer.set_wireframe(False)
                self.say("model solid view enabled.")

            elif command == "RESET_VIEW":
                self.renderer.reset_view()
                self.reset_hand_reference()
                self.say("View reset.")

            elif command == "CLEAR_WORKSPACE":
                self.renderer.clear_workspace()
                self.reset_hand_reference()
                self.update_title()
                self.say("Workspace cleared.")

            elif command == "CANCEL":
                self.say("Cancelled.")

        except Exception as error:
            print(
                "Voice action error:",
                type(error).__name__,
                str(error),
            )
            self.say(
                "I couldn't complete that action. "
                "Check the terminal for details."
            )

    def find_model(self, requested_name):
        OBJECT_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        requested_name = normalize_words(requested_name)

        # Match local filenames rather than treating speech as a path.
        candidates = sorted(
            (
                path
                for path in OBJECT_DIRECTORY.iterdir()
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
                and normalize_words(path.stem) == requested_name
            ),
            key=lambda path: path.name.lower(),
        )

        if not candidates:
            print(
                f"No model named '{requested_name}' in "
                f"{OBJECT_DIRECTORY}"
            )
            self.say(
                f"I couldn't find {requested_name}. "
                "Place the model in the objects folder "
                "and try again."
            )
            return

        if len(candidates) == 1:
            self.load_path(candidates[0])
            return

        if len(candidates) > 5:
            self.say(
                "Too many models share that name. "
                "Give the files distinct names and try again."
            )
            return

        self.pending_models = candidates
        self.selection_expires = time.monotonic() + 60.0

        choices = []

        for index, path in enumerate(candidates, start=1):
            print(f"Model choice {index}: {path.name}")

            extension = " ".join(
                path.suffix[1:].upper()
            )

            choices.append(
                f"Number {index}, {path.stem}, {extension}."
            )

        self.say(
            "I found multiple models. "
            + " ".join(choices)
            + " Say choose one, "
            "or another number."
        )

    def select_model(self, value):
        if (
            not self.pending_models
            or time.monotonic() > self.selection_expires
        ):
            self.clear_selection()
            self.say(
                "There is no active model selection. "
                "Ask me to load the model again."
            )
            return

        index = int(value) - 1

        if not 0 <= index < len(self.pending_models):
            self.say("That number is not in the choices.")
            return

        path = self.pending_models[index]
        self.clear_selection()
        self.load_path(path)

    def load_path(self, path):
        triangle_count = self.renderer.load_model(str(path))

        self.reset_hand_reference()
        self.controller.open_workspace()
        self.update_title()

        print(
            f"Loaded {path.name}: "
            f"{triangle_count:,} triangles"
        )

        self.say(f"{path.stem} loaded.")