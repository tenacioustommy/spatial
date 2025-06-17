import math
import random
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.object_manager import ObjectManager
from pathlib import Path
import os

class TDWExperiment(Controller):
    def __init__(self,
                 port: int = 1071,
                 check_version: bool = True,
                 launch_build: bool = True,
                 output_path: str = None):
        super().__init__(port=port, check_version=check_version, launch_build=launch_build)

        self.output_directory = Path(output_path)
        if not self.output_directory.exists():
            self.output_directory.mkdir(parents=True, exist_ok=True)
        print(f"Images will be saved to: {self.output_directory.resolve()}")

        # ImageCapture will use default frequency="always", capturing for specified avatar_ids on every communicate()
        self.image_capture = ImageCapture(path=self.output_directory,
                                          avatar_ids=["main_camera", "top_down_camera"],
                                          pass_masks=["_img"],
                                          png=True)
        self.object_manager = ObjectManager(transforms=True, bounds=True, rigidbodies=False)

        self.add_ons.extend([self.image_capture, self.object_manager])

        self.item_id = self.get_unique_id() # Sofa
        self.chair_1_id = self.get_unique_id()
        self.chair_2_id = self.get_unique_id()
        self.camera_ground_marker_id = self.get_unique_id()
        self.fov_area_markers = []  # For FOV area visualization

        self.MAIN_CAMERA_FOV_ANGLE = 54.43223  # Match the actual camera FOV
        self.FOV_MARKER_LINE_LENGTH = 1.5
        self.FOV_AREA_MARKER_SIZE = 0.05  # Size of markers for FOV area

        self.current_main_camera_position = {"x": 0, "y": 0, "z": 0}

    def _create_fov_area_markers(self, cam_ground_pos: dict, look_at_target: dict, ground_y: float) -> list:
        """Create small markers to fill the FOV area for better visualization."""
        commands = []
        
        cam_x, cam_z = cam_ground_pos["x"], cam_ground_pos["z"]
        look_x, look_z = look_at_target["x"], look_at_target["z"]
        
        dx = look_x - cam_x
        dz = look_z - cam_z
        
        center_angle_rad = math.atan2(dz, dx)
        half_fov_rad = math.radians(self.MAIN_CAMERA_FOV_ANGLE / 2.0)
        
        # Create markers along the FOV area
        num_radial_lines = 2  # Number of radial lines within FOV
        num_markers_per_line = 15  # Number of markers along each radial line
        
        for i in range(num_radial_lines):
            angle_offset = (i / (num_radial_lines - 1) - 0.5) * self.MAIN_CAMERA_FOV_ANGLE
            angle_rad = center_angle_rad + math.radians(angle_offset)
            
            for j in range(1, num_markers_per_line + 1):
                distance = (j / num_markers_per_line) * self.FOV_MARKER_LINE_LENGTH
                marker_pos = {
                    "x": cam_x + distance * math.cos(angle_rad),
                    "y": ground_y,
                    "z": cam_z + distance * math.sin(angle_rad)
                }
                
                marker_id = self.get_unique_id()
                self.fov_area_markers.append(marker_id)
                
                commands.extend([
                    self.get_add_object(model_name="sphere", 
                                      object_id=marker_id, 
                                      position=marker_pos, 
                                      library="models_flex.json"),
                    {"$type": "scale_object", 
                     "id": marker_id, 
                     "scale_factor": {"x": self.FOV_AREA_MARKER_SIZE, 
                                    "y": self.FOV_AREA_MARKER_SIZE, 
                                    "z": self.FOV_AREA_MARKER_SIZE}},
                    {"$type": "set_color", 
                     "id": marker_id, 
                     "color": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 0.0}}  # Yellow with transparency
                ])
        
        return commands

    def setup_scene(self):
        """Initialize the scene: empty room, sofa, two chairs, and cameras."""
        print("Setting up scene...")
        commands = [TDWUtils.create_empty_room(12, 12)]

        sofa_position = {"x": 0, "y": 0, "z": 0}
        commands.append(self.get_add_object(
            model_name="arflex_strips_sofa",
            position=sofa_position,
            object_id=self.item_id,
            library="models_core.json"
        ))

        chair_1_position = {"x": 1.8, "y": 0, "z": 0.2}
        commands.append(self.get_add_object(
            model_name="yellow_side_chair",
            object_id=self.chair_1_id,
            position=chair_1_position,
            rotation={"x": 0, "y": -45, "z": 0},
            library="models_core.json"
        ))

        chair_2_position = {"x": -1.8, "y": 0, "z": 0.2}
        commands.append(self.get_add_object(
            model_name="yellow_side_chair",
            object_id=self.chair_2_id,
            position=chair_2_position,
            rotation={"x": 0, "y": 45, "z": 0},
            library="models_core.json"
        ))

        initial_main_camera_position = {"x": 0, "y": 1.6, "z": -2.5}
        initial_main_camera_look_at = {"x": sofa_position["x"], "y": 0.5, "z": sofa_position["z"]}
        self.current_main_camera_position = initial_main_camera_position 

        commands.extend(TDWUtils.create_avatar(
            avatar_type="A_Img_Caps_Kinematic",
            avatar_id="main_camera",
            position=initial_main_camera_position,
            look_at=initial_main_camera_look_at
        ))

        commands.extend(TDWUtils.create_avatar(
            avatar_id="top_down_camera",
            position={"x": 0, "y": 10, "z": 0},
            look_at={"x": 0, "y": 0, "z": 0}
        ))
        # 提高图像分辨率以获得更清晰的图片
        # 可选分辨率: 1920x1080 (高清), 2048x2048 (正方形高清), 4096x4096 (超高清)
        commands.append({"$type": "set_screen_size", "width": 1920, "height": 1080})

        self.communicate(commands)
        print(f"Placed sofa (ID: {self.item_id}) at {sofa_position}, "
              f"chair 1 (ID: {self.chair_1_id}) at {chair_1_position}, and "
              f"chair 2 (ID: {self.chair_2_id}) at {chair_2_position} in the environment.")

    def capture_perspectives_with_top_down_markers(self):
        """
        For each of 10 camera positions, capture 3 views with random orientations.
        main_camera captures view without FOV markers.
        top_down_camera captures view with FOV markers on the ground plane.
        """
        print("Capturing perspectives. Main cam: no FOV. Top-down cam: with 2D FOV lines...")

        camera_positions = [
            {"x": 3.0,  "y": 1.5, "z": 4.0},
            {"x": -3.5, "y": 1.5, "z": 3.0},
            {"x": 3.5,  "y": 1.5, "z": 3.0},
            {"x": -5.0, "y": 1.5, "z": 3.0},
            {"x": 5.0,  "y": 1.5, "z": 3.0},
            {"x": 2.0,  "y": 1.5, "z": 3.0},
            {"x": -4.0, "y": 1.5, "z": 3.0},
            {"x": 4.0,  "y": 1.5, "z": 3.0},
            {"x": 3.0,  "y": 1.5, "z": 3.0},
            {"x": 1.8,  "y": 1.5, "z": 4.5}
        ]
        num_random_orientations_per_position = 3
        ground_y = 0.01

        total_image_sets = len(camera_positions) * num_random_orientations_per_position
        current_image_set_count = 0

        for i, cam_pos in enumerate(camera_positions):
            self.current_main_camera_position = cam_pos 

            for orientation_idx in range(num_random_orientations_per_position):
                current_image_set_count += 1
                self.fov_area_markers = []  # Reset FOV area markers for this set

                # Define a general area of interest for look_at targets
                interest_center_x = 0.0
                interest_center_z = 0.0
                interest_radius = 2 # Look at points within this radius of the interest_center
                min_horizontal_dist_to_target = 1 # Ensure target is not too close to cam XZ

                while True:
                    rand_lx = random.uniform(interest_center_x - interest_radius, interest_center_x + interest_radius)
                    rand_lz = random.uniform(interest_center_z - interest_radius, interest_center_z + interest_radius)
                    
                    # Check horizontal distance from camera's XZ to target's XZ
                    temp_dx = rand_lx - cam_pos["x"]
                    temp_dz = rand_lz - cam_pos["z"]
                    if math.sqrt(temp_dx**2 + temp_dz**2) >= min_horizontal_dist_to_target:
                        break
                # Set look_at target at the same height as camera for horizontal rotation only
                random_look_at_target = {"x": rand_lx, "y": cam_pos["y"], "z": rand_lz}

                print(f"  Processing set {current_image_set_count}/{total_image_sets}: Cam pos {i+1}, Orientation {orientation_idx+1} (-> {random_look_at_target})")                # 1. Setup main_camera and communicate (captures main_camera view without FOV)
                print(f"    Camera position: {cam_pos}")
                print(f"    Look at target: {random_look_at_target}")
                
                main_cam_setup_commands = [
                    {"$type": "set_field_of_view", "field_of_view": 54.43223, "avatar_id": "main_camera"},
                    {"$type": "teleport_avatar_to",
                     "avatar_id": "main_camera",
                     "position": cam_pos},
                    {"$type": "look_at_position", 
                     "avatar_id": "main_camera",
                     "position": random_look_at_target}
                ]

                # 2. Create and add FOV markers, then communicate (captures top_down_camera view with FOV)
                fov_marker_creation_commands = []
                cam_ground_pos_dict = {"x": cam_pos["x"], "y": ground_y, "z": cam_pos["z"]}
                
                # Camera position marker (blue sphere)
                fov_marker_creation_commands.extend([
                    self.get_add_object(model_name="sphere", object_id=self.camera_ground_marker_id, position=cam_ground_pos_dict, library="models_flex.json"),
                    {"$type": "scale_object", "id": self.camera_ground_marker_id, "scale_factor": {"x": 0.08, "y": 0.08, "z": 0.08}},
                    {"$type": "set_color", "id": self.camera_ground_marker_id, "color": {"r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0}}
                ])

                # Add FOV area markers
                fov_marker_creation_commands.extend(self._create_fov_area_markers(cam_ground_pos_dict, random_look_at_target, ground_y))
                main_cam_setup_commands.extend(fov_marker_creation_commands)
                print(f"    Top_down_camera view (with FOV) captured for set {current_image_set_count}.")

                # 3. Cleanup FOV objects and communicate
                cleanup_commands = [{"$type": "destroy_object", "id": self.camera_ground_marker_id}]
                for marker_id in self.fov_area_markers:
                    cleanup_commands.append({"$type": "destroy_object", "id": marker_id})
                main_cam_setup_commands.extend(cleanup_commands)
                # Clear the lists
                self.fov_area_markers = []
                
                self.communicate(main_cam_setup_commands)
                print(f"    FOV objects removed for set {current_image_set_count}.")
        
        print("Finished all image capture sets.")

    def run(self):
        self.setup_scene()
        self.capture_perspectives_with_top_down_markers()
        self.communicate({"$type": "terminate"})
        print("Simulation terminated.")

if __name__ == "__main__":
    output_directory = r"D:\ComputerScience\Leetcode\spatial\images"
    experiment = TDWExperiment(launch_build=True, output_path=output_directory)
    experiment.run()
    print(f"Experiment finished. Check images in {experiment.output_directory.resolve()}")