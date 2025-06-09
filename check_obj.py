
import math
import random
import time
from dataclasses import dataclass, field
from copy import deepcopy

from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.object_manager import ObjectManager
from pathlib import Path

from tdw.object_data.bound import Bound
from tdw.object_data.transform import Transform

"""Useful documentation:
1. Avatar and camera: https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/core_concepts/avatars.md
2. Segmentation: https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/visual_perception/id.md
"""

@dataclass
class Object:
    custom_name: str
    model_name: str
    object_id: int = field(default_factory=int)
    position: dict = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    rotation: dict = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    scale: dict = None


class SimpleTDWScene(Controller):
    def __init__(
            self,
            port: int = 1071,
            output_path: str = "tdw_output",
            field_of_view: float = 90,
            room_size: tuple = (12, 12),  
            screen_size: tuple = (1920, 1080),
            objects: list[Object] = None,
    ):
        super().__init__(port=port, check_version=True, launch_build=False)
        
        # Setup output directory
        self.output_dir = Path(output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir.resolve()}")
        
        # Initialize add-ons
        self.image_capture = ImageCapture(
            path=self.output_dir,
            avatar_ids=["main_camera", "top_down_camera"],
            pass_masks=["_img"],
            png=True
        )
        self.object_manager = ObjectManager(transforms=True, bounds=True, rigidbodies=False)
        self.add_ons.extend([self.image_capture, self.object_manager])
        
        # Track objects
        self.objects, self.static_objects = {}, {}
        for obj in objects:
            obj.object_id = self.get_unique_id()
            self.objects[obj.custom_name] = obj
            self.static_objects[obj.custom_name] = obj

        # settings
        self.field_of_view = field_of_view
        self.room_size = room_size
        self.screen_size = screen_size

        self._setup_scene()
        
    def _setup_scene(self):
        """Create an empty room scene, cameras, and objects."""
        print("Setting up empty scene...")
        commands = [TDWUtils.create_empty_room(self.room_size[0], self.room_size[1])]

        print("Creating default camera")
        # Add default camera
        commands.extend(TDWUtils.create_avatar(
            avatar_type="A_Img_Caps_Kinematic",
            avatar_id="main_camera",
            position={"x": 0, "y": 2, "z": -5},
            look_at={"x": 0, "y": 0, "z": 0}
        ))

        commands.extend(TDWUtils.create_avatar(
            avatar_id="top_down_camera",
            position={"x": 0, "y": 5, "z": 0},
            look_at={"x": 0, "y": 0, "z": 0}
        ))
        
        # Set resolution
        commands.append({"$type": "set_screen_size", "width": self.screen_size[0], "height": self.screen_size[1]})

        # Set field of view
        # if self.field_of_view:
        #     commands.append({"$type": "set_field_of_view", "field_of_view": self.field_of_view, "avatar_id": "main_camera"})

        # add objects
        for custom_name, obj in self.objects.items():
            commands.append(self.get_add_object(
                model_name=obj.model_name,
                object_id=obj.object_id,
                position=obj.position,
                rotation=obj.rotation,
                library='models_core.json'
            ))
            if obj.scale:
                commands.append({
                    "$type": "scale_object",
                    "id": obj.object_id,
                    "scale_factor": obj.scale
                })
        
        self.communicate(commands)
        print("Empty scene created successfully!")

    
    def add_object(self, 
                   model_name: str,
                   custom_name: str,
                   position: dict = None, 
                   rotation: dict = None, 
                   scale: dict = None,
                   library: str = "models_core.json",
                   capture_views: bool = False,
    ):
        """
        Add an object to the scene by model name.
        NOTE: will not be tracked by the object manager
        
        Args:
            model_name: Name of the object model in TDW library
            custom_name: Custom name to reference this object instance
            position: {"x": float, "y": float, "z": float}
            rotation: {"x": float, "y": float, "z": float}
            scale: {"x": float, "y": float, "z": float}
            library: TDW model library to use
            capture_views: Whether to automatically capture multiple views of the object
        """
        position = position or {"x": 0, "y": 0, "z": 0}
        rotation = rotation or {"x": 0, "y": 0, "z": 0}
            
        object_id = self.get_unique_id()
        commands = []
        
        # Add the object
        commands.append(self.get_add_object(
            model_name=model_name,
            object_id=object_id,
            position=position,
            rotation=rotation,
            library=library
        ))
        
        # Apply scaling if specified
        if scale is not None:
            commands.append({
                "$type": "scale_object",
                "id": object_id,
                "scale_factor": scale
            })
        
        self.communicate(commands)
        
        # Store object info
        self.objects[custom_name] = Object(
            model_name=model_name,
            custom_name=custom_name,
            object_id=object_id,
            position=position,
            rotation=rotation,
            scale=scale
        )
        
        print(f"Added {model_name} as '{custom_name}' (ID: {object_id}) at position {position}")
        
        # Automatically capture multiple views if requested
        if capture_views:
            self.capture_multiple_views(custom_name)
        
        return object_id
    
    def capture_multiple_views(self, object_name: str, distance: float = None, height_offset: float = 1.0):
        """
        Capture multiple views of a specific object from different angles.
        
        Args:
            object_name: Name of the object to capture
            distance: Distance from object for camera positioning (auto-calculated if None)
            height_offset: Height offset for camera positioning
        """
        if object_name not in self.objects:
            print(f"Object {object_name} not found in scene")
            return
        
        obj_pos = self.objects[object_name]["position"]
        obj_x, obj_y, obj_z = obj_pos["x"], obj_pos["y"], obj_pos["z"]
        
        # Auto-calculate optimal distance if not provided
        if distance is None:
            distance = self.get_optimal_camera_distance(object_name)
            print(f"Auto-calculated optimal camera distance: {distance:.2f}")
        
        # Define multiple camera positions around the object
        views = [
            # Front view
            {
                "name": "front",
                "position": {"x": obj_x, "y": obj_y + height_offset, "z": obj_z - distance},
                "look_at": obj_pos
            },
            # Right side view
            {
                "name": "right",
                "position": {"x": obj_x + distance, "y": obj_y + height_offset, "z": obj_z},
                "look_at": obj_pos
            },
            # Back view
            {
                "name": "back",
                "position": {"x": obj_x, "y": obj_y + height_offset, "z": obj_z + distance},
                "look_at": obj_pos
            },
            # Left side view
            {
                "name": "left",
                "position": {"x": obj_x - distance, "y": obj_y + height_offset, "z": obj_z},
                "look_at": obj_pos
            },
            # Top-angled view
            {
                "name": "top_angle",
                "position": {"x": obj_x + distance/2, "y": obj_y + distance, "z": obj_z - distance/2},
                "look_at": obj_pos
            },
            # Low-angled view
            {
                "name": "low_angle",
                "position": {"x": obj_x - distance/2, "y": obj_y + 0.5, "z": obj_z - distance/2},
                "look_at": obj_pos
            }
        ]
        
        print(f"Capturing multiple views of {object_name}...")
        
        # Store original camera position to restore later
        original_cam_pos = {"x": 0, "y": 2, "z": -5}
        original_look_at = {"x": 0, "y": 0, "z": 0}
        
        for i, view in enumerate(views):
            # Move camera to the view position
            self.move_camera(view["position"], view["look_at"])
            # self.communicate([])  # Trigger image capture
            print(f"  Captured view {i+1}/6: {view['name']}")
            
            # Small delay to ensure image is captured
            time.sleep(0.1)
        
        # Restore original camera position
        self.move_camera(original_cam_pos, original_look_at)
        print(f"Completed multi-view capture for {object_name}")
    
    def move_camera(self, position: dict, look_at: dict = None):
        """Move the main camera to a new position."""
        look_at = look_at or {"x": 0, "y": 0, "z": 0}
            
        commands = [
            {"$type": "teleport_avatar_to", "avatar_id": "main_camera", "position": position},
            {"$type": "look_at_position", "avatar_id": "main_camera", "position": look_at}
        ]
        
        self.communicate(commands)
        print(f"Camera moved to {position}, looking at {look_at}")
    
    def capture_image(self, filename_suffix: str = ""):
        """Capture an image with optional filename suffix."""
        self.communicate([])  # Trigger image capture
        print(f"Image captured{' with suffix: ' + filename_suffix if filename_suffix else ''}")
    
    def remove_object(self, object_name: str):
        """Remove an object from the scene by name."""
        if object_name in self.objects:
            object_id = self.objects[object_name].object_id
            self.communicate([{"$type": "destroy_object", "id": object_id}])
            del self.objects[object_name]
            print(f"Removed {object_name} (ID: {object_id})")
        else:
            print(f"Object {object_name} not found in scene")
    
    def get_object_bounds(self, object_name: str) -> Bound:
        """
        Get the 3D bounding box information for an object.
        
        Args:
            object_name: Name of the object
            
        Returns:
            Bound: Bounding box information including center, size, and corner points
        """
        if object_name not in self.objects:
            print(f"Object {object_name} not found in scene")
            return None
        
        object_id = self.objects[object_name].object_id
        assert object_id in self.object_manager.bounds
        
        # Get bounds from ObjectManager
        bound: Bound = self.object_manager.bounds[object_id]
        
        return bound
    
    def get_object_transform(self, object_name: str):
        """
        Get the transform information (position, rotation, scale) for an object.
        
        Args:
            object_name: Name of the object
            
        Returns:
            dict: Transform information
        """
        if object_name not in self.objects:
            print(f"Object {object_name} not found in scene")
            return None
        
        object_id = self.objects[object_name].object_id
        
        if object_id in self.object_manager.transforms:
            transform = self.object_manager.transforms[object_id]
            return {
                "position": transform.position,
                "rotation": transform.rotation,
                "forward": transform.forward,
            }
        else:
            print(f"Transform not available for {object_name}")
            return None

    def cleanup(self):
        """Clean up and terminate the simulation."""
        self.communicate({"$type": "terminate"})
        
        if hasattr(self, 'socket') and self.socket:
            self.socket.close()
            print("Socket closed.")
        print("Simulation terminated.")



# Example usage
if __name__ == "__main__":
    scene_controller = SimpleTDWScene(
        output_path="test_tdw_output",
        field_of_view=45,
        room_size=(12, 12),
        screen_size=(1920, 1080)
    )
    scene_controller.add_object("blue_side_chair", position={"x": 0, "y": 0, "z": 0})
    scene_controller.capture_multiple_views("blue_side_chair")
    scene_controller.cleanup()