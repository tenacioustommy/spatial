from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.librarian import ModelLibrarian
from tdw.output_data import Bounds, OutputData
from tdw.add_ons.object_manager import ObjectManager

from pathlib import Path
import time
import numpy as np
import math
import functools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import PIL
import json



@dataclass
class PlacedObject:
    """Dataclass to represent a placed object in the scene"""
    id: int
    name: str
    model: str
    position: Dict[str, float]
    rotation: Dict[str, float]
    scale: Dict[str, float]
    bounds: Tuple[float, float]  # (width, depth)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "model": self.model,
            "position": self.position,
            "rotation": self.rotation,
            "scale": self.scale,
            "bounds": self.bounds,
        }

def without_capture(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # remember if the capture add-on was active
        was_active = self.cap in self.controller.add_ons
        if was_active:
            self.controller.add_ons.remove(self.cap)
        try:
            return method(self, *args, **kwargs)
        finally:
            # restore it
            if was_active:
                self.controller.add_ons.append(self.cap)
    return wrapper

class DataConstructor:
    """
    TODO:
        1. Add a H x W grid, each cell holds a value indicating empty space around it
        2. The initial position of the camera should be also an empty cell with no object around it
    """
    def __init__(
            self,
            output_path: str = "tdw_output",
            random_state: int = 42,
            room_size: tuple = (10, 10),
            num_objects: int = 5,
            object_pool: list = None,
            min_distance: float = 0.5,
            screen_size: tuple = (2048, 2048),
        ):
        self.seed = random_state
        self.random_state = np.random.RandomState(random_state)

        self.model_librarian = ModelLibrarian("models_core.json")
        self.room_size = room_size
        self.num_objects = num_objects
        self.object_pool = object_pool or [record.name for record in self.model_librarian.record]
        self.min_distance = min_distance
        self.screen_size = screen_size

        self.controller = Controller(launch_build=False)
        self.placed_objects: List[PlacedObject] = []  # List of PlacedObject dataclass instances

        # Place objects as part of scene setup
        self._setup_scene()

        # Add object manager
        self.object_manager = ObjectManager()
        self.controller.add_ons.append(self.object_manager)

        # Set cameras
        # main_cam_pos = {
        #     "x": self.random_state.uniform(-self.room_size[0]/2, self.room_size[0]/2), 
        #     "y": 0.8, 
        #     "z": self.random_state.uniform(-self.room_size[1]/2, self.room_size[1]/2)
        # }
        main_cam_pos = {
            "x": 0, 
            "y": 0.8, 
            "z": 0
        }
        # main_cam_look_at = {
        #     "x": main_cam_pos['x'], 
        #     "y": 0.8, 
        #     "z": self.room_size[1] / 2
        # } # look at north at beginning
        self.main_cam = ThirdPersonCamera(
            position=main_cam_pos,
            rotation={'x': 0, 'y': 0, 'z': 0},
            # look_at=main_cam_look_at,
            field_of_view=90, # TODO, set fov to 90 horizontally dynamically
            avatar_id="main_cam",
        )
        self.top_down_cam = ThirdPersonCamera(
            position={"x": 0, "y": 10, "z": 0},
            look_at={"x": 0, "y": 0, "z": 0},
            avatar_id="top_down_cam",
        )
        self.controller.add_ons.append(self.main_cam)
        self.controller.add_ons.append(self.top_down_cam)

        # Set image capture
        self.output_directory = Path(output_path)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        print(f"Images will be saved to: {self.output_directory.resolve()}")
        self.cap = ImageCapture(
            path=self.output_directory,
            avatar_ids=["main_cam", "top_down_cam"],
            pass_masks=["_id", "_img"],
            png=True,
        )
        self.cap._save = False # does not save images to disk
        self.controller.add_ons.append(self.cap)

    def _setup_scene(self):
        """Setup the scene by placing objects"""

        # Create extended room with extra padding space (wall)
        PAD = 1
        self.controller.communicate([
            TDWUtils.create_empty_room(self.room_size[0] + PAD, self.room_size[1] + PAD),
            {"$type": "set_screen_size", "width": self.screen_size[0], "height": self.screen_size[1]},
        ])

        self._place_objects()
        
    @without_capture
    def _get_object_bounds_old(
            self,
            model_name: str,
            scale: dict = {"x": 1.0, "y": 1.0, "z": 1.0},
            rotation: dict | int = {"x": 0, "y": 0, "z": 0},
        ) -> tuple:
        """
        Get object bounding box by placing in extended room's temp area
        
        Args:
            model_name: Name of the object model in TDW library
            scale: {"x": float, "y": float, "z": float}
            rotation: {"x": float, "y": float, "z": float} or int: rotation in degrees
            room_size: tuple: (width, depth) of the room
            
        Returns:
            tuple: (width, depth) of the object bounding box

        NOTE the bounding box will also be rotated, the most left, right, front, ... points are also rotated.
        """
        if isinstance(rotation, int):
            rotation = {"x": 0, "y": rotation, "z": 0}

        # Place in temp area (center of the temporary (right) room)
        temp_obj_id = self.controller.get_unique_id()
        
        resp = self.controller.communicate([
            self.controller.get_add_object(
                model_name=model_name,
                position={'x': 0, 'y': 0, 'z': 0},
                rotation=rotation,
                object_id=temp_obj_id,
            ),
            {"$type": "scale_object", "id": temp_obj_id, "scale_factor": scale},
            {"$type": "send_bounds", "ids": [temp_obj_id], "frequency": "once"}
        ])
        
        # Get bounds
        bound = [Bounds(resp[i]) for i in range(len(resp) - 1) if OutputData.get_data_type_id(resp[i]) == 'boun'][0]
        print(f"Object {model_name} left: {bound.get_left(0)}, right: {bound.get_right(0)}, front: {bound.get_front(0)}, back: {bound.get_back(0)}")
        width = abs(bound.get_right(0)[0] - bound.get_left(0)[0])
        depth = abs(bound.get_front(0)[2] - bound.get_back(0)[2])
        bounds = (width, depth)
        print(f"Bounds of {model_name}: {bounds}")
        
        # Clean up temp object
        self.controller.communicate({"$type": "destroy_object", "id": temp_obj_id})

        return bounds
    
    def _get_object_bounds(
            self,
            model_name: str,
            scale: dict = {"x": 1.0, "y": 1.0, "z": 1.0},
            rotation: dict | int = {"x": 0, "y": 0, "z": 0},
        ) -> tuple:
        """
        Get object bounds
        """
        record = self.model_librarian.get_record(model_name)
        bounds = record.bounds # dict of left, right, front, back, bottom, top
        # Apply scale
        width = (bounds['right']['x'] - bounds['left']['x']) * scale['x']
        depth = (bounds['front']['z'] - bounds['back']['z']) * scale['z']
        
        # Apply rotation
        if isinstance(rotation, dict):
            rotation = rotation['y']
            
        # For y-axis rotation, width and depth are swapped based on angle
        angle = rotation % 360
        if angle in [0, 180]:
            pass
        elif angle in [90, 270]:
            width, depth = depth, width
        else:
            angle_rad = math.radians(angle)
            new_width = abs(width * math.cos(angle_rad)) + abs(depth * math.sin(angle_rad))
            new_depth = abs(width * math.sin(angle_rad)) + abs(depth * math.cos(angle_rad))
            width, depth = new_width, new_depth
            
        return (width, depth)
        
    
    def _check_overlap(self, x, z, width, depth, min_distance=None):
        """Check if position overlaps with existing objects (including min_distance)"""
        if min_distance is None:
            min_distance = self.min_distance
        for obj in self.placed_objects:
            px, pz = obj.position['x'], obj.position['z']
            pw, pd = obj.bounds
            if (abs(x - px) < (width + pw) / 2 + min_distance and 
                abs(z - pz) < (depth + pd) / 2 + min_distance):
                return True
        return False
    
    def _find_valid_position(
            self,
            width: float,
            depth: float,
            max_attempts: int = 100,
        ) -> tuple:
        """
        Find valid non-overlapping position
        
        Args:
            width: Width of the object
            depth: Depth of the object
            max_attempts: Maximum number of attempts to find a valid position
            
        Returns:
            tuple: (x, z) of the valid position

        TODO: maintain a H x W grid, each cell holds a value indicating empty space around it
        """
        for _ in range(max_attempts):
            x = int(self.random_state.randint(int(-self.room_size[0] / 2 + width / 2), 
                              int(self.room_size[0] / 2 - width / 2)))
            z = int(self.random_state.randint(int(-self.room_size[1] / 2 + depth / 2), 
                              int(self.room_size[1] / 2 - depth / 2)))
            
            if not self._check_overlap(x, z, width, depth):
                return x, z
        return None, None
    
    def _place_objects(self) -> None:
        """Main function to place objects in room"""
        
        placed_count = 0
        attempts = 0
        max_total_attempts = self.num_objects * 10
        
        while placed_count < self.num_objects and attempts < max_total_attempts:
            attempts += 1
            
            # Random object and properties
            model = self.random_state.choice(self.object_pool)
            
            scale = {"x": 1.0, "y": 1.0, "z": 1.0} # TODO: use scale to constrain the scale of the object
            rotation = {"x": 0, "y": int(self.random_state.choice([0, 90, 180, 270])), "z": 0}
            
            # Get object bounds
            bounds = self._get_object_bounds(model, scale, rotation)
            print(f"Bounds of {model}: {bounds}")
            if not bounds:
                continue
                
            width, depth = bounds
            
            # Find valid position
            x, z = self._find_valid_position(width, depth, max_attempts=50)
            if x is None:
                continue
            
            # Place object in main room
            obj_id = self.controller.get_unique_id()
            self.controller.communicate([
                self.controller.get_add_object(
                    model_name=model,
                    position={'x': x, 'y': 0, 'z': z},
                    rotation=rotation,
                    object_id=obj_id,
                ),
                {"$type": "scale_object", "id": obj_id, "scale_factor": scale},
            ])
            
            # Create PlacedObject dataclass instance
            placed_obj = PlacedObject(
                id=obj_id,
                name=f"{model}_{obj_id}",
                model=model,
                position={'x': x, 'y': 0, 'z': z},
                rotation=rotation,
                scale=scale,
                bounds=(width, depth),
            )
            self.placed_objects.append(placed_obj)
            placed_count += 1
            print(f"Placed {model} at ({x}, {z}) with rotation {rotation}°")
        
        print(f"Successfully placed {placed_count}/{self.num_objects} objects")
        print(f"Placed objects: {[obj.name for obj in self.placed_objects]}")
    
    def _hide_object(self, obj: PlacedObject) -> None:
        """Hide an object by moving it far away"""
        self.controller.communicate([{
            "$type": "teleport_object",
            "id": obj.id,
            "position": {"x": 1000, "y": -1000, "z": 1000}  # Far away position
        }])

    def _show_object(self, obj: PlacedObject) -> None:
        """Show an object by moving it back to original position"""
        self.controller.communicate([{
            "$type": "teleport_object",
            "id": obj.id,
            "position": obj.position
        }])
    
    def _move_camera(
            self,
            position: dict = None, 
            rotation: dict = None,
            look_at: dict | int = None,
        ) -> None:
        """
        Move the main camera to a new position and/or rotation.
        
        Args:
            position: Dictionary with x, y, z coordinates for camera position
            rotation: Dictionary with x, y, z rotation angles in degrees
            look_at: Dictionary of position or object_id to look at

        Refer to https://github.com/threedworld-mit/tdw/tree/master/Documentation/lessons/camera
        """

        if position is not None:
            self.main_cam.teleport(position)
        if rotation is not None:
            self.main_cam.rotate(rotation)
        if look_at is not None:
            self.main_cam.look_at(look_at)
        self.controller.communicate([])

    def _save_cam_image(self, cam_id: str, filename: str, pass_name: str = '_img'):
        """
        Refer to https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/core_concepts/images.md
        """
        image_dict = self.cap.get_pil_images()[cam_id] # pass -> pil image
        output_path = self.output_directory / f"{filename}.png"
        image_dict[pass_name].save(output_path)

    def create_data(self):
        """
        Render images from the main camera
        - valid positions: initial one and all objects' positions
        - valid rotations: 0, 90, 180, 270
        - when capturing at an object's position, temporarily hide that object
        
        NOTE assume camera face north at beginning
        """

        image_meta = []

        capture_positions = [self.main_cam.position]  # Start with original position
        # Add each object's position
        for obj in self.placed_objects:
            obj_pos = obj.position.copy()
            obj_pos['y'] = 0.8  # Keep camera at consistent height
            capture_positions.append(obj_pos)
        
        # Capture images for each position and rotation combination
        for pos_idx, position in enumerate(capture_positions):
            # Hide object if at object position
            obj = None if pos_idx == 0 else self.placed_objects[pos_idx - 1]
            if obj:
                self._hide_object(obj)
            
            for angle in [0, 90, 180, 270]:
                look_at_pos = {
                    "x": position["x"] + math.sin(math.radians(angle)),
                    "y": position["y"],
                    "z": position["z"] + math.cos(math.radians(angle))
                }
                if obj:
                    print(f"Model {obj.name} looking at {look_at_pos}")
                self._move_camera(position=position, look_at=look_at_pos)
                
                # Set filename based on position
                direction = "north" if angle == 0 else "east" if angle == 90 else "south" if angle == 180 else "west"
                filename = f"original_pos_facing_{direction}" if pos_idx == 0 else f"obj_{obj.name}_facing_{direction}"
                
                self._save_cam_image("main_cam", filename)
                image_meta.append({
                    "filename": filename,
                    'position': obj.name if obj else "original",
                    "direction": direction,
                }) # TODO add some meta information like visible objects in the image
                print(f"Captured: {filename} at ({position['x']:.2f}, {position['y']:.2f}, {position['z']:.2f})")
            if obj:
                self._show_object(obj)
        
        # get meta data
        meta_data = {
            "original_cam_position": self.main_cam.position,
            "room_size": self.room_size,
            "screen_size": self.screen_size,
            "random_state": self.seed,
            "num_objects": self.num_objects,
            "object_pool": self.object_pool,
            "min_distance": self.min_distance,
            "objects": [obj.to_dict() for obj in self.placed_objects],
            "images": image_meta,
        }
        with open(self.output_directory / "meta_data.json", "w") as f:
            json.dump(meta_data, f)
        self._save_cam_image("top_down_cam", "top_down_cam_0")

        print(f"Data creation complete. Total images captured: {len(capture_positions) * 4}")
        
    def cleanup(self):
        """Clean up and terminate the simulation."""
        self.controller.communicate({"$type": "terminate"})
        
        if hasattr(self, 'socket') and self.socket:
            self.socket.close()
            print("Socket closed.")
        print("Simulation terminated.")




if __name__ == "__main__":

    object_pool = [
        'blue_club_chair', 'blue_side_chair', 'brown_leather_dining_chair', 
        'brown_leather_side_chair', 'chair_annabelle', 'chair_billiani_doll', 
        'chair_eames_plastic_armchair', 'chair_thonet_marshall', 'chair_willisau_riale', 
        'dark_red_club_chair', 'emeco_navy_chair', 'green_side_chair', 
        'lapalma_stil_chair', 'ligne_roset_armchair', 'linbrazil_diz_armchair', 
        'linen_dining_chair', 'naughtone_pinch_stool_chair', 'red_side_chair', 
        'tan_lounger_chair', 'tan_side_chair', 'vitra_meda_chair', 
        'white_club_chair', 'white_lounger_chair', 'wood_chair', 'yellow_side_chair'
    ]

    data_constructor = DataConstructor(
        output_path="images/task_1",
        random_state=1,
        room_size=(10, 10),
        num_objects=5,
        object_pool=object_pool,
        min_distance=1,
        screen_size=(2048, 2048)
    )

    


    data_constructor.create_data()  
    data_constructor.cleanup()

        
