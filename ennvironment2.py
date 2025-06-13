import math
import random
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.object_manager import ObjectManager
from pathlib import Path
import os
from PIL import Image, ImageDraw, ImageFont
from tdw.librarian import ModelLibrarian

class TDWExperiment(Controller):
    def __init__(self,
                 port: int = 1071,
                 check_version: bool = True,
                 launch_build: bool = True,
                 output_path: str = None,
                 num_objects: int = 3):
        super().__init__(port=port, check_version=check_version, launch_build=launch_build)

        self.output_directory = Path(output_path)
        if not self.output_directory.exists():
            self.output_directory.mkdir(parents=True, exist_ok=True)
        print(f"Images will be saved to: {self.output_directory.resolve()}")

        # 设置物体数量
        self.num_objects = max(1, num_objects)  # 至少需要1个物体
        print(f"Will place {self.num_objects} objects in the scene")

        # 随机挑选符合要求的物体
        self.librarian = ModelLibrarian("models_core.json")
        self.need = ["chair", "table", "sofa", "bed", "desk", "shelf", "cabinet", "lamp", "couch", "stool"]
        self.selected_objects = self._select_random_objects()

        # 定义多个摄像机的位置和角度，包含前后左右各个角度
        self.camera_configs = [
            # 前方摄像机
            {"id": "camera_1", "position": {"x": 0, "y": 1.5, "z": 4.0}, "look_at": {"x": 0, "y": 1.5, "z": 0}},
            {"id": "camera_2", "position": {"x": 1.5, "y": 1.5, "z": 3.5}, "look_at": {"x": 0, "y": 1.5, "z": 0}},
            {"id": "camera_3", "position": {"x": -1.5, "y": 1.5, "z": 3.5}, "look_at": {"x": 0, "y": 1.5, "z": 0}},
            
            # 背后摄像机
            {"id": "camera_4", "position": {"x": 0, "y": 1.5, "z": -4.0}, "look_at": {"x": 0, "y": 1.5, "z": 0}},
            {"id": "camera_5", "position": {"x": 1.5, "y": 1.5, "z": -3.5}, "look_at": {"x": 0, "y": 1.5, "z": 0}},
            {"id": "camera_6", "position": {"x": -1.5, "y": 1.5, "z": -3.5}, "look_at": {"x": 0, "y": 1.5, "z": 0}},
            
            # 左侧摄像机
            {"id": "camera_7", "position": {"x": -4.0, "y": 1.5, "z": 0}, "look_at": {"x": 0, "y": 1.5, "z": 0}},
            {"id": "camera_8", "position": {"x": -3.5, "y": 1.5, "z": 1.5}, "look_at": {"x": 0, "y": 1.5, "z": 0}},
            {"id": "camera_9", "position": {"x": -3.5, "y": 1.5, "z": -1.5}, "look_at": {"x": 0, "y": 1.5, "z": 0}},
            
            # 右侧摄像机
            {"id": "camera_10", "position": {"x": 4.0, "y": 1.5, "z": 0}, "look_at": {"x": 0, "y": 1.5, "z": 0}},
            {"id": "camera_11", "position": {"x": 3.5, "y": 1.5, "z": 1.5}, "look_at": {"x": 0, "y": 1.5, "z": 0}},
            {"id": "camera_12", "position": {"x": 3.5, "y": 1.5, "z": -1.5}, "look_at": {"x": 0, "y": 1.5, "z": 0}},
            
            # 对角线摄像机
            {"id": "camera_13", "position": {"x": 3.0, "y": 1.5, "z": 3.0}, "look_at": {"x": 0, "y": 1.5, "z": 0}},
            {"id": "camera_14", "position": {"x": -3.0, "y": 1.5, "z": 3.0}, "look_at": {"x": 0, "y": 1.5, "z": 0}},
            {"id": "camera_15", "position": {"x": 3.0, "y": 1.5, "z": -3.0}, "look_at": {"x": 0, "y": 1.5, "z": 0}},
            {"id": "camera_16", "position": {"x": -3.0, "y": 1.5, "z": -3.0}, "look_at": {"x": 0, "y": 1.5, "z": 0}},

            # 俯视摄像机
            {"id": "top_down_camera", "position": {"x": 0, "y": 10, "z": 0}, "look_at": {"x": 0, "y": 0, "z": 0}}
        ]

        # 提取所有摄像机ID
        camera_ids = [config["id"] for config in self.camera_configs]

        # ImageCapture配置包含所有摄像机
        self.image_capture = ImageCapture(path=self.output_directory,
                                          avatar_ids=camera_ids,
                                          pass_masks=["_img"],
                                          png=True)
        self.object_manager = ObjectManager(transforms=True, bounds=True, rigidbodies=False)

        self.add_ons.extend([self.image_capture, self.object_manager])

        # 动态生成物体ID
        self.object_ids = [self.get_unique_id() for _ in range(self.num_objects)]
        self.camera_ground_marker_id = self.get_unique_id()
        self.fov_area_markers = []  # For FOV area visualization

        self.MAIN_CAMERA_FOV_ANGLE = 54.43223  # Match the actual camera FOV
        self.FOV_MARKER_LINE_LENGTH = 1.5
        self.FOV_AREA_MARKER_SIZE = 0.05  # Size of markers for FOV area

        self.current_main_camera_position = {"x": 0, "y": 0, "z": 0}

    def _select_random_objects(self):
        """从符合要求的模型中随机选择指定数量的物体"""
        suitable_objects = []
        for record in self.librarian.records:
            for keyword in self.need:
                if keyword in record.name.lower():
                    suitable_objects.append(record)
                    break
        
        if len(suitable_objects) < self.num_objects:
            print(f"Warning: Only found {len(suitable_objects)} suitable objects, using all of them")
            return suitable_objects
        
        selected = random.sample(suitable_objects, self.num_objects)
        print(f"Selected objects: {[obj.name for obj in selected]}")
        return selected

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

    def _generate_random_positions(self, num_objects=None, room_size=12, min_distance=2.0):
        """生成随机位置，确保物体之间不重合"""
        if num_objects is None:
            num_objects = self.num_objects
            
        positions = []
        max_attempts = 100  # 防止无限循环
        
        for i in range(num_objects):
            attempts = 0
            while attempts < max_attempts:
                # 在房间范围内生成随机坐标，留出边界空间
                margin = 1.5  # 距离墙壁的最小距离
                x = random.uniform(-room_size/2 + margin, room_size/2 - margin)
                z = random.uniform(-room_size/2 + margin, room_size/2 - margin)
                y = 0  # 物体放在地面上
                
                new_position = {"x": x, "y": y, "z": z}
                
                # 检查与已有位置的距离
                valid = True
                for existing_pos in positions:
                    distance = math.sqrt(
                        (new_position["x"] - existing_pos["x"])**2 + 
                        (new_position["z"] - existing_pos["z"])**2
                    )
                    if distance < min_distance:
                        valid = False
                        break
                
                if valid:
                    positions.append(new_position)
                    print(f"Generated position {i+1}: {new_position}")
                    break
                
                attempts += 1
            
            if attempts >= max_attempts:
                print(f"Warning: Could not find valid position for object {i+1}, using fallback position")
                # 使用后备位置
                fallback_x = (i - num_objects//2) * 3.0  # 居中排列
                positions.append({"x": fallback_x, "y": 0, "z": 0})
        
        return positions

    def setup_scene(self):
        """Initialize the scene: empty room and randomly selected objects."""
        print("Setting up scene...")
        commands = [TDWUtils.create_empty_room(12, 12)]

        # 生成随机位置
        positions = self._generate_random_positions(self.num_objects, room_size=12, min_distance=2.0)
        
        for i, (obj_record, position, obj_id) in enumerate(zip(self.selected_objects, positions, self.object_ids)):
            # 为某些物体添加随机旋转
            rotation = {"x": 0, "y": random.uniform(0, 360), "z": 0}
            
            commands.append(self.get_add_object(
                model_name=obj_record.name,
                position=position,
                object_id=obj_id,
                rotation=rotation,
                library="models_core.json"
            ))
            print(f"Placed {obj_record.name} (ID: {obj_id}) at {position} with rotation {rotation}")

        # 一次性创建所有摄像机
        for camera_config in self.camera_configs:
            commands.extend(TDWUtils.create_avatar(
                avatar_type="A_Img_Caps_Kinematic",
                avatar_id=camera_config["id"],
                position=camera_config["position"],
                look_at=camera_config["look_at"]
            ))

        # 提高图像分辨率以获得更清晰的图片
        commands.append({"$type": "set_screen_size", "width": 1024, "height": 1024})

        self.communicate(commands)
        print(f"Scene setup complete. Created {len(self.camera_configs)} cameras.")
    def capture_all_perspectives(self):
        """Capture images from all cameras simultaneously with FOV markers."""
        print("Capturing images from all cameras with FOV markers...")
        
        ground_y = 0.01
        all_fov_marker_ids = []
        all_camera_marker_ids = []
        all_commands = []
        
        # 为每个摄像机（除了俯视摄像机）创建FOV标记
        for camera_config in self.camera_configs:
            if camera_config["id"] == "top_down_camera":
                continue
                
            cam_pos = camera_config["position"]
            look_at = camera_config["look_at"]
            
            # 为每个摄像机创建地面位置标记
            camera_marker_id = self.get_unique_id()
            all_camera_marker_ids.append(camera_marker_id)
            cam_ground_pos = {"x": cam_pos["x"], "y": ground_y, "z": cam_pos["z"]}
            
            # 创建摄像机位置标记
            camera_commands = [
                self.get_add_object(model_name="sphere", 
                                  object_id=camera_marker_id, 
                                  position=cam_ground_pos, 
                                  library="models_flex.json"),
                {"$type": "scale_object", 
                 "id": camera_marker_id, 
                 "scale_factor": {"x": 0.08, "y": 0.08, "z": 0.08}},
                {"$type": "set_color", 
                 "id": camera_marker_id, 
                 "color": {"r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0}}  # 蓝色摄像机标记
            ]
            
            # 为这个摄像机创建FOV区域标记
            self.fov_area_markers = []  # 重置FOV标记列表
            fov_commands = self._create_fov_area_markers(cam_ground_pos, look_at, ground_y)
            camera_commands.extend(fov_commands)
            all_fov_marker_ids.extend(self.fov_area_markers)
            
            # 将所有命令添加到总命令列表
            all_commands.extend(camera_commands)
        all_commands.append({"$type": "terminate"})
        # 一次性执行所有命令并捕获图像
        print(f"Creating {len(all_commands)} FOV markers and capturing images...")
        self.communicate(all_commands)
        
        print(f"Images captured from {len(self.camera_configs)} cameras with FOV visualization.")

    def add_camera_labels_to_top_down_image(self):
        """在俯视图上添加摄像机编号标签"""
        try:
            # 直接使用固定的俯视图文件路径
            top_down_image_path = self.output_directory / "top_down_camera" / "img_0001.png"
            
            if not top_down_image_path.exists():
                print(f"Top-down image not found: {top_down_image_path}")
                return
            
            print(f"Processing top-down image: {top_down_image_path}")
            
            # 打开图片
            img = Image.open(top_down_image_path).convert("RGBA")
            draw = ImageDraw.Draw(img)
            
            # 尝试加载字体，如果失败则使用默认字体
            try:
                font = ImageFont.truetype("arial.ttf", 30)
            except:
                font = ImageFont.load_default()
            
            # 图片尺寸
            img_width, img_height = img.size
            
            # 场景中心在图片中心，计算像素比例
            # 假设场景范围是12x12，图片是1024x1024
            scene_size = 12.0
            pixel_per_unit = img_width / scene_size
            center_x = img_width // 2
            center_y = img_height // 2
            
            # 添加摄像机编号标签（从camera_configs中提取）
            for camera_config in self.camera_configs:
                if camera_config["id"] == "top_down_camera":
                    continue
                    
                cam_pos = camera_config["position"]
                camera_number = camera_config["id"].split("_")[-1]  # 提取数字部分
                
                # 将3D坐标转换为2D图片像素坐标
                pixel_x = center_x + int(cam_pos["x"] * pixel_per_unit)
                pixel_y = center_y - int(cam_pos["z"] * pixel_per_unit)  # 注意Y轴翻转
                
                # 绘制编号文本
                text = camera_number
                
                # 计算文本位置（居中）
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                text_x = pixel_x - text_width // 2
                text_y = pixel_y - text_height // 2
                
                # 直接绘制红色文本
                draw.text((text_x, text_y), text, fill=(255, 0, 0, 255), font=font)
            
            # 保存修改后的图片
            labeled_image_path = top_down_image_path.parent / f"labeled_{top_down_image_path.name}"
            img.save(labeled_image_path)
            print(f"Labeled image saved to: {labeled_image_path}")
            
        except Exception as e:
            print(f"Error adding labels to top-down image: {e}")

    def run(self):
        self.setup_scene()
        self.capture_all_perspectives()
        # self.communicate({"$type": "terminate"})
        print("Simulation terminated.")
        
        # 在场景终止后添加标签
        self.add_camera_labels_to_top_down_image()

if __name__ == "__main__":
    output_directory = r"D:\ComputerScience\Leetcode\spatial\images1"
    experiment = TDWExperiment(launch_build=True, output_path=output_directory, num_objects=5)
    experiment.run()
    print(f"Experiment finished. Check images in {experiment.output_directory.resolve()}")