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
random.seed(42)  # 设置随机种子以确保可重复性
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

        # 定义多个摄像机的位置和角度，从12点方向开始顺时针均匀分配16个摄像机
        self.camera_configs = []
        num_cameras = 16
        radius = 4.0  # 摄像机距离中心的半径
        camera_height = 1.5
        
        for i in range(num_cameras):
            # 从12点方向(0度)开始，顺时针分配
            angle_degrees = i * (360 / num_cameras)  # 每个摄像机间隔22.5度
            angle_radians = math.radians(angle_degrees)
            
            # 计算摄像机位置（12点方向为z轴正方向）
            x = radius * math.sin(angle_radians)
            z = radius * math.cos(angle_radians)
            
            camera_config = {
                "id": f"camera_{i+1}",
                "position": {"x": x, "y": camera_height, "z": z},
                "look_at": {"x": 0, "y": camera_height, "z": 0}
            }
            self.camera_configs.append(camera_config)
        
        # 添加四个中心摄像机，分别朝向前后左右四个方向
        center_cameras = [
            {
                "id": "center_north",
                "position": {"x": 0, "y": camera_height, "z": 0},
                "look_at": {"x": 0, "y": camera_height, "z": 5}  # 朝北(前方，正z方向)
            },
            {
                "id": "center_south", 
                "position": {"x": 0, "y": camera_height, "z": 0},
                "look_at": {"x": 0, "y": camera_height, "z": -5}  # 朝南(后方，负z方向)
            },
            {
                "id": "center_east",
                "position": {"x": 0, "y": camera_height, "z": 0},
                "look_at": {"x": 5, "y": camera_height, "z": 0}  # 朝东(右方，正x方向)
            },
            {
                "id": "center_west",
                "position": {"x": 0, "y": camera_height, "z": 0},
                "look_at": {"x": -5, "y": camera_height, "z": 0}  # 朝西(左方，负x方向)
            }
        ]
        self.camera_configs.extend(center_cameras)
        
        # 添加俯视摄像机
        self.camera_configs.append({
            "id": "top_down_camera", 
            "position": {"x": 0, "y": 10, "z": 0}, 
            "look_at": {"x": 0, "y": 0, "z": 0}
        })

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

        self.HORIZONTAL_FOV = 90.0  # 目标水平视野角度
        self.screen_width = 1024
        self.screen_height = 1024

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

    def calculate_vertical_fov(self, horizontal_fov_degrees, width, height):
        """
        根据水平FOV和屏幕尺寸计算垂直FOV
        
        Args:
            horizontal_fov_degrees: 期望的水平视野角度
            width: 屏幕宽度
            height: 屏幕高度
            
        Returns:
            垂直FOV角度
        """
        aspect_ratio = width / height
        horizontal_fov_rad = math.radians(horizontal_fov_degrees)
        vertical_fov_rad = 2 * math.atan(math.tan(horizontal_fov_rad / 2) / aspect_ratio)
        vertical_fov_degrees = math.degrees(vertical_fov_rad)
        
        print(f"Screen size: {width}x{height}, Aspect ratio: {aspect_ratio:.3f}")
        print(f"Horizontal FOV: {horizontal_fov_degrees}°, Calculated Vertical FOV: {vertical_fov_degrees:.2f}°")
        
        return vertical_fov_degrees

    def setup_scene(self):
        """Initialize the scene: empty room and randomly selected objects."""
        print("Setting up scene...")
        commands = [TDWUtils.create_empty_room(12,12)]  # 房间尺寸：12x12米的正方形

        # 设置统一的屏幕尺寸
        commands.append({"$type": "set_screen_size", "width": self.screen_width, "height": self.screen_height})

        # 计算垂直FOV以保持90度水平视野
        vertical_fov = self.calculate_vertical_fov(self.HORIZONTAL_FOV, self.screen_width, self.screen_height)

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

        # 为所有摄像机设置计算后的垂直FOV以保持90度水平视野
        for camera_config in self.camera_configs:
            if camera_config["id"] != "top_down_camera":
                commands.append({
                    "$type": "set_field_of_view", 
                    "avatar_id": camera_config["id"], 
                    "field_of_view": vertical_fov
                })
        commands.append({"$type": "terminate"})
        self.communicate(commands)
        print(f"Scene setup complete. Created {len(self.camera_configs)} cameras.")
        print(f"All cameras: {self.screen_width}x{self.screen_height} with {self.HORIZONTAL_FOV}° horizontal FOV")

    def capture_all_perspectives(self):
        """Capture images from all cameras simultaneously."""
        print("Capturing images from all cameras...")
        
        # 直接捕获图像，不添加任何3D标记
        self.communicate({"$type": "terminate"})
        
        print(f"Images captured from {len(self.camera_configs)} cameras.")

    def add_camera_labels_to_top_down_image(self):
        """在俯视图上添加摄像机编号标签和视野标记"""
        try:
            # 直接使用固定的俯视图文件路径
            top_down_image_path = self.output_directory / "top_down_camera" / "img_0000.png"
            
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
            
            # 视野相关参数
            fov_length_pixels = int(1.5 * pixel_per_unit)  # FOV线长度（像素）
            
            # 添加摄像机编号标签和视野标记
            for camera_config in self.camera_configs:
                if camera_config["id"] == "top_down_camera":
                    continue
                    
                cam_pos = camera_config["position"]
                look_at = camera_config["look_at"]
                camera_number = camera_config["id"].split("_")[-1]  # 提取数字部分
                
                # 将3D坐标转换为2D图片像素坐标
                pixel_x = center_x + int(cam_pos["x"] * pixel_per_unit)
                pixel_y = center_y - int(cam_pos["z"] * pixel_per_unit)  # 注意Y轴翻转
                
                # 绘制摄像机位置标记（蓝色圆点）
                marker_radius = 8
                draw.ellipse([pixel_x - marker_radius, pixel_y - marker_radius, 
                             pixel_x + marker_radius, pixel_y + marker_radius], 
                             fill=(0, 0, 255, 255))
                
                # 计算视野方向
                dx = look_at["x"] - cam_pos["x"]
                dz = look_at["z"] - cam_pos["z"]
                center_angle_rad = math.atan2(dz, dx)
                
                # 绘制视野扇形区域
                half_fov_rad = math.radians(self.HORIZONTAL_FOV / 2.0)
                
                # 计算视野边界线的端点
                left_angle = center_angle_rad - half_fov_rad
                right_angle = center_angle_rad + half_fov_rad
                
                left_end_x = pixel_x + fov_length_pixels * math.cos(left_angle)
                left_end_y = pixel_y - fov_length_pixels * math.sin(left_angle)  # Y轴翻转
                
                right_end_x = pixel_x + fov_length_pixels * math.cos(right_angle)
                right_end_y = pixel_y - fov_length_pixels * math.sin(right_angle)  # Y轴翻转
                
                # 绘制视野边界线（红色）
                draw.line([pixel_x, pixel_y, left_end_x, left_end_y], fill=(255, 0, 0, 255), width=2)
                draw.line([pixel_x, pixel_y, right_end_x, right_end_y], fill=(255, 0, 0, 255), width=2)
                
                # # 绘制视野中心线（较细的红色线）
                # center_end_x = pixel_x + fov_length_pixels * math.cos(center_angle_rad)
                # center_end_y = pixel_y - fov_length_pixels * math.sin(center_angle_rad)  # Y轴翻转
                # draw.line([pixel_x, pixel_y, center_end_x, center_end_y], fill=(255, 100, 100, 255), width=1)
                
                # 计算编号文本在扇形内部的位置
                text = camera_number
                text_distance = fov_length_pixels * 0.3  # 文本位置在扇形长度的60%处
                text_pos_x = pixel_x + text_distance * math.cos(center_angle_rad)
                text_pos_y = pixel_y - text_distance * math.sin(center_angle_rad)  # Y轴翻转
                
                # 计算文本位置（居中）
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                text_x = text_pos_x - text_width // 2
                text_y = text_pos_y - text_height // 2
                
                # 直接绘制红色文本
                draw.text((text_x, text_y), text, fill=(255, 0, 0, 255), font=font)
            
            # 保存修改后的图片
            labeled_image_path = top_down_image_path.parent / f"labeled_{top_down_image_path.name}"
            img.save(labeled_image_path)
            print(f"Labeled image with FOV visualization saved to: {labeled_image_path}")
            
        except Exception as e:
            print(f"Error adding labels to top-down image: {e}")

    def run(self):
        self.setup_scene()
        # 在场景终止后添加标签
        self.add_camera_labels_to_top_down_image()

if __name__ == "__main__":
    output_directory = r"D:\ComputerScience\Leetcode\spatial\images1"
    experiment = TDWExperiment(launch_build=True, output_path=output_directory, num_objects=5)
    experiment.run()
    print(f"Experiment finished. Check images in {experiment.output_directory.resolve()}")