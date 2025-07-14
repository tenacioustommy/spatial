import math
import random
import json
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.object_manager import ObjectManager
from pathlib import Path
import os
from PIL import Image, ImageDraw, ImageFont
from tdw.librarian import ModelLibrarian
import numpy as np
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
       
        # 定义椅子列表和大物体列表
        self.chair_list = [
            'blue_club_chair', 'blue_side_chair', 'brown_leather_dining_chair',
            'brown_leather_side_chair', 'chair_annabelle', 'chair_billiani_doll',
            'chair_eames_plastic_armchair', 'chair_thonet_marshall', 'chair_willisau_riale',
            'dark_red_club_chair', 'emeco_navy_chair', 'green_side_chair',
            'lapalma_stil_chair', 'ligne_roset_armchair', 'linbrazil_diz_armchair',
            'linen_dining_chair', 'naughtone_pinch_stool_chair', 'red_side_chair',
            'tan_lounger_chair', 'tan_side_chair', 'vitra_meda_chair',
            'white_club_chair', 'white_lounger_chair', 'wood_chair', 'yellow_side_chair'
        ]
        # "cabinet_24_wood_beach_honey","cabinet_36_white_wood",
        self.large_object_list = [
            "fridge_large", 
            # "dining_room_table", "cabinet_24_wood_beach_honey", "cabinet_36_white_wood",
	        # "5ft_wood_shelving", "metal_lab_shelf"
        ]
        
        self.selected_objects = self._select_random_objects()

        # 定义多个摄像机的位置和角度，从12点方向开始顺时针均匀分配16个摄像机
        self.camera_configs = []
        num_cameras = 16
        radius = 4.0  # 摄像机距离中心的半径
        camera_height = 1.5
        
        # 定义四个基本朝向的相对方向向量
        direction_vectors = [
            {"x": 0, "z": 1},   # 朝北(前方，+z方向)
            {"x": 1, "z": 0},   # 朝东(右方，+x方向)
            {"x": 0, "z": -1},  # 朝南(后方，-z方向)
            {"x": -1, "z": 0}   # 朝西(左方，-x方向)
        ]
        
        for i in range(num_cameras):
            # 从12点方向(0度)开始，顺时针分配
            angle_degrees = i * (360 / num_cameras)  # 每个摄像机间隔22.5度
            angle_radians = math.radians(angle_degrees)
            
            # 计算摄像机位置（12点方向为z轴正方向）
            x = radius * math.sin(angle_radians)
            z = radius * math.cos(angle_radians)
            
            # 随机选择一个朝向方向
            direction = random.choice(direction_vectors)
            
            # 基于摄像机位置和选择的方向，计算朝向的绝对坐标点
            look_distance = 5.0  # 朝向点距离摄像机的距离
            look_at_x = x + direction["x"] * look_distance
            look_at_z = z + direction["z"] * look_distance
            
            camera_config = {
                "id": f"camera_{i+1}",
                "position": {"x": x, "y": camera_height, "z": z},
                "look_at": {"x": look_at_x, "y": camera_height, "z": look_at_z}  # 基于位置计算的绝对坐标
            }
            self.camera_configs.append(camera_config)
        
        # 添加四个中心摄像机，分别朝向前后左右四个方向（使用绝对坐标）
        center_cameras = [
            {
                "id": "center_north",
                "position": {"x": 0, "y": camera_height, "z": 0},
                "look_at": {"x": 0, "y": camera_height, "z": 5}   # 朝北(前方，绝对坐标)
            },
            {
                "id": "center_east", 
                "position": {"x": -1, "y": camera_height, "z": 0},
                "look_at": {"x": 5, "y": camera_height, "z": 0}   # 朝东(右方，绝对坐标)
            },
            {
                "id": "center_south",
                "position": {"x": 0, "y": camera_height, "z": 0},
                "look_at": {"x": 0, "y": camera_height, "z": -5}  # 朝南(后方，绝对坐标)
            },
            {
                "id": "center_west",
                "position": {"x": 0, "y": camera_height, "z": 0},
                "look_at": {"x": -5, "y": camera_height, "z": 0}  # 朝西(左方，绝对坐标)
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
        self.big_object_ids = [self.object_ids[0]]  # 第一个物体是大物体
        self.HORIZONTAL_FOV = 90.0  # 目标水平视野角度
        self.screen_width = 1024
        self.screen_height = 1024

        self.current_main_camera_position = {"x": 0, "y": 0, "z": 0}

    def _select_random_objects(self):
        """从椅子列表和大物体列表中随机选择指定数量的物体"""
        selected_objects = []
        
        # 首先选择一个大物体作为第一个物体
        available_large_objects = []
        for record in self.librarian.records:
            if record.name in self.large_object_list:
                available_large_objects.append(record)
        print("large object",available_large_objects)
        # 选择一个大物体
        selected_objects= random.sample(available_large_objects,1)
        
        # 然后选择剩余的椅子
        available_chairs = []
        for record in self.librarian.records:
            if record.name in self.chair_list:
                available_chairs.append(record)
        
        selected_chairs = random.sample(available_chairs, self.num_objects - 1)
        selected_objects.extend(selected_chairs)
        
        print(f"Selected objects: {[obj.name for obj in selected_objects]}")
        return selected_objects

    def _generate_random_positions(self, num_objects=None, room_size=12, min_distance=2.0):
        """生成固定位置，确保物体之间不重合，大物体放在中心附近"""
        if num_objects is None:
            num_objects = self.num_objects
            
        positions = []
        
        # 第一个物体（大物体）放在中心附近
        big_object_position = {"x": 0.5, "y": 0, "z": 0.5}  # 中心附近的固定位置
        positions.append(big_object_position)
        print(f"Generated position 1 (big object): {big_object_position}")
        
        # 其他物体使用预定义的固定位置
        predefined_positions = [
            {"x": -3.0, "y": 0, "z": -3.0},  # 左后
            {"x": 3.0, "y": 0, "z": -3.0},   # 右后
            {"x": -3.0, "y": 0, "z": 3.0},   # 左前
            {"x": 3.0, "y": 0, "z": 3.0},    # 右前
            {"x": -2.0, "y": 0, "z": 0},     # 左中
            {"x": 2.0, "y": 0, "z": 0},      # 右中
            {"x": 0, "y": 0, "z": -3.0},     # 后中
            {"x": 0, "y": 0, "z": 3.0},      # 前中
        ]
        
        # 添加剩余物体的位置
        for i in range(1, num_objects):
            if i-1 < len(predefined_positions):
                position = predefined_positions[i-1]
            else:
                # 如果预定义位置不够，使用后备位置
                fallback_x = ((i-1) % 4 - 1.5) * 2.5
                fallback_z = ((i-1) // 4 - 1.5) * 2.5
                position = {"x": fallback_x, "y": 0, "z": fallback_z}
            
            positions.append(position)
            print(f"Generated position {i+1}: {position}")
        
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
      
        # 定义允许的旋转角度：0, 90, 180, 270 度
        allowed_rotations = [0, 90, 180, 270]
        
        for i, (obj_record, position, obj_id) in enumerate(zip(self.selected_objects, positions, self.object_ids)):
            # 随机选择一个允许的旋转角度
            rotation_y = random.choice(allowed_rotations)
            rotation = {"x": 0, "y": rotation_y, "z": 0}
            
            # 设置物体缩放 - 第一个物体（大物体）保持原始大小，其他物体仅Y轴放大一倍
            

            commands.append(self.get_add_object(
                model_name=obj_record.name,
                position=position,
                object_id=obj_id,
                rotation=rotation,
                library="models_core.json"
            ))
            if i == 0:  # 大物体
                commands.append({
                    "$type": "scale_object",
                    "id": obj_id,
                    "scale_factor": {"x": 1.0, "y": 2.0, "z": 1.0}
                })
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
            
    def extract_coordinate_scalar(self, coord_array, axis_index):
            """从三维坐标数组中提取指定轴的标量值"""
            if hasattr(coord_array, '__getitem__'):
                coord_val = coord_array[axis_index]
                if hasattr(coord_val, 'item'):
                    return coord_val.item()
                else:
                    return float(coord_val)
            else:
                return float(coord_array)
            
    def can_camera_see_object(self, camera_position, camera_look_at, target_object_id):
        """
        检测摄像机在指定位置和朝向是否可以看到目标物体
        要求物体完全在视野内且无遮挡才算看见
        
        Args:
            camera_position: 摄像机位置 {"x": float, "y": float, "z": float}
            camera_look_at: 摄像机朝向的目标点 {"x": float, "y": float, "z": float}
            target_object_id: 目标物体ID
            
        Returns:
            dict: 包含可见性、图片路径和物体朝向的信息
                {
                    "can_see": bool,
                    "image_path": str or None,
                    "object_direction": str or None  # "front", "back", "left", "right"
                }
        """
        result = {
            "can_see": False,
            "image_path": None,
            "object_direction": None
        }
        
        # 获取目标物体的位置和边界
        if target_object_id not in self.object_manager.objects_static:
            print(f"Object {target_object_id} not found")
            return result
            
        target_transform = self.object_manager.transforms[target_object_id]
        target_bounds = self.object_manager.bounds[target_object_id]
        target_position = target_transform.position
        
        # 计算摄像机的朝向角度
        cam_dx = camera_look_at["x"] - camera_position["x"]
        cam_dz = camera_look_at["z"] - camera_position["z"]
        camera_angle = math.degrees(math.atan2(cam_dx, cam_dz))
        if camera_angle < 0:
            camera_angle += 360
        
        # 边界框的每个面都是三维坐标点，从中提取对应轴的坐标值
        # 找到所有边界点中各轴的最值
        all_boundary_points = [
            target_bounds.left, target_bounds.right, target_bounds.front, 
            target_bounds.back, target_bounds.top, target_bounds.bottom
        ]
        
        # 提取所有边界点的x坐标，找最小值和最大值
        x_coords = [self.extract_coordinate_scalar(point, 0) for point in all_boundary_points]
        min_x = min(x_coords)
        max_x = max(x_coords)
        
        # 提取所有边界点的z坐标，找最小值和最大值  
        z_coords = [self.extract_coordinate_scalar(point, 2) for point in all_boundary_points]
        min_z = min(z_coords)
        max_z = max(z_coords)
        
        half_width = (max_x - min_x) / 2
        half_depth = (max_z - min_z) / 2
        
        # target_position 是三维坐标数组 [x, y, z]
        target_x = target_bounds.center[0]  # x 坐标
        target_z = target_bounds.center[2]  # z 坐标
        
        corner_points = [
            [target_x - half_width, target_z - half_depth],  # 左后
            [target_x + half_width, target_z - half_depth],  # 右后
            [target_x - half_width, target_z + half_depth],  # 左前
            [target_x + half_width, target_z + half_depth],  # 右前
        ]
        
        # 检查所有角点是否都在视野内
        for corner_x, corner_z in corner_points:
            # 计算角点相对于摄像机的角度
            corner_dx = corner_x - camera_position["x"]
            corner_dz = corner_z - camera_position["z"]
            corner_angle = math.degrees(math.atan2(corner_dx, corner_dz))
            if corner_angle < 0:
                corner_angle += 360
            
            # 计算角度差
            angle_diff = abs(corner_angle - camera_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
                
            # 如果任何一个角点超出视野范围，则物体不完全可见
            if angle_diff > self.HORIZONTAL_FOV / 2:
                return result
        
        # 检查是否被大物体遮挡 - 需要检查到所有角点的射线（仅考虑x,z平面）
        for big_object_id in self.big_object_ids:
            if big_object_id == target_object_id:
                continue  # 跳过目标物体本身
                
            if big_object_id not in self.object_manager.objects_static:
                continue
                
            # 获取大物体的位置和边界
            big_transform = self.object_manager.transforms[big_object_id]
            big_bounds = self.object_manager.bounds[big_object_id]
            
            # 检查到所有角点的射线是否被遮挡（仅在x,z平面上）
            blocked_corners = 0
            for corner_x, corner_z in corner_points:
                # 仅使用x,z坐标进行2D遮挡检测
                corner_2d = [corner_x, corner_z]
                
                # 检查射线是否与大物体的边界框相交（仅x,z平面）
                if self._ray_intersects_box_2d(camera_position, corner_2d, big_bounds):
                    # 进一步检查遮挡物是否真的在摄像机和角点之间
                    if self._is_object_between_camera_and_point_2d(camera_position, corner_2d, big_transform.position):
                        blocked_corners += 1
            
            # 如果有任何角点被遮挡，则物体不完全可见
            if blocked_corners > 0:
                print(f"Object {target_object_id} is partially blocked by big object {big_object_id} ({blocked_corners}/{len(corner_points)} corners blocked)")
                return result
        
        # 如果没有遮挡，物体可见
        result["can_see"] = True
        
        # 创建临时摄像机并拍照
        # result["image_path"] = self._create_temp_camera_and_capture(camera_position, camera_look_at, target_object_id)
        
        # 计算目标物体在当前视角下的朝向
        result["object_direction"] = self._calculate_object_direction(camera_position, camera_look_at, target_transform)
        
        return result
    
    def _create_temp_camera_and_capture(self, camera_position, camera_look_at, target_object_id):
        """
        创建临时摄像机并拍照
        
        Args:
            camera_position: 摄像机位置
            camera_look_at: 摄像机朝向点
            target_object_id: 目标物体ID
            
        Returns:
            str: 图片保存路径
        """
        # 生成唯一的临时摄像机ID
        temp_camera_id = f"temp_camera_{target_object_id}_{random.randint(1000, 9999)}"
        
        # 创建临时摄像机
        commands = []
        commands.extend(TDWUtils.create_avatar(
            avatar_type="A_Img_Caps_Kinematic",
            avatar_id=temp_camera_id,
            position=camera_position,
            look_at=camera_look_at
        ))
        
        # 设置FOV
        vertical_fov = self.calculate_vertical_fov(self.HORIZONTAL_FOV, self.screen_width, self.screen_height)
        commands.append({
            "$type": "set_field_of_view", 
            "avatar_id": temp_camera_id, 
            "field_of_view": vertical_fov
        })
        
        # 添加临时图像捕获
        temp_output_dir = self.output_directory / "temp_captures"
        temp_output_dir.mkdir(exist_ok=True)
        
        temp_image_capture = ImageCapture(
            path=temp_output_dir,
            avatar_ids=[temp_camera_id],
            pass_masks=["_img"],
            png=True
        )
        
        # 暂时添加临时图像捕获组件
        self.add_ons.append(temp_image_capture)
        
        # 执行命令
        commands.append({"$type": "terminate"})
        self.communicate(commands)
        
        # 返回图片路径
        image_path = temp_output_dir / temp_camera_id / "img_0000.png"
        
        # 重命名图片为更有意义的名称
        final_image_name = f"view_object_{target_object_id}_{temp_camera_id}.png"
        final_image_path = temp_output_dir / final_image_name
        
        if image_path.exists():
            image_path.rename(final_image_path)
            # 删除临时目录
            import shutil
            shutil.rmtree(temp_output_dir / temp_camera_id, ignore_errors=True)
            return str(final_image_path)
        else:
            print(f"Warning: Image not found at {image_path}")
            return None

    def _calculate_object_direction(self, camera_position, camera_look_at, target_transform):
        """
        计算目标物体在当前视角下的朝向（前后左右）
        
        Args:
            camera_position: 摄像机位置
            camera_look_at: 摄像机朝向点
            target_transform: 目标物体的变换信息
            
        Returns:
            str: "front", "back", "left", "right"
        """
        # 计算摄像机朝向向量
        cam_dx = camera_look_at["x"] - camera_position["x"]
        cam_dz = camera_look_at["z"] - camera_position["z"]
        cam_forward = np.array([cam_dx, cam_dz])
        cam_forward = cam_forward / np.linalg.norm(cam_forward)  # 归一化
        
        # 获取物体的朝向向量（仅考虑x,z平面）
        obj_forward = target_transform.forward
        obj_forward_2d = np.array([obj_forward[0], obj_forward[2]])
        obj_forward_2d = obj_forward_2d / np.linalg.norm(obj_forward_2d)  # 归一化
        
        # 计算相对角度
        dot_product = np.dot(cam_forward, obj_forward_2d)
        cross_product = cam_forward[0] * obj_forward_2d[1] - cam_forward[1] * obj_forward_2d[0]
        
        # 使用点积和叉积确定相对方向
        angle = math.atan2(cross_product, dot_product)
        angle_degrees = math.degrees(angle)
        
        # 将角度映射到四个基本方向
        if -45 <= angle_degrees <= 45:
            return "front"  # 物体朝向与相机朝向相同
        elif 45 < angle_degrees <= 135:
            return "left"   # 物体朝向相机的左侧
        elif 135 < angle_degrees or angle_degrees <= -135:
            return "back"   # 物体朝向与相机朝向相反
        else:  # -135 < angle_degrees < -45
            return "right"  # 物体朝向相机的右侧

    def _ray_intersects_box_2d(self, ray_start, ray_end_2d, box_bounds):
        """
        检测射线是否与边界框相交（仅考虑x,z平面的2D投影）
        
        Args:
            ray_start: 射线起点 {"x": float, "y": float, "z": float}
            ray_end_2d: 射线终点 [x, z] (仅2D)
            box_bounds: 边界框信息
            
        Returns:
            bool: True表示相交
        """
        # 计算边界框在x,z平面的最小和最大坐标
        all_boundary_points = [
            box_bounds.left, box_bounds.right, box_bounds.front,
            box_bounds.back, box_bounds.top, box_bounds.bottom
        ]
        
        # 找x,z轴的最值
        x_coords = [self.extract_coordinate_scalar(point, 0) for point in all_boundary_points]
        z_coords = [self.extract_coordinate_scalar(point, 2) for point in all_boundary_points]
        
        box_min_2d = [min(x_coords), min(z_coords)]
        box_max_2d = [max(x_coords), max(z_coords)]
        
        # 使用线段与2D AABB的相交测试
        return self._line_segment_intersects_aabb_2d(
            [ray_start["x"], ray_start["z"]], 
            ray_end_2d, 
            box_min_2d, 
            box_max_2d
        )
    
    def _line_segment_intersects_aabb_2d(self, start, end, box_min, box_max):
        """
        检测线段是否与轴对齐边界框(AABB)相交（仅2D x,z平面）
        """
        # 计算线段方向
        direction = [end[i] - start[i] for i in range(2)]  # 仅x,z两个维度
        
        t_min = 0.0
        t_max = 1.0
        
        for i in range(2):  # 仅检查x,z轴
            if abs(direction[i]) < 1e-8:  # 射线平行于某个轴
                if start[i] < box_min[i] or start[i] > box_max[i]:
                    return False
            else:
                t1 = (box_min[i] - start[i]) / direction[i]
                t2 = (box_max[i] - start[i]) / direction[i]
                
                if t1 > t2:
                    t1, t2 = t2, t1
                    
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                
                if t_min > t_max:
                    return False
                    
        return True
    
    def _is_object_between_camera_and_point_2d(self, camera_pos, target_point_2d, obstacle_pos):
        """
        检查障碍物是否在摄像机和目标点之间（仅考虑x,z平面）
        """
        # 计算摄像机到目标点的距离（仅x,z平面）
        cam_to_target_dist = math.sqrt(
            (target_point_2d[0] - camera_pos["x"])**2 + 
            (target_point_2d[1] - camera_pos["z"])**2
        )
        
        # 计算摄像机到障碍物的距离（仅x,z平面）
        obstacle_x = self.extract_coordinate_scalar(obstacle_pos, 0)  # x 坐标
        obstacle_z = self.extract_coordinate_scalar(obstacle_pos, 2)  # z 坐标
        cam_to_obstacle_dist = math.sqrt(
            (obstacle_x - camera_pos["x"])**2 + 
            (obstacle_z - camera_pos["z"])**2
        )
        
        # 障碍物必须在摄像机和目标之间才能造成遮挡
        return cam_to_obstacle_dist < cam_to_target_dist

    def run(self):
        self.setup_scene()
        # 在场景终止后添加标签
        self.add_camera_labels_to_top_down_image()
        
        # 示例调用can_camera_see_object函数
        if self.object_ids:
            # 使用第一个摄像机的配置
            camera_config = self.camera_configs[17]
            # camera_config = self.camera_configs[6]
            camera_pos = camera_config["position"]
            camera_look_at = camera_config["look_at"]
            
            # 测试第一个物体
            target_object_id = self.object_ids[4]
            print(f"Target object ID: {target_object_id}")
            
            result = self.can_camera_see_object(
                camera_position=camera_pos,
                camera_look_at=camera_look_at,
                target_object_id=target_object_id
            )
            
            print(f"Camera position: {camera_pos}")
            print(f"Camera look_at: {camera_look_at}")
            print("\nResult:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    output_directory = r"D:\ComputerScience\Leetcode\spatial\images"
    experiment = TDWExperiment(launch_build=True, output_path=output_directory, num_objects=5)
    experiment.run()
    print(f"Experiment finished. Check images in {experiment.output_directory.resolve()}")
