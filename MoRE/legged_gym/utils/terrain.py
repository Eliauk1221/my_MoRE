# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
import random
from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from pydelatin import Delatin
import pyfqmr
from scipy.ndimage import binary_dilation


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        cfg.terrain_proportions = np.array(cfg.terrain_proportions) / np.sum(cfg.terrain_proportions)
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)

        self.stuck_mask = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16).astype(np.bool_)

        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            if hasattr(cfg, "max_difficulty"):
                self.curiculum(random=True, max_difficulty=cfg.max_difficulty)
            else:
                self.curiculum(random=True)
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            print("Converting heightmap to trimesh...")
            if cfg.hf2mesh_method == "grid":
                self.vertices, self.triangles, self.x_edge_mask, self.stair_pen_mask = convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                                self.cfg.horizontal_scale,
                                                                                                self.cfg.vertical_scale,
                                                                                                self.proportions,
                                                                                                self.cfg.slope_treshold)
                # add stuck mask
                structure = np.ones((7, 1))
                self.stuck_mask = binary_dilation(self.x_edge_mask, structure=structure)
                # down stair mask, concatenate with up stair mask
                self.stair_pen_mask[:-1, :] |= self.stair_pen_mask[1:, :]
                stair_y_start = round(self.proportions[-2] * self.tot_cols)
                up_stair_mask = self.x_edge_mask.copy()
                up_stair_mask[:, :stair_y_start] = False
                self.stair_pen_mask = np.stack([up_stair_mask, self.stair_pen_mask], axis=0)

                if self.cfg.simplify_grid:
                    mesh_simplifier = pyfqmr.Simplify()
                    mesh_simplifier.setMesh(self.vertices, self.triangles)
                    mesh_simplifier.simplify_mesh(target_count = int(0.05*self.triangles.shape[0]), aggressiveness=7, preserve_border=True, verbose=10)

                    self.vertices, self.triangles, normals = mesh_simplifier.getMesh()
                    self.vertices = self.vertices.astype(np.float32)
                    self.triangles = self.triangles.astype(np.uint32)
            else:
                assert cfg.hf2mesh_method == "fast", "Height field to mesh method must be grid or fast"
                self.vertices, self.triangles = convert_heightfield_to_trimesh_delatin(self.height_field_raw, self.cfg.horizontal_scale, self.cfg.vertical_scale, max_error=cfg.max_error)
            print("Created {} vertices".format(self.vertices.shape[0]))
            print("Created {} triangles".format(self.triangles.shape[0]))

    def curiculum(self, random=False, max_difficulty=False):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / (self.cfg.num_rows-1)
                choice = j / self.cfg.num_cols + 0.001
                if random:
                    if max_difficulty:
                        difficulty_level = self.cfg.difficulty_level
                        terrain = self.make_terrain(choice, difficulty_level)
                    else:
                        terrain = self.make_terrain(choice, np.random.uniform(0, 1))
                else:
                    terrain = self.make_terrain(choice, difficulty)

                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.length_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def add_roughness(self, terrain, difficulty=1):
        max_height = (self.cfg.height[1] - self.cfg.height[0]) * difficulty + self.cfg.height[0]
        height = random.uniform(self.cfg.height[0], max_height)
        terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg.downsampled_scale)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.length_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        if choice < self.proportions[0]:
            # stepping_stones 地形（从 extreme-parkour 迁移）
            idx = 0
            stepping_stones_terrain(terrain, 
                                    stone_size=0.72 - 0.06*difficulty, 
                                    pitch_x=0.80,
                                    lane_offset=0.05,
                                    pad_height=0)
            # 不添加表面粗糙度，依赖域随机化提供鲁棒性
        elif choice < self.proportions[1]:
            # parkour 地形（从 extreme-parkour 迁移）
            idx = 1
            x_range = [0.08, 0.16 + 0.14*difficulty]
            y_range = [0.1, 0.14 + 0.06*difficulty]
            stone_len = 0.8
            incline_height = 0.25*difficulty
            last_incline_height = incline_height + 0.1 - 0.1*difficulty
            parkour_terrain(terrain,
                            num_stones=8,
                            x_range=x_range, 
                            y_range=y_range,
                            incline_height=incline_height,
                            stone_len=stone_len,
                            stone_width=1.0, 
                            last_incline_height=last_incline_height,
                            pad_height=0,
                            pit_depth=[0.2, 1])
            # 不添加表面粗糙度，依赖域随机化提供鲁棒性
        elif choice < self.proportions[2]:
            idx = 2
            parkour_pit_terrain(terrain,
                            platform_len=2.5, 
                            platform_height=0., 
                            num_stones=4,
                            x_range=[0.5, 1.2],
                            y_range=[-0.01, 0.01],
                            half_valid_width=1.5,
                            step_height = 0.05 + 0.25 * difficulty,
                            pad_width=0.1,
                            pad_height=0,
                            single_pit_len=1)
        elif choice < self.proportions[3]:
            idx = 3
            parkour_gap_terrain(terrain,
                                platform_len=2.5, 
                                platform_height=0, 
                                num_gaps=7,
                                gap_size=0.05 + 0.4 * difficulty,   
                                x_range=[0.75, 2], # platform x length
                                y_range=[-0.1, 0.1],
                                half_valid_width=1.5,
                                gap_depth=[0.2, 0.4],
                                pad_width=0.1,
                                pad_height=0,
                                flat=False)
        elif choice < self.proportions[4]:
            idx = 4
            parkour_stair_terrain(terrain,
                platform_len=1.5, 
                platform_height=0., 
                num_stones=10,
                x_range=0.31,
                y_range=[-0.01, 0.01],
                half_valid_width=1.5,
                step_height = 0.10 + 0.15 * difficulty,
                pad_width=0.1,
                pad_height=0,
                num_groups=3,
                middle_platform_len=1.5)
        # square terrain
        elif choice < self.proportions[5]:
            idx = 5
            step_height = 0.15 + 0.10 * difficulty
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=-step_height, platform_size=3.)
        elif choice < self.proportions[6]:
            idx = 6
            gap_terrain(terrain, gap_size=0.05 + 0.4 * difficulty, platform_size=3.)
        elif choice < self.proportions[7]:
            idx = 7
            pit_terrain(terrain, depth=0.05 + 0.25 * difficulty, platform_size=4.)

        terrain.idx = idx
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        if terrain.idx in [5, 6, 7]:
            env_origin_x = (i + 0.5) * self.env_length
        else:
            env_origin_x = i * self.env_length + 1.0
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 0.5) / terrain.horizontal_scale) # within 1 meter square range
        x2 = int((self.env_length/2. + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 0.5) / terrain.horizontal_scale)
        
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.terrain_type[i, j] = terrain.idx
        # self.goals[i, j, :, :2] = terrain.goals + [i * self.env_length, j * self.env_width]
        # self.env_slope_vec[i, j] = terrain.slope_vector

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0
    
def parkour_gap_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_gaps=8,
                           gap_size=0.3,
                           x_range=[1.6, 2.4],
                           y_range=[-1.2, 1.2],
                           half_valid_width=1,
                           gap_depth=-200,
                           pad_width=0.1,
                           pad_height=0.5,
                           flat=False):
    mid_y = terrain.length // 2  # length is actually y width

    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    gap_depth = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale)
    
    half_valid_width = round(half_valid_width / terrain.horizontal_scale)

    terrain.height_field_raw[0:platform_len, :] = platform_height

    gap_size = round(gap_size / terrain.horizontal_scale)
    dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size
    dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size

    dis_x = platform_len
    last_dis_x = dis_x
    for i in range(num_gaps):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if not flat:
            terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, :] = gap_depth

        terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = gap_depth
        terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = gap_depth
        
        last_dis_x = dis_x
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def parkour_stair_terrain(terrain,
                        platform_len=2.5,
                        platform_height=0.,
                        num_stones=10,
                        x_range=0.13,
                        y_range=[-0.15, 0.15],
                        half_valid_width=1,
                        step_height=0.15,
                        pad_width=0.1,
                        pad_height=0.5,
                        num_groups=2,
                        middle_platform_len=3):
    mid_y = terrain.length // 2
    dis_x_min = round((x_range)/ terrain.horizontal_scale)

    step_height = round(step_height / terrain.vertical_scale)
    half_valid_width = round(half_valid_width / terrain.horizontal_scale)
    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    middle_platform_len = round(middle_platform_len / terrain.horizontal_scale)

    terrain.height_field_raw[0:platform_len, :] = platform_height

    dis_x = platform_len
    last_dis_x = dis_x

    for group in range(num_groups):
        stair_height = 0
        
        for i in range(num_stones // 2): 
            rand_x = dis_x_min
            rand_y = 0
            stair_height += step_height
            
            terrain.height_field_raw[dis_x:dis_x + rand_x, :] = stair_height
            dis_x += rand_x
            terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = 0
            terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = 0
            last_dis_x = dis_x
        
        terrain.height_field_raw[dis_x:dis_x+middle_platform_len, mid_y+rand_y-half_valid_width:mid_y+rand_y+half_valid_width] = stair_height
        dis_x += middle_platform_len
        last_dis_x = dis_x
        
        for i in range(num_stones // 2, num_stones):
            rand_x = dis_x_min
            rand_y = 0
            stair_height -= step_height
            
            terrain.height_field_raw[dis_x:dis_x + rand_x, :] = stair_height
            dis_x += rand_x
            terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = 0
            terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = 0
            last_dis_x = dis_x

        if group < num_groups - 1:
            interval = dis_x_min * 10
            dis_x += interval
            last_dis_x = dis_x

    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def parkour_pit_terrain(terrain,
                        platform_len=2.5,
                        platform_height=0.,
                        num_stones=8,
                        x_range=[0.2, 0.4],
                        y_range=[-0.15, 0.15],
                        half_valid_width=1,
                        step_height=0.2,
                        pad_width=0.1,
                        pad_height=0.5,
                        single_pit_len=1):
    mid_y = terrain.length // 2

    dis_x_min = round((x_range[0] + step_height) / terrain.horizontal_scale)
    dis_x_max = round((x_range[1] + step_height) / terrain.horizontal_scale)
    step_height = round(step_height / terrain.vertical_scale)
    half_valid_width = round(half_valid_width / terrain.horizontal_scale)
    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    single_pit_len = round(single_pit_len / terrain.horizontal_scale)

    terrain.height_field_raw[0:platform_len, :] = platform_height

    dis_x = platform_len
    last_dis_x = dis_x

    stair_height = 0

    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = 0

        if i < num_stones // 2:
            stair_height += step_height
        elif i > num_stones // 2:
            stair_height -= step_height

        terrain.height_field_raw[dis_x:dis_x + rand_x, :] = stair_height
        dis_x += rand_x

        terrain.height_field_raw[last_dis_x:dis_x, :mid_y + rand_y - half_valid_width] = 0
        terrain.height_field_raw[last_dis_x:dis_x, mid_y + rand_y + half_valid_width:] = 0

        last_dis_x = dis_x

    interval = dis_x_max * 2
    dis_x += interval
    last_dis_x = dis_x

    single_step_x = dis_x_max
    single_step_height = stair_height
    terrain.height_field_raw[dis_x:dis_x + single_step_x, :] = single_step_height
    dis_x += single_step_x
    terrain.height_field_raw[last_dis_x:dis_x, :mid_y - half_valid_width] = 0
    terrain.height_field_raw[last_dis_x:dis_x, mid_y + half_valid_width:] = 0
    last_dis_x = dis_x

    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def stepping_stones_terrain(terrain,
                            stone_size, 
                            pitch_x=0.80,
                            lane_offset=0.05,
                            num_cols=10,
                            platform_len=2.5,
                            platform_height=0.,
                            pit_depth=1.0,
                            pad_width=0.1,
                            pad_height=0.5):
    """
    生成梅花桩地形：每个 x 位置只放一个石块，左右交替。
    
    Args:
        terrain: 地形对象
        stone_size: 石块边长 (米)
        pitch_x: x 方向节距 (米)，即相邻石块中心间距
        lane_offset: 石块相对中轴线的左右偏移量 (米)
        num_cols: 石块列数 (x 方向)
        platform_len: 起始/终点平台长度 (米)
        platform_height: 平台高度 (米)
        pit_depth: 深坑深度 (米)
        pad_width: 边缘填充宽度 (米)
        pad_height: 边缘填充高度 (米)
    """
    terrain.height_field_raw[:] = -round(pit_depth / terrain.vertical_scale)
    
    stone_size_px = round(stone_size / terrain.horizontal_scale)
    pitch_x_px = round(pitch_x / terrain.horizontal_scale)
    lane_offset_px = round(lane_offset / terrain.horizontal_scale)
    platform_len_px = round(platform_len / terrain.horizontal_scale)
    platform_height_px = round(platform_height / terrain.vertical_scale)
    pad_width_px = int(pad_width // terrain.horizontal_scale)
    pad_height_px = int(pad_height // terrain.vertical_scale)

    mid_y = terrain.length // 2

    terrain.height_field_raw[0:platform_len_px, :] = platform_height_px

    stone_region_start = platform_len_px

    for col in range(num_cols):
        center_x = stone_region_start + col * pitch_x_px + stone_size_px // 2

        if col % 2 == 0:
            center_y = mid_y - lane_offset_px
        else:
            center_y = mid_y + lane_offset_px

        x_start = center_x - stone_size_px // 2
        x_end = x_start + stone_size_px
        y_start = center_y - stone_size_px // 2
        y_end = y_start + stone_size_px

        terrain.height_field_raw[x_start:x_end, y_start:y_end] = platform_height_px

    stone_region_end = stone_region_start + num_cols * pitch_x_px
    terrain.height_field_raw[stone_region_end:, :] = platform_height_px

    terrain.height_field_raw[:, :pad_width_px] = pad_height_px
    terrain.height_field_raw[:, -pad_width_px:] = pad_height_px
    terrain.height_field_raw[:pad_width_px, :] = pad_height_px
    terrain.height_field_raw[-pad_width_px:, :] = pad_height_px

def parkour_terrain(terrain, 
                    platform_len=2.5, 
                    platform_height=0., 
                    num_stones=8, 
                    x_range=[1.8, 1.9], 
                    y_range=[0., 0.1], 
                    z_range=[-0.2, 0.2],
                    stone_len=1.0,
                    stone_width=0.6,
                    pad_width=0.1,
                    pad_height=0.5,
                    incline_height=0.1,
                    last_incline_height=0.6,
                    last_stone_len=1.6,
                    pit_depth=[0.5, 1.]):
    """
    生成 parkour 地形（从 extreme-parkour 迁移，移除 goals 设置）
    左右交替的倾斜平台跳跃
    
    Args:
        terrain: 地形对象
        platform_len: 起始平台长度 (米)
        platform_height: 平台高度 (米)
        num_stones: 石块数量
        x_range: x 方向间距范围 [min, max] (米)
        y_range: y 方向偏移范围 [min, max] (米)
        z_range: z 方向高度范围 [min, max] (米)
        stone_len: 石块长度或范围 (米)
        stone_width: 石块宽度 (米)
        pad_width: 边缘填充宽度 (米)
        pad_height: 边缘填充高度 (米)
        incline_height: 倾斜高度 (米)
        last_incline_height: 最后石块倾斜高度 (米)
        last_stone_len: 最后石块长度 (米)
        pit_depth: 深坑深度范围 [min, max] (米)
    """
    # 填充深坑
    terrain.height_field_raw[:] = -round(np.random.uniform(pit_depth[0], pit_depth[1]) / terrain.vertical_scale)
    
    mid_y = terrain.length // 2  # length is actually y width
    
    # 处理 stone_len 参数（可能是单值或范围）
    if isinstance(stone_len, (list, tuple)):
        stone_len = np.random.uniform(*stone_len)
    stone_len = 2 * round(stone_len / 2.0, 1)
    stone_len = round(stone_len / terrain.horizontal_scale)
    
    dis_x_min = stone_len + round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = stone_len + round(x_range[1] / terrain.horizontal_scale)
    dis_x_max = max(dis_x_max, dis_x_min + 1)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)
    dis_y_max = max(dis_y_max, dis_y_min + 1)
    dis_z_min = round(z_range[0] / terrain.vertical_scale)
    dis_z_max = round(z_range[1] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_width = round(stone_width / terrain.horizontal_scale)
    last_stone_len = round(last_stone_len / terrain.horizontal_scale)

    incline_height = round(incline_height / terrain.vertical_scale)
    last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len - stone_len // 2
    left_right_flag = np.random.randint(0, 2)
    dis_z = 0
    
    for i in range(num_stones):
        dis_x += np.random.randint(dis_x_min, dis_x_max)
        pos_neg = round(2*(left_right_flag - 0.5))
        dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
        if i == num_stones - 1:
            dis_x += last_stone_len // 4
            heights = np.tile(np.linspace(-last_incline_height, last_incline_height, stone_width), (last_stone_len, 1)) * pos_neg
            terrain.height_field_raw[dis_x-last_stone_len//2:dis_x+last_stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
        else:
            heights = np.tile(np.linspace(-incline_height, incline_height, stone_width), (stone_len, 1)) * pos_neg
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z

        left_right_flag = 1 - left_right_flag
    
    gap_x_min = round(x_range[0] / terrain.horizontal_scale)
    gap_x_max = max(round(x_range[1] / terrain.horizontal_scale), gap_x_min + 1)
    final_platform_start = dis_x + last_stone_len // 2 + np.random.randint(gap_x_min, gap_x_max)
    terrain.height_field_raw[final_platform_start:, :] = platform_height
    
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def multi_sloped_long_terrain(terrain, slope=1, platform_size=1.0, num_slopes=2, slope_size=4.0):
    """
    Generate multiple pyramid-shaped slopes along the x-axis.

    Parameters:
        terrain: terrain object
        slope: slope factor (positive or negative)
        platform_size: flat space between slopes [m]
        num_slopes: number of slopes along x-axis
        slope_size: width/length of each slope (square) [m]
    """
    slope_cells = int(slope_size / terrain.horizontal_scale)
    platform_cells = int(platform_size / terrain.horizontal_scale)
    length_cells = terrain.length  # y方向格子数
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (slope_cells / 2))

    x_pos = 0
    for i in range(num_slopes):
        # 局部坐标 (从 0 到 slope_cells)
        x = np.arange(0, slope_cells)
        y = np.arange(0, length_cells)
        xx, yy = np.meshgrid(x, y, sparse=True)

        # 中心点在局部
        center_x = slope_cells / 2
        center_y = length_cells / 2

        xx = (center_x - np.abs(center_x - xx)) / center_x
        yy = (center_y - np.abs(center_y - yy)) / center_y

        xx = np.clip(xx, 0, 1).reshape(slope_cells, 1)
        yy = np.clip(yy, 0, 1).reshape(1, length_cells)

        # 写入对应位置（覆盖）
        terrain.height_field_raw[x_pos:x_pos + slope_cells, :] = \
            (max_height * xx * yy).astype(terrain.height_field_raw.dtype)

        # 平台间隔（最后一个坡后面不加）
        x_pos += slope_cells
        if i < num_slopes - 1:
            x_pos += platform_cells * 2

    return terrain

def convert_heightfield_to_trimesh_delatin(height_field_raw, horizontal_scale, vertical_scale, max_error=0.01):
    mesh = Delatin(np.flip(height_field_raw, axis=1).T, z_scale=vertical_scale, max_error=max_error)
    vertices = np.zeros_like(mesh.vertices)
    vertices[:, :2] = mesh.vertices[:, :2] * horizontal_scale
    vertices[:, 2] = mesh.vertices[:, 2]
    return vertices, mesh.triangles

def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, proportions, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols)); my_move_x = np.zeros((num_rows, num_cols)); stair_down_move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))

        # except stair down edge mask
        stair_y_start = round(proportions[-2] * num_cols)
        my_move_x[1:num_rows, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        my_move_x[:num_rows-1, :stair_y_start] -= (hf[:num_rows-1, :stair_y_start] - hf[1:num_rows, :stair_y_start] > slope_threshold)
        stair_down_move_x[:num_rows-1, stair_y_start:] -= (hf[:num_rows-1, stair_y_start:] - hf[1:num_rows, stair_y_start:] > slope_threshold)

        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles, my_move_x != 0, stair_down_move_x != 0