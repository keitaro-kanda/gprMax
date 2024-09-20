import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon, GeometryCollection, MultiPoint
import matplotlib.colors as mcolors

# 定数
C = 3e8  # 光速 (m/s)

# シミュレーション空間のサイズ
WIDTH, HEIGHT = 3, 5

# 媒質の定義
boundary_outer = Polygon([(0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)])
boundary_y2 = LineString([(0, 3), (WIDTH, 3)])
boundary_circle = Point(1.5, 1).buffer(0.15)  # 半径0.15の円

# 媒質を判定する関数
def get_refractive_index(point):
    p = Point(point)
    if boundary_circle.contains(p):
        return 9.0  # 媒質2の誘電率
    elif p.y <= 4:
        return 3.0  # 媒質1の誘電率
    elif boundary_outer.contains(p):
        return 1.0  # 空気
    else:
        return 1.0  # シミュレーション外（空気）

# 光線クラスの定義
class Ray:
    def __init__(self, origin, direction, current_refractive_index, total_time=0, depth=0):
        self.origin = np.array(origin, dtype=float)
        self.direction = self.normalize(direction)
        self.current_refractive_index = current_refractive_index
        self.total_time = total_time
        self.depth = depth  # 再帰の深さ

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def get_line(self, length=1000):
        return LineString([
            tuple(self.origin),
            tuple(self.origin + self.direction * length)
        ])

# スネルの法則による屈折の計算
def refract(direction, normal, n1, n2):
    direction = direction / np.linalg.norm(direction)
    normal = normal / np.linalg.norm(normal)
    cos_i = -np.dot(normal, direction)
    sin_t2 = (n1 / n2)**2 * (1.0 - cos_i**2)
    if sin_t2 > 1.0:
        return None  # 全反射
    cos_t = np.sqrt(1.0 - sin_t2)
    return (direction * (n1 / n2) + normal * (n1 / n2 * cos_i - cos_t))

# 反射の計算
def reflect(direction, normal):
    return direction - 2 * np.dot(direction, normal) * normal

# 境界から法線を計算する関数
def compute_normal(intersection_point, boundary):
    if boundary.equals(boundary_y2):
        return np.array([0, 1])  # y=2 の法線は上向き
    elif boundary.equals(boundary_outer.boundary):
        # 外枠の法線は外向き
        x, y = intersection_point
        if np.isclose(y, 0):
            return np.array([0, 1])
        elif np.isclose(y, HEIGHT):
            return np.array([0, -1])
        elif np.isclose(x, 0):
            return np.array([1, 0])
        elif np.isclose(x, WIDTH):
            return np.array([-1, 0])
    elif boundary.equals(boundary_circle.boundary):
        # 円の法線は中心から交点へのベクトル
        center = np.array([1.5, 1])
        normal = np.array(intersection_point) - center
        return normal / np.linalg.norm(normal)
    else:
        return np.array([0, 0])  # デフォルト

# 光線追跡関数
def trace_ray(ray, boundaries, max_depth=5):
    if ray.depth > max_depth:
        return []
    
    lines = []
    
    line = ray.get_line()
    min_dist = np.inf
    intersection_point = None
    intersect_boundary = None
    intersect_normal = None
    
    for boundary in boundaries:
        intersect = line.intersection(boundary)
        if intersect.is_empty:
            continue
        # ポイントをリストとして取得
        if isinstance(intersect, Point):
            pts = [intersect]
        elif isinstance(intersect, MultiPoint):
            pts = list(intersect.geoms)
        elif isinstance(intersect, GeometryCollection):
            pts = [geom for geom in intersect.geoms if isinstance(geom, Point)]
        else:
            pts = []
        
        for pt in pts:
            pt_coords = np.array(pt.coords[0])
            dist = np.linalg.norm(pt_coords - ray.origin)
            if dist < min_dist and dist > 1e-6:
                min_dist = dist
                intersection_point = pt_coords
                intersect_boundary = boundary

    if intersection_point is None:
        # 境界に達していない場合、無限遠まで描画
        end_point = ray.origin + ray.direction * 1000
        lines.append((ray.origin, end_point, ray.total_time))
        return lines

    # セグメントを追加
    segment_length = min_dist
    speed = C / ray.current_refractive_index  # 光速 / 屈折率
    time = segment_length / speed
    lines.append((ray.origin, intersection_point, ray.total_time + time))
    
    # 法線の計算
    normal = compute_normal(intersection_point, intersect_boundary)
    
    # 新しい起点
    new_origin = intersection_point + normal * 1e-6  # 微小なオフセット
    
    # 新しい屈折率
    n1 = ray.current_refractive_index
    n2 = get_refractive_index(new_origin)
    
    # 屈折方向
    refracted_dir = refract(ray.direction, normal, n1, n2)
    if refracted_dir is not None:
        refracted_ray = Ray(new_origin, refracted_dir, n2, ray.total_time + time, ray.depth + 1)
        lines += trace_ray(refracted_ray, boundaries, max_depth)
    
    # 反射方向
    reflected_dir = reflect(ray.direction, normal)
    reflected_ray = Ray(new_origin, reflected_dir, n1, ray.total_time + time, ray.depth + 1)
    lines += trace_ray(reflected_ray, boundaries, max_depth)
    
    return lines

# 光線を生成する関数
def generate_rays(initial_direction, num_rays, angle_range_deg):
    angle_range_rad = np.radians(angle_range_deg)
    angle_step = angle_range_rad / (num_rays - 1)
    initial_angle = np.arctan2(initial_direction[1], initial_direction[0])
    
    rays = []
    for i in range(num_rays):
        angle = initial_angle - (angle_range_rad / 2) + i * angle_step
        direction_x = np.cos(angle)
        direction_y = np.sin(angle)
        rays.append((direction_x, direction_y))
    
    return rays

# 境界をリストで管理
boundaries = [
    boundary_outer.boundary,  # 外枠
    boundary_y2,              # y=2 の水平線
    boundary_circle.boundary  # 円の境界
]

# 光源の位置と初期方向
source_position = (1.5, 4)  # シミュレーション外（下部中央）
initial_direction = np.array([0, -1])  # 上向き

# 初期屈折率（空気）
initial_n = get_refractive_index(source_position)

# ±15度の範囲に10本の光線を生成
num_rays = 10
angle_range_deg = 5  # ±15度の範囲
ray_directions = generate_rays(initial_direction, num_rays, angle_range_deg)

# 各光線について追跡
all_lines = []
for direction in ray_directions:
    initial_ray = Ray(source_position, direction, initial_n)
    lines = trace_ray(initial_ray, boundaries, max_depth=10)
    all_lines.extend(lines)

# 時間の範囲を取得
times = [line[2] for line in all_lines]
min_time = min(times)
max_time = max(times)

# 色マッピングの準備
norm = mcolors.Normalize(vmin=min_time, vmax=max_time)
cmap = plt.cm.viridis

# プロットの準備
plt.figure(figsize=(8,12))

# 光線の描画
for start, end, t in all_lines:
    plt.plot([start[0], end[0]], [start[1], end[1]], color=cmap(norm(t)), linewidth=2)

# カラーバーの追加
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, shrink=0.6)
cbar.set_label('Time (s)')

# 境界の描画
# 外枠
x_outer, y_outer = boundary_outer.exterior.xy
plt.plot(x_outer, y_outer, color='black')

# y=2 の境界
x_y2, y_y2 = boundary_y2.xy
plt.plot(x_y2, y_y2, color='blue', linestyle='--')

# 円の境界
x_circle, y_circle = boundary_circle.exterior.xy
plt.plot(x_circle, y_circle, color='red', linestyle='--')

# 媒質領域の色分け
# 媒質1
plt.fill_between([0, WIDTH], 0, 3, color='cyan', alpha=0.2)

# 媒質2
circle_patch = plt.Circle((1.5, 1), 0.15, color='magenta', alpha=0.4)
plt.gca().add_patch(circle_patch)

# 光源の描画
plt.plot(source_position[0], source_position[1], 'ro', label='source')

# グラフの設定
plt.xlim(-1, WIDTH + 1)
plt.ylim(-1, HEIGHT + 1)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper right')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
