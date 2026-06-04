#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

def detect_monitor(image):
    """
    TODO: Detect the monitor corners in the input BGR image.

    Return:
      top_left, top_right, bottom_right, bottom_left

    Each point should be an (x, y) pair in the original image coordinate system.
    Return (None, None, None, None) if detection fails.
    """
    h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)

    edges = cv2.dilate(
        edges,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1
    )

    # 검정 계열 mask
    rgb_max = np.max(image, axis=2)
    rgb_min = np.min(image, axis=2)
    rgb_diff = rgb_max - rgb_min

    black_mask = np.zeros((h, w), dtype=np.uint8)
    black_mask[
        ((gray < 120) & (rgb_diff < 100)) |
        (gray < 70)
    ] = 255

    black_mask_dilated = cv2.dilate(
        black_mask,
        np.ones((7, 7), dtype=np.uint8),
        iterations=1
    )

    # Canny edge 중 검정색 근처 edge만 우선 사용
    black_edges = cv2.bitwise_and(edges, black_mask_dilated)

    black_edges = cv2.morphologyEx(
        black_edges,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1
    )

    def order_points(pts):
        pts = pts.astype("float32")

        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)

        top_left = pts[np.argmin(s)]
        top_right = pts[np.argmin(diff)]
        bottom_right = pts[np.argmax(s)]
        bottom_left = pts[np.argmax(diff)]

        return np.array(
            [top_left, top_right, bottom_right, bottom_left],
            dtype="float32"
        )

    def side_black_ratio(p1, p2, num_samples=45):
        p1 = np.array(p1, dtype=np.float32)
        p2 = np.array(p2, dtype=np.float32)

        direction = p2 - p1
        length = np.linalg.norm(direction)

        if length < 1:
            return 0.0

        direction = direction / length
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)

        offsets = [-6, -3, 0, 3, 6]

        valid_count = 0
        black_count = 0

        for i in range(num_samples):
            t = i / (num_samples - 1)

            # 꼭짓점 근처는 다른 선과 섞이기 쉬우므로 제외
            if t < 0.08 or t > 0.92:
                continue

            base = p1 * (1 - t) + p2 * t
            valid_count += 1

            found_black = False

            for offset in offsets:
                sample = base + normal * offset
                x = int(round(sample[0]))
                y = int(round(sample[1]))

                if 0 <= x < w and 0 <= y < h:
                    if black_mask[y, x] > 0:
                        found_black = True
                        break

            if found_black:
                black_count += 1

        if valid_count == 0:
            return 0.0

        return black_count / valid_count

    def has_black_border(pts):
        top_left, top_right, bottom_right, bottom_left = pts

        sides = [
            (top_left, top_right),
            (top_right, bottom_right),
            (bottom_right, bottom_left),
            (bottom_left, top_left)
        ]

        ratios = [side_black_ratio(p1, p2) for p1, p2 in sides]

        # 4변 중 3변 이상이 검정 테두리이면 인정
        black_side_count = sum(r >= 0.30 for r in ratios)

        return black_side_count >= 3

    def collect_candidates(edge_img, require_black=True):
        contours, _ = cv2.findContours(
            edge_img,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []

        min_area = h * w * 0.006

        for contour in contours:
            contour_area = cv2.contourArea(contour)

            if contour_area < min_area:
                continue

            perimeter = cv2.arcLength(contour, True)

            if perimeter == 0:
                continue

            approx = None

            for eps_ratio in [
                0.008, 0.01, 0.015, 0.02, 0.025,
                0.03, 0.04, 0.05, 0.07, 0.09
            ]:
                candidate = cv2.approxPolyDP(
                    contour,
                    eps_ratio * perimeter,
                    True
                )

                if len(candidate) == 4:
                    approx = candidate
                    break

            if approx is None:
                continue

            pts = approx.reshape(4, 2).astype("float32")
            pts = order_points(pts)

            if not cv2.isContourConvex(pts.astype(np.int32)):
                continue

            area = cv2.contourArea(pts.astype(np.int32))

            if area < min_area:
                continue

            side_lengths = [
                np.linalg.norm(pts[1] - pts[0]),
                np.linalg.norm(pts[2] - pts[1]),
                np.linalg.norm(pts[3] - pts[2]),
                np.linalg.norm(pts[0] - pts[3])
            ]

            min_side = min(side_lengths)
            max_side = max(side_lengths)

            if min_side < min(h, w) * 0.035:
                continue

            # 벽 전체나 배경 패턴처럼 너무 납작하고 긴 후보 제거
            if max_side / min_side > 4.2:
                continue

            # 이미지 전체 테두리 방지
            xs = pts[:, 0]
            ys = pts[:, 1]

            touches_image_border = (
                np.min(xs) <= 1 or
                np.max(xs) >= w - 2 or
                np.min(ys) <= 1 or
                np.max(ys) >= h - 2
            )

            if touches_image_border:
                continue

            if require_black and not has_black_border(pts):
                continue

            candidates.append((area, pts))

        return candidates

    # 1순위: 검정 근처 Canny edge에서 찾기
    candidates = collect_candidates(black_edges, require_black=True)

    # 2순위: 전체 Canny에서 찾되, 검정 테두리 조건 유지
    if len(candidates) == 0:
        candidates = collect_candidates(edges, require_black=True)

    # 3순위 fallback: 그래도 없으면 전체 Canny에서 가장 큰 사각형
    if len(candidates) == 0:
        candidates = collect_candidates(edges, require_black=False)

    if len(candidates) == 0:
        return None, None, None, None

    candidates.sort(key=lambda x: x[0], reverse=True)

    area, best_rect = candidates[0]

    top_left, top_right, bottom_right, bottom_left = best_rect

    return top_left, top_right, bottom_right, bottom_left

def rectify_monitor(image, top_left, top_right, bottom_right, bottom_left):
    """
    TODO: Perspective-transform the detected monitor into a front-facing view.

    Return:
      rectified BGR image

    Return None if rectification fails.
    """
    if top_left is None or top_right is None or bottom_right is None or bottom_left is None:
        return image
        
    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    max_width = int(max(width_top, width_bottom))

    height_right = np.linalg.norm(bottom_right - top_right)
    height_left = np.linalg.norm(bottom_left - top_left)
    max_height = int(max(height_right, height_left))
    

    # 원래 검출된 모니터가 세로로 더 긴지 먼저 저장
    is_vertical = max_height > max_width

    # 항상 rectified monitor가 가로:세로 = 16:9가 되도록 설정
    # 원래 긴 쪽을 결과 이미지의 가로 방향으로 보냄
    long_side = max(max_width, max_height)

    max_width = int(round(long_side))
    max_height = int(round(long_side * 9 / 16))

    max_width = max(1, max_width)
    max_height = max(1, max_height)

    if is_vertical:
        # 세로가 더 길게 잡힌 경우:
        # 점 순서를 회전시켜서 세로 방향을 결과 이미지의 가로 방향으로 보냄
        src = np.array(
            [bottom_left, top_left, top_right, bottom_right],
            dtype="float32"
        )
    else:
        # 원래 가로가 더 긴 경우는 그대로 사용
        src = np.array(
            [top_left, top_right, bottom_right, bottom_left],
            dtype="float32"
        )
    dst = np.array([[0,0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1],], dtype="float32",)

    transform = cv2.getPerspectiveTransform(src, dst)

    rectified = cv2.warpPerspective(
        image,
        transform,
        (max_width, max_height),
    )
    
    return rectified

def detect_line(rectified):
    """
    TODO: Detect the longest line inside the rectified monitor image.

    Return:
      (x1, y1, x2, y2)

    Return None if line detection fails.
    """
    if rectified is None:
        return None

    h, w = rectified.shape[:2]

    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(rectified, cv2.COLOR_BGR2HSV)

    # 모니터 테두리를 line으로 착각하지 않도록 내부만 사용
    margin = max(3, int(min(h, w) * 0.08))

    min_len = max(10, int(min(h, w) * 0.045))

    best_line = None
    best_length = 0

    kernel = np.ones((2, 2), dtype=np.uint8)

    def remove_border(mask):
        mask[:margin, :] = 0
        mask[h - margin:h, :] = 0
        mask[:, :margin] = 0
        mask[:, w - margin:w] = 0
        return mask

    def line_support_score(mask, x1, y1, x2, y2):
        num_samples = 50

        valid_count = 0
        support_count = 0

        for i in range(num_samples):
            t = i / (num_samples - 1)

            x = int(round(x1 * (1 - t) + x2 * t))
            y = int(round(y1 * (1 - t) + y2 * t))

            if not (0 <= x < w and 0 <= y < h):
                continue

            valid_count += 1

            x1p = max(0, x - 1)
            x2p = min(w, x + 2)
            y1p = max(0, y - 1)
            y2p = min(h, y + 2)

            patch = mask[y1p:y2p, x1p:x2p]

            if cv2.countNonZero(patch) > 0:
                support_count += 1

        if valid_count == 0:
            return 0.0

        return support_count / valid_count

    def check_edges(edges, support_mask):
        nonlocal best_line, best_length

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=10,
            minLineLength=min_len,
            maxLineGap=3
        )

        if lines is None:
            return

        for line in lines:
            x1, y1, x2, y2 = line[0]

            length = np.hypot(x2 - x1, y2 - y1)

            if length < min_len:
                continue

            # 모니터 테두리 근처 선 제거
            if (
                x1 <= margin or x1 >= w - margin or
                x2 <= margin or x2 >= w - margin or
                y1 <= margin or y1 >= h - margin or
                y2 <= margin or y2 >= h - margin
            ):
                continue

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if (
                cx <= margin or cx >= w - margin or
                cy <= margin or cy >= h - margin
            ):
                continue

            dx = x2 - x1
            dy = y2 - y1

            angle_abs = abs(np.degrees(np.arctan2(dy, dx)))

            is_almost_horizontal = angle_abs < 5 or angle_abs > 175
            is_almost_vertical = 85 < angle_abs < 95

            near_outer_area = (
                cy < h * 0.18 or
                cy > h * 0.82 or
                cx < w * 0.12 or
                cx > w * 0.88
            )

            if near_outer_area and (is_almost_horizontal or is_almost_vertical):
                continue

            support = line_support_score(
                support_mask,
                x1, y1, x2, y2
            )

            if support < 0.35:
                continue

            # candidates 리스트에 저장하지 않고 바로 최댓값만 유지
            if length > best_length:
                best_length = length
                best_line = (x1, y1, x2, y2)

    def process_mask(mask):
        mask = remove_border(mask)

        cleaned = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )

        edges = cv2.Canny(cleaned, 50, 150)
        edges = remove_border(edges)

        check_edges(edges, mask)

    # 1. 검은색 / 어두운 선
    dark_mask = cv2.inRange(gray, 0, 149)
    process_mask(dark_mask)

    # 2. 흰색 / 밝은 선
    bright_mask = cv2.inRange(gray, 191, 255)
    process_mask(bright_mask)

    # 3. 채도가 있는 색깔 선
    # hue를 나누어 서로 다른 색의 line이 하나로 붙지 않게 함
    for hue_start in range(0, 180, 10):
        hue_end = min(hue_start + 9, 179)

        color_mask = cv2.inRange(
            hsv,
            np.array([hue_start, 45, 45], dtype=np.uint8),
            np.array([hue_end, 255, 255], dtype=np.uint8)
        )

        process_mask(color_mask)

    # 4. raw edge도 보조로 사용
    raw_edges = cv2.Canny(gray, 50, 150)
    raw_edges = remove_border(raw_edges)

    check_edges(raw_edges, raw_edges)

    if best_line is None:
        return None

    x1, y1, x2, y2 = best_line

    return int(x1), int(y1), int(x2), int(y2)

def calculate_angle(line) -> Optional[float]:
    """
    TODO: Calculate the line angle in degrees.

    The angle must be expressed in the rectified monitor coordinate system.
    Return None if the angle cannot be calculated.
    """
    if line is None:
        return None

    x1, y1, x2, y2 = line

    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)

    # 사진 좌표계에서 위쪽 점과 아래쪽 점을 먼저 정한다
    if y1 < y2:
        x_top, y_top = x1, y1
        x_bottom, y_bottom = x2, y2
    elif y2 < y1:
        x_top, y_top = x2, y2
        x_bottom, y_bottom = x1, y1
    else:
        # 완전한 수평선은 방향 구분이 애매하므로 +90도로 처리
        return 90.0

    dx = x_bottom - x_top
    dy = y_bottom - y_top

    # 세로 방향을 기준으로 한 사진 좌표계 각도
    angle = np.degrees(np.arctan2(dx, dy))

    return angle

class LineDetector(Node):
    def __init__(self) -> None:
        super().__init__("line_detector_node")

        self.declare_parameter("topic_image", "/camera/camera/color/image_raw")
        self.declare_parameter("topic_student", "/student/angle")

        topic_image = str(self.get_parameter("topic_image").value)
        topic_student = str(self.get_parameter("topic_student").value)

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            topic_image,
            self.image_callback,
            10,
        )

        self.angle_pub = self.create_publisher(
            Float32,
            topic_student,
            10,
        )

        self.line_pub = self.create_publisher(
            Image,
            "/debug/line",
            10,
        )

        self.get_logger().info(
            f"Line detector started. Subscribing to {topic_image!r}, "
            f"publishing to {topic_student!r}."
        )

    def image_callback(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Failed to convert image: {exc!r}")
            return

        # 1. Detect monitor.
        top_left, top_right, bottom_right, bottom_left = detect_monitor(image)
        if any(p is None for p in (top_left, top_right, bottom_right, bottom_left)):
            self.get_logger().warning("Monitor not detected.")
            return

        # 2. Rectify monitor.
        rectified = rectify_monitor(image, top_left, top_right, bottom_right, bottom_left)
        if rectified is None:
            self.get_logger().warning("Monitor not rectified.")
            return

        # 3. Detect line.
        line = detect_line(rectified)
        if line is None:
            self.get_logger().warning("Line not detected.")
            return
        self._debug_line(msg, rectified, line)

        # 4. Calculate and publish angle.
        angle = calculate_angle(line)
        if angle is None:
            self.get_logger().warning("Angle not calculated.")
            return

        angle_msg = Float32()
        angle_msg.data = float(angle)
        self.angle_pub.publish(angle_msg)

        self.get_logger().info(f"Line angle: {float(angle):.2f} deg")

    def _debug_line(self, msg, rectified, line) -> None:
        debug_line = rectified.copy()

        x1, y1, x2, y2 = line
        cv2.line(
            debug_line,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            6,
        )

        debug_line_msg = self.bridge.cv2_to_imgmsg(debug_line, encoding="bgr8")
        debug_line_msg.header = msg.header
        self.line_pub.publish(debug_line_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
