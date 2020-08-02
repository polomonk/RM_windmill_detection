import cv2
import numpy as np
from my_bag import *


class Windmill:
    def __init__(self):
        self.Rect = RectangleLinkedList()  # 创建一个链表存放最小矩形的信息
        self.before_angles = []            # 把目标点与中心点的坐标修正后加入
        self.optimal_center_point = []      # 最优的中心点
        self.angular_speed = 0        # 旋转角速度
        self.last_target_point = []     # 保存上一帧的目标点
        self.next_angle = None      # 卡尔曼预测的下一帧的角度
        self.hit_angle_predict = []     # 卡尔曼预测要打击点的角度
        self.frame_count = 0        # 通过帧数计数求平均中心点
        self.last_k = None      # 记录上次目标点到中心点的直线的斜率和偏移
        self.last_b = None
        self.img = None     # 图像

        self.target_point_color = (100, 100, 255)

    def frame_refresh(self, image):     # 更新图像帧
        self.img = image

    @staticmethod
    def distance(point1, point2):  # 测量两个点直接的距离
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def image_processing(self):      # 图像的预处理
        lower_bgr = np.array([200, 0, 0])  # bgr蓝色部分的上下限
        # 通过分别对绿色和红色通道进行限制再合并限制的两个通道的图像，从而过滤了纯白色
        upper_bgr_rl = np.array([255, 255, 100])
        upper_bgr_gl = np.array([255, 100, 255])
        mask_gl = cv2.inRange(self.img, lowerb=lower_bgr, upperb=upper_bgr_gl)  # 限制绿色
        mask_bl = cv2.inRange(self.img, lowerb=lower_bgr, upperb=upper_bgr_rl)  # 限制红色
        _, binary_img_gl = cv2.threshold(mask_gl, 125, 255, cv2.THRESH_BINARY)  # 将图像进行二值化
        _, binary_img_bl = cv2.threshold(mask_bl, 125, 255, cv2.THRESH_BINARY)  # 将图像进行二值化
        binary = cv2.add(binary_img_bl, binary_img_gl)      # 合并
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))      # 腐蚀
        binary = cv2.erode(binary, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))      # 膨胀
        binary = cv2.dilate(binary, kernel)
        return binary

    def find_contours(self, img_binary):        # 找到预处理图像的轮廓并保存
        contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓信息和轮廓的层次信息
        # for contour in contours:
        for i in range(len(contours)):     # 每个轮廓信息都包含【后一个轮廓，前一个轮廓，子轮廓，父轮廓】信息没有则为-1
            if hierarchy[0][i][2] >= 0:     # 这里将有子轮廓的轮廓保存并画出来
                cv2.drawContours(self.img, contours[i], -1, 255, 2)
                rect = cv2.minAreaRect(contours[i])  # 把轮廓信息用最小矩形包裹做成矩形信息
                # print(rect)   # point = rect[0]   side = rect[1]   angle = rect[2]
                self.Rect.append(np.int0(rect[0]), rect[1], rect[2])
        # return img_binary

    def refresh_optimal_center(self, current_center_point):        # 更新最优中心点
        if current_center_point is None:        # 当前中心点没找到
            if len(self.optimal_center_point) is 0:     # 之前的最优中心点也不存在
                self.optimal_center_point = None       # 无法确认最优中心点
            else:
                self.optimal_center_point = self.optimal_center_point      # 将之前得到的最优中心点
        else:       # 找到了当前中心点
            if len(self.optimal_center_point) is 0:     # 之前的最优中心点不存在
                self.optimal_center_point = current_center_point
            else:       # 对所有中心点取平均值
                self.optimal_center_point = [
                    current_center_point[0]/self.frame_count + self.optimal_center_point[0] * (1-(1/self.frame_count)),
                    current_center_point[1]/self.frame_count + self.optimal_center_point[1] * (1-(1/self.frame_count))]

    def linear_compensation(self):      # 提供线性补偿，不推荐使用
        edges = cv2.Canny(self.img, 50, 100)
        # r_error最好为1，θ_error在2左右, 计数结果20左右, 最小长度要>=8, 缺失2个像素点也认为是一条直线
        lines = cv2.HoughLinesP(edges, 2, np.pi / 180 * 2, 20, minLineLength=8, maxLineGap=2)
        if lines is not None:
            lines1 = lines[:, 0, :]  # 提取为二维
            for x1, y1, x2, y2 in lines1:
                cv2.line(self.img, (x1, y1), (x2, y2), (50, 55, 200), 2)
        # cv2.imshow('ddd', edges)
        # cv2.waitKey(0)

    def kalman_filter(self, angles, speed=0):        # 卡尔曼滤波
        prediction = 3
        x_mat = np.mat([[angles[0], speed], [0, 0]])  # 定义初始状态
        p_mat = np.mat(np.eye(2))  # 定义状态转移矩阵
        f_mat = np.mat([[1, 1], [0, 1]])  # 定义观测矩阵，因为每秒钟采一次样，所以delta_t = 1
        h_mat = np.mat([1, 0])  # 定义观测噪声协方差
        r_mat = np.mat([0.5])  # 定义状态转移矩阵噪声
        for angle in angles:
            x_predict = f_mat * x_mat
            p_predict = f_mat * p_mat * f_mat.T
            kalman = p_predict * h_mat.T / (h_mat * p_predict * h_mat.T + r_mat)
            x_mat = x_predict + kalman * (np.mat(angle) - h_mat * x_predict)
            p_mat = (np.eye(2) - kalman * h_mat) * p_predict
        self.next_angle = (f_mat * x_mat).tolist()[0][0]  # f_mat只预测下一帧的角度
        self.hit_angle_predict = (np.mat([[1, prediction], [0, 1]]) * x_mat).tolist()[0][0]  # 预测count帧后的角度

    def angle_correct(self, center_point, target_point, angle):     # 根据坐标关系对目标点的角度进行修正
        # 修正的理想结果是最左边是0°顺时针到最下边是270°（-90°）
        if center_point[0] - target_point[0] >= 0:      # 目标点在中心点的左边
            if center_point[1] - target_point[1] >= 0 and angle < -60:      # 中心点有偏差，目标点的角度已经超过90°了
                self.before_angles.append(angle + 180)
            elif center_point[1] - target_point[1] <= 0 and angle > 60:     # 目标点还没超过270°
                self.before_angles.append(angle + 180)
            else:
                self.before_angles.append(angle)
        else:       # 目标点在中心点的右边
            if center_point[1] - target_point[1] >= 0 and angle > 60:       # 中心点有偏差，目标点的角度还没超过90°
                self.before_angles.append(angle)
            elif center_point[1] - target_point[1] <= 0 and angle > 60:        # 超过了270°
                self.before_angles.append(angle + 180)
            else:
                self.before_angles.append(angle + 180)
        # print(center_point, target_point, angle)
        if len(self.before_angles) > 1:     # 在与之前的角度对比后再修正
            difference = self.before_angles[-1] - self.before_angles[-2]        # 当前修正的角度与之前的角度差值
            if difference < -345:       # 认为差值已经超过了360°即到了270°到-90°的跳变。把之前的点都减360°
                for i in range(len(self.before_angles) - 1):
                    self.before_angles[i] = self.before_angles[i] - 360
            if difference > 345:        # 预防一波风车反转
                for i in range(len(self.before_angles) - 1):
                    self.before_angles[i] = self.before_angles[i] + 360
            elif (difference + 10)//72 >= 1 and (difference - 10) % 72 >= 60:       # 击中目标后新的目标与之前的角度跨度规律
                self.before_angles.clear()      # 上个目标点收集的角度全部清除
                self.angle_correct(center_point, target_point, angle)       # 加入新的角度

            if len(self.before_angles) > 5:     # 只保留5个点进行滤波预测
                self.before_angles.pop(0)

    def hit_predict(self, center_point, target_point):
        length = self.distance(center_point, target_point)  # 中心点与目标点的距离
        kp = np.tan(np.radians(self.hit_angle_predict))  # 求出目标点相对中心点的横纵坐标
        xp = np.sqrt(length ** 2 / (1 + kp ** 2))
        yp = np.sqrt(length ** 2 - xp ** 2)
        if kp >= 0:  # 如果斜率大于0，则目标的应该出现在以中心点为坐标系的一三象限
            first_quadrant_point = (int(center_point[0] + xp), int(center_point[1] + yp))       # 第一象限预测的目标点
            third_quadrant_point = (int(center_point[0] - xp), int(center_point[1] - yp))
            if self.distance(first_quadrant_point, self.last_target_point) < \
                    self.distance(third_quadrant_point, self.last_target_point):  # 根据与目标点的距离判断在哪个象限
                cv2.circle(self.img, first_quadrant_point, 3, self.target_point_color, 3)
            else:
                cv2.circle(self.img, third_quadrant_point, 3, self.target_point_color, 3)
        else:
            beta_quadrant_point = (int(center_point[0] + xp), int(center_point[1] - yp))
            delta_quadrant_point = (int(center_point[0] - xp), int(center_point[1] + yp))
            if self.distance(delta_quadrant_point, self.last_target_point) < \
                    self.distance(beta_quadrant_point, self.last_target_point):
                cv2.circle(self.img, delta_quadrant_point, 3, self.target_point_color, 3)
            else:
                cv2.circle(self.img, beta_quadrant_point, 3, self.target_point_color, 3)

    def run(self):
        h, w, c = self.img.shape
        center_point = None     # 当前的中心点
        self.Rect.clear()       # 清空上一帧的轮廓数据
        img_binary = self.image_processing()        # 图像预处理
        self.find_contours(img_binary)     # 找到并保存轮廓
        cur = self.Rect.head
        while cur is not None:
            if 3200 * 0.8 <= cur.area() <= 3200 * 1.2:
                """
                    挖坑：
                        如果箭头挨着目标矩形框会造成识别的时候把他们认为是一个大矩形，使目标点与中心点的距离减小产生误差
                    参考：
                        使用穿过中心点直线的斜率使中心点的沿斜率方向移动
                        移动距离按垂直于直线的边作为参考计算相邻的边与实际差了多少
                """
                # print("正常高宽比：", cur.width/cur.height)     # 1.4左右
                box = cv2.boxPoints((cur.point, cur.side, cur.orig_angle))      # 转化成轮廓信息
                box = np.int0(box)  # 转换成整数操作
                cv2.drawContours(self.img, [box], -1, (0, 0, 255), 1)       # 画出其最小外接矩形

                k = np.tan(np.radians(cur.angle))       # 计算目标矩形经过中心点的直线
                b = (cur.point[1]) - k * cur.point[0]

                if self.last_k is None or abs(self.last_k - k) < 1e-4:      # 如果第一次测量或当前测量值与上一次测量的斜率相同
                    if len(self.optimal_center_point) > 0:      # 尽可能使用最优中心点
                        center_point = self.optimal_center_point

                    if -45 <= cur.angle <= 45:      # 合理画出直线
                        cv2.line(self.img, (0, int(b)), (w, int(w * k + b)), (0, 255, 255))  # 把拟合的直线画出来
                    else:
                        cv2.line(self.img, (int(-b / k), 0), (int((h - b) / k), h), (0, 255, 255))  # 把拟合的直线画出来
                else:
                    if -45 <= cur.angle <= 45:
                        cv2.line(self.img, (0, int(b)), (w, int(w * k + b)), (0, 255, 255))  # 把拟合的直线画出来
                    else:
                        cv2.line(self.img, (int(-b / k), 0), (int((h - b) / k), h), (0, 255, 255))  # 把拟合的直线画出来
                    x = (b - self.last_b) / (self.last_k - k)
                    y = k * x + b
                    if 0 < x < w and 0 < y < h:       # 舍去太离谱的中心点
                        center_point = [round(x), round(y)]
                        self.frame_count += 1
                    else:
                        center_point = self.optimal_center_point        # 如果检测到了目标就必须赋值新的中心点
                self.last_k = k     # 保存当前直线
                self.last_b = b

                if center_point is not None:
                    self.refresh_optimal_center(center_point)       # 更新最优中心点
                    # cur.angle 在最下方时为-90°顺时针旋转到最上方时为90°（-90°）即下个循环的起点，而后又到达最下方为90°
                    self.angle_correct(self.optimal_center_point, cur.point, cur.angle)     # 修正角度
                    if len(self.before_angles) > 1:     # 平均角速度
                        self.angular_speed = (self.before_angles[-1]-self.before_angles[0])/(len(self.before_angles)-1)
                    self.kalman_filter(self.before_angles, self.angular_speed)      # 卡尔曼滤波进行预测
                    self.hit_predict(center_point, cur.point)       # 预测要打击的点并画出来
                self.last_target_point = cur.point      # 保存当前目标点
                break
            cur = cur.next
        if center_point is None and len(self.before_angles) > 1:        # 如果没有检测到目标，则预测目标的位置
            self.before_angles.append(self.next_angle)      # 使用下一帧的预测值继续进行预测
            self.angular_speed = (self.before_angles[-1] - self.before_angles[0]) / (len(self.before_angles) - 1)
            self.kalman_filter(self.before_angles, self.angular_speed)
            if len(self.before_angles) > 5:
                self.before_angles.pop(0)
            self.hit_predict(self.optimal_center_point, self.last_target_point)
        # print(self.before_angles)
        if center_point is not None:        # 画出中心点
            cv2.circle(self.img, tuple(np.int0(self.optimal_center_point)), 3, color=(0, 255, 255), thickness=3)
        cv2.imshow('dst', self.img)
        cv2.waitKey(0)


if __name__ == '__main__':
    import time

    Wm = Windmill()
    video = r'D:\Document\py\opencv\demo\windmill_blue.mp4'
    img2 = cv2.imread(r'D:\Document\py\opencv\demo/blue_pic/361.jpg')
    img = cv2.imread(r'D:\Document\py\opencv\demo/blue_pic/360.jpg')
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    while ret:
        Wm.frame_refresh(frame)
        Wm.run()
        ret, frame = cap.read()
