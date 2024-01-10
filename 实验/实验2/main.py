import cv2
from image import process_image, generate_mask

input_file = 'input.mp4'
output_file = 'output2.mp4'
# 打开输入视频文件
input_video = cv2.VideoCapture(input_file)

# 获取输入视频的帧率和尺寸
fps = input_video.get(cv2.CAP_PROP_FPS)
width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建输出视频编写器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

mask = generate_mask(w=width, h=height)

# 逐帧处理和写入输出视频
while True:
    # 读取下一帧
    ret, frame = input_video.read()

    # 如果读取失败，则到达视频的末尾
    if not ret:
        break

    # 对帧进行处理
    processed_frame = process_image(frame, mask)

    # 写入输出视频
    output_video.write(processed_frame)

    # 显示处理后的帧（可选）
    cv2.imshow('Processed Frame', processed_frame)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频对象
input_video.release()
output_video.release()

# 关闭所有窗口
cv2.destroyAllWindows()
