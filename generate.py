from PIL import Image, ImageDraw, ImageFont
import random
import string
import math
import os
import uuid

def generate_captcha_batch():
    # 参数配置
    width, height = 100, 40
    bg_color = (255, 255, 255)  # 白色背景
    font_size = 24  # 放大字体
    num_chars = 4
    output_dir = "captchas"
    num=2000
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for _ in range(num):
        image = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(image, 'RGBA')
        
        # 生成大小写混合验证码
        chars = ''.join(random.choice(string.ascii_letters) for _ in range(num_chars))
        
        # 加载字体
        try:
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
                font.size = font_size

        # 紧凑字符布局
        positions = []
        char_width = 20
        start_x = (width - (char_width * num_chars)) // 2
        
        for i, char in enumerate(chars):
            x = start_x + i * char_width + random.randint(-3, 3)
            y = 10 if char.isupper() else 8
            y += random.randint(-4, 4)
            x = max(5, min(width - char_width - 5, x))
            y = max(5, min(height - 25, y))
            
            char_img = Image.new('RGBA', (char_width, 25), (255, 255, 255, 0))
            char_draw = ImageDraw.Draw(char_img)
            char_draw.text((2, 0), char, fill=(0, 0, 0), font=font)
            
            distorted = Image.new('RGBA', (char_width, 25), (255, 255, 255, 0))
            for dy in range(25):
                amp = 2
                offset = int(amp * math.sin(dy/5 + i*0.5))
                region = char_img.crop((0, dy, char_width, dy+1))
                distorted.paste(region, (offset, dy))
            
            image.paste(distorted, (x, y), distorted)
            positions.append((x, y, char_width))

        # ===== 改进的干扰线生成 =====
        line_color = (255, 255, 0, 100)  # 亮黄色，80%不透明
        line_width = random.randint(5, 8)  # 随机线宽增加变化
        
        # 生成贯穿整个图像的干扰线（确保覆盖整个区域）
        for _ in range(2):  # 两条干扰线
            # 随机选择起点位置（图像边缘）
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                start_x, start_y = random.randint(0, width), 0
            elif edge == 'bottom':
                start_x, start_y = random.randint(0, width), height-1
            elif edge == 'left':
                start_x, start_y = 0, random.randint(0, height)
            else:  # right
                start_x, start_y = width-1, random.randint(0, height)
            
            # 随机选择终点位置（对边）
            if edge in ['top', 'bottom']:
                end_x, end_y = random.randint(0, width), height-1 if edge == 'top' else 0
            else:
                end_x, end_y = width-1 if edge == 'left' else 0, random.randint(0, height)
            
            # 添加随机偏移使线条更自然
            start_x += random.randint(-5, 5)
            start_y += random.randint(-5, 5)
            end_x += random.randint(-5, 5)
            end_y += random.randint(-5, 5)
            
            # 确保点仍在图像范围内
            start_x = max(0, min(width-1, start_x))
            start_y = max(0, min(height-1, start_y))
            end_x = max(0, min(width-1, end_x))
            end_y = max(0, min(height-1, end_y))
            
            # 绘制干扰线
            draw.line([(start_x, start_y), (end_x, end_y)], fill=line_color, width=line_width)

        # 保存文件
        file_id = uuid.uuid4().hex[:6]
        image.save(f"{output_dir}/{chars}_{file_id}.png")

    print(f"已生成{num}个验证码到 {output_dir}/ 目录")

# 执行生成
generate_captcha_batch()