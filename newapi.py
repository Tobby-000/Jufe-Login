from flask import Flask, request, jsonify
import requests
import rsa
import binascii
from bs4 import BeautifulSoup
import re
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
import one_hot
import model
from model import mymodel
import common
import logging
from logging.handlers import RotatingFileHandler
import time
import urllib.parse
import traceback
import os

app = Flask(__name__)

# == 配置参数修改 (开始) ==
DEBUG_MODE = False  # 修改：默认关闭调试模式
# == 配置参数修改 (结束) ==

# == 日志目录创建 (开始) ==
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# == 日志目录创建 (结束) ==

# == 日志系统重构 (开始) ==
def setup_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
    
    # 清除现有的处理程序
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 文件处理程序
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'app.log'),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
    
    # 控制台处理程序
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 修改：控制台固定输出INFO及以上
    
    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理程序
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Werkzeug日志配置
    if DEBUG_MODE:
        werkzeug_log = logging.getLogger('werkzeug')
        werkzeug_log.setLevel(logging.DEBUG)
        werkzeug_log.addHandler(file_handler)
        werkzeug_log.addHandler(console_handler)

# 初始化日志
setup_logging()
app.logger = logging.getLogger(__name__)
app.logger.info(f"应用程序启动，调试模式: {DEBUG_MODE}")
# == 日志系统重构 (结束) ==

# 实例化模型并加载权重
try:
    model = mymodel()
    checkpoint = torch.load("best_model.pth", map_location=torch.device('cpu'))
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    app.logger.info("模型加载成功。")
except Exception as e:
    app.logger.error(f"模型加载失败: {e}")
    traceback.print_exc()
    model = None

# 配置参数
BASE_URL = "https://ssl.jxufe.edu.cn"
LOGIN_PAGE = f"{BASE_URL}/cas/login"
SERVICE_URL = "http://ehall.jxufe.edu.cn"

# == 请求日志优化 (开始) ==
@app.before_request
def log_request_info():
    # 修改：只在DEBUG_MODE下记录详细的请求头信息
    if DEBUG_MODE:
        app.logger.info(f"处理请求: {request.method} {request.url}")
        app.logger.info(f"请求头: {dict(request.headers)}")
    
    if request.method == 'POST':
        try:
            if request.is_json:
                data = request.get_json()
                # 修改：只在DEBUG_MODE下记录JSON数据
                if DEBUG_MODE:
                    app.logger.info(f"请求JSON数据: {data}")
            else:
                app.logger.info(f"请求表单数据: {request.form.to_dict()}")
        except Exception as e:
            app.logger.warning(f"解析请求数据失败: {e}")
# == 请求日志优化 (结束) ==

# 定义图片预处理函数
def preprocess_image(image):
    tensor_img = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((60, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img = tensor_img(image)
    img = torch.reshape(img, (1, 1, 60, 160))
    return img

# == 表单数据获取优化 (开始) ==
def get_login_form_data_and_public_key(session):
    try:
        response = session.get(LOGIN_PAGE, params={'service': SERVICE_URL}, timeout=10)
        response.raise_for_status()
        # 修改：只在DEBUG_MODE下记录状态码
        if DEBUG_MODE:
            app.logger.info(f"登录页面状态码: {response.status_code}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 获取隐藏字段
        lt = soup.find('input', {'name': 'lt'})['value']
        # 修改：只在DEBUG_MODE下记录lt值
        if DEBUG_MODE:
            app.logger.debug(f"获取到lt字段值: {lt}")
        
        # 记录所有隐藏字段的值
        hidden_fields = {}
        for hidden in soup.find_all('input', {'type': 'hidden'}):
            name = hidden.get('name')
            value = hidden.get('value')
            hidden_fields[name] = value
            app.logger.debug(f"隐藏字段 {name} = {value}")
        
        # 获取公钥参数
        n, e = None, None
        script_tags = soup.find_all('script')
        for script in script_tags:
            if script.string and 'var n =' in script.string:
                n_match = re.search(r'var n = "(.*?)"', script.string)
                e_match = re.search(r'var e = "(.*?)"', script.string)
                if n_match and e_match:
                    n = n_match.group(1)
                    e = e_match.group(1)
                    break
        
        if not n or not e:
            raise ValueError("未找到公钥参数")
            
        app.logger.debug(f"公钥参数 n: {n}, e: {e}")
        return lt, n, e
    except Exception as e:
        app.logger.error(f"获取登录表单数据失败: {e}")
        traceback.print_exc()
        raise
# == 表单数据获取优化 (结束) ==

# == 密码加密优化 (开始) ==
def encrypt_password(password, n, e):
    try:
        # 修改：只在DEBUG_MODE下记录密码加密详情
        if DEBUG_MODE:
            app.logger.debug(f"开始加密密码...")
            app.logger.debug(f"转换公钥参数 n: {n}, e: {e}")
        
        public_key = rsa.PublicKey(
            int(n, 16) if isinstance(n, str) else n,
            int(e, 16) if isinstance(e, str) else e
        )
        
        if DEBUG_MODE:
            app.logger.debug(f"创建公钥对象成功")
        
        # 计算最大可加密长度
        key_size = rsa.common.byte_size(public_key.n)
        max_length = key_size - 11
        if DEBUG_MODE:
            app.logger.debug(f"公钥大小: {key_size} 字节, 最大加密块大小: {max_length} 字节")
        
        # 显式编码密码
        password_bytes = password.encode('utf-8')
        if DEBUG_MODE:
            app.logger.debug(f"密码字节长度: {len(password_bytes)}")
        
        # 如果密码过长则分段加密
        encrypted = b''
        if len(password_bytes) > max_length:
            chunks = [password_bytes[i:i+max_length] for i in range(0, len(password_bytes), max_length)]
            app.logger.debug(f"密码过长, 分{len(chunks)}段加密")
            
            for i, chunk in enumerate(chunks):
                encrypted_chunk = rsa.encrypt(chunk, public_key)
                encrypted += encrypted_chunk
                if DEBUG_MODE:
                    app.logger.debug(f"段 {i+1} 加密完成, 长度: {len(encrypted_chunk)} 字节")
        else:
            encrypted = rsa.encrypt(password_bytes, public_key)
            if DEBUG_MODE:
                app.logger.debug(f"加密完成, 长度: {len(encrypted)} 字节")
            
        hex_encrypted = binascii.hexlify(encrypted).decode('utf-8')
        # 修改：优化加密后的日志记录
        if DEBUG_MODE:
            app.logger.debug(f"加密后的字节长度: {len(encrypted)}")
            app.logger.debug(f"加密后的十六进制字符串长度: {len(hex_encrypted)}")
            app.logger.debug(f"加密后的HEX字符串: {hex_encrypted[:50]}...(共{len(hex_encrypted)}字符)")
        return hex_encrypted
    except Exception as e:
        app.logger.error(f"密码加密失败: {e}")
        traceback.print_exc()
        raise
# == 密码加密优化 (结束) ==

# == 验证码获取优化 (开始) ==
def get_captcha(session):
    try:
        # 修改：只在DEBUG_MODE下记录验证码获取详情
        if DEBUG_MODE:
            app.logger.info(f"开始获取验证码...")
        
        # 获取登录页面
        response = session.get(LOGIN_PAGE, params={'service': SERVICE_URL}, timeout=10)
        response.raise_for_status()
        if DEBUG_MODE:
            app.logger.info(f"验证码页面状态码: {response.status_code}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 查找验证码图片
        img_tag = None
        
        # 1. 查找包含特定样式的div内的图片
        div_tags = soup.find_all('div', style=re.compile(r'position\s*:\s*absolute'))
        for div in div_tags:
            img_tag = div.find('img', src=True)
            if img_tag:
                break
                
        # 2. 如果没找到，尝试查找src包含"/cas/codeimage"的图片
        if not img_tag:
            img_tag = soup.find('img', src=re.compile(r'/cas/codeimage'))
        
        # 3. 如果还是没找到，尝试其他常见选择器
        if not img_tag:
            img_tag = soup.select_one('img#captchaImg, img.captcha-img, img[alt="验证码"]')
        
        if not img_tag or not img_tag.get('src'):
            raise ValueError("未找到验证码图片")
        
        # 使用 urllib.parse.urljoin 安全拼接 URL
        captcha_src = img_tag['src']
        captcha_url = urllib.parse.urljoin(response.url, captcha_src)
        # 修改：只在DEBUG_MODE下记录URL
        if DEBUG_MODE:
            app.logger.info(f"验证码图片URL: {captcha_url}")
        
        # 确保 URL 是有效的
        if not captcha_url.startswith('http'):
            raise ValueError(f"无效的验证码 URL: {captcha_url}")
        
        # 获取验证码图片
        response = session.get(captcha_url, timeout=10)
        response.raise_for_status()
        if DEBUG_MODE:
            app.logger.info(f"验证码图片下载完成, 大小: {len(response.content)} 字节")
        
        # 处理验证码图片
        try:
            image = Image.open(BytesIO(response.content))
            if DEBUG_MODE:
                app.logger.info(f"验证码图片格式: {image.format}, 大小: {image.size}, 模式: {image.mode}")
        except Exception as e:
            # 调试：保存原始数据
            if DEBUG_MODE:
                timestamp = int(time.time())
                error_path = os.path.join(log_dir, f"captcha_error_{timestamp}.bin")
                with open(error_path, "wb") as f:
                    f.write(response.content)
                app.logger.error(f"验证码图片解析失败，原始数据已保存为 {error_path}")
            raise ValueError(f"无法解析验证码图片: {str(e)}")
        
        # 调试：保存验证码图片
        if DEBUG_MODE:
            timestamp = int(time.time())
            image_path = os.path.join(log_dir, f"captcha_{timestamp}.png")
            image.save(image_path)
            app.logger.info(f"验证码图片已保存为 {image_path}")
        
        # 预处理和识别
        img = preprocess_image(image)
        if DEBUG_MODE:
            app.logger.info(f"预处理后张量形状: {img.shape}")
        
        with torch.no_grad():
            outputs = model(img)
        outputs = outputs.view(-1, len(common.captcha_array))
        captcha_text = one_hot.vectotext(outputs)
        if DEBUG_MODE:
            app.logger.info(f"识别的验证码: {captcha_text}")
        return captcha_text
    except Exception as e:
        app.logger.error(f"验证码处理失败: {e}")
        traceback.print_exc()
        raise
# == 验证码获取优化 (结束) ==

# == 登录流程优化 (开始) ==
def login(username, password, session, max_retries=3):
    for attempt in range(max_retries):
        app.logger.info(f"登录尝试 #{attempt+1}/{max_retries}")
        try:
            # 修改：只在DEBUG_MODE下记录详情
            if DEBUG_MODE:
                app.logger.info("获取登录表单数据和公钥...")
            lt, n, e = get_login_form_data_and_public_key(session)
            
            if DEBUG_MODE:
                app.logger.info("加密密码...")
            encrypted_password = encrypt_password(password, n, e)
            
            if DEBUG_MODE:
                app.logger.info("获取验证码...")
            captcha_text = get_captcha(session)
            
            # 根据实际表单结构构建数据
            form_data = {
                'username': username,
                'password': encrypted_password,
                'errors': '0',
                'imageCodeName': captcha_text,
                '_rememberMe': 'on',
                'cryptoType': '1',
                'lt': lt,
                '_eventId': 'submit'
            }
            
            # 详细记录表单数据
            if DEBUG_MODE:
                app.logger.info(f"准备提交的完整表单数据: {form_data}")
                app.logger.info(f"提交登录请求到: {LOGIN_PAGE}")
                
            response = session.post(
                LOGIN_PAGE, 
                data=form_data, 
                params={'service': SERVICE_URL}, 
                timeout=15,
                allow_redirects=False
            )
            
            if DEBUG_MODE:
                app.logger.info(f"登录响应状态码: {response.status_code}")
            
            # 检查登录是否成功
            if response.status_code == 302:
                location = response.headers.get('location', '')
                if DEBUG_MODE:
                    app.logger.info(f"重定向到: {location}")
                if SERVICE_URL in location:
                    app.logger.info("登录成功")
                    return True, "登录成功"
                else:
                    app.logger.warning(f"重定向到非服务URL: {location}")
            else:
                if DEBUG_MODE:
                    app.logger.warning(f"非302响应状态码: {response.status_code}")
            
            # 分析错误信息
            if DEBUG_MODE:
                app.logger.info("分析登录响应中的错误信息...")
            soup = BeautifulSoup(response.text, 'html.parser')
            error_div = soup.find('div', id='msg')
            if error_div:
                error_text = error_div.get_text(strip=True)
                app.logger.info(f"提取的错误信息: {error_text}")
                if '验证码' in error_text:
                    app.logger.warning(f"验证码错误，重试 {attempt+1}/{max_retries}")
                    continue
                elif '密码' in error_text or '账号' in error_text:
                    return False, "用户名或密码错误"
            
            # 如果无法识别特定错误，尝试查找其他错误元素
            errors = []
            for div in soup.find_all('div', class_=re.compile('error')):
                errors.append(div.get_text(strip=True))
            
            if errors:
                if DEBUG_MODE:
                    app.logger.warning(f"发现的错误信息: {', '.join(errors)}")
                return False, " | ".join(errors)
            else:
                app.logger.warning("响应中未找到明显的错误信息")
            
            # 调试：保存响应内容
            if DEBUG_MODE:
                timestamp = int(time.time())
                response_path = os.path.join(log_dir, f"login_response_{timestamp}.html")
                with open(response_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                app.logger.info(f"登录响应已保存为 {response_path}")
            
            return False, "登录失败，未知错误"
        except Exception as e:
            app.logger.error(f"登录尝试 {attempt+1} 失败: {e}")
            traceback.print_exc()
            if attempt == max_retries - 1:
                return False, f"登录失败: {str(e)}"
    
    return False, "达到最大重试次数"
# == 登录流程优化 (结束) ==

# == API端点优化 (开始) ==
@app.route('/vpwd', methods=['POST'])
def vpwd():
    if DEBUG_MODE:
        app.logger.info("收到 /vpwd 请求")
    try:
        data = request.get_json()
        if not data or 'username' not in data or 'password' not in data:
            app.logger.error("缺少用户名或密码")
            return jsonify({'result': False, 'success': False, 'message': '缺少用户名或密码'}), 400
        
        app.logger.info(f"验证用户名: {data['username']}")
        
        if model is None:
            app.logger.error("模型未加载")
            return jsonify({'result': False, 'success': False, 'message': '模型未加载'}), 500
        
        if DEBUG_MODE:
            app.logger.info("初始化会话...")
        
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive',
            'Referer': LOGIN_PAGE
        })
        if DEBUG_MODE:
            app.logger.debug(f"会话头信息: {session.headers}")
            app.logger.info("开始登录过程...")
        success, message = login(data['username'], data['password'], session)
        app.logger.info(f"登录结果: {'成功' if success else '失败'}, 消息: {message}")
        return jsonify({'result': success, 'success': True, 'message': message})
    except Exception as e:
        app.logger.exception("处理请求时发生异常")
        return jsonify({
            'result': False,
            'success': False,
            'message': f'服务器错误: {str(e)}' if DEBUG_MODE else '服务器内部错误'
        }), 500
# == API端点优化 (结束) ==

# == 响应日志优化 (开始) ==
@app.after_request
def log_response(response):
    """记录响应的基本信息"""
    # 修改：只在DEBUG_MODE下记录响应详情
    if DEBUG_MODE:
        app.logger.info(f"响应状态码: {response.status_code}")
        app.logger.info(f"响应头: {dict(response.headers)}")
    return response
# == 响应日志优化 (结束) ==

if __name__ == '__main__':
    app.logger.info(f"启动应用，监听端口: 8090")
    app.run(debug=False, port=8090, use_reloader=False)