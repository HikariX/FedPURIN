import numpy as np
import base64
from io import BytesIO

def save_params(filename, param_array):
    """安全保存ndarray到文件（二进制模式）"""
    buffer = BytesIO()
    np.save(buffer, param_array)
    buffer.seek(0)

    # Base64编码为字节串
    b64_bytes = base64.b64encode(buffer.read())

    # 二进制模式追加写入
    with open(filename, 'ab') as f:  # 'ab' = 追加二进制
        f.write(b64_bytes)
        f.write(b'\n')  # 用字节串写换行符


def load_params(filename):
    """从文件加载ndarray列表（二进制模式）"""
    arrays = []

    with open(filename, 'rb') as f:  # 'rb' = 读取二进制
        for line in f:
            # 去除二进制换行符
            b64_bytes = line.strip()

            try:
                # 直接解码二进制数据
                byte_data = base64.b64decode(b64_bytes)
                buffer = BytesIO(byte_data)
                array = np.load(buffer)
                arrays.append(array)
            except Exception as e:
                print(f"加载错误: {e}") # 调试信息
                print(f"问题数据开头: {b64_bytes[:20]}")
                raise

    return arrays