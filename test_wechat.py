"""
test_wechat.py — 微信发消息功能测试脚本

用法:
    uv run test_wechat.py                        # 测试发给老妈（默认）
    uv run test_wechat.py 导师                   # 测试发给导师
    uv run test_wechat.py 班级群                 # 测试发给班级群
    uv run test_wechat.py 老妈 "自定义消息内容"  # 指定消息
"""
import sys
from config import WECHAT_CONTACTS
from tools import send_wechat_shame_message

def main():
    target  = sys.argv[1] if len(sys.argv) >= 2 else "老妈"
    message = sys.argv[2] if len(sys.argv) >= 3 else "[测试] Cyber-Superego 测试消息，请忽略"

    print(f"联系人映射: {WECHAT_CONTACTS}")
    print(f"目标: {target} → {WECHAT_CONTACTS.get(target, '未找到')}")
    print(f"消息: {message}")
    print("─" * 40)

    result = send_wechat_shame_message.invoke({"target": target, "message": message})
    print(f"结果: {result}")

if __name__ == "__main__":
    main()
