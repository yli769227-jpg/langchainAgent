"""
使用示例 - 演示增强版 Agent 的新功能
"""
import requests
import json
import time

API_BASE = "http://localhost:8000"
API_KEY = "sk-your-api-key-here"  # 替换为你的 API Key

def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def chat(message, session_id="demo", kb_ids=None):
    """发送聊天消息"""
    response = requests.post(
        f"{API_BASE}/chat",
        json={
            "message": message,
            "session_id": session_id,
            "api_key": API_KEY,
            "knowledge_base_ids": kb_ids or []
        }
    )
    result = response.json()
    print(f"🤖 Agent: {result['response']}\n")

    if result.get('steps'):
        print("📋 执行步骤:")
        for step in result['steps']:
            print(f"   - 工具: {step['tool']}")
            print(f"     参数: {step['args']}")
            print(f"     结果: {step['result'][:100]}...")

    return result


# ==================== 示例 1: 基础对话 ====================

def demo_basic_chat():
    print_section("示例 1: 基础对话和工具调用")

    print("👤 用户: 现在几点了？")
    chat("现在几点了？", session_id="demo1")

    time.sleep(1)

    print("\n👤 用户: 计算 sin(45度) * sqrt(2)")
    chat("计算 sin(45度) * sqrt(2)", session_id="demo1")


# ==================== 示例 2: 记忆系统 ====================

def demo_memory_system():
    print_section("示例 2: 记忆系统")

    session_id = "demo_memory"

    # 第一轮对话：建立记忆
    print("👤 用户: 我叫张三，今年 25 岁，是一名软件工程师")
    chat("我叫张三，今年 25 岁，是一名软件工程师", session_id=session_id)

    time.sleep(1)

    print("\n👤 用户: 我喜欢用 Python 和 JavaScript 编程")
    chat("我喜欢用 Python 和 JavaScript 编程", session_id=session_id)

    time.sleep(2)

    # 第二轮对话：测试记忆召回
    print("\n👤 用户: 我之前说过我叫什么名字？")
    chat("我之前说过我叫什么名字？", session_id=session_id)

    time.sleep(1)

    print("\n👤 用户: 根据我的背景，推荐一些适合我的技术书籍")
    chat("根据我的背景，推荐一些适合我的技术书籍", session_id=session_id)

    # 查看记忆
    print("\n📊 查看会话记忆:")
    response = requests.get(f"{API_BASE}/memory/{session_id}?limit=5")
    memories = response.json()["memories"]
    for i, mem in enumerate(memories, 1):
        print(f"{i}. {mem['content'][:80]}...")


# ==================== 示例 3: 知识库 ====================

def demo_knowledge_base():
    print_section("示例 3: 知识库系统")

    # 创建知识库
    print("📚 创建产品知识库...")
    response = requests.post(
        f"{API_BASE}/knowledge-base",
        params={
            "name": "产品手册",
            "description": "公司产品使用手册"
        }
    )
    kb_id = response.json()["kb_id"]
    print(f"✅ 知识库 ID: {kb_id}\n")

    # 添加文档
    print("📄 添加产品文档...")
    documents = [
        """
        产品名称：智能助手 Pro
        版本：2.0

        主要功能：
        1. 智能对话：支持多轮对话，理解上下文
        2. 任务自动化：可以执行各种任务和工具调用
        3. 知识库集成：接入企业知识库，提供准确答案
        4. 多语言支持：支持中文、英文、日语等 10+ 语言

        价格：
        - 基础版：免费
        - 专业版：99 元/月
        - 企业版：联系销售
        """,
        """
        常见问题：

        Q: 如何升级到专业版？
        A: 登录后进入"设置" > "订阅管理"，选择专业版套餐并完成支付。

        Q: 专业版有哪些额外功能？
        A: 专业版包含：无限对话次数、优先响应速度、高级工具访问、API 调用权限。

        Q: 支持团队协作吗？
        A: 企业版支持团队协作，包括共享知识库、权限管理、使用统计等功能。
        """
    ]

    for doc in documents:
        requests.post(
            f"{API_BASE}/knowledge-base/{kb_id}/document",
            params={"content": doc}
        )
    print("✅ 文档添加完成\n")

    # 使用知识库对话
    print("👤 用户: 智能助手 Pro 有哪些功能？")
    chat("智能助手 Pro 有哪些功能？", session_id="demo_kb", kb_ids=[kb_id])

    time.sleep(1)

    print("\n👤 用户: 专业版多少钱？包含什么功能？")
    chat("专业版多少钱？包含什么功能？", session_id="demo_kb", kb_ids=[kb_id])


# ==================== 示例 4: 综合场景 ====================

def demo_comprehensive():
    print_section("示例 4: 综合场景 - 客服助手")

    # 创建客服知识库
    print("📚 创建客服知识库...")
    response = requests.post(
        f"{API_BASE}/knowledge-base",
        params={
            "name": "客服知识库",
            "description": "客服标准话术和处理流程"
        }
    )
    kb_id = response.json()["kb_id"]

    # 添加客服话术
    requests.post(
        f"{API_BASE}/knowledge-base/{kb_id}/document",
        params={"content": """
        退款政策：
        1. 7 天内未使用可全额退款
        2. 已使用按比例退款
        3. 退款 3-5 个工作日到账

        标准话术：
        "我理解您的需求。请问是什么原因需要退款呢？我们会根据退款政策为您处理。"
        """}
    )
    print("✅ 知识库创建完成\n")

    session_id = "customer_service"

    # 模拟客服对话
    print("👤 客户: 你好，我想退款")
    chat("你好，我想退款", session_id=session_id, kb_ids=[kb_id])

    time.sleep(1)

    print("\n👤 客户: 我 5 天前购买的，还没使用过")
    chat("我 5 天前购买的，还没使用过", session_id=session_id, kb_ids=[kb_id])

    time.sleep(1)

    print("\n👤 客户: 大概多久能到账？")
    chat("大概多久能到账？", session_id=session_id, kb_ids=[kb_id])


# ==================== 示例 5: 工具组合使用 ====================

def demo_tool_combination():
    print_section("示例 5: 多工具组合使用")

    print("👤 用户: 今天是几号？然后帮我计算一下距离 2026 年 3 月 1 日还有多少天")
    chat("今天是几号？然后帮我计算一下距离 2026 年 3 月 1 日还有多少天", session_id="demo_tools")

    time.sleep(1)

    print("\n👤 用户: 分析这段文字有多少字：人工智能正在改变世界，从医疗到教育，从交通到娱乐，AI 的应用无处不在。")
    chat("分析这段文字有多少字：人工智能正在改变世界，从医疗到教育，从交通到娱乐，AI 的应用无处不在。", session_id="demo_tools")


# ==================== 主函数 ====================

def main():
    print("\n" + "🤖" * 30)
    print("  LangChain Agent Enhanced - 功能演示")
    print("🤖" * 30)

    # 检查后端连接
    try:
        response = requests.get(f"{API_BASE}/health")
        health = response.json()
        print(f"\n✅ 后端服务正常")
        print(f"   向量存储: {'✅' if health['features']['vector_store'] else '❌'}")
        print(f"   记忆系统: {'✅' if health['features']['memory_system'] else '❌'}")
        print(f"   知识库: {'✅' if health['features']['knowledge_base'] else '❌'}")
    except:
        print("\n❌ 错误：无法连接到后端服务")
        print("请先启动后端：python main_enhanced.py")
        return

    # 检查 API Key
    if API_KEY == "sk-your-api-key-here":
        print("\n⚠️  警告：请在脚本中设置你的 OpenAI API Key")
        print("修改 demo_usage.py 中的 API_KEY 变量")
        return

    print("\n选择演示场景：")
    print("1. 基础对话和工具调用")
    print("2. 记忆系统")
    print("3. 知识库系统")
    print("4. 综合场景（客服助手）")
    print("5. 多工具组合")
    print("0. 运行所有演示")

    choice = input("\n请选择（0-5）: ").strip()

    demos = {
        "1": demo_basic_chat,
        "2": demo_memory_system,
        "3": demo_knowledge_base,
        "4": demo_comprehensive,
        "5": demo_tool_combination,
    }

    if choice == "0":
        for demo_func in demos.values():
            demo_func()
            time.sleep(2)
    elif choice in demos:
        demos[choice]()
    else:
        print("❌ 无效选择")
        return

    print("\n" + "=" * 60)
    print("  ✅ 演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 演示已取消")
    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
