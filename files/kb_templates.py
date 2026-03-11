"""
知识库模板 - 快速创建常用知识库
使用方法：python kb_templates.py
"""
import requests
import json
from typing import List, Dict

# 配置
API_BASE = "http://localhost:8000"

class KnowledgeBaseTemplate:
    """知识库模板基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.kb_id = None

    def create(self) -> str:
        """创建知识库"""
        response = requests.post(
            f"{API_BASE}/knowledge-base",
            params={"name": self.name, "description": self.description}
        )
        self.kb_id = response.json()["kb_id"]
        print(f"✅ 创建知识库: {self.name} (ID: {self.kb_id})")
        return self.kb_id

    def add_documents(self, documents: List[str]):
        """批量添加文档"""
        for i, doc in enumerate(documents, 1):
            response = requests.post(
                f"{API_BASE}/knowledge-base/{self.kb_id}/document",
                params={"content": doc}
            )
            print(f"   📄 添加文档 {i}/{len(documents)}")

    def setup(self):
        """子类实现：定义知识库内容"""
        raise NotImplementedError


# ==================== 模板 1: 产品 FAQ ====================

class ProductFAQTemplate(KnowledgeBaseTemplate):
    """产品常见问题知识库"""

    def __init__(self):
        super().__init__(
            name="产品 FAQ",
            description="产品使用常见问题解答"
        )

    def setup(self):
        documents = [
            """
            Q: 如何重置密码？
            A: 1. 点击登录页面的"忘记密码"链接
               2. 输入注册邮箱
               3. 查收邮件中的重置链接
               4. 设置新密码（至少 8 位，包含字母和数字）
            """,
            """
            Q: 如何升级到专业版？
            A: 1. 登录账户后进入"设置"页面
               2. 点击"订阅管理"
               3. 选择"专业版"套餐
               4. 完成支付即可立即生效
            """,
            """
            Q: 支持哪些支付方式？
            A: 我们支持以下支付方式：
               - 信用卡（Visa、MasterCard、American Express）
               - 支付宝
               - 微信支付
               - PayPal
            """,
            """
            Q: 数据安全如何保障？
            A: 我们采用多重安全措施：
               - 数据传输使用 TLS 1.3 加密
               - 数据存储采用 AES-256 加密
               - 定期安全审计和渗透测试
               - 符合 ISO 27001 和 SOC 2 标准
            """,
            """
            Q: 如何导出数据？
            A: 1. 进入"设置" > "数据管理"
               2. 点击"导出数据"
               3. 选择导出格式（JSON、CSV、Excel）
               4. 下载生成的文件
               注意：导出可能需要几分钟，完成后会发送邮件通知
            """
        ]

        self.create()
        self.add_documents(documents)
        print(f"✅ 产品 FAQ 知识库创建完成！")


# ==================== 模板 2: 技术文档 ====================

class TechnicalDocsTemplate(KnowledgeBaseTemplate):
    """技术文档知识库"""

    def __init__(self):
        super().__init__(
            name="技术文档",
            description="API 文档和开发指南"
        )

    def setup(self):
        documents = [
            """
            # API 认证
            所有 API 请求需要在 Header 中包含认证令牌：
            ```
            Authorization: Bearer YOUR_API_KEY
            ```

            获取 API Key：
            1. 登录开发者控制台
            2. 进入"API 密钥"页面
            3. 点击"生成新密钥"
            4. 妥善保管密钥，不要泄露
            """,
            """
            # 创建用户 API
            POST /api/v1/users

            请求体：
            {
              "username": "string",
              "email": "string",
              "password": "string"
            }

            响应：
            {
              "user_id": "uuid",
              "username": "string",
              "created_at": "timestamp"
            }

            错误码：
            - 400: 参数错误
            - 409: 用户名或邮箱已存在
            - 500: 服务器错误
            """,
            """
            # 速率限制
            API 调用受以下限制：
            - 免费版：100 次/小时
            - 基础版：1000 次/小时
            - 专业版：10000 次/小时
            - 企业版：无限制

            超出限制返回 429 状态码，Header 包含：
            - X-RateLimit-Limit: 限制次数
            - X-RateLimit-Remaining: 剩余次数
            - X-RateLimit-Reset: 重置时间戳
            """,
            """
            # Webhook 配置
            接收实时事件通知：

            1. 在控制台配置 Webhook URL
            2. 选择订阅的事件类型
            3. 验证 Webhook 签名：
               ```python
               import hmac
               import hashlib

               def verify_signature(payload, signature, secret):
                   expected = hmac.new(
                       secret.encode(),
                       payload.encode(),
                       hashlib.sha256
                   ).hexdigest()
                   return hmac.compare_digest(expected, signature)
               ```
            """,
            """
            # SDK 使用示例

            Python:
            ```python
            from myapi import Client

            client = Client(api_key="YOUR_KEY")
            user = client.users.create(
                username="john",
                email="john@example.com"
            )
            print(user.id)
            ```

            JavaScript:
            ```javascript
            const client = new MyAPIClient({ apiKey: 'YOUR_KEY' });
            const user = await client.users.create({
              username: 'john',
              email: 'john@example.com'
            });
            console.log(user.id);
            ```
            """
        ]

        self.create()
        self.add_documents(documents)
        print(f"✅ 技术文档知识库创建完成！")


# ==================== 模板 3: 客服话术 ====================

class CustomerServiceTemplate(KnowledgeBaseTemplate):
    """客服话术知识库"""

    def __init__(self):
        super().__init__(
            name="客服话术",
            description="标准客服应答话术和处理流程"
        )

    def setup(self):
        documents = [
            """
            场景：用户投诉产品质量问题

            标准话术：
            "非常抱歉给您带来不好的体验。我们非常重视产品质量，请您详细描述遇到的问题，我会立即为您处理。"

            处理流程：
            1. 表达歉意，安抚情绪
            2. 详细记录问题（截图、订单号等）
            3. 判断问题类型：
               - 产品缺陷 → 申请退换货
               - 使用不当 → 提供使用指导
               - 物流损坏 → 联系物流索赔
            4. 提供解决方案并跟进
            5. 24 小时内回访确认
            """,
            """
            场景：用户要求退款

            标准话术：
            "我理解您的需求。请问是什么原因需要退款呢？我们会根据退款政策为您处理。"

            退款政策：
            - 7 天无理由退款（未使用）
            - 质量问题随时退款
            - 已使用产品按比例退款
            - 虚拟商品不支持退款

            处理流程：
            1. 确认退款原因
            2. 核实订单状态
            3. 审核退款资格
            4. 发起退款流程
            5. 3-5 个工作日到账
            """,
            """
            场景：用户咨询优惠活动

            标准话术：
            "感谢您的关注！目前我们有以下优惠活动：[具体活动内容]。您可以通过 [参与方式] 参加。"

            注意事项：
            - 主动告知活动时间和规则
            - 提醒优惠券使用条件
            - 推荐适合用户的活动
            - 记录用户偏好，用于精准营销
            """,
            """
            场景：用户反馈 Bug

            标准话术：
            "非常感谢您的反馈！这对我们改进产品非常重要。请您提供以下信息，我会立即提交给技术团队：
            1. 问题出现的具体步骤
            2. 您使用的设备和系统版本
            3. 问题截图或录屏（如有）"

            处理流程：
            1. 详细记录 Bug 信息
            2. 判断严重程度（P0-P3）
            3. 提交工单给技术团队
            4. 告知用户预计修复时间
            5. 修复后主动通知用户
            6. 赠送补偿（根据影响程度）
            """,
            """
            场景：用户咨询功能使用

            标准话术：
            "我很乐意帮您解答。[功能名称] 的使用方法是：[详细步骤]。如果还有疑问，我可以为您提供视频教程或远程协助。"

            技巧：
            - 使用简单易懂的语言
            - 分步骤说明，避免一次性信息过多
            - 提供多种学习资源（文档、视频、示例）
            - 询问是否需要进一步帮助
            - 记录常见问题，优化产品设计
            """
        ]

        self.create()
        self.add_documents(documents)
        print(f"✅ 客服话术知识库创建完成！")


# ==================== 模板 4: 公司政策 ====================

class CompanyPolicyTemplate(KnowledgeBaseTemplate):
    """公司政策知识库"""

    def __init__(self):
        super().__init__(
            name="公司政策",
            description="公司规章制度和政策文档"
        )

    def setup(self):
        documents = [
            """
            # 隐私政策

            我们承诺保护用户隐私，遵循以下原则：

            1. 数据收集
               - 仅收集必要的个人信息
               - 明确告知收集目的
               - 获得用户明确同意

            2. 数据使用
               - 仅用于声明的目的
               - 不出售或出租用户数据
               - 不用于未经授权的营销

            3. 数据保护
               - 采用行业标准加密技术
               - 定期安全审计
               - 员工签署保密协议

            4. 用户权利
               - 查看个人数据
               - 修改或删除数据
               - 撤回授权同意
               - 投诉和申诉
            """,
            """
            # 服务条款

            使用本服务即表示您同意以下条款：

            1. 账户责任
               - 您对账户安全负责
               - 不得共享账户
               - 发现异常立即通知我们

            2. 禁止行为
               - 发布违法内容
               - 侵犯他人权益
               - 恶意攻击系统
               - 滥用服务资源

            3. 知识产权
               - 服务内容归我们所有
               - 用户内容归用户所有
               - 授予我们使用许可

            4. 免责声明
               - 服务按"现状"提供
               - 不保证无中断或无错误
               - 不对间接损失负责
            """,
            """
            # 退款政策

            我们提供灵活的退款政策：

            1. 全额退款条件
               - 7 天内未使用服务
               - 服务存在重大缺陷
               - 我们单方面终止服务

            2. 部分退款条件
               - 已使用部分服务
               - 按使用时长比例退款
               - 扣除手续费（5%）

            3. 不予退款情况
               - 违反服务条款被封禁
               - 超过退款期限
               - 虚拟商品已消费

            4. 退款流程
               - 提交退款申请
               - 审核（1-3 个工作日）
               - 原路退回（3-5 个工作日）
            """,
            """
            # 内容审核规范

            我们对用户生成内容进行审核：

            1. 禁止内容
               - 违法违规信息
               - 色情暴力内容
               - 虚假误导信息
               - 侵权盗版内容
               - 垃圾广告信息

            2. 审核机制
               - AI 自动审核
               - 人工复审
               - 用户举报
               - 定期抽查

            3. 处理措施
               - 删除违规内容
               - 警告通知
               - 限制功能
               - 封禁账户

            4. 申诉流程
               - 提交申诉说明
               - 提供证明材料
               - 7 天内答复
               - 恢复或维持原判
            """
        ]

        self.create()
        self.add_documents(documents)
        print(f"✅ 公司政策知识库创建完成！")


# ==================== 模板 5: 行业知识 ====================

class IndustryKnowledgeTemplate(KnowledgeBaseTemplate):
    """行业知识库（示例：AI 行业）"""

    def __init__(self):
        super().__init__(
            name="AI 行业知识",
            description="人工智能行业基础知识和术语"
        )

    def setup(self):
        documents = [
            """
            # 大语言模型（LLM）

            定义：基于 Transformer 架构，在海量文本数据上训练的深度学习模型。

            代表模型：
            - GPT 系列（OpenAI）
            - Claude（Anthropic）
            - Gemini（Google）
            - LLaMA（Meta）

            核心能力：
            - 文本生成
            - 问答对话
            - 代码编写
            - 翻译总结
            - 推理分析

            技术特点：
            - 参数规模：数十亿到数千亿
            - 训练数据：TB 级文本语料
            - 涌现能力：规模增大带来质变
            """,
            """
            # RAG（检索增强生成）

            定义：结合信息检索和生成式 AI 的技术，通过检索外部知识库增强 LLM 回答准确性。

            工作流程：
            1. 用户提问
            2. 向量化查询
            3. 检索相关文档
            4. 构建增强提示词
            5. LLM 生成回答

            优势：
            - 减少幻觉（hallucination）
            - 提供可溯源的答案
            - 支持实时知识更新
            - 降低模型训练成本

            应用场景：
            - 企业知识库问答
            - 客服机器人
            - 文档助手
            - 代码搜索
            """,
            """
            # Prompt Engineering（提示词工程）

            定义：设计和优化输入提示词，以获得更好的 AI 输出结果。

            核心技巧：
            1. 明确指令：清晰描述任务
            2. 提供示例：Few-shot learning
            3. 角色设定：让 AI 扮演专家
            4. 分步思考：Chain-of-Thought
            5. 输出格式：指定结构化输出

            示例：
            差：写一篇文章
            好：你是一位科技记者，请写一篇 800 字的文章，介绍 AI 在医疗领域的应用，包括诊断、药物研发和个性化治疗三个方面，使用通俗易懂的语言。
            """,
            """
            # 向量数据库

            定义：专门存储和检索高维向量的数据库，用于语义搜索和相似度匹配。

            主流产品：
            - Pinecone：托管服务，易用
            - Weaviate：开源，功能丰富
            - Milvus：高性能，可扩展
            - ChromaDB：轻量级，适合原型

            核心概念：
            - Embedding：文本转向量
            - 相似度：余弦/欧氏距离
            - 索引：HNSW、IVF 等
            - 过滤：元数据筛选

            应用：
            - 语义搜索
            - 推荐系统
            - 去重检测
            - 异常检测
            """,
            """
            # AI Agent

            定义：能够感知环境、自主决策并执行任务的智能体。

            核心组件：
            1. 感知：接收输入（文本、图像等）
            2. 规划：制定行动计划
            3. 执行：调用工具完成任务
            4. 记忆：存储和检索信息
            5. 反思：评估和改进

            能力层级：
            - L1：单轮对话
            - L2：多轮对话
            - L3：工具调用
            - L4：多步规划
            - L5：自主学习

            应用案例：
            - 代码助手（GitHub Copilot）
            - 研究助手（Perplexity）
            - 自动化测试
            - 数据分析
            """
        ]

        self.create()
        self.add_documents(documents)
        print(f"✅ AI 行业知识库创建完成！")


# ==================== 主函数 ====================

def main():
    print("=" * 50)
    print("知识库模板快速创建工具")
    print("=" * 50)
    print("\n可用模板：")
    print("1. 产品 FAQ - 常见问题解答")
    print("2. 技术文档 - API 和开发指南")
    print("3. 客服话术 - 标准应答流程")
    print("4. 公司政策 - 规章制度文档")
    print("5. 行业知识 - AI 领域知识（示例）")
    print("0. 创建所有模板")
    print()

    choice = input("请选择模板编号（0-5）: ").strip()

    templates = {
        "1": ProductFAQTemplate,
        "2": TechnicalDocsTemplate,
        "3": CustomerServiceTemplate,
        "4": CompanyPolicyTemplate,
        "5": IndustryKnowledgeTemplate,
    }

    if choice == "0":
        print("\n开始创建所有知识库...\n")
        for template_class in templates.values():
            template = template_class()
            template.setup()
            print()
    elif choice in templates:
        print(f"\n开始创建知识库...\n")
        template = templates[choice]()
        template.setup()
    else:
        print("❌ 无效选择")
        return

    print("\n" + "=" * 50)
    print("✅ 完成！现在可以在对话中使用这些知识库了")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ 错误：无法连接到后端服务")
        print("请确保后端服务已启动：python main_enhanced.py")
    except Exception as e:
        print(f"\n❌ 错误：{e}")
