"""
AI 文本分流处理工具
通过 OpenRouter API 实现文章的多任务并行处理
"""

import streamlit as st
import requests
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# 页面配置
st.set_page_config(
    page_title="AI 文本处理工具",
    page_icon="✨",
    layout="wide"
)

# 常量
BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODELS = [
    "deepseek/deepseek-chat",
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o"
]

# 默认 System Prompts
DEFAULT_PROMPT_A = """你是一位社交媒体文案专家。请根据用户提供的文章，提炼核心内容，生成一段适合发朋友圈的吸睛短文。

要求：
1. 简洁有力，控制在 200 字以内
2. 适当使用 Emoji 增加视觉吸引力
3. 突出文章的核心价值或亮点
4. 语气亲切自然，适合社交分享
5. 可以适当设置悬念或引发好奇"""

DEFAULT_PROMPT_B = """你是一位专业的文字编辑。请对用户提供的文章进行润色和排版。

要求：
1. 修正语法和表达问题
2. 优化文章结构和逻辑
3. 严格使用 Markdown 格式排版：
   - 使用标题层级（#、##、###）
   - 使用列表（有序/无序）组织要点
   - 重点内容使用**加粗**或*斜体*
   - 适当使用引用块（>）
   - 代码或专业术语使用 `行内代码`
4. 保持原文核心意思不变
5. 提升文章的专业性和可读性"""


def fetch_models(api_key: str) -> list:
    """从 OpenRouter 获取可用模型列表"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://ai-text-tool.app",
            "X-Title": "AI-Text-Tool"
        }
        response = requests.get(
            f"{BASE_URL}/models",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        # 提取模型 ID 并排序
        models = [model["id"] for model in data.get("data", [])]
        return sorted(models) if models else DEFAULT_MODELS

    except Exception as e:
        st.sidebar.warning(f"获取模型列表失败: {str(e)}\n使用默认列表")
        return DEFAULT_MODELS


def call_openrouter(api_key: str, model: str, system_prompt: str, user_content: str) -> dict:
    """调用 OpenRouter API"""
    try:
        client = OpenAI(
            base_url=BASE_URL,
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://ai-text-tool.app",
                "X-Title": "AI-Text-Tool"
            }
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7
        )

        return {
            "success": True,
            "content": response.choices[0].message.content
        }

    except Exception as e:
        error_msg = str(e)
        # 解析常见错误
        if "401" in error_msg or "Unauthorized" in error_msg:
            error_msg = "API Key 无效或已过期"
        elif "402" in error_msg or "Payment" in error_msg:
            error_msg = "账户余额不足，请充值"
        elif "429" in error_msg or "rate" in error_msg.lower():
            error_msg = "请求频率过高，请稍后重试"
        elif "timeout" in error_msg.lower():
            error_msg = "请求超时，请重试"

        return {
            "success": False,
            "content": f"错误: {error_msg}"
        }


def process_tasks(api_key: str, model: str, article: str, prompt_a: str, prompt_b: str) -> tuple:
    """并行处理两个任务"""
    results = {"task_a": None, "task_b": None}

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                call_openrouter, api_key, model, prompt_a, article
            ): "task_a",
            executor.submit(
                call_openrouter, api_key, model, prompt_b, article
            ): "task_b"
        }

        for future in as_completed(futures):
            task_name = futures[future]
            results[task_name] = future.result()

    return results["task_a"], results["task_b"]


# ============ 侧边栏配置 ============
st.sidebar.title("⚙️ 配置")

api_key = st.sidebar.text_input(
    "OpenRouter API Key",
    type="password",
    placeholder="sk-or-v1-..."
)

st.sidebar.text_input(
    "Base URL",
    value=BASE_URL,
    disabled=True
)

# 模型选择
if api_key:
    with st.sidebar:
        with st.spinner("正在获取模型列表..."):
            available_models = fetch_models(api_key)
else:
    available_models = DEFAULT_MODELS
    st.sidebar.info("请输入 API Key 以获取完整模型列表")

selected_model = st.sidebar.selectbox(
    "选择模型",
    options=available_models,
    index=0
)

st.sidebar.divider()

# 提示词配置
st.sidebar.subheader("📝 提示词配置")

prompt_a = st.sidebar.text_area(
    "任务 A 提示词（左栏）",
    value=DEFAULT_PROMPT_A,
    height=150,
    help="自定义左侧输出的生成提示词"
)

prompt_b = st.sidebar.text_area(
    "任务 B 提示词（右栏）",
    value=DEFAULT_PROMPT_B,
    height=150,
    help="自定义右侧输出的生成提示词"
)

# 任务标题配置
col_title_a, col_title_b = st.sidebar.columns(2)
with col_title_a:
    title_a = st.text_input("左栏标题", value="📱 朋友圈文案")
with col_title_b:
    title_b = st.text_input("右栏标题", value="📝 Markdown 润色版")

st.sidebar.divider()
st.sidebar.markdown("""
**使用说明**
1. 输入 OpenRouter API Key
2. 选择要使用的模型
3. 自定义两个任务的提示词
4. 在主界面输入文章
5. 点击"开始处理"
6. 使用代码框右上角按钮复制结果
""")


# ============ 主界面 ============
st.title("✨ AI 文本分流处理工具")
st.markdown("将一篇文章同时生成**朋友圈文案**和**Markdown 润色版本**")

# 文章输入
article = st.text_area(
    "请输入要处理的文章",
    height=200,
    placeholder="在此粘贴你的文章内容..."
)

# 处理按钮
col_btn, col_status = st.columns([1, 3])
with col_btn:
    process_btn = st.button("🚀 开始处理", type="primary", use_container_width=True)

# 处理逻辑
if process_btn:
    # 验证输入
    if not api_key:
        st.error("请在侧边栏输入 API Key")
    elif not article.strip():
        st.error("请输入要处理的文章内容")
    else:
        # 开始处理
        with st.spinner("正在处理中，请稍候..."):
            result_wechat, result_markdown = process_tasks(
                api_key, selected_model, article
            )

        st.success("处理完成！")

        # 展示结果 - 左右分栏
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("📱 朋友圈文案")
            if result_wechat["success"]:
                st.code(result_wechat["content"], language=None)
            else:
                st.error(result_wechat["content"])

        with col_right:
            st.subheader("📝 Markdown 润色版")
            if result_markdown["success"]:
                st.code(result_markdown["content"], language=None)
            else:
                st.error(result_markdown["content"])


# 页脚
st.divider()
st.caption("Powered by OpenRouter API | 使用 st.code 展示结果，点击右上角即可复制")
