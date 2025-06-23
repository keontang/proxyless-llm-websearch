# Gradio 是专为机器学习设计的轻量级 Python 库，它以简洁直观的方式将机器学习模型与用户界面相结合。
# 利用 Gradio，用户可以轻松地通过图形界面输入数据并查看模型输出。
# Gradio的最大的价值我认为是缩短了算法与应用的距离，人人都能迅速分享与体验项目成果，做 AI Demo 非常有用，
# 不论是分享开源成果，项目汇报，同行交流，甚至是快速做一个产品。
import gradio as gr
# syncAio是Python标准库中提供的异步编程库，基于协程（coroutines）和事件循环（event loop）的概念。
# 通过async/await关键字，它允许开发者编写异步函数和操作，然后通过一个事件循环在单个线程中调度和执行这些操作。
# Asyncio是一个强大的工具，适用于处理异步IO操作，提高程序性能并实现高并发。
# Asyncio 结合 async/await 就能实现复杂的多个任务异步逻辑。
import asyncio

from agent import ToolsGraph
from pools import BrowserPool, CrawlerPool


async def search_answer(question: str, engine: str):
    browser_pool = BrowserPool(pool_size=1)
    crawler_pool = CrawlerPool(pool_size=1)
    graph = ToolsGraph(browser_pool, crawler_pool, engine=engine)
    result = await graph.run(question)
    return result


# 用 sync wrapper 包装 async 函数（Gradio 不直接支持 async）
def sync_search_answer(question, engine):
    # asyncio.run 完成了一些重要的事情，首先创建了一个全新的事件循环。一旦成功创建，
    # 就会接受我们传递给它的任何协程，并运行它直到完成，然后返回结果。
    return asyncio.run(search_answer(question, engine))

# Blocks 是 Gradio 的低级 API，它允许你创建比 Gradio Interfaces 更多的自定义 web 应用程序和演示。
# Blocks 提供了更大的灵活性和对以下方面的控制：
#   (1) 组件布局 
#   (2) 触发函数执行的事件 
#   (3) 数据流（例如，输入可以触发输出，输出又可以触发下一级输出）。
#  Blocks 还提供了将相关演示组合在一起的方法，例如使用标签页。
#
# 启动界面
with gr.Blocks() as demo:
    gr.Markdown("# 🔍 多引擎搜索问答")
    question = gr.Textbox(label="请输入你的问题")
    engine = gr.Radio(["bing", "quark", "baidu", "sougou"], value="bing", label="选择搜索引擎")
    output = gr.Textbox(label="答案")

    btn = gr.Button("提交查询")
    btn.click(fn=sync_search_answer, inputs=[question, engine], outputs=output)

if __name__ == "__main__":
    # 调用launch()方法启动演示
    demo.launch()
