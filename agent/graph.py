# langchain 把 chat 消息分成了这几种：AIMessage、HumanMessage、SystemMessage 和 ChatMessage。
# HumanMessage 是用户输入的消息，AIMessage 是大语言模型的消息，SystemMessage 是系统的消息，
# ChatMessage 是一种可以自定义的消息。
from langchain_core.messages import HumanMessage, SystemMessage
# langchain 的 chat models 有：ChatAnthropic、AzureChatOpenAI、ChatVertexAI、ChatOpenAI、
# PromptLayerChatOpenAI 等。
from langchain_openai import ChatOpenAI
# LangGraph 是 LangChain 的一个扩展，旨在通过将步骤建模为图中的边和节点，构建强大且有状态的多角色应用程序。
# 它通过图结构（Graph）实现复杂的动态工作流，尤其擅长与大型语言模型（LLMs）结合，
# 支持循环、持久性、人工干预等核心功能，被视为AI代理开发的“终结者”。
# LangGraph 将代理工作流建模为图形。你可以使用三个关键组件来定义代理的行为：
#   State（状态）：一个共享的数据结构，表示应用程序的当前快照。
#   Nodes（节点）：编码代理逻辑的Python函数。它们接收当前的State作为输入，执行一些计算或副作用，并返回一个更新后的State。
#   Edges（边）：Python函数，基于当前的State决定接下来执行哪个Node。它们可以是条件分支或固定转换。
# 通过组合Nodes和Edges，你可以创建复杂的、循环的工作流。
#
# MemorySaver，这是LangGraph提供的内存检查点保存器，用于在对话过程中保存和恢复状态。
from langgraph.checkpoint.memory import MemorySaver
# START 和 END 是两个特殊节点，START 表示开始节点，接受用户的输入，是整个图的入口，
# END 表示结束节点，执行到它之后就没有后续动作了。
#
# StateGraph 是 LangGraph 中的核心概念。StateGraph 定义状态图。
# 它以图的形式表示代理的工作流，其中图中的每个节点代表流程中的一个步骤。
#
# MessagesState 很简单，仅包含一个 LangChain 格式的消息列表，一般在构造聊天机器人或示例代码时使用，
# 在正式环境中用的并不多，因为大多数应用程序需要的状态比消息列表更为复杂。
from langgraph.graph import END, START, StateGraph, MessagesState
# ToolNode 是一个通用的节点，用于执行单个或多个工具调用。 
# 它接收一个或多个 ToolCall 对象作为输入（通常来自LLM 的输出），然后执行这些工具，
# 并将工具的执行结果作为 ToolMessage 返回。
from langgraph.prebuilt import ToolNode
from typing import Literal
from datetime import datetime
from dotenv import load_dotenv
import os

from agent.tools import WebTools
from .prompt import prompts

# 从 .env 文件读取键值对，并将它们添加到环境变量中
# 主要是 OPENAI_API_KEY、OPENAI_BASE_URL、MODEL_NAME
# EMBEDDING_MODEL_NAME、EMBEDDING_API_KEY、EMBEDDING_BASE_URL
load_dotenv()

def get_datetime_str():
    now = datetime.now()
    datetime_str = now.strftime("%Y-%m-%d %H:%M")
    return datetime_str

class ToolsGraph:

    def __init__(self, browser_pool, crawler_pool, engine):
        self.browser_pool = browser_pool
        self.crawler_pool = crawler_pool
        self.engine = engine
        self.ts_manage = WebTools(browser_pool=self.browser_pool, crawler_pool=crawler_pool, engine=self.engine)
        self.tools = [self.ts_manage.web_search, self.ts_manage.link_parser]
        self.tool_node = ToolNode(self.tools)
        self.llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            streaming=False,
            temperature=0,
        ).bind_tools(self.tools)
        # 定义图状态，这里使用的 MessagesState
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools",  self.tool_node)
        # 起始边（Starting Edge）：作为图的开始
        # 设定入口为 agent
        workflow.add_edge(START, "agent") # 开始节点 --> 大模型节点
        # 条件边（Conditional Edges）：使用函数（通常由LLM提供）来确定调用哪个节点。
        # 条件边：决定是否继续调用工具
        # If the latest message (result) from llm is a tool call -> should_continue routes to tools
        # If the latest message (result) from llm is a not a tool call -> should_continue routes to END
        workflow.add_conditional_edges("agent",  self.should_continue) # 大模型节点 -条件边-> 工具节点
        # 普通边（Normal Edges）：表示一个节点执行完就执行另一个节点。普通边就像是确定了任务执行的顺序。
        # 设置普通边：agent 到 agent
        workflow.add_edge("tools", "agent")
        # 将定义好的图结构编译成可执行的工作流
        # 编译工作流成一个 runnable，通过 invoke 调用
        self.graph = workflow.compile()


    def should_continue(self, state: MessagesState) -> Literal["tools", "__end__"]:
        messages = state['messages']
        # 通过最后一条消息做判断
        last = messages[-1]
        # #判断 models 是否返回 tools 调用，有则告诉调用 tools 节点，否则结束
        return "tools" if last.tool_calls else END

    async def call_model(self, state: MessagesState):
        messages = state["messages"]
        print(messages)
        response = await self.llm.ainvoke(messages)
        return {"messages": [response]}

    async def run(self, question):
        inputs = {"messages": [SystemMessage(content=prompts["web_prompt"] + f"\n当前时间：{get_datetime_str()}"),HumanMessage(content=question)]}
        final_state = await self.graph.ainvoke(inputs)
        for i in final_state["messages"]:
            print(i)
        return final_state["messages"][-1].content
