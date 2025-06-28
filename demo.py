# LangGraph Tutorial: Building Agents with LangChain's Agent Framework：
#   https://www.getzep.com/ai-agents/langgraph-tutorial

from pools import BrowserPool, CrawlerPool
from agent import ToolsGraph
import asyncio
#from langchain.globals import set_verbose
#from langchain.globals import set_debug


async def main():
    # Setting the verbose flag will print out inputs and outputs in a slightly more readable format 
    # and will skip logging certain raw outputs (like the token usage stats for an LLM call) 
    # so that you can focus on application logic.
    #set_verbose(True)

    # Setting the global debug flag will cause all LangChain components with callback 
    # support (chains, models, agents, tools, retrievers) to print the inputs they receive 
    # and outputs they generate. This is the most verbose setting and will fully log raw inputs and outputs.
    #set_debug(True)

    browser_pool = BrowserPool(pool_size=2)
    crawler_pool = CrawlerPool(pool_size=2)
    
    graph = ToolsGraph(browser_pool, crawler_pool, engine="bing")

    await browser_pool._create_browser_instance(headless=True)
    await crawler_pool._get_instance()

    result = await graph.run("巴以冲突最近几天消息")

    await browser_pool.cleanup()
    await crawler_pool.cleanup()

    print("Final result: ", result, "\n")

if __name__ == "__main__":
    asyncio.run(main())
