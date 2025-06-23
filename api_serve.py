# Pydantic 是一个流行的库，它可以帮助我们定义数据模型并自动进行数据验证。
# 在Pydantic中，BaseModel是一个核心概念，它用于定义数据模型和验证输入数据。
from pydantic import BaseModel
# FastAPI 是一个高性能、易用且现代的 Python Web 框架，
# 它通过使用最新的 Python 特性和异步编程，提供了快速开发 Web API 的能力。
from fastapi import FastAPI
# CORS（Cross-Origin Resource Sharing）是一种W3C 规范，
# 它定义了一种浏览器和服务器交互的方式来确定是否允许跨源请求。
# 在当下web开发环境下，前后端分离开发是一个比较主流的架构模式。那么因为不再是模板化开发，
# 以至于前后端有可能不在一个域下（即服务器的域名是www.a.com, 客户端域名是 www.b.com),
# 这就可能造成了浏览器跨域禁止的问题。例如：
# 当您发出跨源请求时，请求-响应过程如下：
#   
#      前端web服务                                                     后端服务
#      www.a.com                                                      www.b.com
#               |   Access-Control-Allow-Origin:https://www.a.com      |
#               |  ------------------------------------------------>   |
#               |           服务端判断 Origin 为预期，返回请求内容          ｜
#               |  ✅ <----------------------------------------------   |
#                                                                      ｜
#                                                                      ｜
#                                                                      ｜
#      非预期服务                                                        ｜
#      或者恶意网站                                                      ｜
#      www.d.com                                                       ｜
#               |               Origin:https://www.d.com               |
#               |  ------------------------------------------------>   |
#               |           服务端判断 Origin 为非预期                    ｜
#               ｜      无 Access-Control-Allow-Origin，拒绝访问         ｜
#               |  ❌ <----------------------------------------------   |
#
#
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from agent import ToolsGraph
from pools import BrowserPool, CrawlerPool

browser_pool = BrowserPool(pool_size=1)
crawler_pool = CrawlerPool(pool_size=1)
graph = ToolsGraph(browser_pool, crawler_pool, engine="sougou")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup：可选预热
    await browser_pool._create_browser_instance(headless=True)
    await crawler_pool._get_instance()
    print("✅ Browser pool initialized.")

    # yield 之前（包括 yield 语句）：
    #   定义在应用启动前执行的逻辑（代码）。这意味着在应用开始接收请求之前，这些代码只会被执行一次。
    # yield 之后：
    #   定义在应用关闭时应执行的逻辑。在这种情况下，这段代码将在应用处理可能的多次请求后执行一次。
    yield  # 应用运行中，等待请求

    # shutdown：清理资源
    await browser_pool.cleanup()
    await crawler_pool.cleanup()
    print("✅ Browser pool cleaned up.")

# FastAPI 应用使用 lifespan 参数（异步上下文管理器）来定义启动和关闭的逻辑
app = FastAPI(lifespan=lifespan)

# 可以通过 add_middleware 引入多个中间件，注意遵循 后进先执行 的原则
# 
# CORSMiddleware 中间件支持以下参数：
#   allow_origins - 一个允许跨域请求的源列表
#   allow_origin_regex - 一个正则表达式字符串，匹配的源允许跨域请求
#   allow_methods - 一个允许跨域请求的 HTTP 方法列表
#   allow_headers - 一个允许跨域请求的 HTTP 请求头列表
#   allow_credentials - 指示跨域请求支持 cookies
#     None of allow_origins, allow_methods and allow_headers can be set to ['*'] 
#     if allow_credentials is set to True. All of them must be explicitly specified.
#   expose_headers - 指示可以被浏览器访问的响应头
#   max_age - 设定浏览器缓存 CORS 响应的最长时间，单位是秒。默认为 600。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 在Pydantic中，BaseModel是一个抽象基类，用于定义数据模型。
# 它提供了一种简单而强大的方法来描述数据的结构和验证数据的有效性。
# 使用BaseModel可以帮助我们减少手动验证代码的编写，提高代码的可维护性。
# QueryRequest 对象初始化的时候，Pydantic会去判断（参数）数据是否合法。
class QueryRequest(BaseModel):
    question: str


@app.post("/search")
async def search(query: QueryRequest):
    result = await graph.run(query.question)
    return {"data": result}

if __name__ == "__main__":
    # Uvicorn 是一个异步 Web 服务器网关接口（ASGI）服务器
    # FastAPI 是一个现代化的高性能 Web 框架，它使用 Python 的异步编程特性来提高 Web 应用程序的性能。
    # 而 Uvicorn 则是一个基于 uvloop 和 httptools 实现的高性能 ASGI 服务器，
    # 可以实现异步处理 HTTP 请求。FastAPI 使用 Uvicorn 作为其默认的 Web 服务器，
    # 是因为 Uvicorn 是一个非常快速、可靠且易于使用的 ASGI 服务器，
    # 可以在处理大量并发连接时保持稳定和高效。
    # 此外，Uvicorn 还支持 WebSocket 和 HTTP/2 等新特性，符合 FastAPI 提倡的现代 Web 开发理念。
    # 因此，使用 Uvicorn 作为 FastAPI 的 Web 服务器是一个很好的选择。
    import uvicorn
    port = 8000
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
