import asyncio
import atexit
# Semaphore 是一个信号量，控制对共享资源的访问（如网络请求、数据库连接）或者限制协程的并行执行数量。
#   - async with 是最常用的方式，自动管理资源的获取和释放。
#   - 使用 acquire() 和 release() 手动控制。
from asyncio import Queue, Semaphore
from contextlib import asynccontextmanager
# 市面上已经有很多爬虫框架了，为什么还要选择 Crawl4AI 呢？
#   - 告别传统繁琐的 CSS Selector 和 XPath
#     - 传统爬虫需要你手动编写 CSS Selector 或 XPath 来定位网页元素，这对于非技术人员来说简直是噩梦。
#     - Crawl4AI 提供了两种强大的武器：
#       - 基于 CSS/XPath 的结构化提取： 即使你不懂 CSS Selector 和 XPath，也可以通过预定义的 schema 轻松提取结构化数据
#       - LLM 驱动的智能提取： 结合大语言模型，只需要用自然语言描述你想要提取的内容，Crawl4AI 就能利用 LLM 的强大能力自动帮你完成！
#
# Crawl4AI doc：
#   - https://www.aidoczh.com/crawl4ai/core/quickstart/index.html
#   - https://www.aidoczh.com/crawl4ai/api/async-webcrawler/index.html
#
# Crawl4AI 的核心任务是使网页爬取和数据提取变得简单高效，特别是为大语言模型（LLMs）和 AI 应用提供支持。
# Crawl4AI automatically converts the HTML into Markdown.
# Crawl4AI 默认支持 HTTP(S) 协议，但同时也考虑到网络爬虫的伦理和法律问题，提供了 robots.txt​ 协议的支持，
# 可以通过设置 check_robots_txt=True​ 来遵守网站的爬取规则，避免对网站造成不必要的负担。
#
# AsyncWebCrawler：异步爬虫工具
#   - 可通过 BrowserConfig 和 CrawlerRunConfig 配置浏览器和运行设置
#   - CacheMode​ 枚举类型来控制缓存行为。你可以在 CrawlerRunConfig​ 中指定缓存模式
#   - Automatic HTML-to-Markdown conversion via DefaultMarkdownGenerator (supports optional filters)
#   - 多种提取策略（基于LLM或"传统"基于CSS/XPath）
#     - JSON CSS Extraction：使用 CSS 选择器从结构化的网页中提取数据，速度快、效率高。
#     - JSON XPath Extraction：使用 XPath 表达式从 XML 或 HTML 文档中提取数据，更加灵活。
#     - LLM Extraction：结合大型语言模型，从非结构化的网页中提取信息，适用于处理复杂或语义化的内容。
#
# Crawl4AI的爬虫可以通过两个主要类进行高度自定义：
#   - BrowserConfig: 控制浏览器行为（无头模式或完整UI、用户代理、JavaScript开关等）。
#   - CrawlerRunConfig: 控制每次爬取的运行方式（缓存、提取、超时、钩子等）。
#   - arun(url, config=CrawlerRunConfig) 是用于单页面爬取的主要方法。
#   - arun_many(urls, config=CrawlerRunConfig) 处理多个URL的并发请求。
#
# 调度器（Dispatcher）：Crawl4AI 使用调度器来管理并发任务，主要有两种调度器
#  - MemoryAdaptiveDispatcher（默认）​：根据系统内存使用情况动态调整并发数量，避免内存溢出。
#    - memory_threshold_percent​：内存使用阈值，当内存使用超过该值时，调度器会暂停任务。
#    - max_session_permit​：允许的最大并发任务数。
#    - check_interval​：检查内存使用情况的间隔时间（秒）。
#  - SemaphoreDispatcher​：使用信号量来控制并发数量，简单直接。
#    - semaphore_count​：允许的最大并发任务数。
#
# 深度爬取功能：Crawl4AI 支持深度爬取，可以通过配置 deep_crawl_strategy​ 参数来控制爬取的深度和范围。
#  - BFSDeepCrawlStrategy​（广度优先搜索）：逐层遍历网页。
#  - ​DFSDeepCrawlStrategy​（深度优先搜索）：沿着一条路径深入挖掘，直到无法再深入为止。
#  - BestFirstCrawlingStrategy​：基于评分函数来决定下一个要爬取的链接，优先爬取最有价值的页面。
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode


# 当我们用类来创建上下文管理器时，必须保证这个类包括方法”__enter__()”和方法“__exit__()”。
# 其中，方法“__enter__()”返回需要被管理的资源，方法“__exit__()”里通常会存在一些释放、清理资源的操作。
# 当然与 'async with' 异步执行，那么就需要实现异步上下文管理器（需要实现 __aenter__ 和 __aexit__ 方法）
class CrawlerInstance:
    def __init__(self):
        # a headless browser (Chromium by default)
        self.browser_config = BrowserConfig(headless=True, verbose=False)
        self.run_config = CrawlerRunConfig(cache_mode=CacheMode.ENABLED, stream=False)
        self.crawler = None

    async def __aenter__(self):
        self.crawler = AsyncWebCrawler(config=self.browser_config)
        return self

    # with真正强大之处是它可以处理异常
    # __aexit__ 方法的三个参数：exc_type, exc_val 和 exc_tb，这些参数在异常处理中相当有用。
    # with 后面的代码块抛出任何异常时，__aexit__() 方法被执行
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.crawler:
            await self.crawler.close()

    async def run(self, urls: list[str]) -> list[dict]:
        responses = await self.crawler.arun_many(urls=urls, config=self.run_config)

        results = []
        for r in responses:
            if r.success:
                results.append({"url": r.url, "content": r.markdown})
        return results

class CrawlerPool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.pool = Queue(maxsize=pool_size)
        self.lock = Semaphore(pool_size)
        self.instances = []
        atexit.register(lambda: asyncio.run(self.cleanup()))

    # 在任何一门编程语言中，文件的输入输出、数据库的连接断开等，都是很常见的资源管理操作。
    # 但资源都是有限的，在写程序时，我们必须保证这些资源在使用过后得到释放，不然就容易造成资源泄露，
    # 轻者使得系统处理缓慢，重则会使系统崩溃。为了解决这个问题，不同的编程语言都引入了不同的机制。
    # 而在Python中，对应的解决方式便是上下文管理器（context manager）。
    # 上下文管理器，能够帮助你自动分配并且释放资源，其中最典型的应用便是with语句。
    @asynccontextmanager
    async def get_crawler(self):
        # Semaphore 与 async with 结合是最常用的方式，自动管理有限资源的获取和释放
        async with self.lock:
            crawler = await self._get_instance()
            try:
                # yield 的作用是使所在的函数变成一个生成器，在调用 next() 时执行 yield 语句返回一个值后就中断了，
                # 再次调用 next() 的时候，函数接着上次中断地方继续执行，并在遇到 yield 后再次中断。如果执行到最后
                # 没有 yield 语句了，就会抛出一个 StopIteration 的异常。
                #
                # python中yield的用法详解——最简单，最清晰的解释：https://blog.csdn.net/mieleizhi0522/article/details/82142856
                #
                # python with 后面的对象必须跟一个 enter() 方法和一个 exit() 方法
                # with 语句执行时，该对象的 enter() 方法被调用，enter() 方法的返回值将被赋值给 as 后面的变量。
                #
                # asynccontextmanager 装饰器实现了一个异步上下文管理器（一个具有 __aenter__ 和 __aexit__ 方法的对象），
                # 以便配合 aysnc with：
                #   - yield 之前的语句（包括 yield crawler）在 __aenter__ 方法中执行
                #   - yield之后的语句在 __aexit__ 方法中执行
                yield crawler
            finally:
                await self._release_instance(crawler)

    async def _get_instance(self):
        if self.pool.empty():
            crawler = await CrawlerInstance().__aenter__()
            self.instances.append(crawler)
        else:
            crawler = await self.pool.get()
        return crawler

    async def _release_instance(self, crawler: CrawlerInstance):
        if self.pool.qsize() < self.pool_size:
            await self.pool.put(crawler)

    async def cleanup(self):
        # asyncio.gather(task1, task2, ...):
        #   - 并发执行多个线程
        #   - 等待多个线程完成
        #   - 提取协程返回值：将各协程返回值合并到一个列表中返回
        #
        # 比如，如果我们有两个函数func1和func2，我们可以使用 asyncio.gather(func1(), func2())
        # 来并发执行它们。这个函数会一直运行，直到所有的协程都执行完成。
        #
        # '*' 号运算符将 list 转换成参数：list 中每个元素作为单独参数传递给函数
        await asyncio.gather(*[
            crawler.__aexit__(None, None, None)
            for crawler in self.instances
        ])
