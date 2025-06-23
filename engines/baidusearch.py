from typing import List, Optional
# DOM 是 Document Object Model（文档对象模型）的缩写，定义了访问 HTML 和 XML 文档的标准。
# HTML DOM 将 HTML 文档视作树结构。
#
# XPath 是表示 XML 路径语言。
# CSS（Cascading Style Sheets，层叠样式表）是一种样式表语言，
#   用来为结构化文档（如HTML文档或XML应用）添加样式（字体、间距和颜色等）的。
# 
#
# 一个标签可以命名多个类名字，各类名之间需用空格分隔
# 例如：
#   <div class="red gret background">男孩子</div>
#   red 指定了颜色为红色，background 指定背景填充颜色，gret 指定了高和宽
#
# div 标签中，比较常见的属性是 id 和 class。其作用是让 CSS 或者 JavaScript 找到 DOM 元素并操作。
#   - id 在 CSS 中是以 '#' 开头命名的
#   - class 在 CSS 中是以 '.' 开头命名的
#   - id 是元素的唯一代号，id 具有唯一性，用于区分不同的结构和内容，就像名字，如果一个屋子有2人同名，就会出现混淆
#   - class 是一个样式，多个元素可以使用相同的样式，就像一件衣服，屋子内2人都可以穿相同的衣服
#   - CSS 和 JavaScript 中对某个 class 的操作会反映到每个对应的元素上
#
#
# bs4 是一个HTML/XML的解析器，其主要功能是解析和提取HTML/XML数据。
# 它不仅支持CSS选择器，而且支持Python标准库中的HTML解析器，以及lxml的XML解析器。
# bs4 库会将复杂的HTML文档换成树结构(HITML DoM)，这个结构中的每个节点都是一个Pyhon对象。
# 这些对象可以归纳为如下4种：
#   - bs4.element.Tag 类：表示 HTML 中的标签，是最基本的信息组织单元，它有两个非常重要的属性，
#     分别是表示标签名字的 name 属性和表示标签属性的 attrs 属性。
#   - bs4.element.NavigableString 类：表示 HTML 中标签的文本(非属性字符串)。
#   - bs4.BeautifulSoup 类：表示 HTML DOM 中的全部内容，支持遍历文档树和搜索文档树的大部分方法。
#   - bs4.element.Comment 类：表示标签内字符串的注释部分，是一种特殊的 Navigable String 对象。
# 使用bs4的一般流程如下：
#   - 创建一个BeautifulSoup类型的对象
#     - 根据HTML或者文件创建BeautifulSoup 对象
#   - 通过BeautifulSoup对象的操作方法进行解读搜索
#     - 根据DOM树进行各种节点的搜索（find_all()方法可以搜索出所有满足要求的节点，find()方法只会搜索出第一个满足要求的节点）
#     - 只要获得了一个节点，就可以访问节点的名称、属性和文本
#   - 利用DOM树结构标签的特性，进行更为详细的节点信息提取
#     - 包括名称、属性、文本
from bs4 import BeautifulSoup
import json

from pools import BrowserPool, BrowserPlaywright

class BaiduSearch:

    def __init__(self, browser_pool: BrowserPool):
        self.browser_pool = browser_pool
        self.base_url = "https://www.baidu.com/"

    # 主要获取搜索到的 title 和对应的 url
    async def response(self, questions: Optional[List[str]]) -> Optional[dict]:
        results = {}
        async with self.browser_pool.get_browser() as browser:
            for question in questions:
                html = await self.run(browser=browser, question=question)
                result = self.parsing(html)
                if result:
                    results[question] = result

        return results

    async def run(self, browser: BrowserPlaywright, question: Optional[str]):
        context = await browser.browser.new_context()
        page = await context.new_page()
        await page.goto(self.base_url)

        await page.fill('input[name="wd"]', question)
        await page.wait_for_timeout(1000)
        await page.click('input#su')
        # 百度搜索结果页面一般是这样：
        #   会分多个元素，每个元素的样式是不一样的，但是都有一个 c-container 类名：
        #   <div class="result-op c-container new-pmd" id="1">...</div>  /* 百度百科内容 */
        #   <div class="result-op c-container xpath-log new-pmd" id="2">...</div>  /* 一些新闻 */
        #   <div class="result-op c-container new-pmd" id="3">...</div>  /* 爱奇艺视频内容 */
        #   <div class="result-op c-container new-pmd" id="4">...</div>  /* 大家还在搜的相关'热点搜索'推荐 */
        #   <div class="result-op c-container xpath-log new-pmd" id="5">...</div>  /* 好看视频内容 */
        #   <div class="result c-container xpath-log new-pmd" id="6">...</div>  /* 来自xx网站的内容 */
        #   <div class="result c-container xpath-log new-pmd" id="7">...</div>  /* 来自yy网站的内容 */
        #   <div class="result c-container xpath-log new-pmd" id="8">...</div>  /* 来自zz网站的内容 */
        #   ...
        # 多值 class，指定其中一个即可，所以可以通过定位 'div.c-container' 判断是否已经出来搜索结果
        await page.wait_for_selector('div.c-container')  # 等待搜索结果加载完成
        await page.wait_for_timeout(1000)
        html = await page.content()
        await page.close()
        await context.close()
        return html

    def parsing(self, html: Optional[str]) -> Optional[List[dict]]:
        # 这里创建 BeautifulSoup 实例时共传入了两个参数。
        # 其中，第一个参数表示包含被解析 HTML 文档的字符串；第二个参数指定解析器名称为'lxml'。
        soup = BeautifulSoup(html, "lxml")
        # 获取百度搜索结果列表
        # find_all 返回所有匹配到的对象
        items = soup.find_all("div", class_="c-container")
        results = []
        for item in items:
            # find 只返回第一个匹配到的对象
            # 通过 class 查找，由于 class 属于 Python 的关键字，所以可在 class 的后面加上一个下画线
            # 'class_' 是 BeautifulSoup 中特别关键字参数
            title_tag = item.find('h3', class_='c-title t t tts-title')
            title = title_tag.get_text(strip=True) if title_tag else ''

            publisher_tag = item.find('a', class_='siteLink_9TPP3')
            publisher = publisher_tag.get_text(strip=True) if publisher_tag else ''

            url_tag = item.find('a', class_='siteLink_9TPP3')
            url = url_tag['href'] if url_tag else ''

            summary_tag = item.find('span', class_='content-right_2s-H4')
            summary = summary_tag.get_text(strip=True) if summary_tag else ''

            time_tag = item.find("span", class_="c-color-gray2")
            time = time_tag.get_text(strip=True) if time_tag else ''

            data = {
                "title": title,
                "publisher": publisher,
                "url": url,
                "summary": summary,
                "time": time
            }
            if url:
                results.append(data)
        # json.dumps：将 Python 对象编码成 JSON 字符串
        # json.loads：将已编码的 JSON 字符串解码为 Python 对象
        results = [json.loads(x) for x in set(json.dumps(d, sort_keys=True) for d in results)]
        
        return results
