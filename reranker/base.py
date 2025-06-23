from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional
from openai import AsyncOpenAI
import torch
import os
from dotenv import load_dotenv


class BaseSemanticSearcher(ABC):
    """
    Abstract base class for semantic search implementations.
    """

    # torch.Tensor 是一种包含单一数据类型元素的多维矩阵
    # embedding: 一段文本，转换成 n 维的向量
    @abstractmethod
    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        pass
    
    # 计算 queries 与 documents 的 str 之间的匹配度
    #
    # 比如:
    #   queries=[q1="最近天气？"]
    #   documents=[d1="深圳最近天气不好！", d2="深圳最近下雨!", d3="深圳今天天晴，气温。。。"]
    # calculate_scores 会计算 q1 与 d1, d2, d3 之间的匹配度.
    #
    #   query_embeddings  # n*1 维矩阵
    #   doc_embeddings  # n*1 维矩阵
    #
    #   scores = query_embeddings @ doc_embeddings.T  # n*n 维矩阵
    #   softmax(scores, dim=-1)  # 对 n*n 维矩阵 基于 column 进行 softmax
    #   softmax 输出的矩阵最终意义是 queries 中 q(i) 与 documents 中 d(j) 的匹配度值
    async def calculate_scores(
        self,
        queries: List[str],
        documents: List[str],
    ) -> torch.Tensor:
        # 需要注意，不存在横向量，认为所有一维矩阵/列表都是列向量
        # 列表、一维数组、一维向量都默认是n行1列的列向量
        # embedding 值是一个 n 维列向量：n*1
        query_embeddings = await self._get_embeddings(queries)
        doc_embeddings = await self._get_embeddings(documents)
        # '@' 表示常规的数学上定义的矩阵相乘
        # 在 torch 中，'Tensor.T' 表示张量或者矩阵的转置。在 tensor 中，通过 '.T' 属性来实现。
        # A @ B 可以看成是 B 的每一个列向量 b(j) 在 A 的每一个行向量 a(i) (基向量)上进行投影，
        # 当 A 的每一个基模都为 1 时，投影的结果恰好就是 B 在 A 坐标系中的坐标值。
        scores = query_embeddings @ doc_embeddings.T
        # Softmax可以将数值向量转换为概率分布
        # 'torch.softmax', Alias for 'torch.nn.functional.softmax'
        # torch.nn.functional.softmax takes two parameters: input and dim.
        #   Applies the Softmax function to an n-dimensional input Tensor.
        #   Rescales them so that the elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1.
        #   dim (int) – A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
        #   Input: any number of additional dimensions
        #   Output: same shape as the input
        #
        #
        # n 维向量 [1,1,1,...,n] 的 softmax 比较好理解, 维度为 (n), dim 只能为 0;
        # 对于 [ [1,1,1], [2,2,2] ], 维度为 (2, 3), dim 取值可以为 0 和 1;
        # 对于
        #   [
        #      [
        #         [1,1,1],
        #         [2,2,2]
        #      ],
        #      [
        #         [3,3,3],
        #         [4,4,4]
        #      ]
        #   ]
        # 维度 (2, 2, 3), dim 取值为 0, 1, 2.
        #
        # 那么 (a,b,c,d, ...) 维矩阵的 dim 维的 softmax 怎么理解呢？
        # Pytorch nn.Softmax(dim=?) 深入理解请参考：https://zhuanlan.zhihu.com/p/397695655
        #
        # the dim=-1 argument specifies that the softmax function should be applied to the last dimension of the tensor.
        # in the 2D case: row refers to axis=0, while column refers to axis=1.
        scores = torch.softmax(scores, dim=-1)
        return scores

    async def rerank(
        self,
        query: Union[str, List[str]],
        documents: List[str],
        top_k: int = 5,
    ) -> List[Dict[str, Union[str, float]]]:
        # isinstance(object, classinfo)
        #   object -- 实例对象。
        #   classinfo -- 可以是直接或间接类名、基本类型或者由它们组成的元组。
        queries = [query] if isinstance(query, str) else query
        scores = await self.calculate_scores(queries, documents)

        results = []
        for query_scores in scores:
            # torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None)
            #   Returns the k largest elements of the given input tensor along a given dimension.
            #   If dim is not given, the last dimension of the input is chosen.
            #   If largest is False then the k smallest elements are returned.
            #   The boolean option sorted if True, will make sure that the returned k elements are themselves sorted.
            #   A namedtuple of (values, indices) is returned with the values 
            #   and indices of the largest k elements of each row of the input tensor in the given dimension dim.
            #
            # calculate_scores 里面 softmax 时, dim 取的是最后一个, 力求得出的每个 token(每个字或者每个词)都是概率最大的
            # 在求 topk 的时候, dim 取的是 0, 力求整个片段(整个句子或者整段)是概率最大的
            top_indices = torch.topk(query_scores, min(top_k, len(documents)), dim=0)
            query_results = [
                {
                    # Returns the value of this tensor as a standard Python number. 
                    # This only works for tensors with one element. 
                    "document": documents[idx.item()],
                    "score": score.item()
                }
                # 在 python 3, 为了节省内存空间, zip() 返回一个 tuple 迭代器
                # 比如两个 list, a 和 b, list(zip(a, b)) 生成了一个列表, 在该列表中, 每个元素都是一个 tuple,
                # 对于第 i 个元素, 它其中的内容是 (a[i-1], b[i-1]). 这样的操作, 与压缩软件的 '压缩' 十分接近.
                for score, idx in zip(top_indices.values, top_indices.indices)
            ]
            results.append(query_results)

        return results[0] if isinstance(query, str) else results

    async def get_reranked_documents(
        self,
        query: Union[str, List[str]],
        documents: List[str],
        top_k: int = 5
    ) -> Union[List[str], List[List[str]]]:
        results = await self.rerank(query, documents, top_k)
        if isinstance(query, str):
            return [x['document'].strip() for x in results]
        return [[x['document'].strip() for x in r] for r in results]


class OpenAIEmbeddingReranker(BaseSemanticSearcher):
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY")
        self.base_url = base_url or os.getenv("EMBEDDING_BASE_URL")
        self.model = model or os.getenv("EMBEDDING_MODEL_NAME")
        if not self.api_key:
            raise ValueError("No OpenAI API key provided")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = model

    async def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        embeddings = [e.embedding for e in response.data]
        return torch.tensor(embeddings)
