# Modified for the MIRAI project, 2026.

import os
import time
import json
import logging
import asyncio
import requests
from typing import List, Dict, Optional, Any, Union
from qa_manager.llm_api import LLM_Client

import threading
import random
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def is_workflow_valid(workflow_str: str) -> bool:
    if not workflow_str:
        return False

    modules = [m.strip() for m in workflow_str.split(",") if m.strip()]
    allowed_modules = {"QR", "QDP", "QDS", "R", "DS", "AG"}

    # ===== 规则 3：不能重复 =====
    if len(modules) != len(set(modules)):
        return False

    # ===== 基本合法性检查 =====
    if any(m not in allowed_modules for m in modules):
        return False

    # ===== 规则 1：QDP/QDS只能单独出现 =====
    if "QDP" in modules or "QDS" in modules:
        return len(modules) == 1

    # ===== 规则 2：最后必须是 AG（无 QDP/QDS 情况） =====
    if modules[-1] != "AG":
        return False

    # ===== 规则 4：顺序约束 =====
    order_map = {"QR": 0, "R": 1, "DS": 2, "AG": 3}

    # 如果有 QR 必须在第一位
    if "QR" in modules and modules[0] != "QR":
        return False

    # 检查顺序是否递增
    order_indices = [order_map[m] for m in modules]
    if order_indices != sorted(order_indices):
        return False

    return True


class BaseLLMClient:
    TRACE_DIR = Path(os.getenv("MIRAI_TRACE_DIR", "artifacts/traces"))
    _trace_lock = threading.Lock()  # 类级锁，保证写文件线程安全

    def __init__(
        self,
        model_name: str = "gpt4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.5,
        max_retries: int = 3,
        retry_interval: float = 1.0,
        enable_trace: bool = True
    ):
        self.model_name = model_name
        self.system_jinja2_path = system_jinja2_path or os.getenv("MIRAI_SYSTEM_PROMPT")
        self.user_jinja2_path = user_jinja2_path or os.getenv("MIRAI_USER_PROMPT")
        self.default_temperature = default_temperature
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.enable_trace = enable_trace

        logging.info(f"初始化 LLM_Client: model={model_name}")
        self.llm_client = LLM_Client(
            model_name=self.model_name,
            system_jinja2_path=self.system_jinja2_path,
            user_jinja2_path=self.user_jinja2_path
        )

    def get_response(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        """同步调用，带重试机制"""
        temp = temperature if temperature is not None else self.default_temperature
        for attempt in range(1, self.max_retries + 1):
            try:
                res, _ = self.llm_client.query_model(messages=messages, temperature=temp)
                return res[0]['message']['content']
            except Exception as e:
                logging.error(f"调用LLM失败（第{attempt}次）：{e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_interval)
        return "[ERROR] 调用LLM失败"

    async def async_get_response(
        self, 
        messages: List[Dict[str, str]], 
        agent_name: str,  
        temperature: Optional[float] = None
    ) -> str:
        """异步调用并记录 trace 数据（可选）"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.get_response, messages, temperature)

        # 只有 enable_trace 为 True 才保存
        if self.enable_trace:
            self._record_trace(messages, response, temperature, agent_name)

        return response

    def _record_trace(
        self, 
        messages: List[Dict[str, str]], 
        response: str, 
        temperature: Optional[float],
        agent_name: str
    ):
        """将调用数据追加写入 JSONL 文件（线程安全）"""
        self.TRACE_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        trace_file = self.TRACE_DIR / f"{date_str}.jsonl"

        trace_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "agent_name": agent_name,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "messages": messages,
            "response": response
        }

        with BaseLLMClient._trace_lock:
            with open(trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_entry, ensure_ascii=False) + "\n")

    async def async_batch_get_response(
        self,
        batch_messages: List[List[Dict[str, str]]],
        agent_name: str,
        temperature: Optional[float] = None
    ) -> List[str]:
        """批量异步调用，批量记录 trace（受 enable_trace 控制）"""
        tasks = [
            self.async_get_response(msg, agent_name, temperature) 
            for msg in batch_messages
        ]
        results = await asyncio.gather(*tasks)
        return results


class PlanningAgent(BaseLLMClient):
    """
    继承基础 LLM 客户端的 Planning Agent
    功能：
        - 单个 query 转 workflow
        - 并行处理多个 query
        - 后处理合法性检查
    """
    
    def __init__(
        self,
        model_name: str = "gpt4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.5
    ):
        super().__init__(
            model_name=model_name,
            system_jinja2_path=system_jinja2_path,
            user_jinja2_path=user_jinja2_path,
            default_temperature=default_temperature
        )

    def _build_workflow_messages(self, query: str) -> List[dict]:
        """
        构造生成工作流的 message prompt
        要求输出的 workflow 用 <workflow>xxx</workflow> 包裹
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant to plan a Workflow for the given question using the available tools/agents.\n"
                    "Your task: Output ONLY the correct workflow inside <workflow>...</workflow> tags following the rules below.\n\n"

                    "Available Tools/Agents:\n"
                    "  - Query Rewriter (QR): Input: question → Output: rewritten question that is more concise, clearer, and accurate.\n"
                    "  - Query Decomposition Serial (QDS): Input: question → Output: dependent sub-questions where later ones depend on earlier ones.\n"
                    "  - Retrieval (R): Input: question → Output: relevant candidate documents.\n"
                    "  - Document Selector (DS): Input: question + candidate documents → Output: subset of documents helpful for answering.\n"
                    "  - Answer Generator (AG): Input: question [+ optional documents] → Output: final answer.\n\n"

                    "Rules for tool selection:\n"
                    "1. When the question needs to be broken down into sub-questions:\n"
                    "   - If sub-questions have dependencies and must be answered in sequence, use QDS ONLY.\n"
                    "2. When the question can be answered directly without decomposition:\n"
                    "   - Build workflow from QR, R, DS, AG.\n"
                    "   - If DS is in workflow, R must appear before DS.\n"
                    "   - The last module must always be AG.\n"
                    "3. IMPORTANT:\n"
                    "   - If you choose QDS, DO NOT add any other tools/agents.\n"
                    "   - The workflow must ONLY contain QDS in those cases.\n\n"

                    "Example Workflows:\n"
                    "Question: \"Which city will have the highest GDP in 2024: Singapore, Houston, Beijing?\"\n"
                    "Workflow: <workflow>QDS</workflow>\n"
                    "Question: \"Which African country with the fastest economic growth since 1990 had the main export products in 2023?\"\n"
                    "Workflow: <workflow>QDS</workflow>\n"
                    "Question: \"Which team is the 2025 NBA champion?\"\n"
                    "Workflow: <workflow>QR,R,AG</workflow>\n\n"

                    "Now output in the above format:\n"
                )
            },
            {
                "role": "assistant",
                "content": "Ok, you can provide the Question and I will give the appropriate Workflow inside <workflow>...</workflow> tags."
            },
            {
                "role": "user",
                "content": "Please give the appropriate Workflow in the format requested above.\nQuestion: \"{}\"\nWorkflow: <workflow>".format(query)
            }
        ]


        # messages = [
        #     {
        #         "role": "system",
        #         "content": (
        #             "You are a helpful assistant to plan a Workflow for the given question using the available tools/agents.\n"
        #             "Your task: Output ONLY the correct workflow inside <workflow>...</workflow> tags following the rules below.\n\n"

        #             "Available Tools/Agents:\n"
        #             "  - Query Rewriter (QR): Input: question → Output: rewritten question that is more concise, clearer, and accurate.\n"
        #             "  - Query Decomposition Serial (QDS): Input: question → Output: dependent sub-questions where later ones depend on earlier ones.\n"
        #             "  - Query Decomposition Parallel (QDP): Input: question → Output: several sub-questions can be searched independently.\n"
        #             "  - Retrieval (R): Input: question → Output: relevant candidate documents.\n"
        #             "  - Document Selector (DS): Input: question + candidate documents → Output: subset of documents helpful for answering.\n"
        #             "  - Answer Generator (AG): Input: question [+ optional documents] → Output: final answer.\n\n"

        #             "Rules for tool selection:\n"
        #             "1. When the question needs to be broken down into sub-questions:\n"
        #             "   - If sub-questions have dependencies and must be answered in sequence, use QDS or QDP ONLY.\n"
        #             "2. When the question can be answered directly without decomposition:\n"
        #             "   - Build workflow from QR, R, DS, AG.\n"
        #             "   - If DS is in workflow, R must appear before DS.\n"
        #             "   - The last module must always be AG.\n"
        #             "3. IMPORTANT:\n"
        #             "   - If you choose QDS or QDP, DO NOT add any other tools/agents.\n"
        #             "   - The workflow must ONLY contain QDS or QDP in those cases.\n\n"

        #             "Example Workflows:\n"
        #             "Question: \"Which city will have the highest GDP in 2024: Singapore, Houston, Beijing?\"\n"
        #             "Workflow: <workflow>QDP</workflow>\n"
        #             "Question: \"Which African country with the fastest economic growth since 1990 had the main export products in 2023?\"\n"
        #             "Workflow: <workflow>QDS</workflow>\n"
        #             "Question: \"Which team is the 2025 NBA champion?\"\n"
        #             "Workflow: <workflow>QR,R,AG</workflow>\n\n"

        #             "Now output in the above format:\n"
        #         )
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "Ok, you can provide the Question and I will give the appropriate Workflow inside <workflow>...</workflow> tags."
        #     },
        #     {
        #         "role": "user",
        #         "content": "Please give the appropriate Workflow in the format requested above.\nQuestion: \"{}\"\nWorkflow: <workflow>".format(query)
        #     }
        # ]



        # messages = [
        #     {
        #         "role": "system",
        #         "content": (
        #             "You are a helpful assistant to plan a Workflow for the given question using the available tools/agents.\n"
        #             "Your task: Output ONLY the correct workflow inside <workflow>...</workflow> tags following the rules below.\n\n"

        #             "Available Tools/Agents:\n"
        #             "  - Query Rewriter (QR): Input: question → Output: rewritten question that is more concise, clearer, and accurate.\n"
        #             "  - Query Decomposition Serial (QDS): Input: question → Output: dependent sub-questions where later ones depend on earlier ones.\n"
        #             "  - Retrieval (R): Input: question → Output: relevant candidate documents.\n"
        #             "  - Answer Generator (AG): Input: question [+ optional documents] → Output: final answer.\n\n"

        #             "Rules for tool selection:\n"
        #             "1. When the question needs to be broken down into sub-questions:\n"
        #             "   - If sub-questions have dependencies and must be answered in sequence, use QDS ONLY.\n"
        #             "2. When the question can be answered directly without decomposition:\n"
        #             "   - Build workflow from QR, R, AG.\n"
        #             "   - The last module must always be AG.\n"
        #             "3. IMPORTANT:\n"
        #             "   - If you choose QDS, DO NOT add any other tools/agents.\n"
        #             "   - The workflow must ONLY contain QDS in those cases.\n\n"

        #             "Example Workflows:\n"
        #             "Question: \"Which city will have the highest GDP in 2024: Singapore, Houston, Beijing?\"\n"
        #             "Workflow: <workflow>QDS</workflow>\n"
        #             "Question: \"Which African country with the fastest economic growth since 1990 had the main export products in 2023?\"\n"
        #             "Workflow: <workflow>QDS</workflow>\n"
        #             "Question: \"Which team is the 2025 NBA champion?\"\n"
        #             "Workflow: <workflow>QR,R,AG</workflow>\n\n"

        #             "Now output in the above format:\n"
        #         )
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "Ok, you can provide the Question and I will give the appropriate Workflow inside <workflow>...</workflow> tags."
        #     },
        #     {
        #         "role": "user",
        #         "content": "Please give the appropriate Workflow in the format requested above.\nQuestion: \"{}\"\nWorkflow: <workflow>".format(query)
        #     }
        # ]

        return messages

    def parse_response(self, workflow: str) -> Dict[str, object]:
        """
        严格后处理：
        1. 必须且只能是 <workflow>...</workflow>（标签外有内容 → is_valid=False）
        2. 没标签 → is_valid=False，workflow 返回原始内容
        3. 在格式合规时，再进行 is_workflow_valid 校验
        """

        # 正则强制匹配整个字符串只能是标签包裹的内容（允许首尾空白）
        strict_match = re.fullmatch(
            r'\s*<workflow>\s*(.*?)\s*</workflow>\s*',
            workflow,
            flags=re.IGNORECASE | re.DOTALL
        )

        if not strict_match:
            # 格式不合规：没有标签或标签外有内容
            return {
                "workflow": workflow.strip(),
                "is_valid": False
            }

        # 格式合规 → 提取标签中内容
        workflow_content = strict_match.group(1).strip()

        # 再进行工作流规则合法性判断
        return {
            "workflow": workflow_content,
            "is_valid": is_workflow_valid(workflow_content)
        }


class SubPlanningAgent(BaseLLMClient):
    """
    与 PlanningAgent 类似，但 Prompt 只包含 R, DS, AG 三种工具
    用于生成回答单个或多个 query 的工作流。
    """

    def __init__(
        self,
        model_name: str = "gpt4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.5
    ):
        super().__init__(
            model_name=model_name,
            system_jinja2_path=system_jinja2_path,
            user_jinja2_path=user_jinja2_path,
            default_temperature=default_temperature
        )

    def _build_workflow_messages(self, query: str) -> List[dict]:
        """
        构造用于生成工作流的 message prompt（简化版，只包含 R, DS, AG）
        强制要求输出格式为 <workflow>xxx</workflow>
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant to plan a Workflow using the provided tools/agents "
                    "based on the given question.\n"
                    "Your task: Output ONLY the workflow inside <workflow>...</workflow> tags.\n\n"
                    "Available tools/agents:\n"
                    "Retrieval (R): Input: question → Output: relevant candidate documents.\n"
                    "Document Selector (DS): Input: question + candidate documents → Output: documents helpful for answering.\n"
                    "Answer Generator (AG): Input: question [+ optional documents] → Output: final answer.\n\n"
                    "Rules:\n"
                    "1. If DS is in the workflow, R must appear before DS.\n"
                    "2. The final module must always be AG.\n\n"
                    "Examples:\n"
                    "Question: 'Which city is the capital of Australia.'\n"
                    "Workflow: <workflow>AG</workflow>\n\n"
                    "Question: 'Which team is the 2025 NBA champion?'\n"
                    "Workflow: <workflow>R,AG</workflow>\n\n"
                    "Now I will give you a question. Just output the appropriate Workflow inside <workflow>...</workflow> tags, "
                    "and do not include anything else."
                )
            },
            {
                "role": "assistant",
                "content": "Ok, you can provide the question and I will give the appropriate Workflow."
            },
            {
                "role": "user",
                "content": f"Please give the appropriate Workflow in the required format.\nQuestion: \"{query}\"\nWorkflow: "
            },
        ]

        # messages = [
        #     {
        #         "role": "system",
        #         "content": (
        #             "You are a helpful assistant to plan a Workflow using the provided tools/agents "
        #             "based on the given question.\n"
        #             "Your task: Output ONLY the workflow inside <workflow>...</workflow> tags.\n\n"
        #             "Available tools/agents:\n"
        #             "Retrieval (R): Input: question → Output: relevant candidate documents.\n"
        #             "Answer Generator (AG): Input: question [+ optional documents] → Output: final answer.\n\n"
        #             "Rules:\n"
        #             " - The final module must always be AG.\n\n"
        #             "Examples:\n"
        #             "Question: 'Which city is the capital of Australia.'\n"
        #             "Workflow: <workflow>AG</workflow>\n\n"
        #             "Question: 'Which team is the 2025 NBA champion?'\n"
        #             "Workflow: <workflow>R,AG</workflow>\n\n"
        #             "Now I will give you a question. Just output the appropriate Workflow inside <workflow>...</workflow> tags, "
        #             "and do not include anything else."
        #         )
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "Ok, you can provide the question and I will give the appropriate Workflow."
        #     },
        #     {
        #         "role": "user",
        #         "content": f"Please give the appropriate Workflow in the required format.\nQuestion: \"{query}\"\nWorkflow: "
        #     },
        # ]

        return messages

    def parse_response(self, workflow: str) -> Dict[str, object]:
        """
        严格后处理：
        1. 必须且只能是 <workflow>...</workflow>（标签外有内容 → is_valid=False）
        2. 没标签 → is_valid=False，workflow 返回原始内容
        3. 在格式合规时，再进行 is_workflow_valid 校验
        """
        strict_match = re.fullmatch(
            r'\s*<workflow>\s*(.*?)\s*</workflow>\s*',
            workflow,
            flags=re.IGNORECASE | re.DOTALL
        )

        if not strict_match:
            # 格式不合规：没有标签或标签外有内容
            return {
                "workflow": workflow.strip(),
                "is_valid": False
            }

        # 格式合规 → 提取标签中内容
        workflow_content = strict_match.group(1).strip()

        # 再进行工作流规则合法性判断
        return {
            "workflow": workflow_content,
            "is_valid": is_workflow_valid(workflow_content)
        }

class QueryRewriteAgent(BaseLLMClient):
    """
    基于 PlanningAgent 风格的 Query Rewrite Agent
    功能：
        - 单个 query 重写为简洁、可搜索的版本
        - 支持只输入 query 或输入 query + context
        - 并行处理多个 query
    """

    def __init__(
        self,
        model_name: str = "gpt4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.5
    ):
        super().__init__(
            model_name=model_name,
            system_jinja2_path=system_jinja2_path,
            user_jinja2_path=user_jinja2_path,
            default_temperature=default_temperature
        )

    def _build_messages(self, query: str) -> List[dict]:
        """构造无 context 情况下的 prompt，必须把输出放在 <query>...</query> 标签"""
        return [
            {
                "role": "system",
                "content": (
                    "You are a professional assistant skilled at rewriting slightly redundant or overly wordy factual questions "
                    "into a single, concise, and searchable query. Keep all essential names, dates, and terms. "
                    "Do not add explanations or unrelated details.\n"
                    "You must output ONLY the rewritten query inside <query>...</query> tags, without any extra text outside the tags."
                )
            },
            {
                "role": "assistant",
                "content": "Okay, I will return the concise rewritten query inside <query>...</query> tags."
            },
            {
                "role": "user",
                "content": (
                    f"Original question is: {query}\n"
                    "Rewrite the original question into a short, clear query containing only the key searchable information. "
                    "Preserve important names, places, dates, and events.\n\n"
                    "Follow the format shown in the examples:\n\n"
                    "Example 1:\n"
                    "Original question: 'Who was the Prime Minister of the United Kingdom when the Brexit referendum took place in 2016?'\n"
                    "Rewritten query: <query>UK Prime Minister during 2016 Brexit referendum</query>\n\n"
                    "Example 2:\n"
                    "Original question: 'The novel Beloved by Toni Morrison won which major literary award in 1988?'\n"
                    "Rewritten query: <query>Major literary award won by Beloved in 1988</query>\n\n"
                    "Example 3:\n"
                    "Original question: 'In what year did the Voyager 1 spacecraft first enter interstellar space?'\n"
                    "Rewritten query: <query>Year Voyager 1 entered interstellar space</query>\n\n"
                    "Example 4:\n"
                    "Original question: 'Marie Curie was awarded Nobel Prizes in which scientific fields?'\n"
                    "Rewritten query: <query>Scientific fields of Marie Curie’s Nobel Prizes</query>\n\n"
                    "Example 5:\n"
                    "Original question: 'The Great Wall of China spans across how many provinces?'\n"
                    "Rewritten query: <query>Number of provinces spanned by Great Wall of China</query>\n\n"
                    f"Original question is: {query}\n"
                    "Rewritten query: "
                )
            }
        ]

    def _build_messages_with_context(self, query: str, context: dict) -> List[dict]:
        """构造有 context 情况下的 prompt，必须把输出放在 <query>...</query> 标签"""
        sub_queries_info = "\n".join([
            f"Sub-query: {item['query']}\nAnswer: {item.get('answer', '')}"
            for item in context.get("sub-query_docs_a", [])
        ])

        return [
            {
                "role": "system",
                "content": (
                    "You are a professional assistant skilled at synthesizing context from a main query and related sub-queries "
                    "with their answers, then rewriting a follow-up query into a single, concise, and searchable form. "
                    "You must ensure the rewritten query captures the essential meaning of the current query while leveraging the given context. "
                    "Preserve important names, dates, places, and facts. Do not add explanations or unrelated details.\n"
                    "You must output ONLY the rewritten query inside <query>...</query> tags, without any extra text outside the tags."
                )
            },
            {
                "role": "assistant",
                "content": "Okay, I will rewrite the current query inside <query>...</query> tags."
            },
            {
                "role": "user",
                "content": (
                    f"Main original query:\n{context.get('original_query', '')}\n\n"
                    f"Related sub-queries & answers:\n{sub_queries_info}\n\n"
                    f"Current query to rewrite:\n{query}\n\n"
                    "Based on the above, rewrite the current query into a single, clear, and searchable query "
                    "that includes only the essential details needed to find an answer efficiently. Avoid unnecessary context or verbose wording.\n\n"
                    "Follow the format shown in the examples:\n\n"
                    "Example 1:\n"
                    "Original question: 'Who was the Prime Minister of the United Kingdom when the Brexit referendum took place in 2016?'\n"
                    "Rewritten query: <query>UK Prime Minister during 2016 Brexit referendum</query>\n\n"
                    "Example 2:\n"
                    "Original question: 'The novel Beloved by Toni Morrison won which major literary award in 1988?'\n"
                    "Rewritten query: <query>Major literary award won by Beloved in 1988</query>\n\n"
                    "Example 3:\n"
                    "Original question: 'T. A. Sarasvati Amma was a scholar born in a country with a population of over how many people?'\n"
                    "Rewritten query: <query>Population of T. A. Sarasvati Amma’s birth country</query>\n\n"
                    f"Original question is: {query}\n"
                    "Rewritten query: "
                )
            }
        ]

    def parse_response(self, query_rewrited: str, init_question: str) -> Dict[str, object]:
        """
        严格后处理：
        1. 必须且只能包含一次 <query>...</query>，标签外无其他内容，否则 is_valid=False，返回原始
        2. 在符合第一条时，提取标签内容并校验其他条件
        """
        strict_match = re.fullmatch(
            r'\s*<query>\s*(.*?)\s*</query>\s*',
            query_rewrited,
            flags=re.IGNORECASE | re.DOTALL
        )

        if not strict_match:
            # 格式不合规：返回原始内容，is_valid=False
            return {
                "query_rewrited": query_rewrited.strip(),
                "is_valid": False
            }

        # 格式合规 → 提取标签中内容
        extracted_query = strict_match.group(1).strip()

        # 按空格统计非空词数量
        word_count = len([w for w in extracted_query.split() if w.strip()])

        # 合规基础上进行进一步校验：不能等于原问题，且词数 <= 100
        is_valid = (extracted_query != init_question) and (word_count <= 100)

        return {
            "query_rewrited": extracted_query,
            "is_valid": is_valid
        }

import re
from typing import List, Optional

class QueryDecompositionAgentParallel(BaseLLMClient):
    """
    将复杂问题拆成多个独立可搜索的子问题，并支持并行处理。
    """

    def __init__(
        self,
        model_name: str = "gpt4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.3
    ):
        super().__init__(
            model_name=model_name,
            system_jinja2_path=system_jinja2_path,
            user_jinja2_path=user_jinja2_path,
            default_temperature=default_temperature
        )

    def _build_messages(self, query: str) -> List[dict]:
        """
        构造生成子问题的 message prompt
        要求输出的 subquestions 分行，每个子问题放在独立标签 <q1>...</q1>
        """
        return [
            {
                "role": "system",
                "content": (
                    "You are a professional assistant skilled at decomposing complex multi-entity or multi-location "
                    "questions into multiple independent and searchable sub-questions. "
                    "Each sub-question should be specific, logically complete, and avoid duplication.\n"
                    "You must output each sub-question on its own line, wrapped in numbered tags <q1>...</q1>, <q2>...</q2>, etc., "
                    "starting from <q1> in ascending order with no gaps, no repeats, and no extra text outside the tags."
                )
            },
            {
                "role": "assistant",
                "content": "Okay, I will return each sub-question tagged individually in sequence."
            },
            {
                "role": "user",
                "content": (
                    f"Original question is: '{query}'.\n"
                    "Break down this question into the minimum number of specific, logically complete, "
                    "and independently searchable sub-questions needed to fully understand and answer "
                    "the original question. Do not generate more than 4 sub-questions.\n"
                    "Each sub-question should be on a separate line and tagged as follows:\n"
                    "<q1>first sub-question</q1>\n"
                    "<q2>second sub-question</q2>\n"
                    "... etc.\n\n"
                    "Example:\n"
                    "Original question: 'What are the main exports of Germany and Japan in 2023?'\n"
                    "<q1>What were the main exports of Germany in 2023?</q1>\n"
                    "<q2>What were the main exports of Japan in 2023?</q2>\n\n"
                    f"Original question: '{query}'\n"
                    "Sub-questions:\n"
                )
            }
        ]

    def parse_response(self, context: str, init_question: str) -> Dict[str, object]:
        """
        解析子问题：
        1. 必须且只能包含从 <q1> 到 <qn> 的顺序标签，标签外无文本，否则 is_valid=False
        2. 在格式合规时，提取标签内容作为 subquestions
        3. 数量 >4 或 <=1 时 is_valid=False，但 subquestions 正常返回
        """
        # 允许首尾空格，但中间只能是标签按顺序排列
        tag_pattern = re.compile(r'\s*(<q\d+>.*?</q\d+>\s*)+', flags=re.DOTALL | re.IGNORECASE)
        if not tag_pattern.fullmatch(context.strip()):
            return {
                "subquestions": [init_question],
                "is_valid": False
            }

        # 匹配所有标签
        tags = re.findall(r'<q(\d+)>(.*?)</q\1>', context.strip(), flags=re.DOTALL | re.IGNORECASE)
        if not tags:
            return {
                "subquestions": [init_question],
                "is_valid": False
            }

        # 校验标签编号顺序是否严格递增，无跳号、无重复
        expected_numbers = list(range(1, len(tags) + 1))
        actual_numbers = [int(num) for num, _ in tags]
        if actual_numbers != expected_numbers:
            return {
                "subquestions": [init_question],
                "is_valid": False
            }

        # 提取清理内容
        subquestions = [text.strip() for _, text in tags]

        # 基础格式已经合规，再做数量限制检查
        count = len(subquestions)
        # is_valid = not (count > 7 or count <= 1)
        is_valid = 1 <= count <= 7  # 1～7都

        # 如果有重复内容或完全等于原问题，也判失败
        if len(set(subquestions)) != count or any(s == init_question.strip() for s in subquestions):
            is_valid = False

        return {
            "subquestions": subquestions,
            "is_valid": is_valid
        }

class QueryDecompositionAgentSerial(BaseLLMClient):
    """
    将复杂问题拆解为逻辑上有顺序的子问题，并支持串行处理多个 query。
    返回的结果是子问题列表。
    """

    def __init__(
        self,
        model_name: str = "gpt4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.3
    ):
        super().__init__(
            model_name=model_name,
            system_jinja2_path=system_jinja2_path,
            user_jinja2_path=user_jinja2_path,
            default_temperature=default_temperature
        )

    def _build_messages(self, query: str) -> List[dict]:
        """
        构造生成子问题的 message prompt (有逻辑顺序的版本)
        每个子问题必须放在独立的 <qN> 标签中，按顺序递增
        """
        return [
            {
                'role': 'system',
                'content': (
                    'You are a professional assistant skilled at decomposing complex questions into a minimal sequence '
                    'of logically dependent, independently searchable sub-questions (serial decomposition). '
                    'Each sub-question must:\n'
                    '- Be self-contained and specific\n'
                    '- Be suitable for direct information retrieval from search engines or structured databases\n'
                    '- Be strictly necessary to answer the original question\n'
                    '- Form a logical chain, where later sub-questions depend on earlier sub-questions\n'
                    'You must keep the number of sub-questions as low as possible, never exceed 4 in total, and avoid redundancy.\n'
                    'You must output each sub-question inside numbered tags <q1>...</q1>, <q2>...</q2>, etc., starting at q1 in ascending order, '
                    'with no gaps, no repeats, and no extra text outside tags.'
                )
            },
            {
                'role': 'assistant',
                'content': 'Understood. I will return each sub-question wrapped in its tag, in strict sequence.'
            },
            {
                'role': 'user',
                'content': (
                    f'Original question is: {query}\n'
                    'Now decompose the original question into a logically ordered sequence of dependent sub-questions. '
                    'Output one sub-question per line, each inside the correct numbered tag.\n\n'
                    'Example:\n'
                    "Original question: 'Which African country with the fastest economic growth since 1990 had the main export products in 2023?'\n"
                    "<q1>Which African country had the fastest economic growth since 1990?</q1>\n"
                    "<q2>What were the main export products of that country in 2023?</q2>\n\n"
                    f'Original question: {query}\n'
                    'Sub-questions:\n'
                )
            }
        ]

    def parse_response(self, context: str, init_question: str) -> Dict[str, object]:
        """
        严格后处理：
        1. 必须且只能由 <q1>...</q1>, <q2>...</q2> 按顺序组成；标签外无额外内容
        2. 格式不合规则直接 is_valid=False，subquestions = [init_question]
        3. 格式合规则提取标签内容并继续规则校验
        """
        # 允许首尾空格，但中间只能是若干正确编号标签
        tag_structure_pattern = re.compile(r'\s*(<q\d+>.*?</q\d+>\s*)+', flags=re.DOTALL | re.IGNORECASE)
        if not tag_structure_pattern.fullmatch(context.strip()):
            return {
                "subquestions": [init_question],
                "is_valid": False
            }

        # 提取所有标签内容和编号
        tags = re.findall(r'<q(\d+)>(.*?)</q\1>', context.strip(), flags=re.DOTALL | re.IGNORECASE)
        if not tags:
            return {
                "subquestions": [init_question],
                "is_valid": False
            }

        # 检查编号顺序是否严格递增，无跳号、无重复
        expected_numbers = list(range(1, len(tags) + 1))
        actual_numbers = [int(num) for num, _ in tags]
        if actual_numbers != expected_numbers:
            return {
                "subquestions": [init_question],
                "is_valid": False
            }

        # 提取并清理标签内内容
        subquestions = [text.strip() for _, text in tags]
        count = len(subquestions)

        # 初始合规标记
        is_valid = True

        # 数量限制：>7 为无效，但返回正常提取内容
        if count > 7:
            is_valid = False

        # 内容重复或有等于原问题的情况 → 无效
        if len(set(subquestions)) != count or any(s == init_question.strip() for s in subquestions):
            is_valid = False

        return {
            "subquestions": subquestions,
            "is_valid": is_valid
        }


class DocumentSelectionAgent(BaseLLMClient):
    """
    根据问题和候选文档，选出最相关的文档 ID 列表。
    支持单个和批量并行处理。
    """

    def __init__(
        self,
        model_name: str = "gpt4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.0  # 选文档任务最好稳定输出
    ):
        super().__init__(
            model_name=model_name,
            system_jinja2_path=system_jinja2_path,
            user_jinja2_path=user_jinja2_path,
            default_temperature=default_temperature
        )

    def _build_messages(self, query: str, documents: List[str]) -> List[dict]:
        """
        构造选出相关文档 ID 的 prompt
        输出必须放在 <id>...</id> 标签中
        """
        doc_content = ''.join([f"Document {doc_id}: {doc}\n\n" for doc_id, doc in enumerate(documents)])
        max_id = len(documents) - 1  # 动态计算最高ID

        return [
            {
                'role': 'system',
                'content': (
                    f'You are a helpful, respectful and honest assistant. '
                    f'Your task is to output the IDs of the candidate Documents (0, 1, 2,..., {max_id}) '
                    f'which are helpful in answering the Question. '
                    f'You must output ONLY inside <id>...</id> tags, as comma-separated values like '
                    f'<id>Document0,Document2,Document4</id>. There must be no text outside the tags.'
                )
            },
            {
                'role': 'assistant',
                'content': 'Okay, I will provide the IDs of candidate Documents inside <id>...</id> tags.'
            },
            {
                'role': 'user',
                'content': f'Question is: {query}\n\n{doc_content}'
            },
            {
                'role': 'assistant',
                'content': "OK, I received the Question and the candidate Documents."
            },
            {
                'role': 'user',
                'content': (
                    f"Now, output the IDs of the candidate Documents (0,1,2,...,{max_id}) "
                    "which are helpful in answering the Question. Format them as comma-separated values inside <id>...</id> tags.\n"
                    "Example: <id>Document0,Document2,Document4</id>\n"
                    "Do not output any other text outside the tags."
                )
            }
        ]

    def parse_response(self, context: str, total_docs: int) -> Dict[str, object]:
        """
        严格模式解析：
        1. 必须且只能包含一次 <id>...</id>，标签外无内容，否则 is_valid=False，返回全部文档
        2. 标签内只能出现 DocumentX 且用逗号分隔，不能出现其他内容，否则 is_valid=False
        3. 格式合规则提取 ID 列表：
        - 无 ID → 返回空列表 + True
        - 有超范围 ID → 过滤范围内 + False
        - 有重复 ID → 去重 + False
        - 全合法且无重复 → True
        """
        valid_ids_set = set(range(total_docs))  # 动态计算范围，比如 total_docs=5 → {0,1,2,3,4}
        all_docs_list = sorted(valid_ids_set)

        # Step 1: 格式检查（单标签 + 标签外无内容）
        strict_match = re.fullmatch(r'\s*<id>\s*(.*?)\s*</id>\s*', context, flags=re.DOTALL | re.IGNORECASE)
        if not strict_match:
            return {"docs_ids": all_docs_list, "is_valid": False}

        inside_text = strict_match.group(1).strip()

        # Step 2: 标签内内容必须完全匹配模式：DocumentN(,DocumentN)*
        if not re.fullmatch(r'(Document\d+(?:,Document\d+)*)?', inside_text):
            return {"docs_ids": all_docs_list, "is_valid": False}

        # Step 3: 提取 IDs
        matches = re.findall(r'Document(\d+)', inside_text)
        try:
            ids_list = [int(m) for m in matches]
        except ValueError:
            return {"docs_ids": all_docs_list, "is_valid": False}

        # Step 4: 无 ID → 合法返回空列表
        if not ids_list:
            return {"docs_ids": [], "is_valid": True}

        # Step 5: 范围检查
        if not all(i in valid_ids_set for i in ids_list):
            filtered_ids = sorted(i for i in ids_list if i in valid_ids_set)
            return {"docs_ids": filtered_ids, "is_valid": False}

        # Step 6: 重复检查
        ids_set = sorted(set(ids_list))
        if len(ids_list) != len(ids_set):
            return {"docs_ids": ids_set, "is_valid": False}

        # Step 7: 完全合规则
        return {"docs_ids": ids_set, "is_valid": True}



    # def _build_messages(self, query: str, documents: List[str]) -> List[dict]:
    #     """
    #     构造选出相关文档 ID 的 prompt
    #     输出必须放在 <id>...</id> 标签中
    #     """
    #     doc_content = ''.join([f"Document {doc_id}: {doc}\n\n" for doc_id, doc in enumerate(documents)])
    #     total_count = len(documents) - 1

    #     return [
    #         {
    #             'role': 'system',
    #             'content': (
    #                 f'You are a helpful, respectful and honest assistant. '
    #                 f'Your task is to output the IDs of the candidate Documents (0, 1, 2,..., {total_count}) '
    #                 f'which are helpful in answering the Question. '
    #                 f'You must output ONLY inside <id>...</id> tags, as comma-separated values like '
    #                 f'<id>Document0,Document4,Document6</id>. There must be no text outside the tags.'
    #             )
    #         },
    #         {
    #             'role': 'assistant',
    #             'content': 'Okay, I will provide the IDs of candidate Documents inside <id>...</id> tags.'
    #         },
    #         {
    #             'role': 'user',
    #             'content': f'Question is: {query}\n\n{doc_content}'
    #         },
    #         {
    #             'role': 'assistant',
    #             'content': "OK, I received the Question and the candidate Documents."
    #         },
    #         {
    #             'role': 'user',
    #             'content': (
    #                 f"Now, output the IDs of the candidate Documents (0,1,2,...,{total_count}) "
    #                 "which are helpful in answering the Question. Format them as comma-separated values inside <id>...</id> tags.\n"
    #                 "Example: <id>Document0,Document4,Document6</id>\n"
    #                 "Do not output any other text outside the tags."
    #             )
    #         }
    #     ]

    # def parse_response(self, context: str) -> Dict[str, object]:
    #     """
    #     严格模式解析：
    #     1. 必须且只能包含一次 <id>...</id>，标签外无内容，否则 is_valid=False，返回 0~9 全部
    #     2. 标签内只能出现 DocumentX 且用逗号分隔，不能出现其他内容，否则 is_valid=False
    #     3. 格式合规则提取 ID 列表：
    #        - 无 ID → 返回空列表 + True
    #        - 有超范围 ID → 过滤范围内 + False
    #        - 有重复 ID → 去重 + False
    #        - 全合法且无重复 → True
    #     """
    #     valid_ids_set = set(range(10))
    #     all_docs_list = sorted(valid_ids_set)  # [0,1,...,9]

    #     # Step 1: 格式检查（单标签 + 标签外无内容）
    #     strict_match = re.fullmatch(r'\s*<id>\s*(.*?)\s*</id>\s*', context, flags=re.DOTALL | re.IGNORECASE)
    #     if not strict_match:
    #         return {"docs_ids": all_docs_list, "is_valid": False}

    #     inside_text = strict_match.group(1).strip()

    #     # Step 2: 标签内内容必须完全匹配模式：DocumentN(,DocumentN)*
    #     # N 必须是整数，可以是多位数，但之后会做范围检查
    #     if not re.fullmatch(r'(Document\d+(?:,Document\d+)*)?', inside_text):
    #         return {"docs_ids": all_docs_list, "is_valid": False}

    #     # Step 3: 提取 IDs
    #     matches = re.findall(r'Document(\d+)', inside_text)
    #     try:
    #         ids_list = [int(m) for m in matches]  # 保留原顺序用于重复检查
    #     except ValueError:
    #         return {"docs_ids": all_docs_list, "is_valid": False}

    #     # Step 4: 无 ID → 合法返回空列表
    #     if not ids_list:
    #         return {"docs_ids": [], "is_valid": True}

    #     # Step 5: 范围检查
    #     if not all(i in valid_ids_set for i in ids_list):
    #         filtered_ids = sorted(i for i in ids_list if i in valid_ids_set)
    #         return {"docs_ids": filtered_ids, "is_valid": False}

    #     # Step 6: 重复检查
    #     ids_set = sorted(set(ids_list))
    #     if len(ids_list) != len(ids_set):
    #         return {"docs_ids": ids_set, "is_valid": False}

    #     # Step 7: 完全合规则
    #     return {"docs_ids": ids_set, "is_valid": True}


EXAMPLE_PROMPT = '''
- Example:
Question: When did the simpsons first air on television?
<answer>December 17, 1989</answer>

Question: When did the lightning thief book come out?
<answer>2005</answer>

Question: Who said i'm late i'm late for a very important date?
<answer>The White Rabbit</answer>

Question: Where does the short happy life of francis macomber take place?
<answer>Africa</answer>

Question: What was the fourth expansion pack for sims 2?
<answer>Pets</answer>

Question: Voice of the snake in the jungle book?
<answer>The Jungle Book (2016 film)</answer>

Question: How many seasons are there of star wars the clone wars?
<answer>6</answer>

Question: Which us president appears as a character in the play annie?
<answer>Franklin D. Roosevelt</answer>

Question: Are Calochone and Adlumia both plants?
<answer>yes</answer>

Question: Yukio Mishima and Roberto Bolaño, are Chilean?
<answer>no</answer>
'''

class AnswerGenerationAgent(BaseLLMClient):
    """
    根据问题和候选文档生成简洁准确的答案。
    支持单个和批量并行处理，并在 prompt 中加入 few-shot 样例。
    """

    def __init__(
        self,
        model_name: str = "gpt4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.0
    ):
        super().__init__(
            model_name=model_name,
            system_jinja2_path=system_jinja2_path,
            user_jinja2_path=user_jinja2_path,
            default_temperature=default_temperature
        )

    def _build_messages(self, query: str, documents: List[str]) -> List[dict]:
        """
        构造生成答案的 prompt（带样例 + 明确格式要求，输出必须在 <answer>...</answer> 标签中）
        """
        format_instruction = (
            "Please answer the Question strictly following the format shown in the examples above. "
            "You must output ONLY inside <answer>...</answer> and nothing outside these tags."
        )

        if documents:
            doc_content = ''.join([f"Document {doc_id}: {doc}\n\n" for doc_id, doc in enumerate(documents)])
            return [
                {
                    'role': 'system',
                    'content': (
                        f'You are a helpful, respectful and honest assistant. '
                        f'Your task is to provide a brief and accurate answer to the Question based on the given Documents. '
                        f'If you do not know the answer from the Documents, say "I don\'t know" and do not make up information.\n'
                        f'{EXAMPLE_PROMPT}\n{format_instruction}'
                    )
                },
                {
                    'role': 'assistant',
                    'content': 'Okay, I will answer the Question based on the given Documents inside <answer>...</answer> tags.'
                },
                {
                    'role': 'user',
                    'content': (
                        f'Question: {query}\n\n{doc_content}'
                        "Now, give the brief and accurate answer inside <answer>...</answer> tags."
                    )
                }
            ]
        else:
            return [
                {
                    'role': 'system',
                    'content': (
                        f'You are a helpful, respectful and honest assistant. '
                        f'Your task is to provide a brief and accurate answer to the Question using only the information you know. '
                        f'If you do not know the answer, say "I don\'t know" and do not make up information.\n'
                        f'{EXAMPLE_PROMPT}\n{format_instruction}'
                    )
                },
                {
                    'role': 'assistant',
                    'content': 'Okay, I will answer the Question inside <answer>...</answer> tags.'
                },
                {
                    'role': 'user',
                    'content': (
                        f'Question: {query}\n\n'
                        "Now, give the brief and accurate answer inside <answer>...</answer> tags."
                        "You must not answer \"I don't know.\" Always provide the answer you think is most correct."
                    )
                }
            ]

    def parse_response(self, context: str) -> Dict[str, object]:
        """
        严格模式：
        1. 必须且只能一个 <answer>...</answer> 标签，标签外无内容 → 否则 is_valid=False & 返回原始字符串
        2. 格式合规则 → 提取标签内内容并计算单词数
        3. 单词数 > 40 → is_valid=False & 返回原始字符串
        """
        if not isinstance(context, str) or not context.strip():
            return {"answer": context, "is_valid": False}

        # 格式匹配
        match = re.fullmatch(r'\s*<answer>\s*(.*?)\s*</answer>\s*', context, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            return {"answer": context.strip(), "is_valid": False}

        # 提取标签内的内容
        answer_text = match.group(1).strip()

        # 计算单词数
        word_count = len(re.findall(r'\w+', answer_text))

        # 超过 40 单词 → 无效
        if word_count > 40:
            return {"answer": context.strip(), "is_valid": False}

        # 合规则
        return {"answer": answer_text, "is_valid": True}


class AnswerSummarizationAgent(BaseLLMClient):
    """
    根据原始问题和分解子问题及其答案，生成简洁准确的最终答案。
    支持单个和批量并行处理，并在 prompt 中加入 few-shot 样例。
    """

    def __init__(
        self,
        model_name: str = "gpt4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.0
    ):
        super().__init__(
            model_name=model_name,
            system_jinja2_path=system_jinja2_path,
            user_jinja2_path=user_jinja2_path,
            default_temperature=default_temperature
        )

    def _build_messages(
        self,
        query: str,
        sub_queries: List[str],
        sub_answers: List[str]
    ) -> List[dict]:
        """
        构造生成最终答案的 prompt（带 few-shot 样例 + 明确格式要求，输出必须在 <answer>...</answer> 标签中）
        """
        format_instruction = (
            "Please answer the Original Question strictly following the format shown in the examples above. "
            "You must output ONLY inside <answer>...</answer> and nothing outside these tags."
        )

        # 构造子问题+答案部分
        observation = ''.join(
            [f"Sub-question {i+1}: {sq}\nAnswer: {sa}\n\n" for i, (sq, sa) in enumerate(zip(sub_queries, sub_answers))]
        )

        return [
            {
                'role': 'system',
                'content': (
                    f'You are a helpful, respectful and honest assistant. '
                    f'Your task is to predict the final answer to the Original Question based on the answers to its decomposed sub-questions. '
                    f'If you are not sure about the final answer, do not make up information.\n'
                    f'{EXAMPLE_PROMPT}\n{format_instruction}'
                )
            },
            {
                'role': 'assistant',
                'content': 'Okay, I will provide the final answer inside <answer>...</answer> tags.'
            },
            {
                'role': 'user',
                'content': (
                    f'Original Question: {query}\n\n'
                    f'{observation}'
                    f'Now, based on the above sub-questions and their answers, '
                    f'answer the Original Question: {query} inside <answer>...</answer> tags. '
                    'You must not answer "I don\'t know." Always provide the answer you think is most correct.'
                )
            }
        ]

    def parse_response(self, context: str) -> Dict[str, object]:
        """
        严格模式：
        1. 必须且只能一个 <answer>...</answer> 标签，标签外无内容 → 否则 is_valid=False & 返回原始字符串
        2. 格式合规则 → 提取标签内内容并计算单词数
        3. 单词数 > 40 → is_valid=False & 返回原始字符串
        """
        if not isinstance(context, str) or not context.strip():
            return {"answer": context, "is_valid": False}

        # 格式匹配
        match = re.fullmatch(r'\s*<answer>\s*(.*?)\s*</answer>\s*', context, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            return {"answer": context.strip(), "is_valid": False}

        # 提取标签内的内容
        answer_text = match.group(1).strip()

        # 计算单词数
        word_count = len(re.findall(r'\w+', answer_text))

        # 超过 40 单词 → 无效
        if word_count > 40:
            return {"answer": context.strip(), "is_valid": False}

        # 合规则
        return {"answer": answer_text, "is_valid": True}

class RetrievalTool:
    """
    RetrievalTool 调用远程搜索引擎 API，
    输入 question 和 N，返回 N 个最相关的文档。
    支持多个 API URL 池，每次随机选择，并在失败时重试。
    """

    def __init__(self, api_urls: List[str]):
        """
        初始化 RetrievalTool
        
        Args:
            api_urls (List[str]): 搜索引擎 API 地址池
        """
        if not api_urls:
            raise ValueError("api_urls 列表不能为空")
        self.api_urls = api_urls

    def query(self, question: str, N: int, max_attempts: int = 5) -> List[Dict[str, Any]]:
        """
        查询相关文档（支持随机 URL 及重试策略）
        
        Args:
            question (str): 查询问题
            N (int): 返回文档数量
            max_attempts (int): 失败重试次数上限
        
        Returns:
            List[Dict[str, Any]]: 文档列表
        """
        headers = {'Content-Type': 'application/json'}
        payload = {
            'questions': [question],
            'N': N
        }

        attempts = 0
        tried_urls = set()

        while attempts < max_attempts:
            # 从未尝试过的 URL 中随机选一个
            available_urls = [url for url in self.api_urls if url not in tried_urls]
            if not available_urls:
                # 所有 URL 都尝试过，重新允许重复尝试
                available_urls = self.api_urls

            api_url = random.choice(available_urls)
            tried_urls.add(api_url)

            try:
                response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=5)
                response.raise_for_status()  # HTTP 状态错误抛异常

                data = response.json()
                if not data or "top_k_docs" not in data[0]:
                    raise ValueError("Unexpected API response format")

                return data[0]["top_k_docs"]

            except (requests.exceptions.RequestException, ValueError) as e:
                attempts += 1
                print(f"⚠️ 第 {attempts} 次尝试失败（URL: {api_url}）：{e}")

        # 如果所有尝试都失败
        raise RuntimeError(f"❌ 请求失败次数超过 {max_attempts} 次，无法获取文档")


