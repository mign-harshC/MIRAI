# Modified for the MIRAI project, 2026.

import os

import numpy as np
from typing import Dict, List, Any
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl import DataProto

from qa_manager.base_agent_tag import (
    PlanningAgent,
    SubPlanningAgent,
    QueryRewriteAgent,
    QueryDecompositionAgentParallel,
    QueryDecompositionAgentSerial,
    DocumentSelectionAgent,
    AnswerGenerationAgent,
    AnswerSummarizationAgent,
    RetrievalTool
)

def remove_trailing_marker(text):
    # Check if the text ends with the marker and remove it
    marker = "<|im_end|>"
    if text.endswith(marker):
        return text[:-len(marker)]
    return text

class MIRAIManager:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer

        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", True)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "right")

        # agent name
        self.planning_agent = PlanningAgent()
        self.sub_planning_agent = SubPlanningAgent()
        self.query_rewriter = QueryRewriteAgent()
        self.qdp_agent = QueryDecompositionAgentParallel()
        self.qds_agent = QueryDecompositionAgentSerial()
        self.doc_selector = DocumentSelectionAgent()
        self.answer_gen = AnswerGenerationAgent()
        self.ans_summary = AnswerSummarizationAgent()
        configured_urls = config.get("retrieval_endpoints")
        if configured_urls:
            api_urls = list(configured_urls) if not isinstance(configured_urls, str) else configured_urls.split(",")
        else:
            api_urls = os.getenv("MIRAI_RETRIEVAL_URLS", "http://127.0.0.1:8000/search").split(",")
        api_urls = [url.strip() for url in api_urls if url.strip()]
        self.retrieval_tool = RetrievalTool(api_urls=api_urls)

        # 配置每个 agent_name 的参数规则
        self.AGENT_PARAM_RULES = {
            "PlanningAgent": [(str,)],                       # 一个 str
            "SubPlanningAgent": [(str,)],                       # 一个 str
            "QueryRewriteAgent": [(str, dict)],
            "QueryDecompositionAgentParallel": [(str,)],
            "QueryDecompositionAgentSerial": [(str,)],
            "DocumentSelectionAgent": [(str, list)],
            "AnswerGenerationAgent": [(str, list)],
            "AnswerSummarizationAgent": [(str, list, list)],
        }
        
    def trans_rawprompt_to_ids(self, agent_name: str, *args):
        """
        根据 agent_name 选择不同的参数组合执行。
        agent_name 必须是以下之一：
        """
        if agent_name == "PlanningAgent":
            # 接收 query: str
            if len(args) == 1 and isinstance(args[0], str):
                query = args[0]
                messages = self.planning_agent._build_workflow_messages(query)
            else:
                raise ValueError("PlanningAgent 需要一个字符串 query")

        elif agent_name == "SubPlanningAgent":
            # 接收 query: str
            if len(args) == 1 and isinstance(args[0], str):
                query = args[0]
                messages = self.sub_planning_agent._build_workflow_messages(query)
            else:
                raise ValueError("SubPlanningAgent 需要一个字符串 query")
        
        elif agent_name == "QueryRewriteAgent":
            # 接收 query & context
            if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], dict):
                query, context = args[0], args[1]
                if not context:
                    messages = self.query_rewriter._build_messages(query)
                else:
                    messages = self.query_rewriter._build_messages_with_context(query, context)
            else:
                raise ValueError("QueryRewriteAgent 需要一个query (str) & context (dict)")
        
        elif agent_name == "QueryDecompositionAgentParallel":
            # 接收 query: str
            if len(args) == 1 and isinstance(args[0], str):
                query = args[0]
                messages = self.qdp_agent._build_messages(query)
            else:
                raise ValueError("QueryDecompositionAgentParallel 需要一个字符串 query")

        elif agent_name == "QueryDecompositionAgentSerial":
            # 接收 query: str
            if len(args) == 1 and isinstance(args[0], str):
                query = args[0]
                messages = self.qds_agent._build_messages(query)
            else:
                raise ValueError("QueryDecompositionAgentSerial 需要一个字符串 query")

        elif agent_name == "DocumentSelectionAgent":
            # 接收 query & docs
            if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], list):
                query, docs = args[0], args[1]
                messages = self.doc_selector._build_messages(query, docs)
            else:
                raise ValueError("DocumentSelectionAgent 需要一个query (str) & docs (list)")
        
        elif agent_name == "AnswerGenerationAgent":
            # 接收 query & docs
            if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], list):
                query, docs = args[0], args[1]
                messages = self.answer_gen._build_messages(query, docs)
            else:
                raise ValueError("AnswerGenerationAgent 需要一个query (str) & docs (list)")
        
        elif agent_name == "AnswerSummarizationAgent":
            # 接收 query & subqs & subas
            if len(args) == 3 and isinstance(args[0], str) and isinstance(args[1], list) and isinstance(args[2], list):
                query, subqs, subas = args[0], args[1], args[2]
                messages = self.ans_summary._build_messages(query, subqs, subas)
            else:
                raise ValueError("AnswerSummarizationAgent 需要一个query (str) & subqs & subas")

        else:
            raise ValueError(f"未知的 agent_name: {agent_name}")

        single_dict = self.get_single_ids(messages)

        return single_dict
        
    
    def get_single_ids(self, messages):

        update_dict = {}

        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        update_dict["input_ids"] = input_ids[0]
        update_dict["attention_mask"] = attention_mask[0]
        update_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        update_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            update_dict["raw_prompt"] = messages

        return update_dict

    def get_answers_text(self, data: DataProto):
        
        predicted_answers_list = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            # valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            # valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            # sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences = valid_response_ids
            sequences_str = self.tokenizer.decode(sequences)
            sequences_str = remove_trailing_marker(sequences_str)
            predicted_answers_list.append(sequences_str)

        return predicted_answers_list
