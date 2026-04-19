# Copyright 2025 Individual Contributor: Mert Unsal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import os

import torch

from verl import DataProto
from verl.workers.reward_manager import register

# TANGO_GRPO flag - only create step reward tensors for TANGO_GRPO variants
REWARD_MODEL_TYPE = os.environ.get('REWARD_MODEL_TYPE', '')
IS_TANGO_GRPO = REWARD_MODEL_TYPE in ['PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO', 'PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO_STEP_PENALTY', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT']


@register("batch")
class BatchRewardManager:
    """
    A batch reward manager that computes rewards for a batch of data.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for decoding the responses.
        num_examine (int): The number of responses to examine.
        compute_score (callable): The function to compute the rewards.
        reward_fn_key (str): The key to use for the reward function.
        reward_kwargs (dict): The keyword arguments to pass to the reward function.
    """

    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs
    
    def _create_content_based_step_boundaries(self, response_text, step_positions, tokenizer):
        """
        Create step boundaries based on actual <step> content positions (RL Tango approach).
        
        Args:
            response_text: The decoded response text
            step_positions: List of (start, end) character positions for each <step> tag
            tokenizer: Tokenizer to map character positions to token positions
            
        Returns:
            List of token positions where step rewards should be placed (at end of each step)
        """
        if not step_positions:
            return []
        
        try:
            # Handle empty response text
            if not response_text or not response_text.strip():
                return [0] * len(step_positions)  # Return first token for each step
            
            # Create character-to-token mapping
            encoding = tokenizer(
                response_text, 
                return_offsets_mapping=True, 
                add_special_tokens=False
            )
            offset_mapping = encoding.offset_mapping
            
            # Handle empty tokenization
            if not offset_mapping:
                return [0] * len(step_positions)
            
            boundaries = []
            max_token_idx = len(offset_mapping) - 1
            
            # Map each step's end position to token index
            import os
            DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
            if DEBUG:
                print(f"🔍 STEP BOUNDARY DEBUG: step_positions={step_positions}")
                print(f"🔍 RESPONSE TEXT: {repr(response_text[:100])}...")
            
            for step_idx, (step_start, step_end) in enumerate(step_positions):
                token_end = None
                
                # Validate step positions
                if step_end < 0:
                    boundaries.append(0)
                    continue
                if step_end >= len(response_text):
                    boundaries.append(max_token_idx)
                    continue
                
                # Find the token that contains the step end character (adjusted to avoid overlap)
                # Use step_end - 1 to prevent overlap with next step's start position
                adjusted_step_end = max(0, step_end - 1)
                
                # Priority: token containing > token ending exactly at step end
                for token_idx, (char_start, char_end) in enumerate(offset_mapping):
                    # Skip invalid offset mappings (some tokenizers may have None values)
                    if char_start is None or char_end is None:
                        continue
                        
                    if char_start <= adjusted_step_end < char_end:
                        # Token contains the adjusted step end character - this is preferred
                        token_end = token_idx
                        break
                    elif char_end == adjusted_step_end:
                        # Token ends exactly at adjusted step end - use this if no containing token found
                        token_end = token_idx
                        # Don't break - keep looking for a containing token
                
                # Fallback: if no exact match, find the closest token
                if token_end is None:
                    best_token = 0
                    best_distance = float('inf')
                    for token_idx, (char_start, char_end) in enumerate(offset_mapping):
                        # Skip invalid offset mappings
                        if char_start is None or char_end is None:
                            continue
                            
                        distance = min(abs(char_start - adjusted_step_end), abs(char_end - adjusted_step_end))
                        if distance < best_distance:
                            best_distance = distance
                            best_token = token_idx
                    token_end = best_token
                
                # Ensure token_end is within valid range
                if token_end is not None:
                    token_end = max(0, min(token_end, max_token_idx))
                else:
                    token_end = 0  # Ultimate fallback
                
                if DEBUG:
                    char_at_step_end = response_text[adjusted_step_end-1:adjusted_step_end+1] if 0 <= adjusted_step_end-1 < len(response_text) else "N/A"
                    token_text = tokenizer.decode([encoding.input_ids[token_end]]) if token_end < len(encoding.input_ids) else "N/A"
                    print(f"🔍 STEP {step_idx}: chars({step_start},{step_end}) → adjusted_end={adjusted_step_end} → token[{token_end}]='{token_text}' | char_at_end='{char_at_step_end}'")
                
                boundaries.append(token_end)
            
            return boundaries
            
        except Exception as e:
            print(f"Error in content-based step boundary creation: {e}")
            # Return safe fallback - first token for each step
            return [0] * len(step_positions)

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        
        # DEBUG: Print input data lengths
        import os
        if os.environ.get('DEBUG', 'False').lower() == 'true':
            print(f"🔍 VERIFY INPUT DEBUG: len(data)={len(data)}, data_sources_len={len(data.non_tensor_batch[self.reward_fn_key])}, tensor_batch_size={data.batch.batch_size[0] if data.batch is not None else 'None'}")

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get("extra_info", [None] * len(data))

        # Compute scores for all responses
        scores = self.compute_score(
            data_sources=data_sources,
            solution_strs=responses_str,
            ground_truths=ground_truths,
            extra_infos=extras,
            **self.reward_kwargs,
        )
        
        # DEBUG: Print output scores length
        if os.environ.get('DEBUG', 'False').lower() == 'true':
            print(f"🔍 VERIFY OUTPUT DEBUG: len(scores)={len(scores)}, expected={len(data)}")
        
        return scores

    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        scores = self.verify(data)
        rewards = []
        already_printed = {}

        # DEBUG: Check for length mismatch before processing
        if os.environ.get('DEBUG', 'False').lower() == 'true':
            print(f"🔍 PROCESSING DEBUG: len(data)={len(data)}, len(scores)={len(scores)}, will_process={len(data)}_items")

        # Process rewards for all responses
        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            # For step rewards, use max sequence length to ensure consistent tensor shapes
            max_seq_length = data.batch["responses"].shape[1]
            score = scores[i]

            if isinstance(score, dict):
                reward = score["score"]
                # Extract reward method - must be explicitly provided
                if "reward_method" not in score:
                    raise KeyError(f"reward_method missing from score dict at index {i}: {score}")
                reward_method = score["reward_method"]
                reward_extra_info["reward_methods"].append(reward_method)
                
                # Handle step rewards ONLY for TANGO GRPO
                if IS_TANGO_GRPO and "step_rewards" in score and "step_count" in score:
                    step_rewards_list = score["step_rewards"]
                    step_count = score["step_count"]
                    step_boundary_strategy = score.get("step_boundary_strategy", "content_based")
                    step_positions = score.get("step_positions", [])
                    
                    # Create step reward tensor for this response (use max_seq_length for consistent shapes)
                    step_reward_tensor = torch.zeros(max_seq_length, dtype=torch.float32)
                    step_mask_tensor = torch.zeros(max_seq_length, dtype=torch.bool)
                    
                    if step_count > 0 and len(step_rewards_list) > 0:
                        if step_boundary_strategy == "content_based" and step_positions:
                            # RL Tango approach: use actual <step> content positions
                            response_text = self.tokenizer.decode(
                                data.batch["responses"][i][:length], 
                                skip_special_tokens=True
                            )
                            
                            token_boundaries = self._create_content_based_step_boundaries(
                                response_text, step_positions, self.tokenizer
                            )
                            
                            for step_idx, (step_reward, step_pos) in enumerate(zip(step_rewards_list, token_boundaries)):
                                if step_pos < max_seq_length:
                                    step_reward_tensor[step_pos] = step_reward
                                    step_mask_tensor[step_pos] = True
                                    
                    
                    # Store step reward information
                    reward_extra_info["step_rewards"].append(step_reward_tensor)
                    reward_extra_info["step_rewards_mask"].append(step_mask_tensor)
                    reward_extra_info["step_count"].append(step_count)
                    reward_extra_info["step_boundary_strategy"].append(step_boundary_strategy)
                elif IS_TANGO_GRPO:
                    # TANGO_GRPO but no step rewards - add empty tensors to maintain batch consistency
                    step_reward_tensor = torch.zeros(max_seq_length, dtype=torch.float32)
                    step_mask_tensor = torch.zeros(max_seq_length, dtype=torch.bool)
                    reward_extra_info["step_rewards"].append(step_reward_tensor)
                    reward_extra_info["step_rewards_mask"].append(step_mask_tensor)
                    reward_extra_info["step_count"].append(0)
                    reward_extra_info["step_boundary_strategy"].append("none")
                
                # Add other score information (exclude verbose fields and rename score)
                for key, value in score.items():
                    if key == "score":
                        # Rename "score" to "outcome_score" for rollout saving
                        reward_extra_info["outcome_score"].append(value)
                        if os.environ.get('DEBUG', 'False').lower() == 'true':
                            print(f"🔍 SCORE DEBUG [{i}]: Added score={value} to outcome_score (now {len(reward_extra_info['outcome_score'])} items)")
                    elif key == "step_rewards":
                        # Save original step_rewards list (not the tensor) for rollout analysis
                        reward_extra_info["step_rewards_list"].append(value)
                    elif key not in ["reward_method", "step_count", "step_boundary_strategy", "step_avg_score", "hybrid_score", "score", "step_rewards", "outcome_score"]:  # Don't duplicate these in extra_info, exclude verbose fields AND specially handled keys
                        reward_extra_info[key].append(value)
                        if os.environ.get('DEBUG', 'False').lower() == 'true':
                            if key == "outcome_score":
                                print(f"🚨 DUPLICATE DEBUG [{i}]: Added {key}={value} to reward_extra_info (now {len(reward_extra_info[key])} items)")
                            elif key == "step_penalty_applied":
                                print(f"🔍 PENALTY DEBUG [{i}]: Added step_penalty_applied={value}")
                            elif key in ["model_step_count", "prm_step_count"]:
                                print(f"🔍 COUNT DEBUG [{i}]: Added {key}={value}")
            else:
                reward = score
                print(f"Non-dict score at index {i}: {type(score)} = {score}")
                # For non-dict scores, assume RULE_BASED (legacy compatibility)
                reward_extra_info["reward_methods"].append("empty")
                reward_extra_info["ground_truth"].append(None)
                reward_extra_info["outcome_score"].append(score)  # Add score as outcome_score for consistency
                reward_extra_info["step_rewards_list"].append([])  # Empty step rewards for non-TANGO methods
                # Note: model_step_count, prm_step_count, step_penalty_applied are NOT added here
                # because they don't exist in non-dict scores, and we want to avoid double-counting
                
                # Add empty step reward tensors ONLY for TANGO_GRPO
                if IS_TANGO_GRPO:
                    step_reward_tensor = torch.zeros(max_seq_length, dtype=torch.float32)
                    step_mask_tensor = torch.zeros(max_seq_length, dtype=torch.bool)
                    reward_extra_info["step_rewards"].append(step_reward_tensor)
                    reward_extra_info["step_rewards_mask"].append(step_mask_tensor)
                    reward_extra_info["step_count"].append(0)
                    reward_extra_info["step_boundary_strategy"].append("none")
                    reward_extra_info["step_positions"].append([])  # Safety fallback for edge cases

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)
        
        # DEBUG: Final consistency check before return
        if os.environ.get('DEBUG', 'False').lower() == 'true':
            batch_size = len(data)
            print(f"🔍 FINAL CONSISTENCY CHECK: batch_size={batch_size}")
            for key, value_list in reward_extra_info.items():
                if len(value_list) != batch_size:
                    print(f"🚨 MISMATCH DETECTED: {key} has {len(value_list)} items, expected {batch_size}")
                else:
                    print(f"✅ {key}: {len(value_list)} items (correct)")
        

        # DEBUG: Check reward_extra_info consistency for TANGO_GRPO
        if IS_TANGO_GRPO and os.environ.get('DEBUG', 'False').lower() == 'true':
            print(f"🔍 BATCH EXTRA_INFO DEBUG: Processing {len(data)} responses")
            for key, value_list in reward_extra_info.items():
                print(f"  {key}: length={len(value_list)}")
                if len(value_list) != len(data):
                    print(f"  ⚠️  MISMATCH: {key} has {len(value_list)} items but batch has {len(data)} responses!")

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor