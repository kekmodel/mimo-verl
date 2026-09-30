# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Token and turn metadata for the general trajectory bridge.

These are observations recorded per rollout, consumed by the trainer's
``algorithm.length_penalty`` (``length_signals``) and
``algorithm.tool_call_error_penalty.mask_source=spans`` (turn spans + error flags).
"""


def trajectory_metadata(prompt_ids, response_mask, llm_turn_spans, tool_call_error_flags):
    # ``response_mask`` is the finalized (budget-clipped) response; spans recorded on the full
    # trace can overhang it. Clip them, and drop the flags of turns clipped away entirely, so
    # spans and flags stay aligned turn-for-turn.
    n = len(response_mask)
    llm_turn_spans = [(min(int(s), n), min(int(e), n)) for s, e in llm_turn_spans]
    flags = list(tool_call_error_flags)
    # A span is recorded when the model returns; the flag when the agent processes the turn.
    # A final turn that ended the run before it was processed (e.g. an empty or unparsable
    # tool call raising in the agent) has a span and no flag. It is not known to be a tool-call
    # error, so it gets False instead of making the whole row misaligned (which would drop the
    # flags of every earlier turn).
    if len(llm_turn_spans) == len(flags) + 1:
        flags.append(False)
        tool_call_error_flags = flags
    if len(flags) == len(llm_turn_spans):
        kept = [(span, flag) for span, flag in zip(llm_turn_spans, flags, strict=True) if span[0] < n]
        llm_turn_spans = [span for span, _ in kept]
        tool_call_error_flags = [flag for _, flag in kept]
    decode_length = sum(int(value) for value in response_mask)
    response_length = len(response_mask)
    leading_input = llm_turn_spans[0][0] if llm_turn_spans else response_length
    tool_length = response_length - leading_input - decode_length
    prompt_length = len(prompt_ids) + leading_input
    response_length -= leading_input
    return {
        "llm_turn_spans": [list(span) for span in llm_turn_spans],
        "tool_call_error_flags": list(tool_call_error_flags),
        "length_signals": {
            "prompt_length": prompt_length,
            "response_length": response_length,
            "decode_length": decode_length,
            "tool_length": tool_length,
            "prefill_length": prompt_length + tool_length,
            "turn_count": len(llm_turn_spans),
        },
    }
