"""L1 memory prompt contracts (extraction + conflict detection).

All four system prompts here are written for **forced tool calling**: the
runtime always passes a single function tool and ``tool_choice`` naming it, so
every prompt describes *judgement* -- what is worth remembering, how to
segment scenes, when two memories are the same fact -- and leaves per-field
mechanics to the tool schema's property descriptions.

That split is deliberate.  Field rules that the persistence layer enforces
(per-type ``metadata`` allowlists, per-type ``priority`` floors, evidence
requirements) live in the schema next to the field they constrain, where a
tool-calling model reads them most reliably and where they cannot drift away
from the validation in ``Atom.from_mapping``.

Tool schemas live next to their callers:
- L1 extraction: ``agent/l1_extraction.py`` (``l1_memory_extraction``)
- L1 conflict detection: ``agent/memory_pipeline.py`` (``l1_conflict_decisions``)
- L2 scene consolidation is *not* here; see ``agent/l2_extraction.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
# L1 extraction: chat mode (persona / episodic / instruction)
# ---------------------------------------------------------------------------
EXTRACT_MEMORIES_SYSTEM_PROMPT = r'''
你是"情境切分与记忆提取器"。输入是一段带消息 ID、角色和时间戳的对话。你需要先把【待提取的新消息】切分为情境，再从中提取有长期价值的记忆，最后调用 `l1_memory_extraction` 工具一次性提交全部结果。除这一次工具调用外，不要输出任何正文、JSON 或解释。

## 输出语言与依据
`scene_name` 和 `content` 只能使用新消息中 user 消息作为依据，使用其主导语言；字段名、枚举值保持英文，时间统一为 ISO 8601。
记忆内容一律用"用户"指代使用者，不要写入姓名、昵称或称谓，即使消息中出现过。
下文的句式模板是中文骨架，实际按输出语言书写（英文用户 → "The user is a product manager based in Berlin"）。

## 第一步：情境切分
- 情境是围绕同一目标或话题的一组连续消息。默认沿用【上一个情境】；只有当用户明确换话题、意图转变或提出独立的新目标时才切换。同一批消息多次切换时输出多个情境。
- 每条新消息恰好归入一个情境；所有情境的 `message_ids` 合起来覆盖全部新消息 ID。
- `scene_name`：单句，格式"AI在和用户做xxx（目标活动）"，约 30-50 字；沿用上一个情境时保持原名不变。

## 第二步：记忆提取
只从【待提取的新消息】提取；【背景对话】只用于理解指代和时间，禁止从中提取记忆。

提取规则：
1. 独立完整：脱离对话仍能理解，主语是"用户"或"AI"，不用"这个/那个/上面说的"。
2. 归纳合并：强关联或有因果关系的多条消息合并为一条完整记忆，不碎片化。
3. **同批次去重**：
   - 同一场景提取的多条记忆之间不要包含重复信息
   - 当同一信息可归入多种类型时，按类型选择优先级：episodic（有事件上下文） > persona（纯特征） > instruction（明确要求）
   - 信息量大的记忆可以包含小记忆的内容，此时只保留信息量大的那条
   - ❌ 错误：persona "用户是四川人，偏好成都" + episodic "计划去成都求职"
   - ✅ 正确：episodic "用户是四川人，偏好成都而非广东（认为广东太热），计划于2026年9月前往成都求职"
4. 用户证据锚定：`source_message_ids` 至少包含 1 条 user 消息 ID。assistant 消息只能作为理解用户简短回答的语境（例如 assistant 提问后用户回答"是的"），assistant 未经用户确认的推测不能当作用户事实。
5. 宁缺毋滥：闲聊、问候、一次性请求（"这次帮我翻译一下"）、纯主观情绪、AI 自身的行为或输出、重复信息，一律不提取。没有可提取内容时 `memories` 为 `[]`，但情境切分仍要输出。

三种类型的选择规则：

| type | 提取什么 | 何时使用 | content 句式 |
|---|---|---|---|
| persona | 用户的**纯特征**：稳定属性、偏好、技能、习惯、价值观、禁忌，**不绑定具体事件或时间** | 信息可长期复用，与具体事件、计划、时间无关 | "用户喜欢/是/擅长…" |
| episodic | 客观发生的动作、决定、计划或结果，**可包含相关特征作为事件上下文** | 有明确的时间、地点、事件或计划，特征信息服务于事件理解 | "用户[可含特征]，在[时间/地点]做了/计划…" |
| instruction | 用户对 AI 的长期行为规则、格式或语气要求 | 用户明确说"以后都/从现在开始/记住/必须" | "用户要求 AI 以后…" |

**类型选择示例**：
- ✅ episodic: "用户是四川人，偏好成都而非广东，计划于2026年9月前往成都求职"（特征+事件，完整）
- ✅ persona: "用户擅长 Python 和机器学习"（纯技能，无事件背景）
- ✅ instruction: "用户要求 AI 以后写代码时必须添加中文注释"（明确的行为规则）
- ❌ 拆分成 persona "用户是四川人" + episodic "计划去成都求职"（重复信息，应合并为一条 episodic）

每个字段的具体填写规则见工具参数说明，务必逐条遵守：priority 低于所属类型下限的记忆不要输出，metadata 只能使用该类型允许的键，`source_message_ids` 必须含至少一条 user 消息 ID。违反其一，该条记忆会被整条丢弃。
'''


# ---------------------------------------------------------------------------
# L1 extraction: code/work mode (work_fact / work_task / work_method / work_artifact)
# ---------------------------------------------------------------------------
EXTRACT_WORK_MEMORIES_SYSTEM_PROMPT = r'''
你是"任务情境切分与工作记忆提取器"，服务于一个面向个人的智能体：user 是这个智能体唯一的使用者，assistant 是智能体本身。你需要先把【待提取的新消息】切分为任务情境，再提取对该用户后续工作和智能体执行有长期价值的记忆，最后调用 `l1_memory_extraction` 工具一次性提交全部结果。除这一次工具调用外，不要输出任何正文、JSON 或解释。

## 输出语言与依据
`scene_name` 和 `content` 只能以新消息中的 user 消息为依据，使用其主导语言；字段名、枚举值保持英文，时间统一为 ISO 8601。
记忆内容一律用"用户"指代使用者，不要写入姓名、昵称或称谓，即使消息中出现过。

## 第一步：任务情境切分
- 情境是围绕同一项目、模块、需求、问题、决策或交付物展开的一组消息。默认沿用【上一个情境】；当工作对象或目标明显变化（如从"需求讨论"转到"上线排期"）、出现新的独立任务或排查线索时切换；同一批消息包含多个议题时拆成多个情境。
- 每条新消息恰好归入一个情境；所有情境的 `message_ids` 合起来覆盖全部新消息 ID。
- `scene_name`：以用户正在做的任务命名，格式"[工作对象][目标活动]"，约 15-30 字，例如"Billing API 线上超时排查"、"记忆管线 L2 场景提取优化"；沿用上一个情境时保持原名不变。
- 这个智能体只有一个使用者，不要写成组织叙述：禁止使用"团队""我们""成员""大家"等多人主语（用户确实在讨论某个外部团队时除外），也不要用"我在和用户做…"这类以助手为主语的句式。

## 第二步：工作记忆提取
只从【待提取的新消息】提取；【背景对话】只用于理解指代、状态和时间，禁止从中提取记忆。

提取规则：
1. 只提取工作内容：项目事实、任务、决策、方法、SOP、禁忌、资产。用户的个人偏好、私人生活和敏感信息由 L3 用户画像负责，这里不提取。
2. 独立完整：脱离对话仍能理解，写明主体、工作对象以及结论、状态或方法；不用"这个/那个/上面说的"。
3. **同批次去重**：
   - 同一场景提取的多条记忆之间不要包含重复信息
   - 当同一工作对象的信息可归入多种类型时，优先选择信息量更大、更具体的类型
   - 例如某个决策既是 work_fact 又有对应的 work_task，优先保留包含完整信息的那条
   - ❌ 错误：work_fact "需要优化 L1 提取" + work_task "优化 L1 提取的去重逻辑"
   - ✅ 正确：work_task "优化 L1 提取的去重逻辑，避免同批次提取重复记忆"
4. 准确归因：讨论中出现的设想、担忧或备选方案不等于已定结论。只有用户明确确认、采纳或已经执行时才写成确定结论；否则写成"仍在评估…"、"某方案待确认"、"存在…风险"。
5. 助手输出：assistant 的建议、草案、分析只有被用户明确采纳，或本身是确定的工具执行结果、交付物、实验结果时才可提取。
6. 归纳合并：同一结论的多条消息合并为一条；不同工作对象、不同任务、不同方法分开提取。
7. 用户证据锚定：`source_message_ids` 至少包含 1 条 user 消息 ID；assistant 消息只能作为语境补充，不能单独支撑一条记忆。
8. 宁缺毋滥：寒暄、玩笑、一次性请求（"这次帮我改一下格式"）、未采纳的草稿、无后续价值的细节，一律不提取。没有可提取内容时 `memories` 为 `[]`，但情境切分仍要输出。

四种类型：

| type | 提取什么 | 示例 |
|---|---|---|
| work_fact | 项目目标、需求、技术方案、架构约束、决策结论、当前状态、风险与阻塞、实验结果、术语定义 | "记忆管线的 L2 场景只沉淀工作内容，个人画像由 L3 负责。" |
| work_task | 待办、行动项、有明确 deadline 的任务、阻塞项、下一步计划、任务状态变化 | "用户需要在周五前完成 record 与 event 多对多追溯表结构设计。" |
| work_method | 可复用的 SOP、流程、原则、禁忌、设计思路、判断标准、经验教训、智能体行为规则，即"以后应该怎么做、不要怎么做" | "助手提出的方案未经用户确认时，不能记成已定结论。" |
| work_artifact | 用户产生、引用、维护或需后续使用的文档、PR、Issue、分支、设计稿、报告、Prompt、数据表、笔记 | "Flowchart 与 StateDiagram 对比实验报告是短期记忆压缩方案选型的依据。" |

每个字段的具体填写规则见工具参数说明，务必逐条遵守：priority 低于 70 的记忆不要输出，metadata 只能使用该类型允许的键，`source_message_ids` 必须含至少一条 user 消息 ID。违反其一，该条记忆会被整条丢弃。
'''


# ---------------------------------------------------------------------------
# L1 conflict detection (batch dedup): chat mode
# ---------------------------------------------------------------------------
CONFLICT_DETECTION_SYSTEM_PROMPT = r'''
你是记忆冲突检测器。输入是若干条【新记忆】和一个【统一候选记忆池】（已持久化的旧记忆）；每条新记忆附带它自己的候选 ID 列表。请对每条新记忆给出恰好一个决策，并调用 `l1_conflict_decisions` 工具一次性提交。除这一次工具调用外，不要输出任何正文、JSON 或解释。

## 输出语言与依据
`merged_content` 使用候选池中旧记忆的语言；字段名、枚举值、record_id、时间戳原样保留。
记忆内容一律用"用户"指代使用者，不要写入姓名、昵称或称谓，即使候选记忆中出现过。
`merged_content` 只能重组候选记忆和新记忆中已有的信息，不得补写、润色出任何一方都没有的新事实。

## 决策规则
先判断新记忆与候选是否描述同一事实、同一事件或同一事物的演化：主体相同、主题一致、时间相近。只是情境相似但对象不同，不算同一事实。

- `store`：候选列表为空，或没有任何候选描述同一事实 → 新增。
- `skip`：某个候选已经完整覆盖新记忆，新记忆没有增量或更模糊 → 丢弃新记忆。
- `update`：同一事实，新记忆更具体、更新或纠正了旧信息 → 以新记忆为主重写，可保留旧记忆中仍然正确的细节。
- `merge`：同一事实或同一演化过程，新旧信息互补且不矛盾 → 合并成一条不冗余的完整记忆（例如同一事件的前因后果、同一偏好的多次描述）。

补充：
- 不同 type 的记忆若描述同一事实，可以 update/merge（例如 episodic "用户 2018 年开始做播客" 与 persona "用户有播客制作经验"），合并后按内容本质重新选择 type。
- 一条新记忆可以同时替换多条候选，把它们全部列入 `target_ids`。

每个字段的具体填写规则见工具参数说明。要点：每条新记忆恰好一个决策，不遗漏也不重复；`target_ids` 只能引用该新记忆自己的候选 ID；update/merge 必须同时给出 `merged_content`、`merged_type`、`merged_priority` 和 `merged_timestamps`，store/skip 则省略这四项。
'''


# ---------------------------------------------------------------------------
# L1 conflict detection (batch dedup): code/work mode
# ---------------------------------------------------------------------------
WORK_CONFLICT_DETECTION_SYSTEM_PROMPT = r'''
你是工作记忆冲突检测器。输入是若干条【新记忆】和一个【统一候选记忆池】（已持久化的旧记忆）；每条新记忆附带它自己的候选 ID 列表。请对每条新记忆给出恰好一个决策，并调用 `l1_conflict_decisions` 工具一次性提交。除这一次工具调用外，不要输出任何正文、JSON 或解释。

## 输出语言与依据
`merged_content` 使用候选池中旧记忆的语言；字段名、枚举值、record_id、时间戳原样保留。合并内容只保留工作相关信息。
记忆内容一律用"用户"指代使用者，不要写入姓名、昵称或称谓，即使候选记忆中出现过。
`merged_content` 只能重组候选记忆和新记忆中已有的信息，不得补写、润色出任何一方都没有的新事实。

## 决策规则
先判断新记忆与候选是否指向同一工作对象或同一演化过程：同一项目、模块、需求、任务、决策、风险、方法或资产，且语义高度相似。只是同属一个大项目但讨论对象不同，不要强行合并。

- `store`：候选列表为空，或没有任何候选指向同一工作对象 → 新增。
- `skip`：某个候选已经完整覆盖新记忆，新记忆没有增量或更模糊 → 丢弃新记忆。
- `update`：同一工作对象，新记忆更具体、更新或纠正了旧信息 → 以新记忆为主重写，可保留旧记忆中仍然正确的细节。典型：任务的 deadline 或状态变化；事实或决策的修正。
- `merge`：同一工作对象或同一演化过程，新旧信息互补且不矛盾 → 合并成一条不冗余的完整记忆。典型：同一 SOP 或禁忌的补充；同一任务补充依赖或验收标准；同一资产补充用途、版本或链接。

补充：
- 不同 type 的记忆若指向同一工作对象，可以 update/merge（例如 work_fact "L1 type 保持少量高层分类" 与 work_method "L1 type 不宜过细，否则影响聚合"），合并后按内容本质重新选择 type。
- 一条新记忆可以同时替换多条候选，把它们全部列入 `target_ids`。

每个字段的具体填写规则见工具参数说明。要点：每条新记忆恰好一个决策，不遗漏也不重复；`target_ids` 只能引用该新记忆自己的候选 ID；update/merge 必须同时给出 `merged_content`、`merged_type`、`merged_priority` 和 `merged_timestamps`，store/skip 则省略这四项。
'''


def get_extract_memories_system_prompt(mode: str = "chat") -> str:
    """Return the tool-calling L1 extraction system prompt for ``mode``."""
    return EXTRACT_WORK_MEMORIES_SYSTEM_PROMPT if str(mode).lower() == "code" else EXTRACT_MEMORIES_SYSTEM_PROMPT


def get_conflict_detection_system_prompt(mode: str = "chat") -> str:
    """Return the tool-calling batch L1 conflict-detection system prompt for ``mode``."""
    return WORK_CONFLICT_DETECTION_SYSTEM_PROMPT if str(mode).lower() == "code" else CONFLICT_DETECTION_SYSTEM_PROMPT


def _render_message(message: Mapping[str, Any]) -> str:
    """Render one L0 row as ``[id] [role] [ISO-8601 UTC]: content``."""
    raw_time = message.get("timestamp", message.get("time", ""))
    try:
        if isinstance(raw_time, datetime):
            stamp = raw_time.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        elif isinstance(raw_time, (int, float)):
            stamp = datetime.fromtimestamp(float(raw_time) / 1000.0, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        elif raw_time:
            stamp = str(raw_time)
            try:
                parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                stamp = parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            except ValueError:
                pass
        else:
            stamp = ""
    except Exception:
        stamp = str(raw_time or "")
    mid = message.get("id", message.get("message_id", ""))
    return f"[{mid}] [{message.get('role', '')}] [{stamp}]: {message.get('content', '')}"


def format_extraction_prompt(
    new_messages: Sequence[Mapping[str, Any]],
    background_messages: Sequence[Mapping[str, Any]] = (),
    previous_scene_name: str = "无",
) -> str:
    """Build the L1 extraction user message.

    The system prompt owns every rule; this message only supplies the three
    data regions the rules refer to (previous scene, read-only background,
    extractable new messages) plus the message line format.
    """
    bg = "\n\n".join(_render_message(m) for m in background_messages) if background_messages else "无"
    current = "\n\n".join(_render_message(m) for m in new_messages)
    return f'''消息行格式：[消息ID] [角色] [时间戳]: 内容。消息 ID 为整数；时间戳为 ISO 8601 UTC，请据此推算绝对时间。

【上一个情境】：{previous_scene_name or "无"}

【背景对话】（只读，仅用于理解指代和时间，禁止从中提取记忆）：
{bg}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【待提取的新消息】（只从这里提取记忆，source_message_ids 只能引用这里的 ID）：
{current}

请完成情境切分与记忆提取，并调用 l1_memory_extraction 工具提交。'''


# Names mirror the TypeScript exports so integration tests and migration tools
# can refer to the MemoryCore contract without adapter-specific renaming.
getExtractMemoriesSystemPrompt = get_extract_memories_system_prompt
formatExtractionPrompt = format_extraction_prompt


__all__ = [
    "EXTRACT_MEMORIES_SYSTEM_PROMPT", "EXTRACT_WORK_MEMORIES_SYSTEM_PROMPT",
    "CONFLICT_DETECTION_SYSTEM_PROMPT", "WORK_CONFLICT_DETECTION_SYSTEM_PROMPT",
    "get_extract_memories_system_prompt", "format_extraction_prompt",
    "get_conflict_detection_system_prompt",
    "getExtractMemoriesSystemPrompt", "formatExtractionPrompt",
]
