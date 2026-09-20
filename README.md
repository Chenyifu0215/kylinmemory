# kylinmemory

kylinmemory 是一个独立的 Python 记忆系统，为 AI 应用提供对话存储、记忆提取、场景整理和用户画像能力。它接收对话或历史消息，返回可检索的记忆和可供模型使用的上下文，支持跨会话持久化。

项目提供 Python API、命令行和基于标准输入输出的 JSON Lines 服务，可独立运行，也可集成到已有应用中。记忆提取需要配置模型服务；对话回复由接入它的应用负责生成。

## 功能

- **四层记忆**：从原始对话逐步形成结构化记忆、主题场景和用户画像。
- **记忆检索**：支持全文检索和可选的向量检索，返回带来源消息 ID 的记忆。
- **上下文构建**：提供画像、场景导航和当前查询相关的召回内容。
- **会话管理**：支持持续写入、历史导入、会话切换和重启恢复。
- **本地存储**：使用 SQLite、JSONL 和 Markdown 保存记忆，用户画像加密存储。
- **模型接入**：支持 OpenAI 兼容接口及多种模型提供商，可通过 Python 注入客户端。

## 安装

需要 **Python 3.11 或更高版本**。在项目根目录执行：

```bash
python -m pip install .
kylinmemory --help
```

使用原生 Anthropic 接口时，安装对应的可选依赖：

```bash
python -m pip install '.[anthropic]'
```

也可以使用 uv：

```bash
uv sync --locked
uv run kylinmemory --help
```

以下命令以已安装的 `kylinmemory` 为例；使用 uv 时，在命令前加上 `uv run`。旧命令 `kylin-memory` 和 `kylin-memory-profile` 仍可使用；Python 导入名保持为 `kylin_memory`。

## 配置

创建独立的数据目录，并在其中保存 `config.yaml`：

```bash
mkdir -p data
```

下面是使用 OpenAI 兼容模型接口和全文检索的最小配置。将模型名称和地址替换为实际服务的配置，模型需要支持工具调用：

```yaml
# data/config.yaml
model:
  provider: custom
  model: your-model-name
  base_url: https://your-model-service.example/v1
  api_mode: chat_completions
  api_key_env: OPENAI_API_KEY

memory:
  atom:
    enabled: true
    embedding:
      mode: disabled

user_profile:
  enabled: true
```

通过环境变量提供 API key，也可以放在数据目录的 `.env` 文件中。环境变量优先：

```bash
export OPENAI_API_KEY='your-api-key'
```

向量检索配置见 [examples/config.yaml](examples/config.yaml)。启用前需要填写可用的 embedding 服务地址；设置 `embedding.mode: disabled` 可仅使用全文检索。

默认数据目录是 `~/.kylin-memory/`。可用 `--home`、Python 的 `home` 参数或 `KYLIN_MEMORY_HOME` 指定；未指定时也兼容 `HERMES_HOME`。以下示例统一使用 `./data`。

## 命令行

```bash
# 写入一轮对话
kylinmemory --home ./data --session demo observe '以后请叫我小王。' --assistant '好的，小王。'

# 检索记忆
kylinmemory --home ./data --session demo recall '小王' --limit 5

# 获取供模型使用的上下文
kylinmemory --home ./data --session demo context '怎么称呼用户？'

# 查看记忆状态和场景列表
kylinmemory --home ./data --session demo status
kylinmemory --home ./data --session demo scenes
```

命令输出 JSON。每条独立命令退出时会提交并关闭当前实例；需要持续处理多轮对话时，使用下面的 `serve` 或长期存活的 Python 实例。

## JSON Lines 服务

启动持续运行的本地服务：

```bash
kylinmemory --home ./data --session demo serve
```

标准输入每行接收一个 JSON 请求，标准输出每行返回一个 JSON 结果，日志写入标准错误。它使用本地进程通信，不提供 HTTP 或 MCP 接口。

例如，依次输入：

```json
{"id":1,"method":"observe","params":{"user":"以后请叫我小王。","assistant":"好的，小王。"}}
{"id":2,"method":"commit"}
{"id":3,"method":"consolidate"}
{"id":4,"method":"commit"}
{"id":5,"method":"recall","params":{"query":"小王","limit":5}}
{"id":6,"method":"context","params":{"query":"怎么称呼用户？"}}
```

成功响应格式为 `{"id":1,"result":...}`，失败响应格式为 `{"id":1,"error":{"type":"...","message":"..."}}`。非法请求不会终止服务，输入结束（EOF）时正常提交并关闭。

| 方法 | 用途 |
|---|---|
| `observe` | 写入一轮用户与助手对话 |
| `ingest` | 追加导入历史消息 |
| `recall` | 按查询检索结构化记忆 |
| `context` | 获取画像、场景导航和查询相关上下文 |
| `commit` | 提交 L1 提取，并从已有 L1/L2 更新画像 |
| `consolidate` | 显式执行场景整理 |
| `scenes` / `read_scene` | 列出场景或读取场景正文 |
| `status` | 查看当前会话和记忆状态 |
| `switch_session` | 提交旧会话并切换到新会话 |

## Python API

```python
from kylin_memory import MemorySystem

with MemorySystem(
    "./data",
    session_id="session-1",
    user_id="user-1",
    platform="api",
) as memory:
    memory.observe("以后请叫我小王。", "好的，小王。")
    memory.commit()
    memory.consolidate()
    memory.commit()

    atoms = memory.recall("小王", limit=5)
    context = memory.context("怎么称呼用户？")

    print(atoms)
    print(context["system"])
    print(context["request_context"])
```

`context["system"]` 包含画像和场景导航；`context["request_context"]` 包含当前查询的记忆召回，供调用方作为参考信息传给模型。

`commit()` 不等待后台 L2 整理。如果需要立即得到场景并将其用于画像，按示例执行 `commit()`、`consolidate()`、`commit()`。持续交互时也可使用自动调度。

导入历史对话时，向 `ingest()` 传入消息列表：

```python
with MemorySystem("./data", session_id="history-1") as memory:
    memory.ingest(
        [
            {"role": "user", "content": "以后请叫我小王。", "timestamp": 1700000000},
            {"role": "assistant", "content": "好的，小王。", "timestamp": 1700000001},
        ],
        backfill=True,
    )
    memory.commit()
```

`backfill=True` 仅用于新建的空会话，保留历史时间戳。导入为追加操作，重复导入会重复写入。也可使用 `kylinmemory --home ./data --session history-1 ingest messages.json --backfill` 导入 JSON 文件。

已有应用可通过 `client=` 注入 OpenAI 兼容客户端，或通过 `main_runtime=` 回调提供实时模型配置。

## 记忆层与存储

| 层级 | 内容 | 存储 |
|---|---|---|
| L0 | 从原始会话整理得到的证据 | `l0_memory.db`；完整消息保存在 `state.db` |
| L1 | 带类型、优先级和来源的结构化记忆 | `vectors.db`、`records/*.jsonl` |
| L2 | 按主题组织的场景 | `profiles/<scope>/scene_blocks/` |
| L3 | 用户画像 | `user_profile/profiles/*.profile.enc` |

数据保存在指定的 home 目录中。L2 按 team/agent 共享，L3 按用户隔离；需要完全隔离数据时，应使用不同的 home。一个数据目录应由一个运行实例负责调度。

画像加密不覆盖原始对话和其他记忆文件。备份时需同时保存数据文件及画像密钥。

## 开发

```bash
uv sync --locked --extra dev
bash scripts/run_tests.sh -q
```

核心代码位于 `src/kylin_memory/`，测试位于 `tests/`。运行所需的内置适配代码随包发布，无需安装原 Agent 仓库。

## 许可证

基于 Kylin Agent 的记忆子系统独立封装，采用 [MIT 许可证](LICENSE)。
