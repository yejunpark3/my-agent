import asyncio
import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from google.genai import types

import aiohttp
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentTask,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    JobContext,
    RunContext,
    ToolError,
    cli,
    function_tool,
    get_job_context,
    inference,
    llm,
    room_io,
    utils,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.beta.workflows import TaskGroup
from livekit.agents.llm.chat_context import FunctionCall
from livekit.agents.llm.utils import execute_function_call
from livekit.plugins import ai_coustics, google

logger = logging.getLogger("agent-assistant-capstone-2026")

load_dotenv(".env.local")


def _to_json_serializable(obj):
    """Convert dataclasses and nested structures to JSON-serializable form."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, list):
        return [_to_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    return obj


_JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}


def _load_config(jsonstring) -> dict:
    return json.loads(
        jsonstring
        # Path(__file__).with_name("prompt.json").read_text(encoding="utf-8")
    )


class UserDataTask(AgentTask):
    def __init__(
        self, agent_instructions: str, cfg: dict, extra_tools: list | None = None
    ):
        self._p = cfg["prompts"]["user_data_task"]
        fields = cfg["structured_output"]["fields"]
        wrapped_instructions = (
            self._p["no_greet_prefix"]
            + agent_instructions
            + "\n"
            + self._p["task_instructions"]
            + self._p["no_goodbye_suffix"]
        )
        self._partial_results: list[dict] = []
        # Fields come from prompt.json, so the tool schema is built at runtime.
        edit_tool = function_tool(
            self._edit_user_data_list,
            raw_schema={
                "name": "edit_user_data_list",
                "description": "Update the partial list: add a new data point to the running list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        f["name"]: {
                            "type": _JSON_TYPES[f["type"]],
                            "description": f["description"],
                        }
                        for f in fields
                    },
                    "required": [f["name"] for f in fields],
                },
            },
        )
        super().__init__(
            instructions=wrapped_instructions,
            tools=[*(extra_tools or []), edit_tool],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=self._p["on_enter_instructions"],
            allow_interruptions=True,
            tool_choice="auto",
        )

    async def _edit_user_data_list(self, context: RunContext, raw_arguments: dict):
        self._partial_results.append(raw_arguments)
        return self._p["edit_user_data_list_tool_response"].replace(
            "{count}", str(len(self._partial_results))
        )

    @function_tool(name="record_user_data")
    async def record_user_data(self, context: RunContext):
        """Call when the user has confirmed the list is complete."""
        self.complete(list(self._partial_results))


class DefaultAgent(Agent):
    def __init__(self, metadata) -> None:
        self._cfg = _load_config(metadata)
        self._p = self._cfg["prompts"]
        da = self._p["default_agent"]
        self._agent_instructions = da["agent_instructions"]
        super().__init__(
            instructions="",
            tools=[
                EndCallTool(
                    extra_description=da["end_call_tool"]["extra_description"],
                    end_instructions=da["end_call_tool"]["end_instructions"],
                    delete_room=True,
                ),
            ],
        )

    async def on_enter(self):
        da = self._p["default_agent"]
        greeting_instructions = da["greeting_instructions"]
        # The greeting must not ask a question — the first data collection task
        # asks the opening question. Without this guardrail the LLM tends to end
        # with an open-ended prompt ("How can I help?"), which collides with the
        # task's first turn.
        no_question_guardrail = da["no_question_guardrail"]
        await self.session.generate_reply(
            instructions="\n".join(
                part
                for part in (
                    self._agent_instructions,
                    greeting_instructions,
                    no_question_guardrail,
                )
                if part
            ),
            allow_interruptions=True,
        )
        # Propagate HTTP/client/MCP tools into each data collection task so
        # they're callable mid-task (e.g. looking up a customer record while
        # collecting details). EndCallTool is excluded here — it's invoked
        # programmatically in _finish_data_collection.
        _task_tools = [t for t in self.tools if not isinstance(t, EndCallTool)]
        # TaskGroup can only summarize with session.llm (a RealtimeModel here, which
        # can't), so it's off and _summarize_chat_ctx does it with a text LLM below.
        task_group = TaskGroup(
            chat_ctx=self.chat_ctx, 
            summarize_chat_ctx=False
        )
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: UserDataTask(
                agent_instructions=_ai, cfg=self._cfg, extra_tools=_tools
            ),
            id="user_data",
            description=self._p["user_data_task"]["task_group_description"],
        )
        try:
            group_result = await task_group
        except (ToolError, asyncio.CancelledError):
            logger.info(
                "data collection task group cancelled (participant likely disconnected)"
            )
            return

        await self._finish_data_collection(group_result.task_results)

    async def _finish_data_collection(self, task_results):
        """Serialize results, speak goodbye, and end the session."""
        serialized = _to_json_serializable(task_results)
        get_job_context().proc.userdata["dc_results"] = serialized
        fdc = self._p["finish_data_collection"]
        end_instructions = fdc["end_instructions"]

        summary_task: asyncio.Task | None = None
        summary_task = asyncio.create_task(self._send_dc_summary())

        # Remove EndCallTool from active tools so the LLM cannot call it
        # spontaneously during the goodbye speech (it is invoked programmatically below).
        await self.update_tools(
            [t for t in self.tools if not isinstance(t, EndCallTool)]
        )

        speech_handle = self.session.generate_reply(
            instructions=fdc["goodbye_prompt_template"].replace(
                "{end_instructions}", end_instructions
            ),
            tool_choice="none",
        )

        try:
            await speech_handle
            if summary_task:
                await summary_task
        except ConnectionError:
            logger.debug("user disconnected during goodbye speech")

        try:
            end_call_tool = next(
                (t for t in self.tools if isinstance(t, EndCallTool)), None
            )
            if not end_call_tool:
                end_call_tool = EndCallTool(
                    end_instructions=end_instructions,
                    delete_room=True,
                )

            tools_with_end_call = [*self.tools, end_call_tool]
            tool_ctx = llm.ToolContext(tools_with_end_call)
            end_call_id = utils.shortuuid("fnc_")
            tool_call = llm.FunctionToolCall(
                call_id=end_call_id,
                name="end_call",
                arguments="{}",
            )
            fnc_call = FunctionCall(
                call_id=end_call_id,
                name="end_call",
                arguments="{}",
            )
            call_ctx = RunContext(
                session=self.session,
                speech_handle=speech_handle,
                function_call=fnc_call,
            )
            await execute_function_call(
                tool_call,
                tool_ctx,
                call_ctx=call_ctx,
            )
        except (ConnectionError, RuntimeError):
            logger.debug("room already disconnected during end-call teardown")

    async def _send_dc_summary(self):
        """POST collected results (and a summary when enabled) to the webhook.

        Mirrors the session_end payload shape so downstream webhook consumers
        can handle both events uniformly.
        """
        ended_at = datetime.now(UTC)
        report = get_job_context().make_session_report()
        summary = None

        dc_results = get_job_context().proc.userdata.get("dc_results")
        headers_dict = {}

        body = {
            "job_id": report.job_id,
            "room_id": report.room_id,
            "room": report.room,
            "started_at": datetime.fromtimestamp(report.started_at, UTC)
            .isoformat()
            .replace("+00:00", "Z")
            if report.started_at
            else None,
            "ended_at": ended_at.isoformat().replace("+00:00", "Z"),
            "summary": summary,
            "results": dc_results,
        }

        try:
            session = utils.http_context.http_session()
            timeout = aiohttp.ClientTimeout(total=10)
            resp = await asyncio.shield(
                session.post(
                    "https://localhost:8080",
                    timeout=timeout,
                    json=body,
                    headers=headers_dict,
                )
            )
            if resp.status >= 400:
                raise ToolError(f"error: HTTP {resp.status}: {resp.reason}")
            await resp.release()
        except ToolError:
            raise
        except (TimeoutError, aiohttp.ClientError) as e:
            raise ToolError(f"error: {e!s}") from e


server = AgentServer()


@server.rtc_session(agent_name="assistant-8d0")
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        # llm=inference.RealtimeModel(
        #     model="google/gemini-3.8-live",
        #     voice="Puck",
        # ),
        llm=google.realtime.RealtimeModel(
            model="gemini-3.1-flash-live-preview",
            # model="gemini-3.8-live",
            vertexai=False,
            api_version="v1beta",
            voice="Puck",
            thinking_config={
                "thinking_level": "low",  # "low", "medium", "high" 중 선택
            },
        )
        # llm=google.realtime.RealtimeModel(
        #     model="gemini-3.8-live-extended-thinking",
        #     vertexai=False,
        #     api_version="v1beta",
        #     thinking_config={
        #         "thinking_level": "high",  # "low", "medium", "high" 중 선택
        #     },
        # )
    )
    ctx.proc.userdata["dc_results"] = None

    await session.start(
        agent=DefaultAgent(ctx.job.metadata),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S,
                ),
            ),
        ),
    )

    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=1.0),
    )

    await background_audio.start(room=ctx.room, agent_session=session)


if __name__ == "__main__":
    cli.run_app(server)
