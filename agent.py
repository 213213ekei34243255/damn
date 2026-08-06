
import json
import os
import logging
import requests
from typing import Dict, List, Any, Optional

# ================================
# Configuration
# ================================

LLAMA_URL = os.getenv(
    "LLAMA_URL",
    "http://127.0.0.1:8080/v1/chat/completions"
)

MODEL_NAME = os.getenv(
    "REXY_MODEL",
    "local"
)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("NoahAgent")


# ================================
# Supported Actions
# ================================

SUPPORTED_ACTIONS = {

    "navigate",

    "click",

    "type",

    "pressKey",

    "scroll",

    "wait",

    "observe",

    "extract",

    "reload",

    "goBack",

    "goForward",

    "download",

    "upload",

    "createTab",

    "closeTab",

    "switchTab",

    "complete"

}
ACTION_GO_BACK = "goBack"
ACTION_GO_FORWARD = "goForward"
ACTION_CREATE_TAB = "createTab"


# ================================
# Planner Prompt
# ================================

SYSTEM_PROMPT = """
You are Noah AI.

Codename: Rexy.

You are NOT a chatbot.

You are an autonomous browser agent.

Your responsibility is to convert user goals into executable plans.

You NEVER explain.

You NEVER answer conversationally.

You ONLY produce JSON.

Every response MUST follow this schema:

{
    "complete": false,
    "reason": "",
    "actions": [

    ]
}

Supported action types are:

navigate
click
type
pressKey
scroll
wait
observe
extract
reload
goBack
goForward
download
upload
createTab
closeTab
switchTab
complete

Every action should contain only the parameters required for that action.

Never return markdown.

Never return code fences.

Never return plain text.

Only valid JSON.
"""


# ================================
# Noah Planner
# ================================

class AgentPlanner:

    def __init__(self):

        self.session_memory = {}

        logger.info("🧠 Noah Agent Planner Initialized")

    def remember(self,
                 session_id: str,
                 memory: Dict):

        self.session_memory[session_id] = memory

    def recall(self,
               session_id: str):

        return self.session_memory.get(session_id, {})

    def build_prompt(
            self,
            goal: str,
            observation: Dict,
            memory: Dict
    ):
    
        """
        Build the reasoning prompt for Noah.
        """
    
        observation_json = json.dumps(
            self.summarize_observation(observation),
            indent=2,
            ensure_ascii=False
        )
    
        memory_json = json.dumps(
            memory,
            indent=2,
            ensure_ascii=False
        )
    
        user_prompt = f"""
    =========================
    MISSION
    =========================
    
    Your goal:
    
    {goal}
    
    
    =========================
    CURRENT BROWSER STATE
    =========================
    
    {observation_json}
    
    
    =========================
    MEMORY
    =========================
    
    {memory_json}
    
    
    =========================
    INSTRUCTIONS
    =========================
    
    Think like an autonomous AI agent.
    
    Observe the browser.
    
    Reason about the next step.
    
    If the goal has already been completed,
    return:
    
    {{
        "complete": true,
        "reason": "...",
        "actions":[]
    }}
    
    Otherwise return ONLY the next actions.
    
    Never return explanations.
    
    Never return markdown.
    
    Never return code.
    
    Only valid JSON.
    
    Keep plans short.
    
    Do NOT generate more than 5 actions.
    
    """
    
        return [
    
            {
    
                "role": "system",
    
                "content": SYSTEM_PROMPT
    
            },
    
            {
    
                "role": "user",
    
                "content": user_prompt
    
            }
    
        ]
    def summarize_observation(
            self,
            observation: Dict
    ):
    
        """
        Reduce unnecessary HTML before sending
        to the LLM.
        """
    
        summary = {
    
            "title": observation.get("title"),
    
            "url": observation.get("url"),
    
            "buttons": observation.get("buttons", [])[:20],
    
            "inputs": observation.get("inputs", [])[:20],
    
            "links": observation.get("links", [])[:20],
    
            "text": observation.get("text", "")[:4000]
    
        }

        return summary
    def merge_memory(
            self,
            session_id,
            runtime_memory
    ):

        saved = self.recall(session_id)

        merged = {}

        merged.update(saved)

        merged.update(runtime_memory)

        return merged

    def call_llm(
            self,
            messages: List[Dict]
    ):

        payload = {

            "model": MODEL_NAME,

            "messages": messages,

            "temperature": 0.15,

            "max_tokens": 1024

        }

        logger.info("🧠 Sending planning request to Rexy...")

        response = requests.post(

            LLAMA_URL,

            json=payload,

            timeout=120

        )

        response.raise_for_status()

        result = response.json()

        text = result["choices"][0]["message"]["content"].strip()

        return text
    def extract_json(
            self,
            text: str
    ):

        start = text.find("{")

        end = text.rfind("}")

        if start == -1 or end == -1:

            raise ValueError("No JSON found.")

        return json.loads(text[start:end + 1])

    def validate(
            self,
            plan: Dict
    ):

        if not isinstance(plan, dict):
            raise ValueError("Plan must be an object.")

        if "actions" not in plan:
            plan["actions"] = []

        if "complete" not in plan:
            plan["complete"] = False

        valid_actions = []

        for action in plan["actions"]:

            # NEW: coerce bare strings like "navigate" into {"type": "navigate"}
            if isinstance(action, str):
                action = {"type": action}

            if not isinstance(action, dict):
                continue

            action_type = action.get("type")

            if action_type not in SUPPORTED_ACTIONS:
                logger.warning(f"Unsupported action: {action_type}")
                continue

            valid_actions.append(self.sanitize_action(action))

        plan["actions"] = valid_actions

        return plan
    def generate_plan(
            self,
            messages
    ):

        retries = 3
        last_error = None

        for _ in range(retries):

            try:

                text = self.call_llm(messages)
                plan = self.extract_json(text)
                raw_action_count = len(plan.get("actions", []))

                validated = self.validate(plan)

                # NEW: if the model gave us actions but validation stripped
                # all of them (malformed schema), treat this as a failed
                # attempt and retry rather than silently returning empty.
                if raw_action_count > 0 and len(validated["actions"]) == 0 and not validated.get("complete"):
                    logger.warning(
                        f"Model returned {raw_action_count} action(s) but all failed validation; retrying."
                    )
                    last_error = ValueError("All actions failed schema validation")
                    continue

                logger.info(json.dumps(validated, indent=4))
                return validated

            except Exception as e:
                logger.warning(e)
                last_error = e

        logger.error(last_error)

        return {
            "complete": False,
            "reason": "Planner failed.",
            "actions": [
                {"type": "observe"}
            ]
        }
    def sanitize_action(
            self,
            action
    ):

        if action["type"] == "navigate":

            url = action.get("url", "")

            if not url.startswith("http"):

                action["url"] = "https://" + url

        return action

    def plan(
            self,
            goal: str,
            observation: Dict,
            memory: Dict,
            session_id: str = "default"
    ):

        logger.info("====================================")
        logger.info("🧠 NEW PLANNING CYCLE")
        logger.info("====================================")

        logger.info(f"Goal: {goal}")

        # Merge runtime memory with saved session memory
        merged_memory = self.merge_memory(
            session_id,
            memory
        )

        # Save latest memory
        self.remember(
            session_id,
            merged_memory
        )

        # Build LLM prompt
        messages = self.build_prompt(
            goal,
            observation,
            merged_memory
        )

        # Ask the model
        plan = self.generate_plan(messages)

        logger.info("Planning complete.")

        return plan



planner = AgentPlanner()


def get_agent_plan(
        goal: str,
        observation: Dict,
        memory: Dict,
        session_id: str = "default"
):

    return planner.plan(

        goal=goal,

        observation=observation,

        memory=memory,

        session_id=session_id

    )
