from flask import Flask, request, jsonify, send_file
import os
import boto3


import importlib.util
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IT_ASSISTANT_PATH = BASE_DIR / "IT-Assistant.py"

def load_it_assistant_module():
    spec = importlib.util.spec_from_file_location("it_assistant", str(IT_ASSISTANT_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

it_mod = load_it_assistant_module()
AIAgent = it_mod.AIAgent  # your class

app = Flask(__name__)


agent = AIAgent()

if hasattr(it_mod, "load_sop_persistent"):
    it_mod.load_sop_persistent(agent, sop_folder="local_sops")


@app.get("/")
def home():
    # Serve UI file
    return send_file(str(BASE_DIR / "ui.html"))



@app.post("/ask")
def ask():
    data = request.get_json(force=True) or {}
    query_type = (data.get("query_type") or "ask_llm").strip()

    try:
        if query_type == "sop":
            q = (data.get("query") or "").strip()
            if not q: return jsonify({"error":"Missing query"}), 400
            out = agent.resolve_query("sop", query=q)

        elif query_type == "inventory":
            name = (data.get("name") or "").strip()
            if not name: return jsonify({"error":"Missing name"}), 400
            out = agent.resolve_query("inventory", name=name)

        elif query_type == "cloudscape_api":
            rule_name = (data.get("rule_name") or "").strip()
            api_base_url = (data.get("api_base_url") or "").strip()
            if not rule_name or not api_base_url:
                return jsonify({"error":"Missing rule_name/api_base_url"}), 400
            out = agent.resolve_query("cloudscape_api", api_base_url=api_base_url, rule_name=rule_name)

        else:  # ask_llm
            question = (data.get("question") or "").strip()
            if not question: return jsonify({"error":"Missing question"}), 400
            kwargs = {"question": question}
            if data.get("rule_name") and data.get("api_base_url"):
                kwargs["rule_name"] = data["rule_name"]
                kwargs["api_base_url"] = data["api_base_url"]
            out = agent.resolve_query("ask_llm", **kwargs)

        return jsonify({"answer": out, "error": ""}), 200
    except Exception as e:
        return jsonify({"answer": "", "error": str(e)}), 500
@app.get("/rules")
def rules():
    try:
        dynamodb = boto3.resource("dynamodb", region_name="ap-southeast-1")
        table = dynamodb.Table("cloudscape")

        rules = []
        scan_kwargs = {
            "ProjectionExpression": "rule_name"
        }

        # paginate scan
        while True:
            resp = table.scan(**scan_kwargs)
            for item in resp.get("Items", []):
                rn = item.get("rule_name")
                if rn:
                    rules.append(rn)

            last_key = resp.get("LastEvaluatedKey")
            if not last_key:
                break
            scan_kwargs["ExclusiveStartKey"] = last_key

        rules = sorted(set(rules))
        return jsonify({"rules": rules})

    except Exception as e:
        return jsonify({"rules": [], "error": str(e)}), 500


if __name__ == "__main__":
    # localhost only
    app.run(host="127.0.0.1", port=5000, debug=True)