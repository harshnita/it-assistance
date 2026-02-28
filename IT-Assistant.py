import sqlite3
import boto3
from botocore.exceptions import ClientError
from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
from pathlib import Path
###This is to hide embedding UNEXPECTED message
from transformers.utils import logging
logging.set_verbosity_error()
#####
import faiss
import numpy as np
import os
import chromadb
print(f"Chromadb_version:{chromadb.__version__}")
from transformers.utils import logging
logging.set_verbosity_error()
import logging
import json
from docx import Document
from pypdf import PdfReader
import requests

##### IMPORTANT NOTE
##SOP
#
# -----------------------------
# 1. SQLite Connection (Inventory)
# -----------------------------


##### Call OLLAMA
def call_ollama(prompt: str, model: str = "llama3.2", timeout: int = 120) -> str:
    r = requests.post(
        # "http://localhost:11434/api/generate",
        "http://http://127.0.0.1:5000//api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout
    )
    r.raise_for_status()
    return r.json()["response"]

def generate_engineer_answer(question: str, sop_text: str = "", remediation: str = "", model: str = "llama3.2") -> str:
    prompt = f"""IT Ops assistant.
    Use SOP and remediation if provided.

    Question:
    {question}

    SOP (best match):
    {sop_text}

    Cloudscape remediation (optional):
    {remediation}

    Return for engineer:
    1) Summary
    2) Step-by-step checks
    3) Fix / remediation
    4) Verification
    """
    return call_ollama(prompt, model=model)


def folder_mtime(folder: str) -> float:
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"SOP folder not found: {folder}")
    newest = p.stat().st_mtime
    for f in p.rglob("*"):
        if f.is_file():
            newest = max(newest, f.stat().st_mtime)
    return newest

def load_sop_persistent(agent, sop_folder="local_sops",
                        stamp_file=".sop_index.stamp",
                        index_path="sop_index.faiss",
                        docs_path="sop_docs.json"):
    current_mtime = folder_mtime(sop_folder)

    last_mtime = 0.0
    stamp = Path(stamp_file)
    if stamp.exists():
        try:
            last_mtime = float(stamp.read_text().strip() or "0")
        except ValueError:
            last_mtime = 0.0

    
    # If no change and disk files exist, load from disk
    if current_mtime <= last_mtime and agent.sop_db.load_from_disk(index_path, docs_path):
        print("✅ SOP loaded from disk (no changes)")
        return

    # Else rebuild from text files and save
    agent.sop_db.load_from_local(sop_folder)
    agent.sop_db.save_to_disk(index_path, docs_path)
    stamp.write_text(str(current_mtime))
    print("✅ SOP rebuilt and saved (changes detected or first run)")



def connect_sqlite(db_path="inventory.db"):
    conn = sqlite3.connect(db_path)
    return conn

# def get_inventory_item(conn, item_name):
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM inventory WHERE name=?", (item_name,))
#     # cursor.execute("SELECT * FROM inventory")
#     return cursor.fetchone()

def get_inventory_item(conn, item_name):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM inventory")
    # print("inventory row count:", cursor.fetchone()[0])
    # cursor.execute("SELECT * FROM inventory LIMIT 5")
    cursor.execute("SELECT * FROM inventory WHERE name=?", (item_name,))
    return cursor.fetchone()

# -----------------------------
# 2. AWS API + DynamoDB (Cloudscape Remediation)
# -----------------------------
def connect_dynamodb(region="ap-southeast-1"):
    dynamodb = boto3.resource("dynamodb", region_name=region)
    print("Dynamodb connection successful")
    return dynamodb

def remediate_non_compliant_item(table_name, item_id, remediation_action):
    dynamodb = connect_dynamodb()
    table = dynamodb.Table(table_name)
    try:
        response = table.update_item(
            Key={"rule_name": item_id},
            UpdateExpression="SET remediation_status = :status",
            ExpressionAttributeValues={":status": remediation_action},
            ReturnValues="UPDATED_NEW"
        )
        return response
    except ClientError as e:
        return {"error": str(e)}

def get_cloud_resource_details(resource_id):
    ec2 = boto3.client("ec2")
    try:
        response = ec2.describe_instances(InstanceIds=[resource_id])
        return response
    except ClientError as e:
        return {"error": str(e)}

def get_cloudscape_remediation_api(api_base_url: str, rule_name: str, timeout=2):
    """
    Calls API Gateway endpoint:
      GET {api_base_url}/cloudscape?rule_name=...
    Returns: remediation text (string) or {"error": "..."}.
    """
    try:
        url = api_base_url.rstrip("/") + "/cloudscape"
        resp = requests.get(url, params={"rule_name": rule_name}, timeout=timeout)
        resp.raise_for_status()

        data = resp.json()

        remediation = data.get("remediation")
        if remediation is None:
            return {"error": f"'remediation' key not found in API response: {data}"}

        return remediation

    except Exception as e:
        return {"error": str(e)}
# -----------------------------
# 3. Vector DB (Local Drive SOP Retrieval)
# -----------------------------
class SOPVectorDB:
    
    def __init__(self, model, dimension=384):
        print("Started Vector DB section")
        try:
            self.docs=[]
            self.index = faiss.IndexFlatL2(dimension)

            # self.model = SentenceTransformer("all-MiniLM-L6-v2")
            # self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") 
            self.model = model     
            # self.docs = []
            print("Embedding model loaded")
        except Exception as error1:
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    def add_document(self, doc_text):
        embedding = self.model.encode([doc_text])
        self.index.add(np.array(embedding).astype("float32"))
        self.docs.append(doc_text)

    def save_to_disk(self, index_path="sop_index.faiss", docs_path="sop_docs.json"):
        faiss.write_index(self.index, index_path)
        Path(docs_path).write_text(json.dumps(self.docs, ensure_ascii=False), encoding="utf-8")
        print(f"✅ SOP index saved: {index_path}, {docs_path}")
        
    def load_from_disk(self, index_path="sop_index.faiss", docs_path="sop_docs.json") -> bool:
        if not Path(index_path).exists() or not Path(docs_path).exists():
            return False
        
        self.index = faiss.read_index(index_path)
        self.docs = json.loads(Path(docs_path).read_text(encoding="utf-8"))
        print(f"✅ SOP index loaded from disk: {index_path}, {docs_path}")
        return True
    
    # def load_from_local(self, folder_path):
    #     for filename in os.listdir(folder_path):
    #         if filename.endswith(".txt"):
    #             with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
    #                 content = f.read()
    #                 self.add_document(content)

    # Fine tune
    def load_from_local(self, folder_path: str):
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"SOP folder not found: {folder_path}")

        def read_txt(p: Path) -> str:
            return p.read_text(encoding="utf-8-sig", errors="ignore").strip()

        def read_docx(p: Path) -> str:
            doc = Document(str(p))
            lines = [para.text.strip() for para in doc.paragraphs if para.text and para.text.strip()]
            return "\n".join(lines).strip()

        def read_pdf(p: Path) -> str:
            reader = PdfReader(str(p))
            pages = []
            for page in reader.pages:
                t = page.extract_text() or ""
                t = t.strip()
                if t:
                    pages.append(t)
            return "\n\n".join(pages).strip()

        loaded = 0
        for p in folder.rglob("*"):
            if not p.is_file():
                continue

            suffix = p.suffix.lower()
            try:
                if suffix == ".txt":
                    content = read_txt(p)
                elif suffix == ".docx":
                    content = read_docx(p)
                elif suffix == ".pdf":
                    content = read_pdf(p)
                else:
                    continue  

                if content:
                    self.add_document(content)
                    loaded += 1

            except Exception as e:
                print(f"Skipped {p.name}: {e}")

        print(f"SOP loaded files: {loaded} from {folder_path}")
    # def load_from_local(self, folder_path):
    #     if not os.path.isdir(folder_path):
    #         raise FileNotFoundError(f"SOP folder not found: {folder_path}")

    #     loaded = 0
    #     for filename in os.listdir(folder_path):
    #         if filename.endswith(".txt"):
    #             with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
    #                 content = f.read().strip()
    #                 if content:
    #                     self.add_document(content)
    #                     loaded += 1


    def search(self, query, top_k=2):
        if len(self.docs) ==0:
            return{"error": "No SOP documents loaded. Call load_from_local('local_sops) first."}
        embedding = self.model.encode([query])
        D, I = self.index.search(np.array(embedding).astype("float32"), top_k)

        results =[]
        for idx in I[0]:
            if idx == -1:
                continue
            results.append(self.docs[idx])
        
        return results

        # return [self.docs[i] for i in I[0]]

# -----------------------------
# 4. AI Agent Logic
# -----------------------------
class AIAgent:
    def __init__(self):
        # self.conn = connect_sqlite()
        # self.sop_db = SOPVectorDB()
        self.conn = connect_sqlite()
        self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.sop_db = SOPVectorDB(model=self.embed_model)
        print("No Error declaring connect sqlite and sopvectorDB")
    
    def resolve_query(self, query_type, **kwargs):
        print("Resolve Query")
        # kwarge["action"]
        print(f"query_type:{query_type},kwargs :{kwargs}")
        try:
            if query_type == "inventory":
                # user 'name' (or fallback to item_name)
                item_name=kwargs.get("name") or kwargs.get("item_name")
                # return get_inventory_item(self.conn, kwargs.get("Server-01"))
                return get_inventory_item(self.conn, item_name)
            
            elif query_type == "remediation":
                print("Inside elif query_type='remediation")
                return remediate_non_compliant_item(kwargs["table_name"], kwargs["rule_name"], kwargs["action"])

            elif query_type == "cloud_resource":
                return get_cloud_resource_details(kwargs["resource_id"])

            elif query_type == "sop":
                return self.sop_db.search(kwargs["query"],top_k=2)
            
            elif query_type == "cloudscape_api":
                return get_cloudscape_remediation_api(
                    api_base_url=kwargs["api_base_url"],
                    rule_name=kwargs["rule_name"]
                )
            
            elif query_type == "ask_llm":
                question = kwargs["question"]

                
                sop_results = self.sop_db.search(question, top_k=2)
                sop_text = sop_results[0] if isinstance(sop_results, list) and sop_results else ""

          
                remediation_text = kwargs.get("remediation", "")

                return generate_engineer_answer(question, sop_text, remediation_text, model="llama3.2")
            
            else:
                return {"error": "Unknown query type"}
        except Exception as e1:
            # print(f"Error:{e1}")
            return{"error":str(e1)}

        print("After if elif")
# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    agent = AIAgent()
    # load_sops_if_changed(agent, sop_folder="local_sops")
    # load_sops_on_start(agent, "local_sops")
    load_sop_persistent(agent, sop_folder="local_sops")
    # Load SOPs from local drive folder
    # agent.sop_db.load_from_local("local_sops")


    # Test SQLite inventory (one-by-one service test)
    print(agent.resolve_query("inventory", name="EC2-web-02"))


    print("---- SOP TEST ----")
    results = agent.resolve_query("sop", query="System status check failed")
    # print(results)
    for i, r in enumerate(results, start=1):
        print(f"\nResult {i}:\n{r[:800]}")

    print("---- LLM TEST ----")
    print(agent.resolve_query("ask_llm", question="How to troubleshoot ALB 504 errors?"))
    # print(agent.resolve_query("sop", query="How to fix 504 error")). # enable later

    # # Example Inventory lookup
    # print(agent.resolve_query("inventory", item_name="Server-01"))

    # # Example Cloud remediation
    # print(agent.resolve_query("remediation", table_name="CloudscapeTable", item_id="123", action="resolved"))
    # print(agent.resolve_query("remediation", table_name="cloudscape", rule_name="Lambda functions should be connected to a VPC", action="non-compliant"))
    # # Example Cloud resource details
    # print(agent.resolve_query("cloud_resource", resource_id="i-0abcd1234efgh5678"))

    print("---- CLOUDSCAPE API TEST ----")
    print(agent.resolve_query(
        "cloudscape_api",
        api_base_url="https://dohytw54cl.execute-api.ap-southeast-1.amazonaws.com",
        rule_name="WAF should be enabled on ALBs"
    ))