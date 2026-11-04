import requests
import streamlit as st

st.set_page_config(page_title="Codebase Assistant", page_icon="🧭", layout="wide")

API = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")


def api_get(path):
    r = requests.get(f"{API}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def api_post(path, payload, timeout=600):
    r = requests.post(f"{API}{path}", json=payload, timeout=timeout)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(detail)
    return r.json()


# ---------- state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_repo" not in st.session_state:
    st.session_state.selected_repo = None

st.title("🧭 AI-Powered Codebase Assistant")
st.caption("Index a public GitHub repo, then ask questions, get summaries, or debug errors. "
           "Answers cite the exact files and lines.")

# ---------- sidebar: indexing ----------
with st.sidebar:
    st.header("1. Index a repo")
    repo_url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo")
    force = st.checkbox("Re-index if already indexed")
    if st.button("Index repository", use_container_width=True):
        if not repo_url.strip():
            st.warning("Enter a GitHub URL first.")
        else:
            with st.spinner("Cloning, chunking and embedding... this can take a minute or two"):
                try:
                    res = api_post("/index", {"repo_url": repo_url.strip(), "force": force})
                    st.session_state.selected_repo = res["repo_id"]
                    st.session_state.messages = []
                    if res.get("chunks"):
                        st.success(f"Indexed {res['files']} files into {res['chunks']} chunks.")
                    else:
                        st.info("This repo was already indexed.")
                except requests.exceptions.ConnectionError:
                    st.error("Can't reach the API. Is the backend running?")
                except Exception as e:
                    st.error(str(e))

    st.header("2. Choose a repo")
    try:
        repos = api_get("/repos")["repos"]
    except Exception:
        repos = []
        st.error("Can't reach the API. Start it with:\n\n`uvicorn main:app --app-dir backend --port 8000`")

    if repos:
        default = repos.index(st.session_state.selected_repo) if st.session_state.selected_repo in repos else 0
        repo_id = st.selectbox("Indexed repos", repos, index=default)
        if repo_id != st.session_state.selected_repo:
            st.session_state.selected_repo = repo_id
            st.session_state.messages = []
    else:
        repo_id = None
        st.info("No repos indexed yet.")

    st.header("3. Mode")
    mode = st.radio(
        "What do you want?",
        ["ask", "summarize", "debug"],
        format_func=lambda m: {"ask": "Ask a question",
                               "summarize": "Summarize a topic",
                               "debug": "Debug an error"}[m],
    )
    include_tests = st.checkbox("Include test files", value=False)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- chat ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.code(f'{s["file_path"]}:{s["start_line"]}-{s["end_line"]}  [{s["symbol"]}]',
                            language=None)

placeholder = {
    "ask": "e.g. Where is authentication handled?",
    "summarize": "e.g. the routing system",
    "debug": "Paste an error message or describe the problem",
}[mode]

question = st.chat_input(placeholder, disabled=not repo_id)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the code and thinking..."):
            try:
                res = api_post("/ask", {
                    "repo_id": repo_id,
                    "question": question,
                    "mode": mode,
                    "include_tests": include_tests,
                }, timeout=120)
                st.markdown(res["answer"])
                if res["sources"]:
                    with st.expander(f"Sources ({len(res['sources'])})"):
                        for s in res["sources"]:
                            st.code(f'{s["file_path"]}:{s["start_line"]}-{s["end_line"]}  [{s["symbol"]}]',
                                    language=None)
                st.session_state.messages.append(
                    {"role": "assistant", "content": res["answer"], "sources": res["sources"]})
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the API. Is the backend running?")
            except Exception as e:
                st.error(str(e))