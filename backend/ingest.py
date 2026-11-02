import os
from git import Repo

SKIP_DIRS = {".git", "node_modules", "venv", "__pycache__", "dist", "build"}
CODE_EXTS = {".py", ".js", ".ts", ".java", ".go", ".md"}

def clone_repo(url: str, dest: str = "repos/target") -> str:
    if os.path.exists(dest):
        return dest
    Repo.clone_from(url, dest)
    return dest

def load_code_files(root: str):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if os.path.splitext(name)[1] in CODE_EXTS:
                path = os.path.join(dirpath, name)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    files.append({"path": path, "content": f.read()})
    return files

if __name__ == "__main__":
    root = clone_repo("https://github.com/pallets/flask")
    files = load_code_files(root)
    print(f"Loaded {len(files)} files")