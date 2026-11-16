"""Modal 镜像准备的确定性构建文本；由运行配置编译器调用。"""
import hashlib
import json
import shlex

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cpu_image_recipe(image: str, repo: str, has_setup: bool, entrypoint=None) -> str:
    text=(f'FROM {image}\nUSER root\n'
          'RUN apt-get update && apt-get install -y --no-install-recommends git sudo ca-certificates && rm -rf /var/lib/apt/lists/*\n'
          'RUN python3 -m pip install modal==1.6.1\n')
    if has_setup:
        text+='COPY runtime-setup.sh /tmp/runtime-setup.sh\nRUN bash /tmp/runtime-setup.sh\n'
        text+=f'ENV PYTHONPATH={repo}\n'
    text+=f'COPY sandbox_bootstrap.py /tmp/sandbox_bootstrap.py\nRUN python3 /tmp/sandbox_bootstrap.py {shlex.quote(repo)}\n'
    if entrypoint is not None:
        if not isinstance(entrypoint,list) or any(not isinstance(x,str) for x in entrypoint):
            raise ValueError('Image entrypoint override must be a list of strings')
        text+='ENTRYPOINT '+json.dumps(entrypoint)+'\n'
    return text


