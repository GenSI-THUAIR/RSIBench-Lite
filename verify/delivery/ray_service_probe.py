"""通过已创建的命名 actor 查询全部 Ray rank 状态，不启动模型或重新创建 actor。

参数：期望节点数。示例：python ray_service_probe.py 4
缺 rank、连接错误原样失败；stdout 唯一 HARBOR_RAY_STATUS 帧供宿主解析。
"""
import json
import sys


def probe(count):
    import ray
    from serve_controller import ACTOR_PREFIX, NAMESPACE, ray_address
    ray.init(address=ray_address(), namespace=NAMESPACE, logging_level="ERROR")
    try:
        actors = [ray.get_actor(f"{ACTOR_PREFIX}{rank}", namespace=NAMESPACE) for rank in range(count)]
        return ray.get([actor.diagnostics.remote() for actor in actors], timeout=15)
    finally:
        ray.shutdown()


if __name__ == '__main__':
    count = int(sys.argv[1])
    if count < 1:
        raise ValueError('expected positive rank count')
    print('HARBOR_RAY_STATUS=' + json.dumps(probe(count)))
