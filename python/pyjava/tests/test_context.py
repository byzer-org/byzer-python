import unittest

from pyjava.api.mlsql import RayContext

import ray
# from ray.runtime_env import RuntimeEnv

@ray.remote(num_cpus=1, num_gpus=1)
class TestUDFWorker(object):
    def __int__(self):
        pass

class RayContextTestCase(unittest.TestCase):
    def test_raycontext_collect_as_file(self):
        ray_context = RayContext.connect(globals(), None)
        dfs = ray_context.collect_as_file(32)

        for i in range(2):
            print("======={}======".format(str(i)))
            for df in dfs:
                print(df)

        ray_context.context.build_result([{"content": "jackma"}])

    def test_ray(self):


if __name__ == '__main__':
    unittest.main()
