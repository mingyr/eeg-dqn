import os
import math
import numpy as np
import scipy.io
from collections import OrderedDict

class Env():
    def __init__(self, decay = 0.25, filename = None):
        if filename:
            self._load(filename)
 
        self._decay = decay
        print("decay factor for vigilance tracing process: {}".format(decay))

        self._cur = 0;

        self._last_rt = 1.0

        # self._act = OrderedDict({'excited': 0.875, 'attentive': 1.25, 'alert' : 1.75, 'normal': 2.5, 'drowsy' : 3.5, 'sleepy' : 5.0, 'asleep': 7.0})
        # self._act = OrderedDict({'excited': 0.75, 'alert' : 1.75, 'drowsy' : 3.5, 'sleepy' : 6.0})
        
        rt_min = 0.5
        rt_max = 8.0
        rt_res = 0.1
        self._act = OrderedDict({i:v for i, v in enumerate(list(np.arange(rt_min, rt_max, rt_res)))})

        # self._act = OrderedDict({i:v for i, v in enumerate(list(np.arange(0, 1, 0.1)))})

    def _load(self, filename):
        assert(os.path.isfile(filename)), "Invalid file name"

        content = scipy.io.loadmat(filename)

        self._data = content['data']
        self._rt = content['RT']

        self._rt = np.squeeze(self._rt)

        print("file name -> {}".format(filename))
        print("total steps -> {}".format(len(self._rt)))

    def step(self, act):
        assert(act in self._act.keys()), "Invalid action"

        # print("internal state {}".format(self._last_rt))

        if self._cur < self._rt.size:
            state = self._data[:, :, self._cur]
            terminal = False

            self._last_rt = self._decay * self._last_rt + (1 - self._decay) * self._act[act]

            reward = 0.0 if self._rt[self._cur] == 0.0 else abs(self._last_rt - self._rt[self._cur])
            # reward = - math.log(1 + reward)
            reward = - reward

            self._cur += 1

            return state, reward, terminal

        else:
            state = None
            reward = None
            terminal = True
        
            return state, reward, terminal

    def reset(self, filename = None):
        if filename:
            self._load(filename)

        self._cur = 0
        self._last_rt = 1.0

    def __exit__(self):
        pass
 
    @property
    def actions(self):
        return self._act

    @property
    def predicted_rt(self):
        return self._last_rt

    @property
    def measured_rt(self):
        return self._rt[self._cur] if self._cur < self._rt.size else 0

def test_actions():
    env = Env()
    
    actions = env.actions

    for act in list(actions.keys()):
        print(act)

    
def test():
    import glob
    env = Env(0.2, glob.glob('data/train-128/*.mat')[0])

    acts = list(env.actions.keys())

    count = 0
    while True:
        print("{} - measured RT: {}".format(count, env.measured_rt))
        print("{} - predicted RT: {}".format(count, env.predicted_rt))

        act = acts[np.random.randint(0, len(acts))]

        print("{} - action: {}".format(count, act))
        state, reward, terminal = env.step(act)

        if not terminal:
            # print(state.shape)
            print("{} - reward: {}".format(count, reward))

            count += 1
        else:
            break
    
if __name__ == "__main__":
    # test_actions()
    test()

                
