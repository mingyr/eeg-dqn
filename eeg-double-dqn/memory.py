import numpy as np
import tensorflow as tf
from collections import deque

class TransitionMemory():
    def __init__(self, args):
        self._batchSize = args['batchSize']
        self._stateDim = args['stateDim']
        self._numActions = args['numActions']
        self._maxSize = args.get('maxSize', (1024 << 1))

        self._numEntries = 0
        self._insertIndex = 0
        self._seqNum = 0

        # list data structure used here for accessing efficiency
        self._s   = [None for i in range(self._maxSize)]
        self._a   = [0    for _ in range(self._maxSize)]
        self._r   = [0.0  for _ in range(self._maxSize)]
        self._t   = [1    for _ in range(self._maxSize)]
        self._seq = [0    for _ in range(self._maxSize)]

        # numpy array used here for compatibility with tensorflow
        self._batch_a      = np.zeros([self._batchSize], np.int32)
        self._batch_r      = np.zeros([self._batchSize], np.float32)
        self._batch_s      = np.zeros([self._batchSize] + self._stateDim, np.float32)
        self._batch_s2     = np.zeros([self._batchSize] + self._stateDim, np.float32)

        self._zero_state = np.zeros(self._stateDim, np.float32)

    # useless
    def reset(self):
        self._numEntries = 0
        self._insertIndex = 0

    @property
    def size(self):
        return self._numEntries

    def empty(self):
        return self._numEntries == 0

    # def sample(self, batch_size = 1)
    def sample(self):

        for i in range(self._batchSize):

            while True:
                index = np.random.randint(0, self._numEntries - 1)

                # print("{} -> {}".format(i, index))

                if self._t[index] == 0 and self._seq[index] < self._seq[index + 1]:
                    break

            self._batch_s[i, ...] = self._s[index]
            self._batch_a[i] = self._a[index]
            self._batch_r[i] = self._r[index]
            self._batch_s2[i, ...] = self._s[index + 1]

            # self.dump_entry(index)

        return self._batch_s, self._batch_a, self._batch_r, self._batch_s2
        
    # before entry, self._numEntries indicates the effective available elements
    # self._insertIndex indicates the next available slot
    # upon entry, self._numEntries ++
    # upon insertion, self._insertIndex ++ with potential wrap around
    def add(self, s, a, r, term):
        # assert(s != None), 'State cannot be None'
        assert(a != None), 'Action cannot be None'
        assert(r != None), 'Reward cannot be None'

        # Incremenet until at full capacity
        if self._numEntries < self._maxSize:
            self._numEntries += 1

        # Overwrite (s,a,r,t) at insertIndex
        self._s[self._insertIndex] = s
        self._a[self._insertIndex] = a
        self._r[self._insertIndex] = r
        self._t[self._insertIndex] = 1 if term else 0

        # update pivot
        self._seq[self._insertIndex] = self._seqNum
        self._seqNum += 1

        # Always insert at next index, then wrap around
        self._insertIndex += 1

        # Overwrite oldest experience once at capacity
        if self._insertIndex == self._maxSize:
            self._insertIndex = 0
       
    def dump(self):
        for i in range(self._numEntries):
            print("{}th entry:".format(i))
            print("\ts: {}".format(self._s[i]))
            print("\ta: {}".format(self._a[i]))
            print("\tr: {}".format(self._r[i]))
            print("\tt: {}".format(self._t[i]))
            print("")

    def dump_entry(self, i):
        print("{}th entry:".format(i))
        print("\ts: {}".format(self._s[i]))
        print("\ta: {}".format(self._a[i]))
        print("\tr: {}".format(self._r[i]))
        print("\tt: {}".format(self._t[i]))
        print("")
        
def test_syntax():
    
    from config import FLAGS

    transition_args = {
        'batchSize':   FLAGS.batch_size,
        'stateDim':    [30, 750],
        'numActions':  16,
        'maxSize':     FLAGS.replay_memory,
    }

    transitions = TransitionMemory(transition_args)

    print("syntax test passed")

def test_add():

    transition_args = {
        'batchSize':   2,
        'stateDim':    [30, 750],
        'numActions':  16,
        'maxSize':     8,
    }

    transitions = TransitionMemory(transition_args)

    for i in range(12):
        transitions.add(np.random.normal(0.0, 1.0, transition_args['stateDim']), i, float(i), int(np.random.randint(0, 2)))

    transitions.dump()

def test_sample():

    transition_args = {
        'batchSize':   2,
        'stateDim':    [2, 2],
        'numActions':  16,
        'maxSize':     1000,
    }

    transitions = TransitionMemory(transition_args)

    for i in range(1200):
        # transitions.add(np.full(transition_args['stateDim'], float(i)), i, float(i), int(np.random.randint(0, 2)))
        transitions.add(np.full(transition_args['stateDim'], float(i)), i, float(i), 0)

    s, a, r, s2 = transitions.sample()

    print("state: {}".format(s))
    print("action: {}".format(a))
    print("reward: {}".format(r))
    print("state2: {}".format(s2))

def test_flags():

    from config import FLAGS
    stateDim = [int(s) for s in FLAGS.image_dim.split(',')]
    print('stateDim: {}'.format(stateDim))

    transition_args = {
        'batchSize':   2, #FLAGS.batch_size,
        'stateDim':    stateDim,
        'numActions':  16,
        'maxSize':     FLAGS.replay_memory,
    }

    transitions = TransitionMemory(transition_args)

    for i in range(1200):
        # transitions.add(np.full(transition_args['stateDim'], float(i)), i, float(i), int(np.random.randint(0, 2)))
        transitions.add(np.full(transition_args['stateDim'], float(i)), i, float(i), 0)

    s, a, r, s2 = transitions.sample()

    print("state: {}".format(s))
    print("action: {}".format(a))
    print("reward: {}".format(r))
    print("state2: {}".format(s2))


if __name__ == "__main__":
    # test_syntax()
    # test_add()
    test_sample()
    # test_flags()



