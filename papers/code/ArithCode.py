import numpy as np

class ArithCode(object):
    def __init__(self, ptab):
        self.ctab = np.cumsum(ptab)

    def decode(self, len, code):
        out = []
        for lx in xrange(len):
            pos = np.searchsorted(self.ctab, code)
            #print code, pos
            out.append(pos)
            low = 0 if pos == 0 else self.ctab[pos-1]
            high = self.ctab[pos]
            assert low < high
            code = (code - low) / (high - low)
        return out

    def restrict(self, low, high, sym):
        low_bound = 0 if sym == 0 else self.ctab[sym-1]
        high_bound = self.ctab[sym]
        range = high - low
        new_low = low + range * low_bound
        new_high = low + range * high_bound
        return (new_low, new_high)
        
    def encode(self, syms):
        low, high = 0, 1
        for sym in syms:
            low, high = self.restrict(low, high, sym)
        return low

testcases = (
    (2, 0, 1, 2),
    (0, 2, 2, 1, 0)
    )

ac = ArithCode([.2, .3, .5])
for test in testcases:
    code = ac.encode(test)
    print test, code, ac.decode(len(test), code)
