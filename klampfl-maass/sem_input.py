
import numpy

import pylab,matplotlib
import sys
import shelve

from sem_utils import *
#from speech_input import *

class InputGenerator(object):
    def __init__(self, n, dt):
        self.n = n
        self.dt = dt
        self.idx = -1
    def generate(self):
        pass
    def reset(self):
        pass
    def get_idx(self):
        return self.idx

class PoissonInputGenerator(InputGenerator):
    def __init__(self, n, dt, r):
        InputGenerator.__init__(self, n, dt)
        self.r = r
    def generate(self):
        return (numpy.random.exponential(1.0/self.r, self.n) <= self.dt).astype(int)

class RateInputGenerator(InputGenerator):
    def __init__(self, n, dt, tPattern, tau, nGroups, rate_range):
        InputGenerator.__init__(self, n, dt)
        self.tPattern = tPattern
        self.nGroups = nGroups
        self.group_size = n/nGroups
        self.min_rate = rate_range[0]
        self.max_rate = rate_range[1]
        self.tau = tau
        self.spikes = numpy.zeros(((int)(self.tau/self.dt),n), 'int')
        self.draw_pattern()
    def draw_pattern(self):
        self.current_rates = numpy.zeros(self.n)
        rates = numpy.zeros(self.nGroups)
        for g in range(self.nGroups):
            r = (self.max_rate-self.min_rate)*numpy.random.rand()
            new_rate = self.min_rate + r
            #if g==self.nGroups-1: #for now
                ##new_rate = self.min_rate + self.max_rate - self.current_rates[g*self.group_size-1]
                #new_rate = (self.nGroups-1)*self.max_rate - rates[:g].sum()
                #self.current_rates[g*self.group_size:] = new_rate
            #else:
                #self.current_rates[g*self.group_size:(g+1)*self.group_size] = new_rate
            self.current_rates[g*self.group_size:(g+1)*self.group_size] = new_rate
            rates[g] = new_rate
        #print self.current_rates, numpy.sum(self.current_rates)
    def generate(self, t):
        tp = t % (int)(self.tPattern/self.dt)
        if tp==0:
            self.draw_pattern()
        y = [(int)(numpy.random.exponential(1.0/r) <= self.dt) for r in self.current_rates]
        self.spikes = numpy.roll(self.spikes, -1, axis=0)
        self.spikes[-1,:] = y
        self.num_spikes = numpy.sum(self.spikes, axis=0)/self.tau
        #print self.current_rates
        #print self.num_spikes
        return numpy.asarray(y)

class AnalogRateInputGenerator(InputGenerator):
    def __init__(self, n, dt, tPattern, tau, nGroups, rate_range):
        InputGenerator.__init__(self, n, dt)
        self.tPattern = tPattern
        self.nGroups = nGroups
        self.group_size = n/nGroups
        self.min_rate = rate_range[0]
        self.max_rate = rate_range[1]
        self.tau = tau
        self.draw_pattern()
    def draw_pattern(self):
        self.current_rates = numpy.zeros(self.n)
        rates = numpy.zeros(self.nGroups)
        for g in range(self.nGroups):
            r = (self.max_rate-self.min_rate)*numpy.random.rand()
            new_rate = self.min_rate + r
            #if g==self.nGroups-1: #for now
                ##new_rate = self.min_rate + self.max_rate - self.current_rates[g*self.group_size-1]
                #new_rate = (self.nGroups-1)*self.max_rate - rates[:g].sum()
                #self.current_rates[g*self.group_size:] = new_rate
            #else:
                #self.current_rates[g*self.group_size:(g+1)*self.group_size] = new_rate
            self.current_rates[g*self.group_size:(g+1)*self.group_size] = new_rate
            rates[g] = new_rate
        #print self.current_rates, numpy.sum(self.current_rates)
    def generate(self, t):
        tp = t % (int)(self.tPattern/self.dt)
        if tp==0:
            self.draw_pattern()
        y = self.current_rates/self.max_rate
        #print self.current_rates
        #print self.num_spikes
        return numpy.asarray(y)

class RatePatternInputGenerator(InputGenerator):
    def __init__(self, n, dt, rmax, nPatterns, tPattern, nGroups, group_size_lims):
        InputGenerator.__init__(self, n, dt)
        self.rmax = rmax
        self.nPatterns = nPatterns
        self.tPattern = tPattern
        self.nGroups = nGroups
        self.min_group_size = group_size_lims[0]
        self.max_group_size = group_size_lims[1]
        self.generate_patterns()
    def generate_patterns(self):
        self.patterns = []
        self.sizes = numpy.zeros(self.nGroups, dtype='int')
        for group in range(self.nGroups-1):
            num_assigned = numpy.sum(self.sizes[:group])
            num_remaining = self.n - num_assigned
            max_group_size = min(self.max_group_size, num_remaining-(self.nGroups-group)*self.min_group_size)
            min_group_size = max(self.min_group_size, num_remaining-(self.nGroups-group)*self.max_group_size)
            group_size = numpy.random.randint(min_group_size,max_group_size+1)
            self.sizes[group] = group_size
        num_assigned = numpy.sum(self.sizes[:-1])
        group_size = self.n - num_assigned
        self.sizes[-1] = group_size
        for p in range(self.nPatterns):
            rate_pattern = numpy.zeros(self.n)
            for group in range(self.nGroups):
                rates = numpy.random.rand(self.sizes[group])
                num_assigned = numpy.sum(self.sizes[:group])
                group_size = self.sizes[group]
                rate_pattern[num_assigned:num_assigned+group_size] = self.rmax*rates/numpy.sum(rates)
            self.patterns.append(rate_pattern)
            #print "Pattern",p,rate_pattern,len(rate_pattern)
        #print "Sizes:",self.sizes,len(self.sizes),numpy.sum(self.sizes)
        self.draw_pattern()
    def draw_pattern(self):
        if hasattr(self,'idx'):
            self.idx = (self.idx+1)%self.nPatterns
        else:
            self.idx = numpy.random.randint(self.nPatterns)
        #self.idx = numpy.where(numpy.random.multinomial(1, self.prob))[0][0]
        self.current_pattern = self.patterns[self.idx]
    def generate(self, t):
        tp = t % (int)(self.tPattern/self.dt)
        if tp==0:
            self.draw_pattern()
        y = [(int)(numpy.random.exponential(1.0/r) <= self.dt) for r in self.current_pattern]
        return numpy.asarray(y)
    def __repr__(self):
        desc = "RatePatternInputGenerator:\n"
        desc += " input group sizes: " + str(self.sizes)
        desc += ", #groups: " + str(len(self.sizes))
        desc += ", #inputs: " + str(numpy.sum(self.sizes))
        desc += ", group rate: %dHz" % (self.rmax)
        return desc

class PatternInputGenerator(InputGenerator):
    def __init__(self, n, dt, r, nPatterns, tPattern, rNoise=0.0, prob=None):
        InputGenerator.__init__(self, n, dt)
        self.r = r
        self.nPatterns = nPatterns
        if not hasattr(tPattern,'__iter__'):
            self.tPattern = numpy.ones(self.nPatterns)*tPattern
        else:
            self.tPattern = numpy.asarray(tPattern)
        assert(len(self.tPattern)==self.nPatterns)
        self.rNoise = rNoise
        if prob is None:
            prob = numpy.ones(self.nPatterns)
        assert(len(prob)==self.nPatterns)
        self.prob = numpy.asfarray(prob)/numpy.sum(prob)
        self.noise = False
        self.last_pattern_start = 0
        self.time_warp = 1.0
        self.generate_patterns()
    def generate_patterns(self):
        """Generates n poisson spiketrains and stores them in **self.patterns**"""
        self.patterns = []
        self.pattern_count = dict()
        for p in range(self.nPatterns):
            pattern = []
            for n in range(self.n):
                st = generatePoissonSpikeTrain(self.tPattern[p], self.r)
                pattern.append(st)
            self.patterns.append(pattern)
            self.pattern_count[p] = 0
        self.draw_pattern()
    def draw_pattern(self):
        self.idx = numpy.where(numpy.random.multinomial(1, self.prob))[0][0]
        self.pattern_count[self.idx] += 1
        self.current_pattern = self.patterns[self.idx]
    def set_noise(self, noise):
        self.noise = noise
    def get_idx(self):
        if self.noise:
            return -1
        else:
            return self.idx
    def get_current_pattern_length(self):
        return (int)(self.tPattern[self.idx]/self.dt)
    def get_pattern_step(self, t):
        return t - self.last_pattern_start
    def generate(self, t):
        tp = t - self.last_pattern_start
        if tp%self.get_current_pattern_length()==0:
            self.draw_pattern()
            self.last_pattern_start = t
        if self.noise:
            y = (numpy.random.exponential(1.0/self.r, self.n) <= self.dt).astype(int)
        else:
            y = [numpy.any((st*self.time_warp>=tp*self.dt) & (st*self.time_warp<(tp+1)*self.dt)) for st in self.current_pattern]
            y = numpy.asarray(y).astype(int)
        if self.rNoise>0.0:
            y_noise = (numpy.random.exponential(1.0/self.rNoise, self.n) <= self.dt).astype(int)
            y = y | y_noise
        return y
    def reset(self, t=0):
        self.last_pattern_start = t

class SequentialPatternInputGenerator(PatternInputGenerator):
    def __init__(self, n, dt, r, nPatterns, tPattern, pattern_sequences, rNoise=0.0, mode='random_switching', sprob=0.5, time_warp_range=(1.0,1.0)):
        self.pattern_sequences = pattern_sequences
        self.mode = mode
        self.sidx = 0
        self.last_pattern_sequence_start = 0
        self.pattern_sequence_count = dict()
        for s in range(len(self.pattern_sequences)):
            self.pattern_sequence_count[s] = 0
        self.sprob = sprob
        self.next_idx = None
        self.time_warp_range = time_warp_range
        PatternInputGenerator.__init__(self, n, dt, r, nPatterns, tPattern, rNoise)
    def draw_pattern(self):
        if self.next_idx is not None:
            self.sidx = self.next_idx
            self.next_idx = None
        elif self.mode=='random_switching':
            if numpy.random.rand()<=self.sprob:
                sidx = numpy.random.randint(len(self.pattern_sequences))
                while sidx==self.sidx:
                    sidx = numpy.random.randint(len(self.pattern_sequences))
                #print "old: %d, new: %d" % (self.sidx, sidx)
                self.sidx = sidx
        elif self.mode=='random_ind':
            self.sidx = numpy.where(numpy.random.multinomial(1, self.sprob/numpy.sum(self.sprob)))[0][0]
        elif self.mode=='alternating':
            self.sidx = (self.sidx+1)%len(self.pattern_sequences)
        elif type(self.mode)==int:
            self.sidx = self.mode
        else:
            raise Exception("Unknown mode: %s" % (self.mode))
        self.pidx = 0
        self.current_pattern_sequence = self.pattern_sequences[self.sidx]
        self.idx = self.current_pattern_sequence[self.pidx]
        self.current_pattern = self.patterns[self.idx]
        if not self.pattern_sequence_count.has_key(self.sidx):
            self.pattern_sequence_count[self.sidx] = 0
        self.pattern_sequence_count[self.sidx] += 1
        #print self.sidx, self.pidx, self.idx
        self.ids = self.idx*numpy.ones(self.n, 'int')
        self.time_warp = self.time_warp_range[0] + (self.time_warp_range[1]-self.time_warp_range[0])*numpy.random.rand()
    def get_idx(self):
        if self.noise:
            return -1
        else:
            return self.ids
    def get_sidx(self):
        if self.noise:
            return -1
        else:
            return self.sidx
    def get_current_pattern_length(self):
        return (int)(self.time_warp*self.tPattern[self.idx]/self.dt)
    def get_current_pattern_sequence_length(self):
        return numpy.sum([(int)(self.time_warp*self.tPattern[i]/self.dt) for i in self.current_pattern_sequence])
    def get_pattern_step(self, t):
        return t - self.last_pattern_start
    def generate(self, t):
        tp = t - self.last_pattern_start
        tps = t - self.last_pattern_sequence_start
        #print " ",t,tp,tps,self.last_pattern_start,self.last_pattern_sequence_start
        #if tps%self.get_current_pattern_sequence_length()==0:
        self.ids = self.idx*numpy.ones(self.n, 'int')
        if tps==self.get_current_pattern_sequence_length():  # if end of the sequence is reached
            self.draw_pattern()
            #print t, "new sequence", self.current_pattern_sequence, "pattern", self.idx
            self.last_pattern_sequence_start = t
            self.last_pattern_start = t
        elif tp==self.get_current_pattern_length():  # if end of the pattern is reached
            self.pidx += 1
            self.idx = self.current_pattern_sequence[self.pidx]
            self.current_pattern = self.patterns[self.idx]
            self.last_pattern_start = t
            self.ids = self.idx*numpy.ones(self.n, 'int')
            #print t, "pattern", self.idx
        if self.noise:
            y = (numpy.random.exponential(1.0/self.r, self.n) <= self.dt).astype(int)
            assert False, "self.noise is true."
        else:
            y = [numpy.any((st*self.time_warp>=tp*self.dt) & (st*self.time_warp<(tp+1)*self.dt)) for st in self.current_pattern]
            y = numpy.asarray(y).astype(int)
        if self.rNoise>0.0:
            y_noise = (numpy.random.exponential(1.0/self.rNoise, self.n) <= self.dt).astype(int)
            #print numpy.where(y_noise)[0]
            self.ids[numpy.where(y_noise)[0]] = -1
            y = y | y_noise
        return y
    def reset(self, t=0, draw=True):
        self.last_pattern_sequence_start = t
        self.last_pattern_start = t
        self.pidx = 0
        if draw:
            self.draw_pattern()

class EmbeddedPatternInputGenerator(InputGenerator):
    def __init__(self, inpgen, tNoiseRange=(100e-3, 100e-3), prob=0.5):
        self.inpgen = inpgen  # inpgen_seq
        self.dt = self.inpgen.dt
        self.r = self.inpgen.r# + self.inpgen.rNoise
        self.n = self.inpgen.n
        self.tNoiseRange = tNoiseRange
        self.tNoise = self.tNoiseRange[0]+numpy.random.rand()*(self.tNoiseRange[1]-self.tNoiseRange[0])
        self.prob = prob
        self.noise_phase = False
        self.last_pattern_start = 0
        self.inpgen.generate_patterns()
        self.draw_pattern()
    def draw_pattern(self):
        #print "EmbeddedPatternInputGenerator.drawPattern"
        self.noise_phase = (numpy.random.rand()<=self.prob) and not self.noise_phase
        if not self.noise_phase:
            self.inpgen.draw_pattern()
            self.idx = self.inpgen.idx
            if hasattr(self.inpgen, 'current_pattern_sequence'):
                self.time_to_draw = self.inpgen.get_current_pattern_sequence_length()
            else:
                self.time_to_draw = self.inpgen.get_current_pattern_length()
        else:
            self.idx = -1
            self.tNoise = self.tNoiseRange[0]+numpy.random.rand()*(self.tNoiseRange[1]-self.tNoiseRange[0])
            self.time_to_draw = self.get_current_noise_length()
        #print self.noise_phase, self.idx, self.time_to_draw
        #if self.idx>=0:
            #print self.inpgen.stimulus.file, self.inpgen.stimulus.Tsim
    def get_idx(self):
        if self.noise_phase:
            return -1*numpy.ones(self.n, 'int')
        else:
            return self.inpgen.get_idx()
    def get_sidx(self):
        if self.noise_phase:
            return -1
        else:
            if hasattr(self.inpgen, 'sidx'):
                return self.inpgen.sidx
            else:
                return self.inpgen.idx
    def get_current_noise_length(self):
        return (int)(self.tNoise/self.dt)
    def get_pattern_step(self, t):
        return self.inpgen.get_pattern_step(t)
    def generate(self, t):
        tp = t-self.last_pattern_start
        #print t,tp,self.time_to_draw
        if tp==self.time_to_draw:
            #print "EmbeddedPatternInputGenerator: t =", t
            self.draw_pattern()
            self.last_pattern_start = t
            self.inpgen.reset(t,draw=False)
        if self.noise_phase:
            y = (numpy.random.exponential(1.0/(self.r), self.n) <= self.dt).astype(int)
        else:
            y = self.inpgen.generate(t)
        return y
    def reset(self, t=0):
        self.noise_phase = False
        self.inpgen.reset(draw=False)
        self.last_pattern_start=t
        self.draw_pattern()

class CombinedInputGenerator(InputGenerator):
    def __init__(self, inpgen_list):
        self.inpgen_list = inpgen_list
        self.draw_pattern()
    def draw_pattern(self):
        for inpgen in self.inpgen_list:
            inpgen.draw_pattern()
        #self.ids = numpy.concatenate([[inpgen.idx]*inpgen.n for inpgen in self.inpgen_list])
        self.ids = numpy.concatenate([inpgen.get_idx() for inpgen in self.inpgen_list])
    def get_idx(self):
        #self.ids = numpy.concatenate([[inpgen.idx]*inpgen.n for inpgen in self.inpgen_list])
        self.ids = numpy.concatenate([inpgen.get_idx() for inpgen in self.inpgen_list])
        return self.ids
    def generate(self, t):
        return numpy.concatenate([inpgen.generate(t) for inpgen in self.inpgen_list])
    def reset(self, t=0):
        for inpgen in self.inpgen_list:
            inpgen.reset(t)

class SwitchableInputGenerator(InputGenerator):
    def __init__(self, inpgen_list):
        self.inpgen_list = inpgen_list
        self.inpgen_idx = 0
        self.inpgen = self.inpgen_list[self.inpgen_idx]
        self.draw_pattern()
    def draw_pattern(self):
        for inpgen in self.inpgen_list:
            inpgen.draw_pattern()
        self.idx = self.inpgen.get_idx()
    def get_idx(self):
        return self.inpgen.get_idx()
    def switch(self, new_idx):
        self.inpgen_idx = new_idx
        self.inpgen = self.inpgen_list[self.inpgen_idx]
    def generate(self, t):
        return self.inpgen.generate(t)
    def reset(self, t=0):
        for inpgen in self.inpgen_list:
            inpgen.reset(t)

class PreprocessedSpeechInputGenerator(InputGenerator):
    def __init__(self, n, dt, r, rNoise=0.0, sprob=0.5, poisson=False, time_warp_range=(1.0, 1.0), **kwds):
        InputGenerator.__init__(self, n, dt)
        self.path = os.environ["HOME"] + '/research/SEM/speech'
        self.inpgen = PreprocessedSpeech(path=self.path, rev_st=True, **kwds)
        self.rNoise = rNoise
        self.r = r
        self.sprob = sprob
        self.poisson = poisson
        self.noise = False
        self.last_pattern_start = 0
        self.digits = numpy.asarray(self.inpgen.digits, 'int')
        self.rev_digits = numpy.asarray(self.inpgen.rev_digits, 'int')
        self.all_digits = numpy.asarray(self.digits.tolist()+self.rev_digits.tolist(), 'int')
        self.idx = 0
        self.next_idx = None
        self.ids = self.idx*numpy.ones(self.n, 'int')
        self.noise_period = None
        self.mixed = False
        self.time_warp_range = time_warp_range
        self.draw_pattern()
    def generate_patterns(self):
        pass
    def get_idx_from_stim(self, stim):
        if stim.reversed:
            idx = numpy.where(self.all_digits==int(stim.digit))[0][-1]
        else:
            idx = numpy.where(self.all_digits==int(stim.digit))[0][0]
        return idx
    def draw_pattern(self):
        if self.mixed:
            self.draw_mixed()
        else:
            self.draw_single()
    def draw_single(self):
        #print "PreprocessedSpeechInputGenerator.drawPattern"
        if len(self.all_digits)==1:
            stim = self.inpgen.generate()
        else:
            if self.next_idx is not None:
                self.idx = self.next_idx
                self.next_idx = None
            #print "idx:", self.idx
            if numpy.random.rand()<=self.sprob:
                #print "switch"
                stim = self.inpgen.generate()
                didx = self.get_idx_from_stim(stim)
                #print stim.file, stim.Tsim, stim.reversed, didx
                while didx==self.idx:
                    stim = self.inpgen.generate()
                    didx = self.get_idx_from_stim(stim)
                    #print stim.file, stim.Tsim, stim.reversed
            else:
                #print "no switch"
                stim = self.inpgen.generate()
                didx = self.get_idx_from_stim(stim)
                #print stim.file, stim.Tsim, stim.reversed, didx
                while not didx==self.idx:
                    stim = self.inpgen.generate()
                    didx = self.get_idx_from_stim(stim)
                    #print stim.file, stim.Tsim, stim.reversed
        print stim.file, stim.Tsim
        self.stimulus = stim
        self.cochleagram = getCochleagram(stim.speaker,stim.utterance,stim.digit,h5filename=self.path+'/cochleagrams.h5')
        self.idx = self.get_idx_from_stim(self.stimulus)
        if stim.reversed:
            self.cochleagram = self.cochleagram[:,::-1]
        #print self.idx
        self.time_warp = self.time_warp_range[0] + (self.time_warp_range[1]-self.time_warp_range[0])*numpy.random.rand()
        self.Tsim = self.stimulus.Tsim*self.time_warp
    def draw_mixed(self):
        print "mixing..."
        self.draw_single()
        stim1 = self.stimulus
        self.draw_single()
        stim2 = self.stimulus
        coch1 = getCochleagram(stim1.speaker,stim1.utterance,stim1.digit,h5filename=self.path+'/cochleagrams.h5')
        coch2 = getCochleagram(stim2.speaker,stim2.utterance,stim2.digit,h5filename=self.path+'/cochleagrams.h5')
        self.cochleagram = getMeanCochleagram(coch1,coch2)

        self.stimulus = PreprocessedSpeechStimulus()
        for c in range(self.cochleagram.shape[0]):
            channel = Channel(BSA(self.cochleagram[c,:],50))
            self.stimulus.channel.append(channel)
        self.stimulus.Tsim = max([numpy.max(ch.data) for ch in self.stimulus.channel if len(ch.data)>0])
        self.Tsim = self.stimulus.Tsim*self.time_warp
        self.idx = self.get_idx_from_stim(stim1) + self.get_idx_from_stim(stim2)+1
    def set_noise(self, noise):
        self.noise = noise
    def get_idx(self):
        if self.noise:
            return -1
        else:
            return self.ids
    def get_current_pattern_length(self):
        return (int)(self.Tsim/self.dt)
    def get_pattern_step(self, t):
        return t - self.last_pattern_start
    def generate(self, t):
        tp = t - self.last_pattern_start
        if tp==self.get_current_pattern_length():
            #print "PreprocessedSpeechInputGenerator: t =",t
            self.draw_pattern()
            self.last_pattern_start = t
            tp = 0
        self.ids = self.idx*numpy.ones(self.n, 'int')
        if self.noise:
            y = (numpy.random.exponential(1.0/self.r, self.n) <= self.dt).astype(int)
        else:
            nc = self.stimulus.getNumChannels()
            assert(nc==self.cochleagram.shape[0])
            chids = numpy.round(numpy.linspace(0,nc-1,self.n)).astype(int)
            if self.poisson:
                rates = numpy.clip(self.r * self.cochleagram[chids,tp], 0, 1)
                rates *= self.n*self.r/numpy.sum(rates)
                y = numpy.asarray([(numpy.random.exponential(1.0/r) <= self.dt) for r in rates], 'int')
            else:
                channels = numpy.take(self.stimulus.channel, chids)
                y = [numpy.any((ch.data*self.time_warp>=tp*self.dt) & (ch.data*self.time_warp<(tp+1)*self.dt)) for ch in channels]
                y = numpy.asarray(y).astype(int)
            if self.noise_period is not None:
                if tp>=self.noise_period[0] and tp<self.noise_period[1]:
                    y = numpy.zeros(self.n,dtype='int')
        if self.rNoise>0.0:
            y_noise = (numpy.random.exponential(1.0/self.rNoise, self.n) <= self.dt).astype(int)
            self.ids[numpy.where(y_noise)[0]] = -1
            y = y | y_noise
        return y
    def reset(self, t=0, draw=False):
        self.last_pattern_start = t


def test():
    #st = generatePoissonSpikeTrain(1, 20)
    #print st, len(st)

    n = 100
    dt = 1e-3
    r = 5
    rmax = 100
    nPatterns  = 2
    #tPattern = 50e-3
    tPattern = numpy.array([200e-3]*nPatterns)
    #pattern_sequences = [[0,2],[0,3],[1,2],[1,3]]
    pattern_sequences = [[1],[0,1],[0]]
    #pattern_sequences = [[0,1,2],[3]]
    inpgen1 = SequentialPatternInputGenerator(n, dt, r, nPatterns, tPattern, pattern_sequences, mode='alternating',
        sprob=1.0, rNoise=2.0, time_warp_range=(0.5,2.0))
    inpgen2 = SequentialPatternInputGenerator(n/2, dt, r, nPatterns, tPattern, pattern_sequences)
    inpgen3 = SequentialPatternInputGenerator(n/2, dt, r, nPatterns, tPattern, pattern_sequences)
    inpgen4 = CombinedInputGenerator([inpgen2,inpgen3])
    inpgen_speech = PreprocessedSpeechInputGenerator(n, dt, r, poisson=False, scale=50,
        digits=[1,2], rev_digits=[], sprob=1.0, rNoise=2.)
    #inpgen1 = PatternInputGenerator(n, dt, r, nPatterns, tPattern, rNoise=2.0, prob=None)
    #inpgen = EmbeddedPatternInputGenerator(n, dt, r, nPatterns, tPattern, rNoise=0.0, tNoiseRange=[50e-3,200e-3])
    #inpgen = RatePatternInputGenerator(n, dt, rmax, nPatterns, tPattern, 5, (2, 10))
    #inpgen = RateInputGenerator(n, dt, 30e-3, 3, (10,80))
    inpgen5 = EmbeddedPatternInputGenerator(inpgen_speech, tNoiseRange=[200e-3,300e-3], prob=1.0)
    inpgen6 = SwitchableInputGenerator([inpgen4,inpgen5])
    T = 4000
    sts = []; ids = []; ids2 = [];
    X = [[] for i in range(n)]
    r = numpy.zeros((T,n))
    #inpgen1.patterns[2] = inpgen1.patterns[1]
    #inpgen_speech.noise_period=(200,400)
    #inpgen_speech.mixed=True
    inpgen = inpgen5
    inpgen_speech.time_warp_range=(2.0,2.0)
    #inpgen1.idx=0
    #inpgen.inpgen.next_idx=0
    #inpgen.reset()
    #inpgen.switch(1)
    for t in range(T):
        #inpgen.set_noise((t>110) and (t<160))
        x = inpgen.generate(t)
        for i in numpy.where(x)[0]:
            X[i].append(t)
        sts.append(x)
        ids.append(inpgen.get_idx())
        #print t,inpgen.get_idx()
        #ids2.append(inpgen.get_sidx())
        #r[t,:] = inpgen.current_rates
    sts = numpy.asarray(sts)
    ids = numpy.asarray(ids)
    ids2 = numpy.asarray(ids2)
    #print ids
    #print inpgen.pattern_sequence_count
    #return

    #ids2 = numpy.random.randint(0,25,n)
    pylab.figure()
    #pylab.subplot(211)
    plot_spike_trains2(X, T, pylab.gca(), ids, ms=3)
    #pylab.subplot(212)
    #pylab.plot(r)
    #pylab.show()

    #print
    #inpgen.switch(0)
    #sts = []; ids = []; ids2 = [];
    #X = [[] for i in range(n)]
    #r = numpy.zeros((T,n))
    #inpgen.reset()
    ##inpgen.inpgen.next_idx=1
    #for t in range(T):
        #x = inpgen.generate(t)
        #for i in numpy.where(x)[0]:
            #X[i].append(t)
        #sts.append(x)
        #ids.append(inpgen.get_idx())
    #sts = numpy.asarray(sts)
    #ids = numpy.asarray(ids)

    #pylab.figure()
    #plot_spike_trains2(X, T, pylab.gca(), ids, ms=3)

    pylab.show()

    #for digit in range(10):
        #Ts = []
        #for speaker in [1,2,5,6,7]:
            #for utterance in range(1,11):
                #stim = inpgen1.inpgen.generate((speaker,utterance,digit,False))
                #Ts.append(stim.Tsim)
        #Ts = numpy.asarray(Ts)
        #print digit, numpy.mean(Ts)

def test_cochleagram():
    n = 100
    dt = 1e-3
    r = 5
    inpgen = PreprocessedSpeechInputGenerator(n, dt, r, poisson=False, scale=50,
        digits=[1,2], rev_digits=[], sprob=1.0)
    stim1 = inpgen.inpgen.generate()
    stim2 = inpgen.inpgen.generate()
    coch1 = getCochleagram(stim1.speaker,stim1.utterance,stim1.digit,h5filename=inpgen.path+'/cochleagrams.h5')
    coch2 = getCochleagram(stim2.speaker,stim2.utterance,stim2.digit,h5filename=inpgen.path+'/cochleagrams.h5')
    coch_mean = getMeanCochleagram(coch1,coch2)

    stimulus = PreprocessedSpeechStimulus()
    for c in range(coch_mean.shape[0]):
        channel = Channel(BSA(coch_mean[c,:],50))
        stimulus.channel.append(channel)
    stimulus.Tsim = max([numpy.max(ch.data) for ch in stimulus.channel if len(ch.data)>0])

    pylab.figure()
    pylab.subplot(221)
    plotCochleagram(coch1)
    pylab.subplot(222)
    plotCochleagram(coch2)
    pylab.subplot(223)
    plotCochleagram(coch_mean)
    pylab.subplot(224)
    stimulus.plot(color='k')

    pylab.show()

if __name__ == '__main__':
    test()
    #test_cochleagram()
