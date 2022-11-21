
import numpy
import sys,os

from dataformats import *
from inputs.inputgenerator import *

class PreprocessedSpeechStimulus(Stimulus):
    def __init__(self, Tsim=0.5, speaker=1, digit=1, utterance=1, reversed=False):
        super(PreprocessedSpeechStimulus, self).__init__(Tsim)
        self.speaker = speaker
        self.utterance = utterance
        self.digit = digit
        self.reversed = reversed
        self.file = "s%d_u%d_d%d" % (speaker,utterance,digit)
        self.save_attrs += ['speaker','utterance','digit','reversed','file']

    def __str__(self):
        desc = '''  PreprocessedSpeechStimulus
  channel   : [1x%s struct]
  Tsim      : %s
  speaker   : %d
  utterance : %d
  digit     : %d
  reversed  : %s
  file      : %s\n''' % (len(self.channel), self.Tsim, self.speaker, self.utterance,
            self.digit,self.reversed,self.file)
        return desc

    def getNumChannels(self):
        return len(self.channel)

    def getTitle(self):
        return str(self.file) + {True:"_rev",False:""}[bool(self.reversed)]


class PreprocessedSpeech(InputGenerator):
    def __init__(self, **kwds):
        self.scale = kwds.get('scale',-1)
        self.speakers = kwds.get('speakers',[1,2,5,6,7])
        self.digits = kwds.get('digits',[0,1,2,3,4,5,6,7,8,9])
        self.utterances = kwds.get('utterances',[1,2,3,4,5,6,7,8,9,10])
        self.rev_speakers = kwds.get('rev_speakers',self.speakers)
        self.rev_digits = kwds.get('rev_digits',self.digits)
        self.rev_utterances = kwds.get('rev_utterances',self.utterances)
        self.rev_st = kwds.get('rev_st',False)
        self.path = os.path.realpath(kwds.get('path', '..'))
        self.stimlist = kwds.get('stimlist',[])
        if len(self.stimlist)==0:
            self.size = len(self.speakers)*len(self.digits)*len(self.utterances)
            self.rev_size = len(self.rev_speakers)*len(self.rev_digits)*len(self.rev_utterances)
        else:
            self.size = len([(s,u,d,rev) for (s,u,d,rev) in self.stimlist if rev==False])
            self.rev_size = len([(s,u,d,rev) for (s,u,d,rev) in self.stimlist if rev==True])
        self.randomized = kwds.get('randomized',False)
        if self.randomized:
            self.idxmap = numpy.zeros(self.size+self.rev_size).astype(int)
            self.idxmap[:self.size] = numpy.random.permutation(self.size)
            self.idxmap[self.size:] = numpy.random.permutation(self.rev_size)+self.size
        else:
            self.idxmap = numpy.arange(self.size+self.rev_size)
        if self.scale==-1:
            self.h5filename = 'spkdata.h5'
        elif self.scale==-2:
            self.h5filename = 'etimes.h5'
        else:
            self.h5filename = 'spkdata_%d.h5' % (self.scale)
        #print self.getFullH5Path()
        self.subsample = kwds.get('subsample', -1)
        self.maxspikes = kwds.get('maxspikes', -1)

    def __str__(self):
        desc = '''  PREPROCESSED_SPEECH
  file           : %s
  speakers       : %s
  digits         : %s
  utterances     : %s
  rev_speakers   : %s
  rev_digits     : %s
  rev_utterances : %s
  size           : %s
  rev_size       : %s
''' % (self.h5filename, self.speakers, self.digits, self.utterances,
            self.rev_speakers, self.rev_digits, self.rev_utterances,
            self.size, self.rev_size)
        return desc
 
    def get_sud(self, index, reversed=False):
        i = index
        if len(self.stimlist)==0:
            if reversed:
                i = i-self.size;
                d = self.rev_digits[i % len(self.rev_digits)]
                i = i/len(self.rev_digits)
                s = self.rev_speakers[i % len(self.rev_speakers)]
                i = i/len(self.rev_speakers)
                u = self.rev_utterances[i]
            else:
                d = self.digits[i % len(self.digits)]
                i = i/len(self.digits)
                s = self.speakers[i % len(self.speakers)]
                i = i/len(self.speakers)
                u = self.utterances[i]
        else:
            s,u,d,reversed = self.stimlist[i]
        return (s,u,d)

    def getTotalSize(self):
        return self.size + self.rev_size

    def getFullH5Path(self):
        return self.path+"/"+self.h5filename

    def generate(self, what=-1):
        if type(what)==int:
            if what<0:
                return self.generateRandom()
            else:
                return self.generateByIdx(what)
        elif type(what)==tuple:
            if len(what)!=4:
                raise TypeError("requires tuple of length 4")
            s,u,d,rev = what
            return self.generateBySUD(s,u,d,rev)
        else:
            raise TypeError("invalid type; expected <int> or <tuple>")

    def generateBySUD(self, speaker, utterance, digit, reversed):
        stimulus = PreprocessedSpeechStimulus()
        filename = self.getFullH5Path()
        grpname = "s%d_u%d_d%d" % (speaker,utterance,digit)
        #print "loading %s from %s" % (grpname,filename)
        if not self.rev_st:
            revstr = {False:"",True:"_rev"}[reversed]
            grpname = "%s%s" % (grpname,revstr)
            stimulus.load(filename=filename, grpname=grpname)
        else:
            stimulus.load(filename=filename, grpname=grpname)
            if reversed:
                for c in stimulus.channel:
                    c.data = numpy.sort(stimulus.Tsim - c.data)
                stimulus.reversed = True
                stimulus.file += "_rev"
        if self.subsample > 0:
            nch = len(stimulus.channel)
            intvl = nch/self.subsample
            offs = intvl/2
            stimulus.channel = stimulus.channel[offs::intvl][:self.subsample]
        if self.maxspikes>0:
            for c in stimulus.channel:
                l = len(c.data)
                if len(c.data)>self.maxspikes:
                    #c.data = c.data[numpy.random.permutation(len(c.data))[:self.maxspikes]]
                    #c.data = numpy.sort(c.data)
                    c.data = numpy.array([])
                #print "%d -> %d" % (l,len(c.data))
        return stimulus

    def generateByIdx(self, index):
        if index < 0 or index > self.size+self.rev_size:
            raise IndexError("index %d out of bounds" % (index))
        elif index < self.size:
            reversed = False
        else:
            reversed = True
        s,u,d = self.get_sud(self.idxmap[index],reversed)
        return self.generateBySUD(s,u,d,reversed)

    def generateRandom(self):
        idx = numpy.random.randint(self.size+self.rev_size)
        return self.generateByIdx(idx)

def generateBSAinput(scale = -1, start = 0):
    from mlabwrap import mlab
    mlab.addpath('../')
    mlab.startup()
    if scale==-1:
        InputDist = mlab.preprocessed_speech()
        h5filename = 'spkdata.h5'
    else:
        InputDist = mlab.preprocessed_speech2('scale',scale)
        h5filename = 'spkdata_%d.h5' % scale
    N = mlab.get(InputDist,'size').flatten()
    N_rev = mlab.get(InputDist,'rev_size').flatten()
    NN = N + N_rev
    #print "Loading inputs..."
    #S = mlab.generate_input(InputDist)
    #NN = mlab.length(S).flatten()
    #nc = mlab.length(S[0].channel).flatten()
    print "Saving inputs..."
    for i in range(start,NN):
        #mlab.clear("all")
        stimulus = PreprocessedSpeechStimulus()
        S = mlab.generate_input(InputDist,i+1)
        nc = mlab.length(S.channel).flatten()
        for c in range(nc):
            channel = Channel(S.channel[c].data.flatten())
            stimulus.channel.append(channel)
        stimulus.Tsim = S.info.Tstim.flatten()[0]
        stimulus.file = S.info.file
        stimulus.speaker = S.info.speaker.flatten()[0]
        stimulus.utterance = S.info.utterance.flatten()[0]
        stimulus.digit = S.info.digit.flatten()[0]
        stimulus.reversed = S.info.reversed.flatten()[0]==1
        if stimulus.reversed:
            grpname = stimulus.file + "_rev"
        else:
            grpname = stimulus.file
        print "%d: %s" % (i,grpname)
        stimulus.save(filename=h5filename, grpname=grpname)

def generateBSAinput2(scale = -1, start = 0):
    if scale==-1:
        h5filename = 'spkdata.h5'
    else:
        h5filename = 'spkdata_%d.h5' % scale

    speakers = [1,2,5,6,7]
    digits = range(10)
    utterances = range(1,11)
    rev = False
    print "Saving inputs..."
    i = 0
    for s in speakers:
        for d in digits:
            for u in utterances:
                #for rev in [False,True]:
                    stimulus = PreprocessedSpeechStimulus()
                    spec = getCochleagram(s,u,d,'cochleagrams.h5')
                    nc = spec.shape[0]
                    for c in range(nc):
                        channel = Channel(BSA(spec[c,:],scale))
                        stimulus.channel.append(channel)
                    #stimulus.Tsim = spec.shape[1]*1e-3
                    stimulus.Tsim = max([numpy.max(ch.data) for ch in stimulus.channel if len(ch.data)>0])
                    stimulus.file = "s%d_u%d_d%d" % (s,u,d)
                    stimulus.speaker = s
                    stimulus.utterance = u
                    stimulus.digit = d
                    stimulus.reversed = rev
                    if stimulus.reversed:
                        grpname = stimulus.file + "_rev"
                    else:
                        grpname = stimulus.file
                    print "%d: %s" % (i,grpname)
                    i += 1
                    stimulus.save(filename=h5filename, grpname=grpname)

def generateHBinput():
    h5file_in = tables.openFile('etimes_tmp.h5')
    h5filename = 'etimes.h5'
    speakers = [1,2,5,6,7]
    digits = range(10)
    utterances = range(1,11)
    rev = False
    print "Saving inputs..."
    i = 0
    for s in speakers:
        for d in digits:
            for u in utterances:
                #for rev in [False,True]:
                    stimulus = PreprocessedSpeechStimulus()
                    grpname = "s%d_u%d_d%d" % (s,u,d)
                    et = numpy.asarray(h5file_in.getNode("/"+grpname+"/et").read())
                    ei = numpy.asarray(h5file_in.getNode("/"+grpname+"/ei").read()).astype(int)
                    et -= numpy.min(et)
                    nc = len(ei)
                    for c in range(nc):
                        start = ei[c]-1
                        if c==nc-1:
                            end = len(et)
                        else:
                            end = ei[c+1]
                        #channel = Channel(et[start:end])
                        if start<len(et):
                            channel = Channel(numpy.array([et[start]]))
                        else:
                            channel = Channel(numpy.array([]))
                        stimulus.channel.append(channel)
                    stimulus.Tsim = numpy.max(et)
                    stimulus.file = grpname
                    stimulus.speaker = s
                    stimulus.utterance = u
                    stimulus.digit = d
                    stimulus.reversed = rev
                    if stimulus.reversed:
                        grpname = stimulus.file + "_rev"
                    else:
                        grpname = stimulus.file
                    print "%d: %s" % (i,grpname)
                    i += 1
                    stimulus.save(filename=h5filename, grpname=grpname)
    h5file_in.close()

def BSA(wave,scale,tau_filter=30e-3,threshold=0.97):
    ad_sampling_rate=16000
    decimation_factor=128
    input_dt=10e-3
    internal_dt=1e-3
    output_dt=1e-3
    #tau_filter=30e-3
    ts = numpy.arange(0.0, 6.0*tau_filter+internal_dt/2.0, internal_dt)
    filter = scale/(tau_filter/internal_dt)*numpy.exp(-ts/tau_filter)
    #threshold=0.97

    N = len(wave)
    P = len(filter)
    spike = numpy.zeros(N)
    wave = numpy.hstack((wave,numpy.zeros(P)))
    for i in range(N):
        segment = wave[i:i+P]
        filt = filter.copy()
        filt[numpy.arange(i,i+P)>N] = 0
        if numpy.sum(numpy.abs(segment-filt)) <= numpy.sum(numpy.abs(segment))-threshold:
            spike[i] = 1
            wave[i:i+P] -= filt
    #spiketimes = spiketrain2spiketimes(spike, out_dt);
    spiketimes = numpy.nonzero(spike)[0] * output_dt
    return spiketimes

def getCochleagram(s,u,d,h5filename=None):
    if h5filename is None:
        from mlabwrap import mlab
        mlab.addpath('../')
        spec = mlab.cochleagram(s,u,d)
    else:
        h5f = tables.openFile(h5filename)
        grpname = "/s%d_u%d_d%d" % (s,u,d)
        spec = numpy.asarray(h5f.getNode(grpname).read()).T
        h5f.close()
    return spec



if __name__ == '__main__':
    #generateBSAinput()
    #generateBSAinput(scale=1)
    #generateBSAinput(scale=5,start=710)
    #generateBSAinput(scale=10)
    generateBSAinput2(scale=40)
    #generateBSAinput2(scale=75)
    #generateHBinput()
