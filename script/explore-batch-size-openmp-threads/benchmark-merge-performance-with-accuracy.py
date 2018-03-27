#! /usr/bin/python
import ck.kernel as ck
import os

def do(i):

    top1={}
    top5={}

    # List accuracy entries
    r=ck.access({'action':'search',
                 'module_uoa':'experiment',
                 'data_uoa':'ck-request-asplos18-caffe-intel-performance-*',
                 'repo_uoa':'local',
                 'add_meta':'yes'})
    if r['return']>0: return r
    lst=r['lst']

    for q in lst:
        duid=q['data_uid']
        duoa=q['data_uoa']

        path=q['path']

        if 'inception-v3' in duoa:
           model='inception-v3'
           model_species='1b339ddb13408f8f'
        elif 'resnet50' in duoa:
           model='resnet50'
           model_species='d777f6335496db61'

        if model=='':
           return {'return':1, 'error':'model is not recognized'}

        prec=''
        if '-fp32' in duoa:
           prec='fp32'
        elif '-int8' in duoa:
           prec='int8'

        if prec=='':
           return {'return':1, 'error':'model precision is not recognized'}

        ck.out('* '+duoa+' / '+model+' / '+prec)

        # Search matching accuracy entry (if intel-request)
        x='ck-request-asplos18-caffe-intel-accuracy.*.'+model+'-'+prec
        r=ck.access({'action':'search',
                     'module_uoa':'experiment',
                     'data_uoa':x,
                     'repo_uoa':'local'})
        if r['return']>0: return r
        alst=r['lst']
        if len(alst)!=1:
           return {'return':1, 'error':'ambiguity when search for accuracy'}

        a=alst[0]
        apath=a['path']             

        # There is only one point normally (no model tuning)
        dacc={}
        xacc=os.listdir(apath)

        for f in xacc:
            if f.endswith('.flat.json'):
               r=ck.load_json_file({'json_file':os.path.join(apath,f)})
               if r['return']>0: return r

               dx=r['dict']

               # Get only accuracy keys (convert to common format)
               for k in dx:
                   if k.startswith('##characteristics#run#acc/top-'):
                      k1='##characteristics#run#accuracy_top'+k[30:]
                      dacc[k1]=dx[k]
                   elif k.startswith('##characteristics#run#accuracy/top-'):
                      k1='##characteristics#run#accuracy_top'+k[35:]
                      dacc[k1]=dx[k]

               break 
        
        if len(dacc)==0:
           return {'return':1, 'error':'strange - no match for accuracy entries'}

        # Iterating over points to aggregate
        dperf=os.listdir(path)
        for f in dperf:
            if f.endswith('.flat.json'):
               ck.out(' * '+f)

               # Load performance file 
               p1=os.path.join(path, f)

               r=ck.load_json_file({'json_file':p1})
               if r['return']>0: return r
               d=r['dict']

               # Merge accuracy
               for k in dacc:
                   d[k]=dacc[k]

               # Save updated dict
               r=ck.save_json_to_file({'json_file':p1, 'dict':d, 'sort_keys':'yes'})
               if r['return']>0: return r

    return {'return':0}

r=do({})
if r['return']>0: ck.err(r)
