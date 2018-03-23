#! /usr/bin/python
import ck.kernel as ck
import os

def do(i):

    # List performance entries
    r=ck.access({'action':'search',
                 'module_uoa':'experiment',
#                 'repo_uoa':'ck-request-asplos18-results'})
                 'data_uoa':'ck-request-asplos18-caffe-intel-performance-*'})
    if r['return']>0: return r
    lst=r['lst']

    for q in lst:
        duid=q['data_uid']
        duoa=q['data_uoa']
        ruid=q['repo_uid']
        path=q['path']

        ck.out(duoa)

        # Load entry
        r=ck.access({'action':'load',
                     'module_uoa':'experiment',
                     'data_uoa':duid,
                     'repo_uoa':ruid})
        if r['return']>0: return r

        dd=r['dict']
        ruid=r['repo_uid']
        apath=r['path']             

        # Check extra info
        if 'inception-v3' in duoa:
           model='inception-v3'
           model_species='1b339ddb13408f8f'
           model_size=95533753
        elif 'resnet50' in duoa:
           model='resnet50'
           model_species='d777f6335496db61'
           model_size=102462397

        if model=='':
           return {'return':1, 'error':'model is not recognized'}

        prec=''
        if '-fp32' in duoa:
           prec='fp32'
        elif '-int8' in duoa:
           prec='int8'
           model_size=model_size/4 # Guess

        if prec=='':
           return {'return':1, 'error':'model precision is not recognized'}

        # Updating meta if needed
        dd['meta']['scenario_module_uoa']='a555738be4b65860' # module:request.asplos18

        dd['meta']['model_species']=model_species # model.species:mobilenets

        dd['meta']['dataset_species']='ImageNet' # dataset species (free format)
        dd['meta']['dataset_size']=50000 # number of images ...

        dd['meta']['platform_species']='server' # embedded vs server (maybe other classifications such as edge)

       # Unified full name for some deps
        ds=dd['meta']['deps_summary']

        x=ds['caffemodel']
        r=ck.access({'action':'make_deps_full_name','module_uoa':'request.asplos18','deps':x})
        if r['return']>0: return r
        dd['meta']['model_design_name']=r['full_name']

        x=ds['lib-caffe']
        r=ck.access({'action':'make_deps_full_name','module_uoa':'request.asplos18','deps':x})
        if r['return']>0: return r
        dd['meta']['library_name']=r['full_name']

        x=x['deps']['compiler']
        r=ck.access({'action':'make_deps_full_name','module_uoa':'request.asplos18','deps':x})
        if r['return']>0: return r
        dd['meta']['compiler_name']=r['full_name']

        used_gpgpu=False
        if ds.get('lib-caffe',{}).get('deps',{}).get('compiler-cuda',{}).get('data_name','')!='':
           used_gpgpu=True

        if used_gpgpu:
           # GPU used
           dd['meta']['platform_peak_power']=180 #Watts
           dd['meta']['platform_price']=700 # $
           dd['meta']['platform_price_date']='20180101' # date

        else:
           dd['meta']['gpgpu_name']=''
           dd['meta']['gpgpu_vendor']=''
           dd['meta']['platform_peak_power']=105 #Watts
           dd['meta']['platform_price']=1166 # $
           dd['meta']['platform_price_date']='20141212' # date

        dd['meta']['artifact']='e7cc77d72f13441e' # artifact description

        dd['meta']['model_precision']=prec

        # Updating entry
        r=ck.access({'action':'update',
                     'module_uoa':'experiment',
                     'data_uoa':duid,
                     'repo_uoa':ruid,
                     'dict':dd,
                     'substitute':'yes',
                     'ignore_update':'yes',
                     'sort_keys':'yes'
                    })
        if r['return']>0: return r

        # Checking points to aggregate
        os.chdir(path)
        dperf=os.listdir(path)
        for f in dperf:
            if f.endswith('.cache.json'):
               os.system('git rm -f '+f)

            elif f.endswith('.flat.json'):
               ck.out(' * '+f)

               # Load performance file 
               p1=os.path.join(path, f)

               r=ck.load_json_file({'json_file':p1})
               if r['return']>0: return r
               d=r['dict']

               # Unify execution time + batch size
               batch=int(d.get('##characteristics#run#REAL_ENV_CK_CAFFE_BATCH_SIZE#min',''))
               d['##features#batch_size#min']=batch

               tall=d.get('##characteristics#run#time_fw_s#all',[])

               tnew=[]
               for t in tall:
                   t1=t/batch
                   tnew.append(t1)
               
               r=ck.access({'action':'stat_analysis',
                            'module_uoa':'experiment',
                            'dict':d,
                            'dict1':{'##characteristics#run#prediction_time_avg_s':tnew}
                           })
               if r['return']>0: return r

               d['##features#model_size#min']=model_size

               if not used_gpgpu:
                  d['##features#cpu_freq#min']=2600
                  d['##features#freq#min']=d['##features#cpu_freq#min']
               else:
                  d['##features#gpu_freq#min']=1600
                  d['##features#freq#min']=d['##features#gpu_freq#min']

               # Save updated dict
               r=ck.save_json_to_file({'json_file':p1, 'dict':d, 'sort_keys':'yes'})
               if r['return']>0: return r

    return {'return':0}

r=do({})
if r['return']>0: ck.err(r)
