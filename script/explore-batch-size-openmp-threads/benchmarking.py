#! /usr/bin/python
import ck.kernel as ck
import copy
import re
import argparse


# ReQuEST description.
request_dict={
  'report_uid':'e7cc77d72f13441e', # unique UID for a given ReQuEST submission generated manually by user (ck uid)
                                   # the same UID will be for the report (in the same repo)

  'repo_uoa':'ck-request-asplos18-caffe-intel',
  'repo_uid':'0e40b194adc51b5a',

  'repo_cmd':'ck pull repo:ck-request-asplos18-caffe-intel',

  'farm':'', # if farm of machines

  'algorithm_species':'4b8bbc192ec57f63' # image classification
}

# Platform tags.
platform_tags='xeon-e5-2650-v3-icc'

# Batch size.
# NB: This script uses the choice list.
bs={
  'choice':[1, 8, 16, 24, 32, 40, 48, 56, 64],
  'default':1
}

# Number of OpenMP threads.
# NB: Test the number of cores (10) and hyperthreads (20).
nt={
  'start':10,
  'stop':20,
  'step':10,
  'default':1
}

# Number of statistical repetitions.
num_repetitions=5

def do(i, arg):

    random_name = arg.random_name

    # Detect basic platform info.
    ii={'action':'detect',
        'module_uoa':'platform',
        'out':'out'}
    r=ck.access(ii)
    if r['return']>0: return r

    # Keep to prepare ReQuEST meta.
    platform_dict=copy.deepcopy(r)

    # Host and target OS params.
    hos=r['host_os_uoa']
    hosd=r['host_os_dict']

    tos=r['os_uoa']
    tosd=r['os_dict']
    tdid=r['device_id']

    # Program and command.
    program='caffe'
    cmd_key='time_cpu'

    # Load Caffe program meta and desc to check deps.
    ii={'action':'load',
        'module_uoa':'program',
        'data_uoa':program}
    rx=ck.access(ii)
    if rx['return']>0: return rx
    mm=rx['dict']

    # Get compile-time and run-time deps.
    cdeps=mm.get('compile_deps',{})
    rdeps=mm.get('run_deps',{})

    # Merge rdeps with cdeps for setting up the pipeline (which uses
    # common deps), but tag them as "for_run_time".
    for k in rdeps:
        cdeps[k]=rdeps[k]
        cdeps[k]['for_run_time']='yes'

    # Caffe libs.
    depl=copy.deepcopy(cdeps['lib-caffe'])
    if (arg.tos is not None) and (arg.did is not None):
        tos=arg.tos
        tdid=arg.did

    ii={'action':'resolve',
        'module_uoa':'env',
        'host_os':hos,
        'target_os':tos,
        'device_id':tdid,
        'out':'con',
        'quiet':'yes',
        'deps':{'lib-caffe':copy.deepcopy(depl)}
    }
    r=ck.access(ii)
    if r['return']>0: return r

    udepl=r['deps']['lib-caffe'].get('choices',[]) # All UOAs of env for Caffe libs.
    if len(udepl)==0:
        return {'return':1, 'error':'no installed Caffe libs'}

    # Caffe models.
    depm=copy.deepcopy(cdeps['caffemodel'])

    ii={'action':'resolve',
        'module_uoa':'env',
        'host_os':hos,
        'target_os':tos,
        'device_id':tdid,
        'out':'con',
        'quiet':'yes',
        'deps':{'caffemodel':copy.deepcopy(depm)}
    }
    r=ck.access(ii)
    if r['return']>0: return r

    udepm=r['deps']['caffemodel'].get('choices',[]) # All UOAs of env for Caffe models.
    if len(udepm)==0:
        return {'return':1, 'error':'no installed Caffe models'}

    # ImageNet datasets.
    depd=copy.deepcopy(cdeps['dataset-imagenet-lmdb'])

    ii={'action':'resolve',
        'module_uoa':'env',
        'host_os':hos,
        'target_os':tos,
        'device_id':tdid,
        'out':'con',
        'quiet':'yes',
        'deps':{'dataset-imagenet-lmdb':copy.deepcopy(depd)}
    }
    r=ck.access(ii)
    if r['return']>0: return r

    udepd=r['deps']['dataset-imagenet-lmdb'].get('choices',[]) # All UOAs of env for ImageNet datasets.
    if len(udepd)==0:
        return {'return':1, 'error':'no installed ImageNet datasets'}

    # Prepare pipeline.
    cdeps['lib-caffe']['uoa']=udepl[0]
    cdeps['caffemodel']['uoa']=udepm[0]
    cdeps['dataset-imagenet-lmdb']['uoa']=udepd[0]

    ii={'action':'pipeline',
        'prepare':'yes',
        'dependencies':cdeps,

        'module_uoa':'program',
        'data_uoa':program,
        'cmd_key':cmd_key,

        'target_os':tos,
        'device_id':tdid,

        'no_state_check':'yes',
        'no_compiler_description':'yes',
        'skip_calibration':'yes',

        'cpu_freq':'max',
        'gpu_freq':'max',

        'flags':'-O3',
        'speed':'no',
        'energy':'no',

        'skip_print_timers':'yes',
        'out':'con'
    }

    r=ck.access(ii)
    if r['return']>0: return r

    fail=r.get('fail','')
    if fail=='yes':
        return {'return':10, 'error':'pipeline failed ('+r.get('fail_reason','')+')'}

    ready=r.get('ready','')
    if ready!='yes':
        return {'return':11, 'error':'pipeline not ready'}

    state=r['state']
    tmp_dir=state['tmp_dir']

    # Remember resolved deps for this benchmarking session.
    xcdeps=r.get('dependencies',{})

    # Clean pipeline.
    if 'ready' in r: del(r['ready'])
    if 'fail' in r: del(r['fail'])
    if 'return' in r: del(r['return'])

    pipeline=copy.deepcopy(r)

    # For each Caffe lib.*******************************************************
    for lib_uoa in udepl:
        # Load Caffe lib.
        ii={'action':'load',
            'module_uoa':'env',
            'data_uoa':lib_uoa}
        r=ck.access(ii)
        if r['return']>0: return r

        real_tags=r['dict']['tags']

        # Get the tags from e.g. 'BVLC Caffe framework (intel, request)'
        lib_name=r['data_name']
        lib_tags=re.match('BVLC Caffe framework \((?P<tags>.*)\)', lib_name)
        lib_tags=lib_tags.group('tags').replace(' ', '').replace(',', '-')
        # Skip some libs with "in [..]" or "not in [..]".

        # Check if Intel artifact and select OpenMP tuning
        tuning_dims={
                'choices_order':[
                    ['##choices#env#CK_CAFFE_BATCH_SIZE']
                ],
                'choices_selection':[
                    {'type':'loop', 'default':bs['default'], 'choice':bs['choice']}
                ],
                'features_keys_to_process':[
                    '##choices#env#CK_CAFFE_BATCH_SIZE'
                ]
        }

        intel_artifact=False
        if 'intel' in real_tags and 'vrequest' in real_tags:
           intel_artifact=True

           # Add extra tuning dimension (OpenMP)
           tuning_dims['choices_order'].append(['##choices#env#OMP_NUM_THREADS'])
           tuning_dims['choices_selection'].append({'type':'loop', 'default':nt['default'], 'start':nt['start'], 'stop':nt['stop'], 'step':nt['step']})
           tuning_dims['features_keys_to_process'].append('##choices#env#OMP_NUM_THREADS')

        # ReQuEST CUDA/CUDNN
        cuda=False
        rtags=r['dict'].get('tags',[])
        if 'vcuda' in rtags:
           # Detect gpgpu
           r=ck.access({'action':'detect',
                        'module_uoa':'platform.gpgpu',
                        'cuda':'yes',
                        'select':'yes'})
           if r['return']>0: return r

           platform_dict['features'].update(r['features'])

           cuda=True

        # Check proper command line for CPU or GPU
        cmd_key='time_cpu'
        if cuda: cmd_key='time_gpu'

        # Remark next one if you want to check other libs
#        if lib_tags not in [ 'intel-request' ]: continue

        skip_compile='no'

        # For each Caffe model.*************************************************
        for model_uoa in udepm:
            # Load Caffe model.
            ii={'action':'load',
                'module_uoa':'env',
                'data_uoa':model_uoa}
            r=ck.access(ii)
            if r['return']>0: return r

            model_real_tags=r['dict']['tags']

            if 'vint8' in model_real_tags and not intel_artifact:
               continue

            # Get the tags from e.g. 'Caffe model (net and weights) (inception-v3, fp32)'
            model_name=r['data_name']
            model_tags = re.match('Caffe model \(net and weights\) \((?P<tags>.*)\)', model_name)
            model_tags = model_tags.group('tags').replace(' ', '').replace(',', '-')

            # Skip some models with "in [..]" or "not in [..]".
            if model_tags not in [ 'resnet50-fp32', 'resnet50-int8', 'inception-v3-fp32', 'inception-v3-int8' ]: continue

            record_repo='local'
            record_uoa='ck-request-asplos18-caffe-intel-performance-'+platform_tags+'.'+lib_tags+'.'+model_tags

            # Prepare pipeline.
            ck.out('---------------------------------------------------------------------------------------')
            ck.out('%s - %s' % (lib_name, lib_uoa))
            ck.out('%s - %s' % (model_name, model_uoa))
            ck.out('Experiment - %s:%s' % (record_repo, record_uoa))

            # Prepare autotuning input.
            cpipeline=copy.deepcopy(pipeline)

            # Reset deps and change UOA.
            new_deps={'lib-caffe':copy.deepcopy(depl),
                      'caffemodel':copy.deepcopy(depm)}

            new_deps['lib-caffe']['uoa']=lib_uoa
            new_deps['caffemodel']['uoa']=model_uoa

            jj={'action':'resolve',
                'module_uoa':'env',
                'host_os':hos,
                'target_os':tos,
                'device_id':tdid,
                'deps':new_deps}
            r=ck.access(jj)
            if r['return']>0: return r

            cpipeline['dependencies'].update(new_deps)

            cpipeline['no_clean']=skip_compile
            cpipeline['no_compile']=skip_compile

            cpipeline['cmd_key']=cmd_key

            cpipeline['extra_run_cmd']='-phase TEST'

            # Prepare common meta for ReQuEST tournament
            features=copy.deepcopy(cpipeline['features'])
            platform_dict['features'].update(features)

            r=ck.access({'action':'prepare_common_meta',
                         'module_uoa':'request.asplos18',
                         'platform_dict':platform_dict,
                         'deps':cpipeline['dependencies'],
                         'request_dict':request_dict})
            if r['return']>0: return r

            record_dict=r['record_dict']

            meta=r['meta']

            if random_name:
               rx=ck.gen_uid({})
               if rx['return']>0: return rx
               record_uoa=rx['data_uid']

            tags=r['tags']

            tags.append('explore-batch-size-openmp-threads')
            tags.append(program)
            tags.append(model_tags)
            tags.append(lib_tags)
            tags.append(platform_tags)

            ii={'action':'autotune',

                'module_uoa':'pipeline',
                'data_uoa':'program',

                'iterations':-1,
                'repetitions':num_repetitions,

                'record':'yes',
                'record_failed':'yes',
                'record_params':{
                    'search_point_by_features':'yes'
                },

                'tags':tags,
                'meta':meta,

                'record_dict':record_dict,

                'record_repo':record_repo,
                'record_uoa':record_uoa,

                'pipeline':cpipeline,
                'out':'con'}

            ii.update(copy.deepcopy(tuning_dims))

            r=ck.access(ii)
            if r['return']>0: return r

            fail=r.get('fail','')
            if fail=='yes':
                return {'return':10, 'error':'pipeline failed ('+r.get('fail_reason','')+')'}

            skip_compile='yes'

    return {'return':0}

##############################################################################################

parser = argparse.ArgumentParser(description='Pipeline')
parser.add_argument("--target_os", action="store", dest="tos")
parser.add_argument("--device_id", action="store", dest="did")
parser.add_argument("--random_name", action="store_true", default=False, dest="random_name")
myarg=parser.parse_args()

r=do({}, myarg)
if r['return']>0: ck.err(r)
