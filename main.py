from experiment_group import ExperimentGroup
import numpy as np, argparse, os
from board import Board
from arghierarchy import ArgHierarchy
from sequential import Sequential

if __name__ == '__main__':

    presets = ['HDSexp', 'IRWexp', 'DSexp', 'CBRWexp', 'HDSbern', 'IRWbern', 'DSbern', 'CBRWbern',
            'HDSmulti', 'IRWmulti', 'DSmulti', 'CBRWmulti',
            'HDSreal', 'IRWreal', 'DSreal', 'CBRWreal']
    parser = argparse.ArgumentParser()
    # required/positional arguments
    parser.add_argument('algorithm', type=str,
                            help='algorithm',
                            choices=list(Sequential.algorithms.keys())+presets
                            )


    # problem parameters
    parser.add_argument('-d','--dist', type=str,
                            help='dist',
                            choices=Board.dists)
    parser.add_argument('-f','--fork', type=int, default=None)
    parser.add_argument('-l','--levels', type=int, default=None)
    parser.add_argument('-m','--n_cells', type=int, default=None)
    parser.add_argument('-i','--iterate', action='store_true',
                    help='iterate over the levels')
    parser.add_argument('-ap','--anomaly_parameter', type=float, default=None,
                    help='anomaly parameter index, higher index -> harder problem')
    parser.add_argument('-k', '--n_anom', type=int, default=1,
                    help='number of anomalies')

    # algorithm arguments
    parser.add_argument('-min_ap','--min_anomaly_parameter', type=float, default=None,
                    help='minium anomaly parameter index, higher index -> harder problem')
    parser.add_argument('-fixed','--fixed_size', action='store_true',
                            help='use fixed size local test')
    parser.add_argument('-lgllr','--lgllr', action='store_true',
                            help='use LGRLL statistics instead of LALLR (for DS/HDS only)')
    parser.add_argument('-c','--sampling_cost', type=float, default=1e-2,
                    help='relative cost of sampling')    
    parser.add_argument('-max_samples','--max_samples', type=int, default=-1,
                    help='trajectorioes with more samples are discarded')
                    
    # simulation/table settings
    parser.add_argument('-p', '--processes', type=int, default=1,
                    help='processes per experiment')
    parser.add_argument('-v', '--verbose', action='store_true',
                    help='print to terminal')
    parser.add_argument('-noload', '--noload', action='store_true',
                    help='does not load previous data')
    parser.add_argument('-folder', '--folder', type=str, default=None,
                    help='relative data folder location')
    parser.add_argument('-nosave','--nosave', action='store_true')
    parser.add_argument('-max_sim','--max_sim', type=float, default=-1,
                            help='maximum number of simulations')
    parser.add_argument('-nint','--notification_interval', type=int, default=60,
                            help='notification interval in seconds')
    parser.add_argument('-sint','--save_interval', type=int, default=60,
                            help='save interval in seconds')

    args = parser.parse_args()

    if args.algorithm in presets:
        args.iterate = True
        args.sampling_cost = 1e-2
        args.n_anom = 1
        if 'exp' in args.algorithm:
            args.dist = 'EXP'
            args.anomaly_parameter = 1e3
            r = 'exp'
        elif 'bern' in args.algorithm:
            args.dist = 'BERN'
            args.anomaly_parameter = 10
            r = 'bern'
        elif 'multi' in args.algorithm:
            args.dist = 'EXP'
            args.anomaly_parameter = 1e3
            args.n_anom = 5
            r = 'multi'
        elif 'real' in args.algorithm:
            args.anomaly_parameter = 1 # this is ignored
            args.dist = 'REAL'
            r = 'real'
        args.algorithm = args.algorithm.replace(r,'')


    fork_default = 2
    if args.n_cells is None:
        if args.levels is None:
            if args.fork is None:
                args.fork = fork_default
                args.levels = 7
            else:
                args.levels = 1
        else:
            if args.fork is None:
                args.fork = fork_default

    elif args.fork is None and args.levels is None:
        args.fork = args.n_cells
        args.levels = 1
    else:
        raise ValueError('Can either pass -fork/f and -levels/l or -n_cells/m but not both.')


    base_parameters = {
        'k' : args.n_anom,
        'algorithm' : args.algorithm,
        'fixed_size' : args.fixed_size,
        'lgllr' : args.lgllr,
        'max_samples' : args.max_samples,
    }

    if args.save_interval <= 0: # only saves when simulation is stopped
        args.save_interval = np.inf

    # folder structure
    folder = 'data'
    if args.folder is not None:
        folder = os.path.join(folder,args.folder)
    else:
        folder = os.path.join(folder,args.algorithm)
    if not os.path.isdir(folder) and not args.nosave:
        try:
            os.makedirs(folder)
        except:
            pass
        
    settings = {
        'datafolder' : folder,
        'max_sim' : int(args.max_sim),
        'load_previous' : not args.noload,
        'notification_interval' : args.notification_interval,
        'save_interval' : args.save_interval,
        'nosave' : args.nosave,
        'processes' : args.processes,
        'verbose' : args.verbose,
    }

    experiments = ExperimentGroup(base_parameters, settings)
    
    params = {}

    iterate_over = {}
    params = ArgHierarchy(args.algorithm)
        
    params.dist = args.dist
    if args.dist == 'EXP':
        pname = 'lambda1'
        params.lambda0 = 1
    elif args.dist == 'BERN':
        pname = 'loc2'
        params.loc1 = -6
        params.lambda0 = 0.1
    elif args.dist == 'REAL':
        pname = 'nothing'
    else:
        raise NotImplementedError
    if args.anomaly_parameter is not None:
        params[pname] = args.anomaly_parameter
    else:
        raise NotImplementedError
    min_pname = 'min_'+pname
    if args.min_anomaly_parameter is not None:
        params[min_pname] = args.min_anomaly_parameter
    else:
        if Sequential.algorithms[args.algorithm].knows_anom_param:
            if args.dist == 'EXP':
                params[min_pname] = params[pname]
            elif args.dist == 'BERN':
                pass
            elif args.dist == 'REAL':
                pass
            else:
                raise NotImplementedError
        else:
            if args.dist == 'AWGNC':
                params[min_pname] = params[pname]/2
            elif args.dist == 'EXP':
                params[min_pname] = (params[pname]+params.lambda0)/2
            elif args.dist == 'BERN':
                pass
            elif args.dist == 'REAL':
                pass
            else:
                raise NotImplementedError
    if args.dist == 'REAL':
        iterate_over['levels'] = np.arange(2,8)
    else:
        iterate_over['levels'] = np.arange(2,8)
        if args.n_anom > 1:
            iterate_over['levels'] += 2
    iterate_over['k'] = 2**np.arange(1,5)
    params.levels = args.levels
    params.fork = args.fork
    params.c = args.sampling_cost

    if args.iterate:
        for i,p in enumerate(iterate_over['levels']):
            params['levels'] = p
            print(params.dictionary)
            experiments.add_experiment(params)
    else:
        experiments.add_experiment(params)

    # commands
    experiments.run()