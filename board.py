from scipy.stats import norm, expon, binom, bernoulli, ncx2
import numpy as np
import random
from util import exp_mean_cdf, exp_mean_ppf, sample_entropy, permute_2nd_column
from util import load_float_array, save_1dim_arrays, load_1dim_lists

class Board:
    dist_parametertypes = {
        'EXP' : ['lambda0', 'lambda1','min_lambda1'],
        'BERN' : ['lambda0', 'loc1', 'loc2'],
        'REAL' : ['nothing'],
        }
    dist_min_samples = {
        'EXP' : 1,
        'BERN' : 1,
        'REAL' : 2,
        }
    dists = list(dist_parametertypes.keys()) 
    dist_parameters = [p for l in dist_parametertypes.values() for p in l]

    def __init__(self, dist: str, k:int, fork: int,
            levels:int, dist_parameters:dict) -> None:
        
        self.size = fork**levels
        self.levels = levels
        self.k = k
        self.min_samples = Board.dist_min_samples[dist]
        self.fork = fork
        self.n_nodes = round((self.fork**(self.levels+1)-1)\
                            /(self.fork-1))
        self.dist = dist
        self.__dict__.update(dist_parameters)
        if dist == 'EXP':
            self.lambda_thresh_leaf = [(l1-self.lambda0)/np.log(l1/self.lambda0)\
                for l1 in [self.min_lambda1,self.lambda1]]
        elif dist == 'BERN':
            if self.loc1 >= 0:
                raise NotImplementedError
            self.loc2_set = np.array([10,5,1])
            self.cbrw_thresh = 1
            self.xi = .05
            if self.loc2 not in self.loc2_set:
                raise ValueError
        elif dist == 'REAL':
            self.check_one_anomaly()
            self.chunk_length = 0.1
            timeseries = load_float_array('dataset/converted_dataset.csv')
            normal_mask = timeseries[:,2]==0
            #timeseries = timeseries[timeseries[:, 0].argsort()] already sorted by time

            data_time = timeseries[-1,0]

            print('Calculating offline estimates...')
            training_split = .5

            training_time = training_split*data_time
            training_mask = timeseries[:,0]<=training_time
            #training_mask = ~training_mask

            # testing
            self.anomalous_testseries = timeseries[~training_mask,:2].copy()
            self.anomalous_testseries[:,0] -= self.anomalous_testseries[0,0]-0.001
            self.normal_testseries = timeseries[~training_mask & normal_mask,:2].copy()
            self.normal_testseries[:,0] -= self.normal_testseries[0,0]-0.001
            self.n_test_samples = round(np.floor(self.anomalous_testseries[-1,0]/self.chunk_length))

            # training
            reshuffle = 1000
            anomalous_trainseries = timeseries[training_mask,:2].copy()
            anomalous_trainseries[:,0] -= anomalous_trainseries[0,0]-0.001
            normal_trainseries = timeseries[training_mask & normal_mask,:2].copy()
            normal_trainseries[:,0] -= normal_trainseries[0,0]-0.001
            n_train_samples = round(np.floor(self.anomalous_testseries[-1,0]/self.chunk_length))

            filename = f'dataset/estimates_split_{training_split}_shuffle_{reshuffle}.csv'
            load_estimate = False
            load_estimate = True # comment this to overwrite previous estimates
            
            self.a2, self.a2hds = [], []
            self.a1, self.a1hds = [], []
            self.a0n, self.a0nhds = [], []
            self.mumid, self.sigmamid = [], []
            self.mumidcbrw, self.sigmamidcbrw = [], []
            self.min_sigma = []
            try:
                if not load_estimate:
                    raise IOError
                self.__dict__.update(load_1dim_lists(filename))
                start = len(self.mu0)
                print(f'Loaded previous estimates up to {start-1} levels.')
            except IOError:
                self.mu0 = []
                self.mu1 = []
                self.sigma0 = []
                self.sigma1 = []
                start = 0
            loaded = True
            for l in range(self.levels):
                if not load_estimate or l >= start:
                    loaded = False
                    normal_samples, anoamly_samples = [], []
                    for _ in range(reshuffle):
                        normal_permuted = np.stack([permute_2nd_column(normal_trainseries)
                            for _ in range(self.fork**l)])
                        anomalous_permuted = permute_2nd_column(anomalous_trainseries)
                        for j in range(n_train_samples):
                            normal_chunk = normal_permuted[:,
                                (j*self.chunk_length<=normal_permuted[0,:,0]) &
                                (normal_permuted[0,:,0]<(j+1)*self.chunk_length),1]
                            normal_samples += [sample_entropy(normal_chunk.flatten())]
                            abnormal_chunk = anomalous_permuted[
                                (j*self.chunk_length<=anomalous_permuted[:,0]) &
                                (anomalous_permuted[:,0]<(j+1)*self.chunk_length),1]
                            anoamly_samples += [sample_entropy(np.concatenate(
                                (normal_chunk[:-1].flatten(),abnormal_chunk)))]
                    normal_samples = np.array(normal_samples)
                    anoamly_samples = np.array(anoamly_samples)
                    self.mu0 += [np.mean(normal_samples)]
                    self.sigma0 += [np.sqrt(np.sum((normal_samples-self.mu0[l])**2)/(normal_samples.shape[0]-1))]
                    self.mu1 += [np.mean(anoamly_samples)]
                    self.sigma1 += [np.sqrt(np.sum((anoamly_samples-self.mu1[l])**2)/(anoamly_samples.shape[0]-1))]
                self.mumid += [0.5*(self.mu0[l]+self.mu1[l])]
                self.sigmamid += [0.5*(self.sigma0[l]+self.sigma1[l])]
                self.min_sigma += [np.nextafter(1,2)-1]
                self.mumidcbrw += [(3*self.mu0[l]+self.mu1[l])/4]
                self.sigmamidcbrw += [(3*self.sigma0[l]+self.sigma1[l])/4]
                self.a2 += [0.5*(self.sigma0[l]**-2-self.sigma1[l]**-2)]
                self.a1 += [self.mu1[l]/self.sigma1[l]**2-self.mu0[l]/self.sigma0[l]**2]
                self.a0n += [(0.5*((self.mu0[l]/self.sigma0[l])**2-(self.mu1[l]/self.sigma1[l])**2)+
                    np.log(self.sigma0[l]/self.sigma1[l]))]
                self.a2hds += [0.5*(self.sigma0[l]**-2-self.sigmamid[l]**-2)]
                self.a1hds += [self.mumid[l]/self.sigmamid[l]**2-self.mu0[l]/self.sigma0[l]**2]
                self.a0nhds += [(0.5*((self.mu0[l]/self.sigma0[l])**2-(self.mumid[l]/self.sigmamid[l])**2)+
                    np.log(self.sigma0[l]/self.sigmamid[l]))]
            if not loaded:
                d = {'mu0':self.mu0,'mu1':self.mu1,
                    'sigma0':self.sigma0,'sigma1':self.sigma1}
                save_1dim_arrays(d, filename)
            print('Done loading data and calculating offline estimates.')
            
            self.sample_counter = 0
        else:
            raise ValueError
                
        if self.size <= 0:
            raise ValueError('Board size must be greater than 0.')

        if k >= self.size:
            raise ValueError('Number of anomalies must be lower than the board size.')
        self.initialize()
    
    def exp_glrt_crossover(self, n_cells:int) -> float:
        l0 = n_cells*self.lambda0
        l1 = (n_cells-1)*self.lambda0+self.min_lambda1
        return (l1-l0)/np.log(l1/l0)

    def anomaly_bound(self, node:int, mean:float, p:float, n:int) -> bool:
        n_cells = self.n_cells(node)
        if self.dist == 'EXP':
            return mean < exp_mean_ppf(p, self.exp_glrt_crossover(n_cells), n)
        elif self.dist == 'BERN': # not justified subgaussian bound
            return mean > self.cbrw_thresh+np.sqrt(-2*self.xi*np.log(p)/n)
        elif self.dist == 'REAL':
            l = self.level_from_n_cells(n_cells)
            return mean < self.mumidcbrw[l]-self.sigmamidcbrw[l]/np.sqrt(n)*norm.ppf(1-p)
        else:
            raise NotImplementedError
    
    def normal_bound(self, node:int, mean:float, p:float, n:int) -> float:
        n_cells = self.n_cells(node)
        if self.dist == 'EXP':
            return mean > exp_mean_ppf(1-p, self.exp_glrt_crossover(n_cells), n)
        elif self.dist == 'BERN': # not justified subgaussian bound
            return mean <= self.cbrw_thresh-np.sqrt(-2*self.xi*np.log(p)/n)
        elif self.dist == 'REAL':
            l = self.level_from_n_cells(n_cells)
            return mean >= self.mumid[l]+self.sigmamid[l]/np.sqrt(n)*norm.ppf(1-p)
        else:
            raise NotImplementedError

    def initialize(self) -> None:
        self.removed = []
        self.hiders = random.sample(range(self.size),self.k)
        self.hider_mask = np.zeros(self.size, int)
        self.hider_mask[self.hiders] = 1

        if self.dist == 'REAL': # new anomaly locations -> new chunks array
            self.shuffle_if_necessary(force=True)         
    
    def shuffle_if_necessary(self, force=False) -> None:
        if self.sample_counter > self.n_test_samples:
            raise Exception('This should not have happened.')
        if force or self.sample_counter==self.n_test_samples:
            timeseries = [permute_2nd_column(
                self.anomalous_testseries
                if a else self.normal_testseries)
                 for a in self.hider_mask]
            self.chunks = [[t[(j*self.chunk_length<=t[:,0]) &
                (t[:,0]<(j+1)*self.chunk_length),1]
                for j in range(self.n_test_samples)] for t in timeseries]
            for normal in range(self.size):
                if normal not in self.hiders:
                    break
            self.sample_counter = 0

    def min_samples_for_llr_local_test(self, algorithm:str) -> int:
        n = np.zeros(self.levels, int)
        for l in range(self.levels):
            try_n = 1
            if self.dist == 'EXP':
                lam0 = self.normal_parameter(self.fork**l)
                if algorithm == 'IRW':
                    lam1 = self.abnormal_parameter(self.fork**l, 1)
                elif algorithm == 'HDS':
                    lam1 = self.border_abnormal_parameter(self.fork**l)
                else:
                    raise ValueError
                thresh = np.log(lam1/lam0)/(lam1-lam0)
            elif self.dist == 'BERN':
                lam0 = self.fork**l*self.lambda0
            elif self.dist == 'REAL':
                pass
            else:
                raise NotImplementedError
            while True:
                if self.dist == 'EXP':
                    p_correct_0 = 1-exp_mean_cdf(thresh,lam0,try_n)
                    p_correct_1 = exp_mean_cdf(thresh,lam1,try_n)
                elif self.dist == 'BERN':
                    def calc_p_err_0(loc2) -> float:
                        alpha = np.log(2)-lam0*self.loc1
                        beta = np.log(np.exp(lam0*self.loc1)+np.exp(lam0*loc2))-lam0*self.loc1
                        ratio = alpha/beta
                        p = np.exp(-lam0*loc2)
                        if ratio <= p:
                            raise ValueError('No n exists')
                        return 1-binom.cdf(try_n*ratio,try_n,p)
                    
                    if algorithm == 'IRW':      
                        p_correct_0 = 1-calc_p_err_0(self.loc2)
                    elif algorithm == 'HDS':
                        p_correct_0 = max(0, 1-np.sum([calc_p_err_0(l2) for l2 in self.loc2_set]))
                    p_correct_1 = 1-((1+np.exp(lam0*self.loc1))/2)**try_n
                elif self.dist == 'REAL':
                    if algorithm == 'IRW':
                        a0 = try_n*self.a0n[l]
                        def p0(mu,sigma) -> float:
                            cdf = ncx2.cdf((try_n*self.a1[l]**2/(4*self.a2[l])-a0)/
                            (self.a2[l]*sigma**2),try_n,
                            try_n/sigma**2*(mu+0.5*self.a1[l]/self.a2[l])**2)
                            return cdf if self.a2[l]>0 else 1-cdf

                        p_correct_0 = p0(self.mu0[l],self.sigma0[l])
                        p_correct_1 = 1-p0(self.mu1[l],self.sigma1[l])
                    elif algorithm == 'HDS':
                        a0 = try_n*self.a0nhds[l]
                        def p0(mu,sigma) -> float:
                            cdf = ncx2.cdf((try_n*self.a1hds[l]**2/(4*self.a2hds[l])-a0)/
                            (self.a2hds[l]*sigma**2),try_n,
                            try_n/sigma**2*(mu+0.5*self.a1hds[l]/self.a2hds[l])**2)
                            return cdf if self.a2hds[l]>0 else 1-cdf
                        
                        p_correct_0 = p0(self.mu0[l],self.sigma0[l])
                        p_correct_1 = 1-p0(self.mumid[l],self.sigmamid[l])
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                p_comp = 0.5/p_correct_0**(self.fork-1)
                if p_correct_0>p_comp and p_correct_1>p_comp:
                    n[l] = try_n
                    break
                try_n += 1
        print(n)
        return n

    def check_one_anomaly(self) -> None:
        if self.k != 1:
            raise NotImplementedError('Currently only support one anomaly.')


    def est_leaf(self, knows_anom:bool, samples:np.ndarray) -> tuple:

        if isinstance(samples, (float,int)):
            n = 1
        elif samples.ndim == 1:
            n = samples.shape[0]
        else:
            raise ValueError

        mean = np.mean(samples, axis=-1)
        if self.dist == 'EXP':
            lam = 1/mean
            return lam, lam>self.lambda_thresh_leaf[knows_anom]
        elif self.dist == 'BERN':
            est, logp = self.est1_and_stat(1,samples,n)
            anom = logp > self.logp('0',0,samples,n)
            return est, anom
        elif self.dist == 'REAL':
            sigma = np.sqrt(np.mean((samples-mean)**2))
            sigma = max(sigma, self.min_sigma[0])
            return np.array([mean, sigma]), mean<self.mumid[0] and sigma < self.sigmamid[0]
        else:
            raise NotImplementedError
                
    def est1(self, n_cells:int, samples:np.ndarray, n:np.ndarray=None) -> np.ndarray:
        return self.est1_and_stat(n_cells, samples, n)[0]

    np.seterr(all='raise')
    def est1_and_stat(self, n_cells:int, samples:np.ndarray, n:np.ndarray=None) -> tuple:
        if samples.ndim > 1:
            raise Exception
        n = samples.shape[0]

        mean = np.mean(samples, axis=-1)
        if self.dist == 'EXP':
            return np.maximum(1/mean, self.min_lambda1+(n_cells-1)*self.lambda0), mean
        elif self.dist == 'BERN':
            logp = [self.logp((1,loc2),n_cells,samples,n) for loc2 in self.loc2_set]
            i = np.argmax(logp)
            stat = logp[i]
            est = [(1,self.loc2_set[i])]
            return est[0], stat
        elif self.dist == 'REAL':
            l = self.level_from_n_cells(n_cells)
            mean = min(mean,self.mumid[l])
            sigma = np.sqrt(np.sum((samples-mean)**2)/n)
            sigma = max(sigma, self.min_sigma[l])
            sigma = min(sigma,self.sigmamid[l])
            return np.array([mean, sigma]), None
        else:
            raise NotImplementedError

    def normal_parameter(self, n_cells:int) -> float:
        if n_cells < 1:
            raise ValueError

        if self.dist == 'EXP':
            return self.lambda0*n_cells
        elif self.dist == 'BERN':
            return 0,0
        elif self.dist == 'REAL':
            l = self.level_from_n_cells(n_cells)
            return np.array([self.mu0[l], self.sigma0[l]])
        else:
            raise NotImplementedError
    
    def abnormal_parameter(self, n_cells:int, n_anomalies:int) -> float:
        if n_cells < 1:
            raise ValueError

        if self.dist == 'EXP':
            return (n_cells-n_anomalies)*self.lambda0+n_anomalies*self.lambda1
        elif self.dist == 'BERN':
            return 1, self.loc2
        elif self.dist == 'REAL':
            l = self.level_from_n_cells(n_cells)
            return np.array([self.mu1[l], self.sigma1[l]])
        else:
            raise NotImplementedError
    
    def level_from_n_cells(self, n_cells:int) -> int:
        level = np.log(n_cells)/np.log(self.fork)
        if np.ceil(level)!=np.floor(level):
            raise NotImplementedError('Only implemented for homogenous trees.')
        return  round(level)
    
    def border_abnormal_parameter(self, n_cells:int) -> float:
        if n_cells < 1:
            raise ValueError

        if self.dist == 'EXP':
            return (n_cells-1)*self.lambda0+self.min_lambda1
        else:
            raise NotImplementedError

    def n_cells(self, node:int) -> int:
        def rec(node:int) -> int:
            try:
                return np.sum([rec(c) for c in self.child_nodes(node)])
            except ValueError:
                return 1
        return rec(node)

    def logp(self, parameter, node:int, samples:np.ndarray, n:np.ndarray=None) -> float:
        n_cells = self.n_cells(node)

        
        if n is None:
            if isinstance(samples, (float,int)):
                n = 1
            elif samples.ndim == 1:
                n = samples.shape[0]
            else:
                raise ValueError

        stat = None
        if isinstance(parameter, str):
            if parameter == '0':
                parameter = self.normal_parameter(n_cells)
            elif parameter == '1':
                parameter = self.abnormal_parameter(n_cells, 1)
            elif parameter == 'est1':
                parameter, stat = self.est1_and_stat(n_cells, samples, n)
            else:
                raise NotImplementedError
            
        if self.dist == 'EXP':
            mean = np.sum(samples, axis=-1)/n if stat is None else stat
            if self.dist == 'EXP':
                return n*(np.log(parameter)-parameter*mean)
            else:
                NotImplementedError
        elif self.dist == 'BERN':
            one = samples.ndim == 1
            if one:
                samples = [samples]
                parameter = [parameter]
                stat = [stat]
            else:
                raise Exception
            logp = []
            lam = n_cells*self.lambda0
            for i,s,param in zip(range(len(parameter)),samples,parameter):
                if param[0] == 0: # normal
                    if np.any(s<0):
                        lp = -np.inf
                    else:
                        mean = np.mean(s)
                        lp = n*(np.log(lam)-lam*mean)
                elif param[0] == 1: # anomaly
                    if stat is not None and stat[0] is not None:
                        lp = stat[i]
                    else:
                        mask = s<param[1]
                        n1 = np.sum(mask)
                        n2 = n-n1
                        lp = n*np.log(lam/2)
                        try:
                            lp += np.log(1+np.exp(-lam*n2*(param[1]-self.loc1)))
                        except FloatingPointError:
                            pass
                        if n1>0:
                            mean1 = np.mean(s[mask])
                            lp -= lam*n1*(mean1-self.loc1)
                        if n2>0:
                            mean2 = np.mean(s[~mask])
                            lp -= lam*n2*(mean2-param[1])
                else:
                    raise ValueError
                logp += [lp]
            res = logp[0] if one else np.array(logp)
            return res
        elif self.dist == 'REAL':
            return np.sum(norm.logpdf(samples,loc=parameter[0],scale=parameter[1]))
        else:
            raise NotImplementedError

    def gllr(self, node:int, samples:np.ndarray) -> float:
        return self.llr(node, samples, True)
    
    def llr(self, node:int, samples:np.ndarray, estimate=False) -> float:
        logp1 = self.logp('est1' if estimate else '1', node, samples)
        logp0 = self.logp('0', node, samples)
        if type(logp0) is np.ndarray:
            llr = np.zeros(logp0.shape[0])
            mask = logp1 != logp0 # problems at infinity
            llr[mask] += logp1[mask]-logp0[mask]
        else:
            llr = 0
            if logp1 != logp0: # problems at infinity
                llr += logp1-logp0
        return llr

    def leafs_beneath(self, node:int) -> int:
        def rec(node:int) -> int:
            try:
                return [a for c in self.child_nodes(node) for a in rec(c)]
            except ValueError:
                return [node]
        return rec(node)

    def sample(self, node:int, n_samples:int=1) -> np.ndarray:
        if n_samples == 0: # saves a lot of time!
            return np.array([])
        leafs = self.leafs_beneath(node)
        n_cells = len(leafs)
        n_anomalies = np.sum(self.hider_mask[leafs])
        if self.dist == 'EXP':
            if n_anomalies:
                lam = self.abnormal_parameter(n_cells, n_anomalies)
            else:
                lam = self.normal_parameter(n_cells)
            return expon.rvs(scale=1/lam,size=n_samples)
        elif self.dist == 'BERN':
            lam = n_cells*self.lambda0
            rv = expon.rvs(scale=1/lam,size=n_samples)
            if n_anomalies:
                rv += self.loc1
                rv += bernoulli.rvs(.5,size=n_samples)*(self.loc2-self.loc1)
            return rv
        elif self.dist == 'REAL':
            samples = np.zeros(n_samples)
            for i in range(n_samples):
                self.shuffle_if_necessary()
                chunk = np.concatenate([self.chunks[m][self.sample_counter] for m in leafs])
                self.sample_counter += 1
                samples[i] = sample_entropy(chunk)
            return samples
        else:
            raise ValueError
            
    def check_coords(self, level:int, index:int) -> None:
        if index >= round(self.fork**(self.levels-level)) or\
            not (0 <= level <= self.levels):
            raise Exception(f'Coordinates ({level},{index}) are out of range.')

    def coords_to_node(self, level:int, index:int):
        self.check_coords(level, index)
        lower_layer_nodes = round((self.fork**(self.levels+1)\
            -self.fork**(self.levels-level+1))/(self.fork-1))
        return lower_layer_nodes + index

    def node_to_coords(self, node:int) -> tuple:
        if not (0 <= node < self.n_nodes):
            raise Exception('Index out of range.')
        level = 0
        lower_layer_nodes = 0
        layer_nodes = self.size
        while node >= lower_layer_nodes+layer_nodes:
            level += 1
            lower_layer_nodes += layer_nodes
            layer_nodes //= self.fork
        return level, node-lower_layer_nodes
    
    def child_nodes_and_level(self, node:int) -> list:
        level, index = self.node_to_coords(node)
        if level == 0:
            raise ValueError('Leaf node has no children.')
        child_level = level-1
        left_child = self.coords_to_node(child_level, index*self.fork)
        children = list(range(left_child,left_child+self.fork))
        for r in self.removed:
            try:
                children.remove(r)
            except ValueError:
                pass
        return children, child_level
    
    def remove(self, r:int) -> None:
        if r in self.removed:
            raise ValueError
        self.removed += [r]
        if all([b in self.removed for b in self.brothers(r)]):
            self.remove(self.parent_node(r))

    def child_nodes(self, node:int) -> list:
        return self.child_nodes_and_level(node)[0]

    def root_node(self) -> int:
        return self.n_nodes-1

    def parent_node_and_level(self, node:int) -> int:
        level, index = self.node_to_coords(node)
        parent_level = min(level+1, self.levels)
        parent_index = index//self.fork
        return self.coords_to_node(parent_level, parent_index), parent_level

    def parent_node(self, node:int) -> int:
        return self.parent_node_and_level(node)[0]

    def brothers(self, node:int) -> list:
        return self.child_nodes(self.parent_node(node))

    def level_nodes(self, level:int) -> list:
        if not (0 <= level <= self.levels):
            raise Exception('Level out of range.')
        if level == self.levels:
            return [self.coords_to_node(self.levels,0)]
        else:
            return list(range(self.coords_to_node(level,0)
                    ,self.coords_to_node(level+1,0)))