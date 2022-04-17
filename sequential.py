import numpy as np
from board import Board
from util import fwer
from multiprocessing import Queue
import queue, time

def add_registry(cls):
    cls.algorithms = {
            f.__name__:f for f in cls.__dict__.values()
            if hasattr(f,'is_algorithm')
            }
    cls.hierarchical = {
            f.__name__:f.hierarchical for f in cls.__dict__.values()
            if hasattr(f,'is_algorithm')
            }
    return cls

@add_registry
class Sequential:
    def __init__(self, board:Board, algorithm:str, c:float,
                max_samples:int, fixed_size:bool=None, lgllr:bool=None) -> None:
        self.board = board
        self.max_samples = max_samples
        if algorithm not in Sequential.algorithms:
            raise NotImplementedError
        self.hierarchical = Sequential.hierarchical[algorithm]
        if self.hierarchical and fixed_size is None:
            raise Exception
        if not self.hierarchical and fixed_size is not None:
            raise Exception
        self.algorithm = algorithm
        self.leaf_nodes = board.level_nodes(0)
        self.c = c
        if algorithm in ['HDS', 'IRW']:
            self.llr_thresh = np.log(board.levels/c)
            if fixed_size:
                self.ltn = board.min_samples_for_llr_local_test(algorithm)
            else: # active test thresholds
                p = np.nextafter(0.5,1) # confidence
                self.v0 = -np.log((self.board.fork*p)/(1-p))
                self.v1 = -self.v0
        if algorithm in ['HDS', 'DS', 'DSFB']:
            if algorithm == 'DS':
                self.llr_thresh = -np.log(c)
            self.lgllr = lgllr
        elif lgllr is not None:
            raise ValueError
        if fixed_size:
            if algorithm in ['CBRW']:
                raise Exception
        self.fixed_size = fixed_size

        if algorithm == 'CBRW':
            self.cbrw_p = .2
            if self.cbrw_p >= 1-0.5**(1/board.fork):
                raise Exception
            epsilon = c
            self.cbrw_alpha0 = epsilon/(2*board.levels)*(1-np.exp(-2*(1-2*(1-self.cbrw_p)**2)**2))**2

        self.run_algorithm = getattr(self,algorithm)
        self.knows_anom_param = self.run_algorithm.knows_anom_param

    def register_algorithm(knows_anom_param:bool, hierarchical:bool):
        def decorator(f):
            f.is_algorithm = True
            f.knows_anom_param = knows_anom_param
            f.hierarchical = hierarchical
            return f
        return decorator

    def reset(self) -> None:
        self.board.initialize()
        if not hasattr(self,'samples'):
            self.samples = np.zeros((self.board.n_nodes-1, 256))
            self.n = np.zeros(self.board.n_nodes-1, dtype=int)
        else:
            self.samples[:] = 0
            self.n[:] = 0
        self.restart = False
        self.t = 0 # sample counter i.e. self.t = np.sum(self.n)

    def simulate(self, q:Queue, stop) -> dict:
        run = True
        while not stop.value:
            if run:
                self.reset()
                res = self.run_algorithm()
            try:
                q.put(res, False)
                run = True
            except queue.Full:
                run = False
                time.sleep(.01)
        stop.value = 0

    def wipe_node_samples(self, nodes:list) -> None:        
        self.samples[nodes] = 0
        self.n[nodes] = 0

    ### hierarchical algorithms ###

    @register_algorithm(knows_anom_param=True, hierarchical=True)
    def IRW(self) -> dict:
        return self.IRW_or_HDS_or_CBRW()

    @register_algorithm(knows_anom_param=False, hierarchical=True)
    def HDS(self) -> dict:
        return self.IRW_or_HDS_or_CBRW()

    @register_algorithm(knows_anom_param=False, hierarchical=True)
    def CBRW(self) -> dict:
        return self.IRW_or_HDS_or_CBRW()

    def IRW_or_HDS_or_CBRW(self) -> dict:
        anomalies = []
        for _ in range(self.board.k):
            if self.algorithm == 'CBRW':
                a = self.CBRW_one_anomaly()
            else:
                a = self.IRW_or_HDS_one_anomaly()
            anomalies += [a]
            self.board.remove(a)
        return self.return_dict(anomalies)


    def CBRW_one_anomaly(self) -> dict:
        test_level = self.board.levels
        test_node = self.board.root_node()
        while True:
            while True: # high level tests
                children, child_level = self.board.child_nodes_and_level(test_node)
                self.wipe_node_samples(children)
                winner = -1
                alpha = self.cbrw_p if child_level>0 else self.cbrw_alpha0
                for i,m in enumerate(children):
                    mean = 0
                    while True:
                        s = self.sample_update(m, 1)
                        mean = (mean*(self.n[m]-1)+s)/self.n[m]
                        if self.board.anomaly_bound(m, mean, alpha, self.n[m]):
                            winner = i
                            break
                        if self.board.normal_bound(m, mean, self.cbrw_p, self.n[m]):
                            break
                    if winner != -1:
                        break
                if winner == -1:
                    test_node, test_level =\
                        self.board.parent_node_and_level(test_node)
                else:
                    children, test_level = self.board.child_nodes_and_level(test_node)
                    test_node = children[winner]
                    if test_level == 0:
                        return test_node
    

    def IRW_or_HDS_one_anomaly(self) -> int:
        test_level = self.board.levels
        test_node = self.board.root_node()
        
        while True:
            while True: # internal test
                children, child_level = self.board.child_nodes_and_level(test_node)
                self.wipe_node_samples(children)
                if self.fixed_size:
                    n = self.ltn[child_level]
                    samples = self.uniform_sampling(children, n)
                    if self.knows_anom_param:
                        llr = [self.board.llr(c, s) for c,s in zip(children,samples)]
                    else:
                        # for proper estimates...
                        n_samples = self.board.min_samples-self.n[children[0]]
                        if n_samples > 0:
                            samples = np.hstack((samples,self.uniform_sampling(children, n_samples)))
                        llr = [self.board.gllr(c, s) for c,s in zip(children,samples)]
                    winner = np.argmax(llr)
                else:
                    winner = 0
                    llr = [0 for _ in children]
                    while True:
                        child = children[winner]
                        s = self.sample_update(child, 1)
                        if self.knows_anom_param:
                            llr[winner] += self.board.llr(child, s)
                        else:
                            # for proper estimates...
                            n_samples = self.board.min_samples-self.n[child]
                            if n_samples > 0:
                                self.sample_update(child, n_samples)

                            llr[winner] = self.board.gllr(child, self.current_samples(child))
                        winner = np.argmax(llr)
                        if llr[winner]>self.v1 or llr[winner]<=self.v0:
                            break
                if llr[winner]<=0:
                    test_node, test_level =\
                        self.board.parent_node_and_level(test_node)
                else:
                    test_node = children[winner]
                    test_level -= 1
                    if test_level == 0:
                        break
            # leaf test
            self.wipe_node_samples([test_node])
            llr = 0 # lallr or lgllr for HDS
            if not self.knows_anom_param and not self.lgllr:
                est = self.board.normal_parameter(1) # initial estimate
            # for proper estimates...
            n_samples = self.board.min_samples-self.n[test_node]-1
            if n_samples > 0:
                self.sample_update(test_node, n_samples)
            state = True
            while True or state:
                # DS phase 2
                s = self.sample_update(test_node, 1)
                # DS phase 3
                if self.knows_anom_param:
                    llr += self.board.llr(test_node,s)
                else:
                    if self.lgllr:
                        llr = self.board.gllr(test_node, self.current_samples(test_node))
                    else:
                        logp1 = self.board.logp(est,test_node,s)
                        logp0 = self.board.logp('0',test_node,s)
                        if logp1 != logp0: # problems at infinity
                            llr += logp1-logp0
                if llr>=self.llr_thresh:
                    return test_node
                if llr<0:
                    break
                if not self.knows_anom_param:
                    # for proper estimates...
                    n_samples = self.board.min_samples-self.n[test_node]
                    if n_samples > 0:
                        self.sample_update(test_node, n_samples)
                    if not self.knows_anom_param and not self.lgllr:
                        est = self.board.est1(1, self.current_samples(test_node))
            test_node, test_level =\
                self.board.parent_node_and_level(test_node)


    ### non-hierarchical algorithms ###

    @register_algorithm(knows_anom_param=False, hierarchical=False)
    def DS(self) -> dict:
        # initial estimate = normal parameter
        est = [self.board.normal_parameter(1) for _ in range(self.board.size)]
        llr = np.zeros(self.board.size)
        estimated_states = [False for _ in self.leaf_nodes]
        est_anoms = []
        prioritize_anomaly = True
        if not prioritize_anomaly:
            raise NotImplementedError

        m = 0
        while True:
            if len(est_anoms) == self.board.k:
                first = True
                t=0
                while first or prioritize_anomaly == estimated_states[borderline]:
                    t += 1
                    first = False
                    borderline = est_anoms[np.argmin(llr[est_anoms])]
                    if llr[borderline]>=self.llr_thresh:
                        return self.return_dict(est_anoms)   
                    # phase 2
                    s = self.sample_update(borderline, 1)
                    if self.restart: # if stuck, restarts
                        return self.DS()
                    # phase 3
                    if not self.lgllr: # takes old estimate
                        logp_denominator = self.board.logp('0',borderline,s)
                        logp = self.board.logp(est[borderline],borderline,s)
                        if logp != logp_denominator: # problems at infinity...
                            llr[borderline] += logp-logp_denominator
                    est[borderline], estimated_states[borderline] = self.board.est_leaf(False, self.current_samples(borderline))
                    if self.lgllr:
                        llr[borderline] = self.board.logp(est[borderline],borderline,self.current_samples(borderline))\
                            -self.board.logp('0',borderline,self.current_samples(borderline))
                est_anoms.remove(borderline)

            # phase 1
            # for proper estimates
            n_samples = self.board.min_samples-self.n[m]-1
            if n_samples > 0:
                self.sample_update(m, n_samples)

            s = self.sample_update(m, 1)

            if self.restart: # if stuck, restarts
                self.reset()
                return self.DS()

            if not self.lgllr: # takes old estimate
                logp_denominator = self.board.logp('0',m,s)
                logp = self.board.logp(est[m],m,s)
                if logp != logp_denominator: # problems at infinity...
                    llr[m] += logp-logp_denominator
            est[m], new_state = self.board.est_leaf(False, self.current_samples(m))
            if self.lgllr:
                llr[m] = self.board.logp(est[m],m,self.current_samples(m))\
                    -self.board.logp('0',m,self.current_samples(m))

            if not estimated_states[m]:
                if new_state:
                    est_anoms.append(m)
            else:
                if not new_state:
                    est_anoms.remove(m)
            estimated_states[m] = new_state
            m = (m+1)%self.board.size
            
    def current_samples(self,node:int) -> np.ndarray:
        return self.samples[node,:self.n[node]]

    def uniform_sampling(self, nodes:list, n_samples:int) -> np.ndarray:
        s = np.zeros((len(nodes), n_samples))
        for i,m in enumerate(nodes):
            s[i] = self.sample_update(m, n_samples)
        return s

    def sample_update(self, node:int, n_samples:int) -> np.ndarray:        
        # extend array if necessary
        while self.n[node]+n_samples>self.samples.shape[1]:
            self.samples = np.hstack((self.samples, np.zeros(self.samples.shape)))
        
        new_samples = self.board.sample(node, n_samples)
        self.samples[node,self.n[node]:self.n[node]+n_samples] = new_samples
        self.n[node] += n_samples
        self.t += n_samples
        if self.max_samples > 0 and self.t>=self.max_samples:
            self.restart = True
        return new_samples

    def return_dict(self, anomalies:list) -> dict:
        return {
            'tau': self.t,
            'fwer': fwer(self.board.hiders, anomalies),
            }