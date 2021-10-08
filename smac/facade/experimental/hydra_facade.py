# type: ignore
# mypy: ignore-errors

import logging
import os
import datetime
import time
import typing
import copy
import json
from collections import defaultdict

import pickle

import neptune.new as neptune
import numpy as np
import pandas as pd

from ConfigSpace.configuration_space import Configuration

from smac.tae.base import BaseRunner
from smac.tae.execute_ta_run_hydra import ExecuteTARunHydra
from smac.tae.execute_ta_run_hydra import ExecuteTARunOld
from smac.scenario.scenario import Scenario
from smac.facade.experimental.psmac_facade import PSMAC
from smac.utils.io.output_directory import create_output_directory
from smac.runhistory.runhistory import RunHistory
from smac.epm.util_funcs import get_rng
from smac.utils.constants import MAXINT
from smac.optimizer.pSMAC import read
from smac.utils.neptune import object_to_dict
import smac.utils.neptune as smac_neptune

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


class Hydra(object):
    """
    Facade to use Hydra default mode

    Attributes
    ----------
    logger
    stats : Stats
        loggs information about used resources
    solver : SMBO
        handles the actual algorithm calls
    rh : RunHistory
        List with information about previous runs
    portfolio : list
        List of all incumbents

    """

    def __init__(self,
                 scenario: typing.Type[Scenario],
                 n_iterations: int,
                 val_set: str = 'train',
                 incs_per_round: int = 1,
                 n_optimizers: int = 1,
                 rng: typing.Optional[typing.Union[np.random.RandomState, int]] = None,
                 run_id: int = 1,
                 tae: typing.Type[BaseRunner] = ExecuteTARunOld,
                 tae_kwargs: typing.Union[dict, None] = None,
                 **kwargs):
        """
        Constructor

        Parameters
        ----------
        scenario : ~smac.scenario.scenario.Scenario
            Scenario object
        n_iterations: int,
            number of Hydra iterations
        val_set: str
            Set to validate incumbent(s) on. [train, valX].
            train => whole training set,
            valX => train_set * 100/X where X in (0, 100)
        incs_per_round: int
            Number of incumbents to keep per round
        n_optimizers: int
            Number of optimizers to run in parallel per round
        rng: int/np.random.RandomState
            The randomState/seed to pass to each smac run
        run_id: int
            run_id for this hydra run
        tae: BaseRunner
            Target Algorithm Runner (supports old and aclib format as well as AbstractTAFunc)
        tae_kwargs: Optional[dict]
            arguments passed to constructor of '~tae'

        """
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        self.n_iterations = n_iterations
        self.scenario = scenario
        self.run_id, self.rng = get_rng(rng, run_id, self.logger)
        self.kwargs = kwargs
        self.output_dir = None
        self.top_dir = None
        self.portfolio = None
        self.rh = RunHistory()
        self._tae = tae
        self._tae_kwargs = tae_kwargs
        if incs_per_round <= 0:
            self.logger.warning('Invalid value in %s: %d. Setting to 1', 'incs_per_round', incs_per_round)
        self.incs_per_round = max(incs_per_round, 1)
        if n_optimizers <= 0:
            self.logger.warning('Invalid value in %s: %d. Setting to 1', 'n_optimizers', n_optimizers)
        self.n_optimizers = max(n_optimizers, 1)
        self.val_set = self._get_validation_set(val_set)
        self.cost_per_inst = {}
        self.optimizer = None
        self.portfolio_cost = None

    def _get_validation_set(self, val_set: str, delete: bool = True) -> typing.List[str]:
        """
        Create small validation set for hydra to determine incumbent performance

        Parameters
        ----------
        val_set: str
            Set to validate incumbent(s) on. [train, valX].
            train => whole training set,
            valX => train_set * 100/X where X in (0, 100)
        delete: bool
            Flag to delete all validation instances from the training set

        Returns
        -------
        val: typing.List[str]
            List of instance-ids to validate on

        """
        if val_set == 'none':
            return None
        if val_set == 'train':
            return self.scenario.train_insts
        elif val_set[:3] != 'val':
            self.logger.warning('Can not determine validation set size. Using full training-set!')
            return self.scenario.train_insts
        else:
            size = int(val_set[3:]) / 100
            if size <= 0 or size >= 1:
                raise ValueError('X invalid in valX, should be between 0 and 1')
            insts = np.array(self.scenario.train_insts)
            # just to make sure this also works with the small example we have to round up to 3
            size = max(np.floor(insts.shape[0] * size).astype(int), 3)
            ids = np.random.choice(insts.shape[0], size, replace=False)
            val = insts[ids].tolist()
            if delete:
                self.scenario.train_insts = np.delete(insts, ids).tolist()
            return val

    def optimize(self) -> typing.List[Configuration]:
        """
        Optimizes the algorithm provided in scenario (given in constructor)

        Returns
        -------
        portfolio : typing.List[Configuration]
            Portfolio of found configurations

        """
        run = smac_neptune.get_run('hydra')

        # Setup output directory
        self.portfolio = []
        portfolio_cost = np.inf
        if self.output_dir is None:
            self.top_dir = "hydra-output_%s" % (
                datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f'))
            self.scenario.output_dir = os.path.join(self.top_dir, "psmac3-output_%s" % (
                datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f')))
            self.output_dir = create_output_directory(self.scenario, run_id=self.run_id, logger=self.logger)

        scen = copy.deepcopy(self.scenario)
        scen.output_dir_for_this_run = None
        scen.output_dir = None
        run['optimizer'] = object_to_dict(self)
        for i in range(self.n_iterations):
            run_iteration = run.child(f'iteration/{i}')
            run['scenario_output_dir'].log(self.scenario.output_dir)
            run['output_dir'].log(self.output_dir)
            self.logger.info("=" * 120)
            self.logger.info("Hydra Iteration: %d", (i + 1))
            run['hydra_iteration'].log(i)

            if i == 0:
                tae = self._tae
                tae_kwargs = self._tae_kwargs
            else:
                tae = ExecuteTARunHydra
                if self._tae_kwargs:
                    tae_kwargs = self._tae_kwargs
                else:
                    tae_kwargs = {}
                tae_kwargs['cost_oracle'] = self.cost_per_inst
            self.optimizer = PSMAC(
                scenario=self.scenario,
                run_id=self.run_id,
                rng=self.rng,
                tae=tae,
                tae_kwargs=tae_kwargs,
                shared_model=False,
                validate=True if self.val_set else False,
                n_optimizers=self.n_optimizers,
                val_set=self.val_set,
                n_incs=self.n_optimizers,  # return all configurations (unvalidated)
                neptune_run=run_iteration,
                **self.kwargs
            )
            self.optimizer.output_dir = self.output_dir
            incs = self.optimizer.optimize()

            def save_configs(incs, name):
                # TODO: Save a Vampire call with each config.
                run_iteration_configs = run_iteration.child(f'configs/{name}')
                run_iteration_configs['pkl'] = neptune.types.File.as_pickle(incs)

                fn = os.path.join(self.optimizer.scenario.output_dir, 'hydra', 'configs', f'{name}.json')
                os.makedirs(os.path.dirname(fn), exist_ok=True)
                records = [conf.get_dictionary() for conf in incs]
                with open(fn, "w") as fp:
                    json.dump(records, fp, indent=2)
                run_iteration_configs['json'] = neptune.types.File(fn)

                fn = os.path.join(self.optimizer.scenario.output_dir, 'hydra', 'configs', f'{name}.csv')
                pd.DataFrame.from_records(records, columns=self.scenario.cs).to_csv(fn)
                run_iteration_configs['csv'] = neptune.types.File(fn)

            save_configs(incs, 'incs_all')

            cost_per_conf_v, val_ids, cost_per_conf_e, est_ids = self.optimizer.get_best_incumbents_ids(incs)
            if self.val_set:
                to_keep_ids = val_ids[:self.incs_per_round]
            else:
                to_keep_ids = est_ids[:self.incs_per_round]
            run['incs/to_keep_ids'].log(to_keep_ids)
            config_cost_per_inst = {}
            incs = [incs[i] for i in to_keep_ids]
            save_configs(incs, 'incs_kept')
            self.logger.info('Kept incumbents')
            for inc in incs:
                self.logger.info(inc)
                config_cost_per_inst[inc] = cost_per_conf_v[inc] if self.val_set else cost_per_conf_e[inc]

            cur_portfolio_cost = self._update_portfolio(incs, config_cost_per_inst)
            run['portfolio/cost'].log(cur_portfolio_cost)
            run['portfolio/cost_normalized'].log(self.scenario.normalize_cost(cur_portfolio_cost))
            costs = np.fromiter(self.cost_per_inst.values(), float)
            crashes = np.count_nonzero(costs >= self.scenario.cost_for_crash)
            run['portfolio/crashes'].log(crashes)
            run['portfolio/crashes_relative'].log(crashes / len(costs))
            if self.scenario.run_obj == 'runtime':
                timeouts = np.count_nonzero(costs >= self.scenario.cutoff * self.scenario.par_factor)
                run['portfolio/timeouts'].log(timeouts)
                run['portfolio/timeouts_relative'].log(timeouts / len(costs))
            save_configs(self.portfolio, 'portfolio')
            if portfolio_cost <= cur_portfolio_cost:
                self.logger.info("No further progress (%f) --- terminate hydra", portfolio_cost)
                break
            else:
                portfolio_cost = cur_portfolio_cost
                self.logger.info("Current pertfolio cost: %f", portfolio_cost)

            self.scenario.output_dir = os.path.join(self.top_dir, "psmac3-output_%s" % (
                datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f')))
            self.output_dir = create_output_directory(self.scenario, run_id=self.run_id, logger=self.logger)
        read(self.rh, os.path.join(self.top_dir, 'psmac3*', 'run_' + str(MAXINT)), self.scenario.cs, self.logger)
        self.rh.save_json(fn=os.path.join(self.top_dir, 'all_validated_runs_runhistory.json'), save_external=True)
        run['runhistory'].upload(os.path.join(self.top_dir, 'all_validated_runs_runhistory.json'))
        with open(os.path.join(self.top_dir, 'portfolio.pkl'), 'wb') as fh:
            pickle.dump(self.portfolio, fh)
        run['portfolio/portfolio.pkl'].upload(os.path.join(self.top_dir, 'portfolio.pkl'))
        self.logger.info("~" * 120)
        self.logger.info('Resulting Portfolio:')
        for configuration in self.portfolio:
            self.logger.info(str(configuration))
        run['portfolio/str'] = self.portfolio
        self.logger.info("~" * 120)

        return self.portfolio

    def _update_portfolio(self, incs: np.ndarray, config_cost_per_inst: typing.Dict) -> typing.Union[np.float, float]:
        """
        Validates all configurations (in incs) and determines which ones to add to the portfolio

        Parameters
        ----------
        incs: np.ndarray
            List of Configurations

        Returns
        -------
        cur_cost: typing.Union[np.float, float]
            The current cost of the portfolio

        """
        if self.val_set:  # we have validated data
            for kept in incs:
                if kept not in self.portfolio:
                    self.portfolio.append(kept)
                    cost_per_inst = config_cost_per_inst[kept]
                    if self.cost_per_inst:
                        if len(self.cost_per_inst) != len(cost_per_inst):
                            raise ValueError('Num validated Instances mismatch!')
                        else:
                            for key in cost_per_inst:
                                self.cost_per_inst[key] = min(self.cost_per_inst[key], cost_per_inst[key])
                    else:
                        self.cost_per_inst = cost_per_inst
            cur_cost = np.mean(list(self.cost_per_inst.values()))  # type: np.float
        else:  # No validated data. Set the mean to the approximated mean
            means = []  # can contain nans as not every instance was evaluated thus we should use nanmean to approximate
            for kept in incs:
                means.append(np.nanmean(list(self.optimizer.rh.get_instance_costs_for_config(kept).values())))
                self.portfolio.append(kept)
            if self.portfolio_cost:
                new_mean = self.portfolio_cost * (len(self.portfolio) - len(incs)) / len(self.portfolio)
                new_mean += np.nansum(means)
            else:
                new_mean = np.mean(means)
            self.cost_per_inst = defaultdict(lambda: new_mean)
            cur_cost = new_mean

        self.portfolio_cost = cur_cost
        return cur_cost
