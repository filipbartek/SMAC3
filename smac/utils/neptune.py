def configspace_to_dict(cs):
    return {
        'hyperparameters': cs.get_hyperparameters_dict(),
        'conditions': cs.get_conditions(),
        'forbidden_clauses': cs.get_forbiddens()
    }
