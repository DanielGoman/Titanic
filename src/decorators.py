import os


def run_all_models(func):
    def run_loop(**kargs):
        """Decorator that runs all chosen models

        Empties the output file and iteratively calls runs a grid search cv
        for each of the models.

        Args:
            **kargs:
              dictionary of key worded parameters to pass to func
        """
        model_names = kargs.pop('model_names')
        if os.path.exists(kargs['out_path']):
            os.remove(kargs['out_path'])

        for model_name in model_names:
            print(f'Running {model_name}:')
            func(model_name=model_name, **kargs)

    return run_loop


def set_logger(func):
    def logger(**kargs):
        """Logger decorator

        Writes output to both the terminal and an output file

        Args:
            **kargs:
              dictionary of key worded parameters to pass to func
        """
        out_path = kargs.pop('out_path')
        model_name = kargs['model_name']
        scorers = kargs['scorers']

        score, best_params = func(**kargs)

        with open(out_path, 'a') as f:
            f.write(f'{model_name} scores:\n')
            for scorer in scorers:
                score_str = f'\t{scorer} score: {round(score[f"mean_test_{scorer}"][0], 4)}'
                f.write(f'{score_str}\n')
                print(score_str)

            f.write(f'\n\t{model_name} best params:\n')
            print(f'{model_name} best params:')
            for param_name, param_value in best_params.items():
                best_params_str = f'\t\t{param_name} = {param_value}'
                f.write(f'{best_params_str}\n')
                print(best_params_str)

            f.write('\n\n')

    return logger
