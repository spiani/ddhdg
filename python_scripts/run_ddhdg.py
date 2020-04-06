import subprocess

from constants import EXECUTABLE


def run_ddhdg(execution_parameters):
    print('-' * 70)
    print('Executing the code with the following parameters:')
    print('  *  V degree:...........{:.>5}'.format(execution_parameters.V_degree))
    print('  *  n degree:...........{:.>5}'.format(execution_parameters.n_degree))
    print('  *  refinements:........{:.>5}'.format(execution_parameters.refinements))
    print()
    args = [EXECUTABLE, '-']
    proc = subprocess.run(
        args,
        encoding='UTF-8',
        input=execution_parameters.to_prm_file(),
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL
    )
    converged = (proc.returncode == 0)

    if converged:
        print('OK!')
    else:
        print('*** FAILED! ***')
    print('-' * 70)
    return converged
