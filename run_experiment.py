from tqdm import tqdm
import pandas as pd

from NoPrivacyModel import TransactionModel

if __name__ == '__main__':

    n_iterations = 5
    n_plebeians = 100
    n_patricians = 100
    file_name = f'results.csv'

    mu_ranges = [ (i/(-2), i/2) for i in range(1,6) ]
    sigma_ranges = [ i/10 for i in range(1,30) ]
    alphas = [ i/100 for i in range(1,100) ]
    betas = [ i/100 for i in range(1,300) ]
    gammas = [ i/300 for i in range(1,300) ]

    results = []

    for mu in tqdm(mu_ranges):
        for sigma in sigma_ranges:
            for alpha in alphas:
                for beta in betas:
                    for gamma in gammas:
                        for symmetric in [True, False]:

                            kwargs = {
                                'n_plebeians': n_plebeians,
                                'n_patricians': n_patricians,
                                'mu_range': mu,
                                'sigma': sigma,
                                'alpha': alpha,
                                'beta': beta,
                                'gamma': gamma,
                                'symmetric': symmetric,
                            }

                            model = TransactionModel(**kwargs)

                            for i in range(n_iterations):
                                model.step()

                            plebeian_wealth = sum([a.wealth for a in model.schedule.agents if a.type == 'plebeian'])
                            patrician_wealth = sum([a.wealth for a in model.schedule.agents if a.type == 'patrician'])

                            plebeian_transactions = sum([a.num_transactions
                                                         for a in model.schedule.agents if a.type == 'plebeian'])
                            patrician_transactions = sum([a.num_transactions
                                                          for a in model.schedule.agents if a.type == 'patrician'])

                            avg_plebeian_wealth = plebeian_wealth/n_plebeians
                            avg_patrician_wealth = patrician_wealth/n_patricians
                            plebeian_wealth_by_trans = plebeian_wealth/plebeian_transactions
                            patrician_wealth_by_trans = patrician_wealth/patrician_transactions

                            diff_avg_wealth = avg_plebeian_wealth - avg_patrician_wealth
                            diff_num_trans = plebeian_transactions - patrician_transactions
                            diff_wealth_by_trans = plebeian_wealth_by_trans - patrician_wealth_by_trans

                            results.append(
                                {
                                    'n_plebeians': n_plebeians,
                                    'n_patricians': n_patricians,
                                    'mu': mu,
                                    'sigma': sigma,
                                    'alpha': alpha,
                                    'beta': beta,
                                    'gamma': gamma,
                                    'symmetric': symmetric,
                                    'avg_plebeian_wealth': avg_plebeian_wealth,
                                    'avg_patrician_wealth': avg_patrician_wealth,
                                    'plebeian_transactions': plebeian_transactions,
                                    'patrician_transactions': patrician_transactions,
                                    'plebeian_wealth_by_trans': plebeian_wealth_by_trans,
                                    'patrician_wealth_by_trans': patrician_wealth_by_trans,
                                    'diff_avg_wealth': diff_avg_wealth,
                                    'diff_num_trans': diff_num_trans,
                                    'diff_wealth_by_trans': diff_wealth_by_trans,

                                }
                            )

    pd.DataFrame(results).to_csv(file_name, index=False)