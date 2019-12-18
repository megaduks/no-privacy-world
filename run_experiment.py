from tqdm import tqdm
import pandas as pd

from NoPrivacyModel import TransactionModel

if __name__ == '__main__':

    n_iterations = 50
    n_plebeians = 100
    n_patricians = 100
    file_name = f'results.csv'

    mu_ranges = [ (-1,1) ]
    sigma_ranges = [ (1) ]
    alphas = [ 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8 ]
    betas = [ 1.1, 1.5, 2, 3 ]

    results = []

    for mu in mu_ranges:
        for sigma in sigma_ranges:
            for alpha in tqdm(alphas):
                for beta in betas:
                    for symmetric in [True]:

                        kwargs = {
                            'n_plebeians': n_plebeians,
                            'n_patricians': n_patricians,
                            'mu_range': mu,
                            'sigma': sigma,
                            'alpha': alpha,
                            'beta': beta,
                            'symmetric': symmetric,
                        }

                        model = TransactionModel(**kwargs)

                        for i in tqdm(range(n_iterations)):
                            model.step()

                        plebeian_wealth = sum([a.wealth for a in model.schedule.agents if a.type == 'plebeian'])
                        patrician_wealth = sum([a.wealth for a in model.schedule.agents if a.type == 'patrician'])

                        plebeian_transactions = sum([a.num_transactions
                                                     for a in model.schedule.agents if a.type == 'plebeian'])
                        patrician_transactions = sum([a.num_transactions
                                                      for a in model.schedule.agents if a.type == 'patrician'])

                        plebeian_rejections = sum([a.num_rejections
                                                     for a in model.schedule.agents if a.type == 'plebeian'])
                        patrician_rejections = sum([a.num_rejections
                                                      for a in model.schedule.agents if a.type == 'patrician'])

                        if n_plebeians > 0:
                            avg_plebeian_wealth = plebeian_wealth/n_plebeians
                        else:
                            avg_plebeian_wealth = 0

                        if n_patricians > 0:
                            avg_patrician_wealth = patrician_wealth/n_patricians
                        else:
                            avg_patrician_wealth = 0

                        if plebeian_transactions > 0:
                            plebeian_wealth_by_trans = plebeian_wealth/plebeian_transactions
                        else:
                            plebeian_wealth_by_trans = 0

                        if patrician_transactions > 0:
                            patrician_wealth_by_trans = patrician_wealth/patrician_transactions
                        else:
                            patrician_wealth_by_trans = 0

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
                                'symmetric': symmetric,
                                'avg_plebeian_wealth': avg_plebeian_wealth,
                                'avg_patrician_wealth': avg_patrician_wealth,
                                'plebeian_transactions': plebeian_transactions,
                                'patrician_transactions': patrician_transactions,
                                'plebeian_rejections': plebeian_rejections,
                                'patrician_rejections': patrician_rejections,
                                'plebeian_wealth_by_trans': plebeian_wealth_by_trans,
                                'patrician_wealth_by_trans': patrician_wealth_by_trans,
                                'diff_avg_wealth': diff_avg_wealth,
                                'diff_num_trans': diff_num_trans,
                                'diff_wealth_by_trans': diff_wealth_by_trans,

                            }
                        )

    pd.DataFrame(results).to_csv(file_name, index=False)