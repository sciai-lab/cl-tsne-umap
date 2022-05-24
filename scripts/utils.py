import numpy as np

def get_noise_in_estimator(n, n_noise, dataset):
    print(dataset)

    # rescale the partition funciton / Z by n_noise and the data distribution
    if dataset == "mnist":
        # with EE
        noise_in_estimator_tsne = 8.13 * 10**6 * n_noise / n / (n-1)

        # without EE
        #noise_in_estimator_tsne = 6.25 * 10**6 * n_noise / n / (n-1)

        # using Z not norm for NCVis
        noise_in_estimator_ncvis = 3.43 * 10**7 * n_noise / n / (n-1)

    elif dataset == "human-409b2":
        noise_in_estimator_tsne = 1.30 * 10**6 * n_noise / n / (n-1)
        noise_in_estimator_ncvis = 3.57 * 10**6 * n_noise / n / (n-1) # using Z not norm

    noise_in_estimator_tsne = float(np.format_float_scientific(noise_in_estimator_tsne,
                                                           precision=2))
    noise_in_estimator_ncvis = float(np.format_float_scientific(noise_in_estimator_ncvis,
                                                            precision=2))

    nbs_noise_in_estimator =  np.array([noise_in_estimator_tsne,
                                        noise_in_estimator_ncvis,
                                        5e-5,
                                        1e-4,
                                        2e-4, 5e-4,
                                        1e-3,
                                        2e-3, 5e-3,
                                        1e-2,
                                        2e-2, 5e-2,
                                        1e-1,
                                        2e-1,
                                        5e-1,
                                        1e-0,2e-0, 5e-0,
                                        1e1, 2e1, 5e1,
                                        1e2, 2e2, 5e2
                                        ])

    return nbs_noise_in_estimator