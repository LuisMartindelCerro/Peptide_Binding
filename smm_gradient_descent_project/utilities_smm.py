def encode(peptides, encoding_scheme, alphabet):
    encoded_peptides = []
    for peptide in peptides:
        encoded_peptide = []
        for peptide_letter in peptide:
            for alphabet_letter in alphabet:
                encoded_peptide.append(encoding_scheme[peptide_letter][alphabet_letter])
        encoded_peptides.append(encoded_peptide)
    return np.array(encoded_peptides)

def cumulative_error(peptides, y, lamb, weights):
    error = 0
    for i in range(0, len(peptides)):
        peptide = peptides[i]
        y_target = y[i]
        y_pred = np.dot(peptide, weights)
        error += 1.0/2 * (y_pred - y_target)**2
    gerror = error + lamb*np.dot(weights, weights)
    error /= len(peptides)
    return gerror, error

def predict(peptides, weights):
    pred = []
    for i in range(0, len(peptides)):
        peptide = peptides[i]
        y_pred = np.dot(peptide, weights)
        pred.append(y_pred)
    return pred

def cal_mse(vec1, vec2):
    mse = 0
    for i in range(0, len(vec1)):
        mse += (vec1[i] - vec2[i])**2
    mse /= len(vec1)
    return mse

def gradient_descent(y_pred, y_target, peptide, weights, lamb_N, epsilon):
    do = y_pred - y_target
    for i in range(0, len(weights)):
        de_dw_i = do * peptide[i] + 2*lamb_N * weights[i]
        weights[i] -= epsilon * de_dw_i


def gradient_descent_batch(peptides, y, weights, lamb_N, epsilon):
    ix = np.random.randint(0, len(peptides))
    peptide = peptides[ix]
    y_target = y[ix]
    y_pred = np.dot(peptide, weights)
    gradient_descent(y_pred, y_target, peptide, weights, lamb_N, epsilon)