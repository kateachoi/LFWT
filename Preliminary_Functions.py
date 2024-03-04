import numpy as np
from scipy.linalg import eig, cholesky, eigh
import matplotlib.pyplot as plt


def onsiteextractor(num, dim1, dim2, matrix):
    if num == 1:
        out = np.zeros((dim1, dim1), dtype=complex)

        for iter in range(1, dim2 + 1):
            state = np.concatenate((np.zeros(iter - 1), [1], np.zeros(dim2 - iter)))
            base = np.transpose(np.kron(np.eye(dim1), [state]))
            out += np.conj(np.transpose(base)) @ matrix @ base

        out /= np.sqrt(dim1 * dim2)
        out -= np.eye(dim1) * np.trace(out) / len(out)

    else:
        out = np.zeros((dim1, dim2), dtype=complex)

        for iter in range(1, dim1 + 1):
            state = np.concatenate((np.zeros(iter - 1), [1], np.zeros(dim1 - iter)))
            base = np.transpose(np.kron([state], np.eye(dim2)))
            out += np.conj(np.transpose(base)) @ matrix @ base

        out /= np.sqrt(dim1 * dim2)
        out -= np.eye(dim1) * np.trace(out) / len(out)

    return (out + np.conj(np.transpose(out))) / 2


# ONSITEEXTRACTOR CHECK !! DONE


def ConstructMeanField(sitenumber, localham, nonlocalham, psis):
    # local hamiltonian : {matrix}
    ham = localham[sitenumber - 1];

    # nonlocal hamiltonion handler : {siteA, siteB, vector, matrix}

    intersiteterms = [term for term in nonlocalham if term[0] == sitenumber]
    if intersiteterms:

        for term in intersiteterms:
            # get site indices
            site1, site2 = term[0], term[1]

            # get hilbert space dimensions
            dim1, dim2 = len(localham[site1 - 1]), len(localham[site2 - 1])

            # get state of second site
            psi2 = psis[site2 - 1]

            # construct two-site basis
            testpsis = np.transpose(np.kron(np.eye(dim1), [psi2]))
            ex = np.conj(np.transpose(testpsis)) @ term[3] @ testpsis
            ex -= onsiteextractor(1, dim1, dim2, term[3])
            ham += ex

    intersiteterms = [term for term in nonlocalham if term[1] == sitenumber]
    if intersiteterms:

        for term in intersiteterms:
            # get site indices
            site1, site2 = term[1], term[0]

            # get hilbert space dimensions
            dim1, dim2 = len(localham[site1 - 1]), len(localham[site2 - 1])

            # get state of second site
            psi2 = psis[site2 - 1]

            # construct two-site basis
            testpsis = np.transpose(np.kron([psi2], np.eye(dim1)))
            ex = np.conj(np.transpose(testpsis)) @ term[3] @ testpsis
            ex -= onsiteextractor(2, dim1, dim2, term[3])
            ham += ex

    return ham


def RunMeanField(localham, nonlocalham, psis, maxrun, printflag):
    nsites = len(localham)
    output = np.array(psis)
    change = 1
    iterationnumber = 0

    while change > 0.0000001 and iterationnumber <= maxrun:
        oldpsis = np.array(output)

        if printflag == 1:
            print(f"Iteration Number {iterationnumber} psis = ")
            print(oldpsis)

        for j in range(1, nsites + 1):
            ham = ConstructMeanField(j, localham, nonlocalham, output)
            vals, vecs = eig(ham)
            order = np.argsort(vals)
            vals = vals[order]
            vecs = vecs[:, order]
            output[j - 1] = np.real(vecs[:, 0])

        change = np.linalg.norm(output - oldpsis)
        iterationnumber += 1

    if iterationnumber <= maxrun:
        print(f"MeanField .. Success. Converged after {iterationnumber - 1} iterations.")
    else:
        print(f"MeanField .. did not converge. Stopped after {iterationnumber - 1} iterations.")

    return output


def ConstructFlavorWave(coords, localham, nonlocalham, psis, kx, ky):
    basis = []
    localvals = []
    nsites = len(localham)

    # first run mean-field to get on-site Hamiltonian and basis
    for k in range(1, nsites + 1):
        ham = ConstructMeanField(k, localham, nonlocalham, psis)
        vals, vecs = eig(ham)
        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]
        basis.append(vecs)
        localvals.append(vals)

    # adjust the second component of the first site's basis
    basis[1][0] = -1j * basis[1][0]

    totalstates = sum(len(localvals[i]) for i in range(nsites))

    # DEBUG
    if not localvals:
        raise ValueError("localvals is empty. Check your input data.")

    index = []
    curr_index = 0
    for i in range(nsites - 1):
        index.append(list(range(curr_index, curr_index + len(localvals[i]))))
        curr_index += len(localvals[i])

    # number = [0] + [len(localvals[i]) - 1 for i in range(nsites)]
    # temp = list(range(1, totalstates + 1))
    # index = [temp[sum(number[:j]): sum(number[:j + 1])] for j in range(nsites - 1)]

    # a+i ai
    exlist = []
    for i in range(nsites):
        exlist.extend(localvals[i][1:] - localvals[i][0])
    # exlist = np.concatenate([localvals[i][1:] - localvals[i][0] for i in range(nsites)])

    # print exlist
    ada = np.diag(exlist)

    # a+i aj
    # nonlocal hamiltonian handler : {siteA, siteB, vector, matrix}

    for term in nonlocalham:
        siteA, siteB = term[0] - 1, term[1] - 1

        dif = -1 * (np.array(term[2]) + coords[siteB - 1] - coords[siteA - 1])
        dif = dif.reshape(-1, 1)  # Reshape to (2, 1)

        psileft = np.conj(np.kron(basis[siteA][1:], [basis[siteB][0]]))
        psiright = np.transpose(np.kron([basis[siteA][0]], basis[siteB][1:]))
        ex = np.dot(np.dot(psileft, term[3]), psiright)


        # DEBUG
        print("siteA: ", siteA)
        print("siteB: ", siteB)
        print("Length of index: ", len(index))

        if ex.shape == (): # ex is scalar
            ada[index[siteA - 1], index[siteB - 1][0]] += ex.reshape(2, 1, -1) * np.exp(1j * np.dot(np.array([kx, ky]), dif))

        else:
            ada[index[siteA - 1], index[siteB - 1][0]] += np.sum(ex.reshape(1, 2, -1) * np.exp(1j * np.dot(np.array([kx, ky]), dif)[:, :, np.newaxis]), axis=(1, 2))


        #ada[index[siteA - 1], index[siteB - 1][0]] += ex * np.exp(1j * np.dot(np.array([kx, ky]), dif))
        #ada[index[siteB - 1], index[siteA - 1][0]] += np.transpose(np.conj(ex)) * np.exp(
            #-1j * np.dot(np.array([kx, ky]), dif))

    adad = np.zeros((len(exlist), len(exlist)), dtype=complex)

    # a+i a+j
    # nonlocal hamiltonian handler : {siteA, siteB, vector, matrix}

    for term in nonlocalham:
        siteA, siteB = term[0], term[1]
        dif = -1 * (np.array(term[2]) + coords[siteB - 1] - coords[siteA - 1])

        psiright = np.transpose(np.kron([basis[siteA - 1][0]], [basis[siteB - 1][0]]))
        psileft = np.conj(np.kron(basis[siteA - 1][1:], basis[siteB - 1][1:]))
        # ex = np.array_split(psileft @ matrix @ psiright, len(basis[siteB - 1]) - 1)
        ex = np.dot(np.dot(psileft, term[3]), psiright)
        ex = np.array(ex).reshape((len(basis[siteB - 1]) - 1, len(basis[siteB - 1]) - 1))

        adad[index[siteA - 1], index[siteB - 1]] += ex * np.exp(1j * np.array([kx, ky]) @ dif)
        adad[index[siteB - 1], index[siteA - 1]] += np.transpose(ex) * np.exp(-1j * np.array([kx, ky]) @ dif)

    result = np.block([[ada, adad], [np.transpose(np.conj(adad)), np.transpose(np.conj(ada))]])
    return basis, np.real_if_close(result)


def DiagChol(matrix):
    dim = len(matrix)
    mat = 0.5 * matrix + 0.5 * np.transpose(np.conj(matrix))

    k = cholesky(mat + 0.0000000 * np.eye(dim), lower=False)
    g = np.block([[np.eye(dim // 2), np.zeros((dim // 2, dim // 2))],
                  [np.zeros((dim // 2, dim // 2)), -np.eye(dim // 2)]])

    vals, vecs = eigh(np.dot(np.dot(k, g), np.transpose(np.conj(k))))
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    t = np.linalg.inv(k) @ np.transpose(vecs) @ np.linalg.matrix_power(g @ np.diag(vals), 1 / 2)

    return vals, np.transpose(t)


def Intensities(basis, eigsys, op, freq_range):
    nsites = len(basis)
    dim = len(np.concatenate(basis, axis=1)) - nsites

    totalstates = sum([len(basis[i]) for i in range(nsites)])
    number = [0] + [len(basis[i]) - 1 for i in range(nsites)]
    temp = list(range(1, totalstates + 1))
    index = [temp[sum(number[:i]): sum(number[:i + 1])] for i in range(nsites - 1)]

    # a + i
    ad = np.zeros(dim, dtype=complex)
    for term in op:
        siteA, dimA, opA = term[0], len(basis[term[0] - 1]) - 1, term[1]

        for stateA in range(1, dimA + 1):
            ME = np.conj(basis[siteA - 1][stateA]) @ opA @ np.transpose(basis[siteA - 1][0])
            ad[index[siteA - 1][stateA - 1]] += ME

    # ai
    a = np.zeros(dim, dtype=complex)
    for term in op:
        siteA, dimA, opA = term[0], len(basis[term[0] - 1]) - 1, term[1]

        for stateA in range(1, dimA + 1):
            ME = np.conj(basis[siteA - 1]) @ opA @ np.transpose(basis[siteA - 1][stateA])
            a[index[siteA - 1][stateA - 1]] += ME

    freqpoints = np.linspace(freq_range[0], freq_range[1], freq_range[2])
    intensities = np.zeros((len(freqpoints), len(eigsys[0])), dtype=float)

    for i, omega in enumerate(freqpoints):
        for j, val in enumerate(eigsys[0]):
            dens = sum([(np.abs(ad[t]) ** 2) / (2 * omega) + (np.abs(a[t]) ** 2) * (2 * omega)
                        for t in range(len(ad))])
            intensities[i, j] = dens

    return intensities

#   all_op = np.concatenate([ad, a])
#   measure = np.kron(np.transpose(all_op), np.conj(all_op))

#   ens = np.real(eigsys[0])
#   ints = np.real(np.diag(np.conj(eigsys[1]) @ measure @ np.transpose(eigsys[1])))

#   def f(w):
#       return sum([ints[i] * (gamma / np.pi) / ((w - ens[i]) ** 2 + gamma ** 2) if wmin <= ens[i]<= wmax
#                   else 0 for i in range(len(ens) // 2, len(ens))])

#   result = [(w0, f(w0)) for w0 in np.linspace(freq[0], freq[1], freq[2])]
#   return result
