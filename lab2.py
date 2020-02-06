import numpy as np
import texttable as tt
from scipy import linalg
import sys
import random


def is_close(float1, float2, epsilon=1e-6):
    return abs(float1 - float2) <= epsilon


def f1(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def f2(x):
    return (x[0] - 4) ** 2 + 4 * (x[1] - 2) ** 2


def f3(x):
    return (x[0] - 2) ** 2 + (x[1] + 3) ** 2


def f4(x):
    return (x[0] - 3) ** 2 + x[1] ** 2


# Enables us to not repeat calculations and keep track of number of evals
class GoalFunction:
    def __init__(self, function, start=None):
        self.f = function
        self.start = np.array(start)
        self.count = 0
        self.store = dict()

    def eval(self, x):
        if str(x) not in self.store.keys():
            self.store[str(x)] = self.f(x)
            self.count += 1
        return self.store[str(x)]

    def reset(self):
        self.count = 0
        self.store = dict()


# Wraps GoalFunction in order to search along an axis
# e_i is direction vector
class LambdaMinWrapper:
    def __init__(self, gf, x, e_i):
        self.gf = gf
        self.x = x
        self.e_i = e_i

    def eval(self, L):
        return self.gf.eval(self.x + L * self.e_i)


'''
Postupak trazenja unimodalnog intervala

Ulazne velicine:
- gf: ciljna funkcija (klasa GoalFunction)
- tocka: pocetna tocka pretrazivanja
- h: pomak pretrazivanja
- f: ciljna funkcija

Izlazne vrijednosti:
- unimodalni interval [l, r]
'''


def get_unimodal_interval(gf, tocka, h=1):
    step = 1
    l, m, r = float(tocka) - h, float(tocka), float(tocka) + h
    fl, fm, fr = gf.eval(l), gf.eval(m), gf.eval(r)

    if (fm < fr) and (fm < fl):
        return [float(l), float(r)]

    elif fm > fr:
        while fm > fr:
            l = float(m)
            m = float(r)
            fm = float(fr)
            step *= 2
            r = float(tocka) + h * step
            fr = gf.eval(r)

    else:
        while fm > fl:
            r = float(m)
            m = float(l)
            fm = float(fl)
            step *= 2
            l = float(tocka) - h * step
            fl = gf.eval(l)

    return [float(l), float(r)]


'''
Algoritam zlatnog reza

ulazne velicine:
- a, b: pocetne granice unimodalnog intervala
- e: preciznost

ako je jedan od a ili b None, obavlja se get_unimodal_interval
'''


def golden_section_search(gf, a=None, b=None, e=1e-6):
    if a is None:
        a, b = get_unimodal_interval(gf, b, 1)
    elif b is None:
        a, b = get_unimodal_interval(gf, a, 1)
    k = 0.5 * (np.sqrt(5) - 1)
    c = b - k * (b - a)
    d = a + k * (b - a)
    fc = gf.eval(c)
    fd = gf.eval(d)
    while (b - a) > e:
        if fc < fd:
            b = float(d)
            d = float(c)
            c = b - k * (b - a)
            fd = float(fc)
            fc = gf.eval(c)
        else:
            a = c
            c = d
            d = a + k * (b - a)
            fc = fd
            fd = gf.eval(d)
    return (a + b) / 2


'''
    Algoritam simpleks postupka po Nelderu i Meadu (Downhill simplex)

    gf - ciljna funkcija
    x0 - pocetna tocka
    step - koef pomaka jednog koraka
    alpha - koef refleksija
    beta - koef kontrakcija
    gamma - koef ekspanzije
    sigma - koef pomicanja u smjeru najbolje tocke
    mat_iter - maksimalan broj iteracija
'''


def nelder_mead(gf, x0, step=1, alpha=1, beta=0.5, gamma=2, sigma=0.5, e=1e-6, max_iter=10000):
    tab = tt.Texttable()
    tab.header(['Iteracija', 'Centroid', 'f(centroid)', 'Simplex'])
    # pocetni simplex
    simplex, centroid = [x0], np.array([])
    for i in range(len(x0)):
        tocka = np.array(x0)
        tocka[i] += step
        simplex.append(tocka)
    simplex = np.array(simplex)
    for i in range(max_iter):
        # indeksi najvece i najmanje vrijednosti funkcije
        l, h = get_max_and_min(gf, simplex)
        centroid = get_centroid(simplex, h)
        tab.add_row([i, centroid, gf.eval(centroid), simplex])
        reflected = reflection(simplex[h], centroid, alpha)
        if gf.eval(reflected) < gf.eval(simplex[l]):
            expanded = expansion(reflected, centroid, gamma)
            if gf.eval(expanded) < gf.eval(simplex[l]):
                simplex[h] = np.array(expanded)
            else:
                simplex[h] = np.array(reflected)
        else:
            # ako F(Xr)>F(X[j]) za svaki j=0..n, j!=h
            condition = True
            for j in range(simplex.shape[0]):
                if j == h:
                    continue
                if gf.eval(reflected) <= gf.eval(simplex[j]):
                    condition = False
                    break
            if condition is True:  # ako F(Xr)>F(X[j]) za svaki j=0..n, j!=h
                if gf.eval(reflected) < gf.eval(simplex[h]):
                    simplex[h] = np.array(reflected)
                contracted = contraction(simplex[h], centroid, beta)
                if gf.eval(contracted) < gf.eval(simplex[h]):
                    simplex[h] = np.array(contracted)
                else:
                    # pomakni sve tocke prema simplex[l]
                    simplex = move_all_to_lowest(simplex, l, sigma)
            else:
                simplex[h] = np.array(reflected)

        # stop if stop value <= epsilon
        stop_value = 0
        for i in range(simplex.shape[0]):
            stop_value += (gf.eval(simplex[i]) - gf.eval(centroid)) ** 2

        stop_value = np.sqrt(stop_value / float(simplex.shape[0]))

        if stop_value <= e:
            print("[+] Cilj dostignut prije max_iter, stop_value ={0}!!".format(stop_value))
            break

    print(tab.draw())
    # l, h = get_max_and_min(gf, simplex)
    # return simplex[l]
    return centroid


# mice sve tocke simplexa prema najmanjoj
def move_all_to_lowest(simplex, l, sigma):
    new_simplex = np.array([simplex[l]])
    for i in range(simplex.shape[0]):
        new_simplex = np.vstack([new_simplex, np.array([sigma * (simplex[i] + simplex[l])])])
    return new_simplex[1:]


# prima najgoru tocku i vraca njenu refleksiju
def reflection(tocka, centroid, alpha):
    return (1 + alpha) * centroid - alpha * tocka


# prima reflektiranu tocku i produljuje ju u smjeru centroida
def expansion(reflected, centroid, gamma):
    return (1 - gamma) * centroid + gamma * reflected


# prima najgoru tocku i pomice ju u smjeru centroida tako da smanji simpleks
def contraction(tocka, centroid, beta):
    return (1 - beta) * centroid + beta * tocka


# vraca centroid svih osim skip_i-te tocke
def get_centroid(simplex, skip_i):
    centroid = np.zeros(simplex.shape[1])
    for i in range(simplex.shape[0]):
        if i == skip_i:
            continue
        for j in range(simplex.shape[1]):
            centroid[j] += simplex[i][j]

    return np.true_divide(centroid, simplex.shape[0] - 1)


# vraca indekse najbolje i najgore tocke
def get_max_and_min(gf, simplex):
    l, h = 0, 0
    max_value = gf.eval(simplex[0])
    min_value = gf.eval(simplex[0])
    for i in range(1, simplex.shape[0]):
        value = gf.eval(simplex[i])
        if value > max_value:
            max_value = value
            h = i
        if value < min_value:
            min_value = value
            l = i
    return l, h


'''
    Algoritam Hooke-Jeeves postupka

    x0 - pocetna tocka
    xB - bazna tocka 
    xP - pocetna tocka pretrazivanja
    xN - tocka dobivena pretrazivanjem
'''


def hooke_jeeves(gf, x0, dx=0.5, e=10e-6, max_iter=200):
    tab = tt.Texttable()
    tab.header(['Iteracija', 'Bazna', 'Pocetna', 'Explored'])

    start = base = x0
    for i in range(max_iter):
        explored = explore(gf, start, dx=dx)
        if gf.eval(explored) < gf.eval(base):  # prihvati baznu tocku
            start = 2 * explored - base  # definiramo novu tocku pretrazivanja
            base = np.array(explored)
        else:
            dx /= 2.0
            start = np.array(base)  # vratimo se na zadnju baznu tocku
        tab.add_row([i, "f({0})={1}".format(base, gf.eval(base)),
                     "f({0})={1}".format(start, gf.eval(start)),
                     "f({0})={1}".format(explored, gf.eval(explored))])
        # uvjet zaustavljanja
        if dx < e:
            print("[+] Kraj prije max_iter, dx=", dx)
            break

    print(tab.draw())
    if gf.eval(start) < gf.eval(base) and gf.eval(start) < gf.eval(explored):
        return start
    elif gf.eval(explored) < gf.eval(base) and gf.eval(explored) < gf.eval(start):
        return explored
    return base


# hooke jeeves helper functions
def explore(gf, tocka, dx=0.5):
    x = np.array(tocka)
    for i in range(x.shape[0]):
        P = gf.eval(x)
        x[i] = float(x[i]) + dx
        N = gf.eval(x)
        if N > P:  # ne valja pozitivni pomak
            x[i] -= 2 * dx
            N = gf.eval(x)
            if N > P:  # ne valja ni negativni
                x[i] += dx  # vratimo na staro
    return x


'''
    Helper for gradient methods
'''


def approx_gradient(f, x, delta=1e-9):
    gradients = []
    for j in range(len(x)):
        tmp_x1, tmp_x2 = list(x), list(x)
        tmp_x1[j] += delta
        tmp_x2[j] -= delta
        gradient_approx = f.eval(tmp_x1) - f.eval(tmp_x2)
        gradient_approx /= (2 * delta)
        gradients.append(gradient_approx)

    return gradients


'''
    Gradient Descent with gradient calculation on the fly

    function        GoalFunction which we are minimizing
    x               Vector with start values
    golden_section  Finds optimal learning rate if True
'''


def gradient_descent(f, x, golden_section=True, epsilon=1e-9, rate=1, iterations=1000, delta=1e-9):
    stuck_count, f_x, f_best = 0, 0, 10e18
    for i in range(iterations):
        if f.eval(x) >= f_best:
            stuck_count += 1
        else:
            stuck_count = 0
        if stuck_count == 100:
            break
        f_x = f.eval(x)
        if f_x < f_best:
            f_best = float(f_x)

        print("{}: f({}): {}".format(i, x, f_x))
        if abs(f_x) <= epsilon:
            print("Success!")
            break

        gradients = approx_gradient(f, x, delta=delta)

        if golden_section is False:
            for j in range(len(x)):
                x[j] -= rate * gradients[j]
        else:  # using golden section search to find optimal learning rate
            for j in range(len(x)):
                Lgf = LambdaMinWrapper(f, x, np.array(gradients))
                unimodal = get_unimodal_interval(Lgf, 0)
                L = golden_section_search(Lgf, unimodal[0], unimodal[1])
                x[j] += L * gradients[j]

    f_x = f.eval(x)
    print("Final result: f({}): {}\n".format(x, f_x))
    return x, f_x, f.count


'''
    Helper for Newton methods
'''


def hesse(f, x, delta=1e-6):
    d = len(x)
    grad_x = approx_gradient(f, x)
    H = []
    for i in range(len(grad_x)):
        tmp_x1, tmp_x2 = list(x), list(x)
        tmp_x1[i] += delta
        tmp_x2[i] -= delta
        sd1 = np.array(approx_gradient(f, tmp_x1))
        sd2 = np.array(approx_gradient(f, tmp_x2))
        second_derivative = (sd1 - sd2) / (2 * delta)
        H.append(second_derivative)

    return H


def supstitute_backward(U, y):
    return linalg.solve(U, y)


def supstitute_forward(L, P, E):
    N = L.shape[0]
    PE = np.dot(P, E)
    return linalg.solve(L, PE)


def inverse(m):
    m = np.array(m)
    E = np.zeros(m.shape)
    for i in range(E.shape[0]):
        E[i][i] = 1

    P, L, U = linalg.lu(m)
    X = []
    for i in range(m.shape[0]):
        y = supstitute_forward(L, P, E)
        x = supstitute_backward(U, y)
        X.append(x)
    return np.array(X)


'''
    Newton Raphson optimization method

    function        GoalFunction which we are minimizing
    x               Vector with start values
    golden_section  Finds optimal learning rate if True
'''


def newton_rhapson(f, x, epsilon=1e-6, iterations=1000):
    counters = {'hesse': 0, 'gradient': 0, "f_evals": 0}
    for i in range(iterations):
        f_x = f.eval(x)
        print("{}: f({}): {}".format(i, x, f_x))

        if abs(f_x) <= epsilon:
            print("Success!")
            break

        gradient = np.array(approx_gradient(f, x))
        H = np.array(hesse(f, x))
        counters['hesse'] += 1
        counters['gradient'] += 2 * len(gradient) + 1
        print("Hesse:\n", H)
        try:
            step = np.dot(inverse(H), gradient)[0][:]
            Lgf = LambdaMinWrapper(f, x, step)
            unimodal = get_unimodal_interval(Lgf, 0)
            L = golden_section_search(Lgf, unimodal[0], unimodal[1])
            x = x + L * step
        except np.linalg.LinAlgError as e:
            print(str(e))
            print("\nCannot find inverse of hesse matrix\n")
            return "FAILED"

    f_x = f.eval(x)
    print("Final result: f({}): {}\n".format(x, f_x))
    counters['f_evals'] = f.count
    return x, f_x, counters


'''
    Helper method used with constrained optimization methods:
        Box Algorithm
        Mixed-Transformation Algorithm
        
    Implicit limitations hardcoded:
    (x2-x1 >= 0), (2-x1 >= 0)
'''


def check_implicit_limitations(x):
    if x[1] - x[0] >= 0 and 2 - x[0] >= 0:
        return True
    else:
        return False


'''
    Helper for Box
    
    returns indices of two worst points
'''


def get_worst_indices(simplex, f):
    to_be_sorted = []
    for i, x in enumerate(simplex):
        to_be_sorted.append([i, f.eval(x)])
    to_be_sorted = sorted(to_be_sorted, key=lambda x: x[1])

    return to_be_sorted[-1][0], to_be_sorted[-2][0]


'''
    Helper for Box
    
    returns centroid of given points
'''


def centroid(simplex):
    N = len(simplex[0])
    xc = np.zeros(N)
    for x in simplex:
        xc += x
    xc /= N

    return xc


'''
    Helper for Box
    
    reflection operator
'''


def reflection(point, xc, alpha):
    return (1 + alpha) * xc - alpha * point


'''
    Box algorithm
    
    f - goalfunction
    x0 - starting point, arbitrary dimensions
    xd, xg - lower, upper limit
    
'''


def box_algorithm(f, x0, xd, xg, eps=1e-6, alpha=1.3):
    # validating starting point
    assert (f.eval(x0) >= 0)
    N = len(x0)
    # expand xd and xg if they are single numbers
    if len(xd) == 1:
        for i in range(N - 1):
            xd.append(float(xd[0]))
    if len(xg) == 1:
        for i in range(N - 1):
            xg.append(float(xg[0]))
    # check for explicit limitations
    for i, xi in enumerate(x0):
        assert (xd[i] <= xi <= xg[i])

    # generating 2*N vectors
    xc = np.array(x0)  # satisfies limitations
    simplex = []
    for t in range(2 * N):
        x = np.zeros(N)
        for i in range(N):
            R = random.uniform(0, 1)
            x[i] = xd[i] * R * (xg[i] - xd[i])  # inside explicit limitation
        while check_implicit_limitations(x) is False:
            x = 0.5 * (x + xc)  # move towards centroid until implicit limitations are satisfied
        simplex.append(x)

    stuck_count, f_x, f_best = 0, 0, 10e18
    while abs(f.eval(xc) > eps):
        if f.eval(xc) >= f_best:
            stuck_count += 1
        else:
            stuck_count = 0
        if stuck_count == 100:
            break
        f_x = f.eval(xc)
        if f_x < f_best:
            f_best = float(f_x)

        worst1, worst2 = get_worst_indices(simplex, f)
        no_worst = []
        for i, x in enumerate(simplex):
            if i != worst1 and i != worst2:
                no_worst.append(x)

        xc = centroid(no_worst)
        print("f({})={}".format(xc, f.eval(xc)))
        xr = reflection(simplex[worst1], xc, alpha)

        for i in range(N):
            if xr[i] < xd[i]:
                xr[i] = float(xd[i])
            elif xr[i] > xg[i]:
                xr[i] = float(xg[i])
        while check_implicit_limitations(xr) is False:
            xr = 0.5 * (xr + xc)  # move towards centroid until implicit limitations are satisfied
        if f.eval(xr) > f.eval(simplex[worst2]):  # still worst point
            xr = 0.5 * (xr + xc)  # another squash towards centroid

        simplex[worst1] = xr

    return xc, f.eval(xc), f.count


class NoLimitForm:
    def __init__(self, f, g, h, t, N, eps=1e-6):
        self.f = f  # arbitrary function
        self.g = g  # >=0
        self.h = h  # ==0
        self.t = t  # t=1/r parameter
        self.N = N  # dimensionality
        self.eps = eps  # precision
        self.count = 0
        self.store = dict()

    def eval(self, x):
        xx = np.insert(x, 0, 1.0, axis=0)
        if str(x) not in self.store.keys():
            # return infinite if x does not satisfy limitations
            if self.check_limits(x) is False:
                self.store[str(x)] = 1e18
                return self.store[str(x)]
            # formula for this form
            g_value, h_value = 0, 0
            for g in self.g:
                g_value += np.log(sum(xx*np.array(g)))
            g_value *= (1 / self.t)
            for h in self.h:
                h_value += sum(xx*np.array(h))**2
            h_value *= self.t
            value = self.f.eval(x) - g_value + h_value
            self.store[str(x)] = value
            self.count += 1
        return self.store[str(x)]

    def reset(self):
        self.count = 0
        self.store = dict()

    def check_limits(self, x):
        xx = np.insert(x, 0, 1.0, axis=0)
        result = True
        # >= 0
        for g in self.g:
            if sum(np.array(g) * xx) < 0:
                result = False
        # == 0
        for h in self.h:
            if abs(sum(np.array(h) * xx)) < self.eps:
                result = False
        return result


'''
    No limit transform algorithm
    
    f - NoLimitForm type class
    x0 - starting point, arbitrary dimensions
'''


def no_limit_transform(f, x, eps=1e-6):
    assert(f.check_limits(x))

    stuck_count, f_x, f_best = 0, 0, 10e18
    while True:
        # stuck exit condition
        if f.eval(x) >= f_best:
            stuck_count += 1
        else:
            stuck_count = 0
        if stuck_count == 100:
            break
        f_x = f.eval(x)
        if f_x < f_best:
            f_best = float(f_x)

        new_x = hooke_jeeves(f, x, max_iter=1000)
        if abs(f.eval(x) - f.eval(new_x)) < eps:
            break
        x = new_x
        f.t *= 10
        print("Current t: {}\t x: {}".format(f.t, x))
    return new_x, f.count, f.f.eval(new_x)


def task1():
    result1 = gradient_descent(GoalFunction(f3), [0, 0], golden_section=False, rate=1)
    result2 = gradient_descent(GoalFunction(f3), [0, 0], golden_section=False, rate=0.1)
    gradient_descent(GoalFunction(f3), [0, 0])
    print("Task1 results:\nrate=1: ", result1)
    print("rate=0.1: ", result2)


def task2():
    result1 = gradient_descent(GoalFunction(f1), [-1.9, 2])
    result2 = gradient_descent(GoalFunction(f2), [0.1, 0.3])

    result3 = newton_rhapson(GoalFunction(f1), [-1.9, 2])
    result4 = newton_rhapson(GoalFunction(f2), [0.1, 0.3])
    print("Gradient descent results on f1:\n ", result1)
    print("Gradient descent results on f2: \n", result2)

    print("Newton Rhapson results on f1:\n", result3)
    print("Newton Rhapson results on f2:\n", result4)


def task3():
    result1 = box_algorithm(GoalFunction(f1), [-1.9, 2], [-100], [100])
    result2 = box_algorithm(GoalFunction(f2), [0.1, 0.3], [-100], [100])
    print("Box algorithm results on f1:\n", result1)
    print("Box algorithm results on f2:\n", result2)


def task4():
    g, h = [[0, -1, 1]], [[2, -1, 0]]
    no_limit_form1 = NoLimitForm(GoalFunction(f1), g, h, t=1, N=2)
    no_limit_form2 = NoLimitForm(GoalFunction(f2), g, h, t=1, N=2)
    result1 = no_limit_transform(no_limit_form1, [-1.9, 2])
    result2 = no_limit_transform(no_limit_form2, [0.1, 0.3])
    print("Results on f1:\n", result1)
    print("Results on f2:\n", result2)
    '''
    p1, p2 = [1.5, 10], [1.5, 10]
    input("Press ENTER to test better starting points: f1:{}\tf2:{}".format(p1, p2))
    no_limit_form1 = NoLimitForm(GoalFunction(f1), g, h, t=1, N=2)
    no_limit_form2 = NoLimitForm(GoalFunction(f2), g, h, t=1, N=2)
    result1 = no_limit_transform(no_limit_form1, [-1.9, 2])
    result2 = no_limit_transform(no_limit_form2, [0.1, 0.3])
    print("Results on f1:\n", result1)
    print("Results on f2:\n", result2)
    '''


def task5():
    g, h = [[3, -1, -1], [3, 1.5, -1]], [[-1, 0, 1]]
    no_limit_form1 = NoLimitForm(GoalFunction(f4), g, h, t=1, N=2)
    result1 = no_limit_transform(no_limit_form1, [0.5, 0.5])
    print("Results on f4:\n", result1)


locals()["task" + str(sys.argv[1])]()
