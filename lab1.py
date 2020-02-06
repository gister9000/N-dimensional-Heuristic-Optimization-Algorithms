import numpy as np
import texttable as tt

# KONFA POCETAK
# testrange pocetnih tocaka za cijeli prvi zadatak
testrange = [10, 100, 1000]
#
# parametri za postupak zlatnog reza
golden_e = 1e-6
# parametri za postupak pretrazivanja po koord. osima
axis_e = 1e-6
# parametri za Nelder-Mead postupak
alpha = 1
beta = 0.5
gamma = 2
sigma = 0.5
step = 1
simplex_e = 1e-6
# parametri za Hooke-Jeeves postupak
dx = 0.5
hook_jeeves_e = 1e-6
# KONFA KRAJ


# Rosenbrockova banana funkcija
# Početna točka: x0 = −( 1.9, 2), minimum: xmin = (1,1) , min f = 0
def f1(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


# Početna točka: x0 = (0.1,0.3) , minimum: xmin = (4,2), min f = 0
def f2(x):
    return (x[0] - 4) ** 2 + 4 * (x[1] - 2) ** 2


# Početna točka: x 0 0 = (nul vektor), minimum: xmin = (1,2,3,...,n)
def f3(x_vector):
    suma = 0
    for i in range(len(x_vector)):
        suma += (x_vector[i] - i) ** 2
    return suma


# Jakobovićeva funkcija
# Početna točka: x0 = (5.1,1.1) , minimum: xmin = (0,0) , min f = 0
def f4(x):
    return abs((x[0] - x[1]) * (x[0] + x[1])) + np.sqrt(x[0] ** 2 + x[1] ** 2)


# minimum: x 0 min = (nul vektor), min f = 0
def f6(x_vector):
    temp_suma = 0.0
    for i in range(len(x_vector)):
        temp_suma += x_vector[i] ** 2
    brojnik = np.sin(np.sqrt(temp_suma)) ** 2 - 0.5
    nazivnik = (1 + 0.001 * temp_suma) ** 2
    return 0.5 + brojnik / nazivnik


# klasa koja sluzi spremanju vec izracunatih vrijednosti i broj poziva
# ciljnih funkcija
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


# klasa koja sluzi minimizaciji lambde u pretrazivanju po osima
# koristi se na isti nacin kao i GoalFunction
class LamdbaMinWrapper:
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
def get_unimodal_interval(gf, tocka, h):
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
def golden_section_search(gf, a, b, e=1e-6):
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
Algoritam pretrazivanja po koordinatnim osima

ulazne velicine:
- x0: pocetna tocka
- e: preciznost
'''
def axis_search(gf, x0, e=1e-6):
    x = np.array(x0)
    n = x.shape[0]  # broj dimenzija
    while True:
        xs = np.array(x)
        for i in range(n):
            # e_i je jedinicni vektor s jedinicom na i-tom mjestu
            e_i = np.zeros(x.shape[0])
            e_i[i] = 1
            # odrediti lambda koji minimizira F(x + lambda * e_i)
            Lgf = LamdbaMinWrapper(gf, x, e_i)
            a, b = get_unimodal_interval(Lgf, 0.0, 1)
            L = golden_section_search(Lgf, a, b)
            # azuriranje tocke
            x = x + L * e_i
        # uvjet zavrsetka
        if abs(x[i] - xs[i]) <= e:
            break
    return x


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
def nelder_mead(gf, x0, step=1, alpha=1, beta=0.5, gamma=2, sigma=0.5, e=1e-6):
    tab = tt.Texttable()
    tab.header(['Iteracija', 'Centroid', 'f(centroid)', 'Simplex'])
    # pocetni simplex
    simplex, centroid = [x0], np.array([])
    for i in range(len(x0)):
        tocka = np.array(x0)
        tocka[i] += step
        simplex.append(tocka)
    simplex = np.array(simplex)
    for i in range(500):
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
    #l, h = get_max_and_min(gf, simplex)
    #return simplex[l]
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

    return np.true_divide(centroid, simplex.shape[0]-1)


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


#################################################################
# ZADATAK 1
def task1_function(x):
    return (x - 3) ** 2


def task1():
    for i in testrange:
        gf = GoalFunction(task1_function, start=[float(i)])
        a, b = get_unimodal_interval(gf, gf.start, 1)
        print("Dobiveni unimodalni interval: [{0}, {1}]".format(a, b))
        print("Metoda zlatnog reza, x0: {0}.".format(i))
        m = golden_section_search(gf, a, b, e=golden_e)
        print("[+] Minimum u {0} u {1} evaluacija funkcije.".format(m, gf.count))
    print("#" * 50, "\n")

    for i in testrange:
        gf = GoalFunction(task1_function, start=[float(i)])
        print("Pretrazivanje po osima, x0: {0}.".format(i))
        m = axis_search(gf, gf.start, e=axis_e)
        print("[+] Minimum u {0} u {1} evaluacija funkcije.".format(m, gf.count))
    print("#" * 50, "\n")

    for i in testrange:
        gf = GoalFunction(task1_function, start=[float(i)])
        print("Nelder Mead simplex, x0: {0}".format(i))
        m = nelder_mead(gf, gf.start, step=1.1, alpha=alpha, beta=beta, gamma=gamma, sigma=sigma, e=simplex_e)
        print("[+] Minimum u {0} u {1} evaluacija funkcije.".format(m, gf.count))
    print("#" * 50, "\n")

    for i in testrange:
        gf = GoalFunction(task1_function, start=[float(i)])
        print("Hooke Jeeves, x0: {0}".format(i))
        m = hooke_jeeves(gf, gf.start, dx=dx, e=hook_jeeves_e)
        print("[+] Minimum u {0} u {1} evaluacija funkcije.".format(m, gf.count))
    print("#" * 50, "\n")


#################################################################
# Zadatak 2
def task2():
    gf1 = GoalFunction(f1, start=[-1.9, 2.0])
    gf2 = GoalFunction(f2, start=[0.1, 0.3])
    gf3 = GoalFunction(f3, start=[0.0, 0.0, 0.0, 0.0, 0.0])
    gf4 = GoalFunction(f4, start=[5.1, 1.1])
    goalfuncs = [gf1, gf2, gf3, gf4]

    i, alg = 1, 1
    algos = {"1": "Nelder Mead",
             "2": "Hooke Jeeves",
             "3": "Axis search"}
    table_data = []
    for f in goalfuncs:
        print("Ciljna funkcija broj ", i)
        print("Nelder Mead simplex, x0: {0}".format(f.start))
        m = nelder_mead(f, f.start, step=step, alpha=alpha, beta=beta, gamma=gamma, sigma=sigma, e=simplex_e)
        print("[+] Minimum u {0} u {1} evaluacija funkcije.".format(m, f.count))
        table_data.append([i, algos[str(alg)], f.count, m])
        f.reset()
        alg += 1

        print("Ciljna funkcija broj ", i)
        print("Hooke Jeeves, x0: {0}".format(f.start))
        m = hooke_jeeves(f, f.start, dx=dx, e=hook_jeeves_e)
        print("[+] Minimum u {0} u {1} evaluacija funkcije.".format(m, f.count))
        table_data.append([i, algos[str(alg)], f.count, m])
        f.reset()
        alg += 1

        print("Ciljna funkcija broj ", i)
        print("Pretrazivanje po osima, x0: {0}.".format(f.start))
        m = axis_search(f, f.start, e=axis_e)
        print("[+] Minimum u {0} u {1} evaluacija funkcije.".format(m, f.count))
        table_data.append([i, algos[str(alg)], f.count, m])
        f.reset()
        alg = 1

        i += 1

    tab = tt.Texttable()
    tab.header(['Funkcija', 'Algoritam', 'Br. evals', 'Minimum'])
    for row in table_data:
        tab.add_row(row)
    print(tab.draw())


#################################################################
# Zadatak 3
def task3():
    gf = GoalFunction(f4, start=[5.0, 5.0])
    print("Hooke Jeeves, x0: {0}".format(gf.start))
    m = hooke_jeeves(gf, gf.start, dx=dx, e=hook_jeeves_e)
    print("[+] Minimum u {0} u {1} evaluacija funkcije.".format(m, gf.count))
    gf.reset()

    print("\nNelder Mead simplex, x0: {0}".format(gf.start))
    m = nelder_mead(gf, gf.start, step=step, alpha=alpha, beta=beta, gamma=gamma, sigma=sigma, e=simplex_e)
    print("[+] Minimum u {0} u {1} evaluacija funkcije.".format(m, gf.count))
    gf.reset()


#################################################################
# Zadatak 4
def task4():
    tab = tt.Texttable()
    tab.header(['Pocetna', 'Br. evals', 'Minimum'])

    testrange4 = range(1, 21)
    for i in testrange4:
        gf = GoalFunction(f1, start=[0.5, 0.5])
        print("Nelder Mead simplex, x0: {0}".format(gf.start))
        m = nelder_mead(gf, gf.start, step=i, alpha=alpha, beta=beta, gamma=gamma, sigma=sigma, e=simplex_e)
        print("[+] Minimum u {0} u {1} evaluacija funkcije.".format(m, gf.count))
        tab.add_row([gf.start, gf.count, m])
        gf.reset()

    for i in testrange4:
        gf = GoalFunction(f1, start=[20, 20])
        print("Nelder Mead simplex, x0: {0}".format(gf.start))
        m = nelder_mead(gf, gf.start, step=i, alpha=alpha, beta=beta, gamma=gamma, sigma=sigma, e=simplex_e)
        print("[+] Minimum u {0} u {1} evaluacija funkcije.".format(m, gf.count))
        tab.add_row([gf.start, gf.count, m])
        gf.reset()

    print(tab.draw())
#################################################################
# Zadatak 5
def task5():
    counter = 0
    trials = 1000
    for i in range(trials):
        gf = GoalFunction(f6, start=[np.random.randint(low=-50, high=50)])
        print("Hooke Jeeves, x0: {0}".format(gf.start))
        m = hooke_jeeves(gf, gf.start, dx=dx, e=hook_jeeves_e)
        print("[+] Minimum u {0} u {1} evaluacija funkcije.".format(m, gf.count))
        if m < 1e-4:
            counter += 1
        gf.reset()
    print("{0}/{1} success rate".format(counter, trials))


import sys

N = sys.argv[1]
locals()["task" + str(N)]()
