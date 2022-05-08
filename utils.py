from itertools import product

_derivative_names = [[""]] + [s.split() for s in [
    "x y z",
    "xx xy xz yy yz",
    "xxy xxz yyz xyy xzz yzz xyz",
    "xxxy xxxz xxyy xxzz xyyy xzzz yyyz yyzz yzzz",
    "xxxyy xxxyz xxxzz xxyyy xxyyz xxyzz xxzzz xyyyz xyyzz yyyzz yyzzz",
    ]]
_derivatives_map = {} # sorted name: (derivative order, derivative index)
_name_map = {} # reverse derivatives_map
_expand_map = [] # derivative order: 3**order list of selected index
# or laplace pair
_select_map = [] # derivative order: 2*order+1 list of indices into
# 3**order expanded
_derive_map = {} # (derivative order, derivative index): ((lower
# derivative order, lower derivative index), axis to derive)

def name_to_idx(name):
    """Return a tuple of axis indices for given derivative
    
    Parameters
    ----------
    name : str
        A derivative name, e.g. `"xxz."`
    Returns
    -------
    idx : tuple of int
        Axis tuple, e.g. `(0, 0, 2)`.
    See also
    --------
    idx_to_name : Inverse
    """
    return tuple("xyz".index(n) for n in name)

def idx_to_name(idx):
    """Return sorted derivative name for axis tuple
    
    Parameters
    ----------
    idx : tuple of int
        An axis tuple.
    
    Returns
    -------
    name : str
        Derivative name.
    See also
    --------
    name_to_idx : Inverse
    """
    return "".join("xyz"[i] for i in sorted(idx))

def idx_to_nidx(idx):
    """Return index into flattened 3**order array for given order-tuple.
    
    Parameters
    ----------
    idx : tuple of int
        Axis tuple.
        
    Returns
    -------
    i : int
        Derivative order.
    j : int
        Index into flattened derivative tensor.
    """
    return sum(j*3**(len(idx)-i-1) for i, j in enumerate(idx))

def find_laplace(c):
    """Finds the two partial derivatives `a` and `b` such that the
    triple `a, b, c` is traceless, `a + b + c == 0`.
    Parameters
    ----------
        c : axis tuple
    Returns
    -------
    generator
        Generator of tuples `(a, b)` such that `a + b + c == 0` for any
        harmonic tensor of any order.
    """
    name = sorted(c)
    letters = list(range(3))
    found = None
    for i in letters:
        if name.count(i) >= 2:
            keep = name[:]
            keep.remove(i)
            keep.remove(i)
            take = letters[:]
            take.remove(i)
            a, b = (tuple(sorted(keep+[j,j])) for j in take)
            yield a, b

def _populate_maps():
    for deriv, names in enumerate(_derivative_names):
        #assert len(names) == 2*deriv+1, names
        for idx, name in enumerate(names):
            assert len(name) == deriv, name
            _derivatives_map[name] = (deriv, idx)
            _name_map[(deriv, idx)] = name
            if deriv > 0:
                for i, n in enumerate(_derivative_names[deriv-1]):
                    for j, m in enumerate("xyz"):
                        if name == "".join(sorted(n+m)):
                            _derive_map[(deriv, idx)] = (deriv-1, i), j
                            break
                assert (deriv, idx) in _derive_map, name
            for lap in find_laplace(name_to_idx(name)):
                a, b = map(idx_to_name, lap)
                assert (a not in names) or (b not in names), (name, a, b)
        idx = tuple(idx_to_nidx(name_to_idx(name)) for name in names)
        _select_map.append(idx)
        _expand_map.append([])
        for idx in product(range(3), repeat=deriv):
            name = idx_to_name(idx)
            if name in names:
                _expand_map[deriv].append(names.index(name))
            else:
                for a, b in find_laplace(idx):
                    a, b = map(idx_to_name, (a, b))
                    if a in names and b in names:
                        ia, ib = (names.index(i) for i in (a, b))
                        _expand_map[deriv].append((ia, ib))
        assert len(_expand_map[deriv]) == 3**deriv

_populate_maps()


def construct_derivative(deriv, idx):
    """Return lower deriv and axis to derive.
    When constructing a higher order derivative, take the value of the
    lower order derivative and evaluate its derivative along the axis
    returned by this function.
    
    Parameters
    ----------
    deriv : int
    idx : int
    
    Returns
    -------
    i : tuple (int, int)
        Lower derivative (derivative order, derivative index)
    j : int
        Axis to derive along
    """
    return _derive_map[(deriv, idx)]