from ylearn.autodiff import Sum, Mul, Var, Const

x = Var('x', 3)
y = Var('y', 2)

z = Sum(
    Sum(
        Mul(x, x),
        Mul(Const(3), Mul(x, y))
    ),
    Const(1)
)

print(f'z = {z} \n  = {z.compute()}')
print(f"z' = {z.backward(x)} \n   = {z.backward(x).compute()}")
print(f"z'' = {z.backward(x).backward(y)} \n    = {z.backward(x).backward(y).compute()}")

k = x - y
print(f'k = {k} \n  = {k.compute()}')
print(f"k'x = {k.backward(x)} \n    = {k.backward(x).compute()}")
print(f"k'y = {k.backward(y)} \n    = {k.backward(y).compute()}")

d = x / y

print(f'd = {d} \n  = {d.compute()}')
print(f"d'x = {d.backward(x)} \n    = {d.backward(x).compute()}")

print(f"d'y = {d.backward(y)} \n    = {d.backward(y).compute()}")
print("Forward")

print(f"d'x = ", d.forward())