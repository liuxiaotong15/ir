from ase.db import connect

db = connect('qm9_ir_spectrum.db')
row = db.get(1)
print(row.toatoms())

print(row.data.ir_spectrum[0])
print(row.data.ir_spectrum[1])

for i,j  in zip(row.data.ir_spectrum[0], row.data.ir_spectrum[1]):
    if j > 0.1:
        print(i, j)