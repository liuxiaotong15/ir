from ase.io import read
from ase.calculators.vasp import Vasp
from ase.vibrations import Infrared
from ase.io import read

a = read('qm9.db@1')
a.set_cell([10, 10, 10])
a.set_pbc([1, 1, 1])

calc = Vasp(prec='Accurate',
    ediff=1E-8,
    isym=0,
    idipol=4,       # calculate the total dipole moment
    dipol=a.get_center_of_mass(scaled=True),
    ldipol=True,
    kpts=[1, 1, 1])

a.calc = calc
ir = Infrared(a)
ir.run()
ir.summary()
