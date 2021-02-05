from ase.io import read
from ase.calculators.vasp import Vasp
from ase.vibrations import Infrared
from ase.io import read
from ase.db import connect
import os


for i in range(133885):
    try:
        print('-' * 100, i)
        os.system('ls | grep -v water.py | grep -v qm9*.db | xargs rm')
        a = read('qm9.db@' + str(i))
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
        with connect('qm9_ir_spectrum.db') as db:
            db.write(a, data={'ir_spectrum':ir.get_spectrum()[0]})
            # print(ir.get_spectrum()[0].shape)
    except:
        pass
