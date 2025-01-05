import os

from PyTightRMSD.rmsd import TightRMSD



files = "testcases/"
try:
    f = open(os.path.join(files, "Alanin.xyz"))
    f.close()
except:
    files = "tests/testcases/"



def test_tight_rmsd():

    rmsd_threshold = 0.000001

    rmsdobj = TightRMSD()

    water1 = os.path.join(files, "H2O.xyz")
    water2 = os.path.join(files, "H2O_different_order_rotated.xyz")
    ala1 = os.path.join(files, "Alanine.xyz")
    ala2 = os.path.join(files, "Alanine_different_order.xyz")
    ala3 = os.path.join(files, "L-Alanine.xyz")
    ala4 = os.path.join(files, "D-Alanine.xyz")
    pep1 = os.path.join(files, "Tyr-Ala-Trp.xyz")
    pep2 = os.path.join(files, "Tyr-Ala-Trp_different_order.xyz")
    benzene1 = os.path.join(files, "benzene.xyz")
    benzene2 = os.path.join(files, "benzene_rotated.xyz")
    benzene3 = os.path.join(files, "benzene_rotated_different_order.xyz")
    ammonia1 = os.path.join(files, "ammonia.xyz")
    ammonia2 = os.path.join(files, "ammonia_different_order.xyz")

    water_test = rmsdobj.calc(water1, water2) <= rmsd_threshold
    ala_test1 = rmsdobj.calc(ala1, ala2) <= rmsd_threshold
    ala_test2 = rmsdobj.calc(ala3, ala4) <= rmsd_threshold
    ala_test3 = rmsdobj.calc(ala1, ala3) <= rmsd_threshold
    pep_test = rmsdobj.calc(pep1, pep2) <= rmsd_threshold
    benzene_test1 = rmsdobj.calc(benzene1, benzene2) <= rmsd_threshold
    benzene_test2 = rmsdobj.calc(benzene1, benzene3) <= rmsd_threshold
    ammonia_test = rmsdobj.calc(ammonia1, ammonia2) <= rmsd_threshold

    assert (
        water_test and 
        ala_test1 and
        not ala_test2 and
        not ala_test3 and
        pep_test and
        benzene_test1 and
        benzene_test2 and
        ammonia_test
    )
