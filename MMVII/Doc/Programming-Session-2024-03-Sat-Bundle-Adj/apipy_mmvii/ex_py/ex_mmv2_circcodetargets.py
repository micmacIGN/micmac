import MMVII
from pathlib import Path


pt2dSet = MMVII.SetMesPtOf1Im.fromFile('/home/JMMuller/tmp/Circ-Code-Target/MMVII-PhgrProj/PointsMeasure/Filt/MesIm-043_0005_Scaled.tif.xml')

print(pt2dSet)

for mes in pt2dSet.measures():
    print(mes.namePt, mes.pt)


pcIntrCalib = MMVII.PerspCamIntrCalib.fromFile('/home/JMMuller/tmp/Circ-Code-Target/MMVII-PhgrProj/Ori/BA_rig/CalibIntr_CamNIKON_D5600_Add043_Foc24000.xml')


pp2i = pcIntrCalib.mapPProj2Im()
i2pp = pp2i.mapInverse()

print(pcIntrCalib.pp, ' -> ', i2pp.value(pcIntrCalib.pp))

dist = pcIntrCalib.dir_Dist()
inv_dist = MMVII.DataInvertOfMapping2D(pcIntrCalib.dir_DistInvertible())

for mes in pt2dSet.measures():
    pt = mes.pt
    print('Pt ', mes.namePt, ' im: ', pt)
    print(' -> central with disto: ', i2pp.value(pt))
    print(' -> central no disto: ', inv_dist.value(i2pp.value(pt)))
    print(' -> lig/col no disto: ', pp2i.value(inv_dist.value(i2pp.value(pt))))



for mes in pt2dSet.measures():
    mes.pt = pp2i.value(inv_dist.value(i2pp.value(mes.pt)))

pt2dSet.toFile('out_no_dist.xml')

