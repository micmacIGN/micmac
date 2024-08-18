import MMVII

# Prepare data: see MMVII/MMVII-TestDir/Input/Saisies-MMV1/readme.md

dirData = '../../MMVII-TestDir/Input/Saisies-MMV1/'
imPath = dirData + 'MMVII-PhgrProj/Ori/toto/Ori-PerspCentral-IMGP4168.JPG.xml'

cam = MMVII.SensorCamPC.fromFile(imPath)

pose = cam.pose
icalib = cam.internalCalib
proj = icalib.dir_Proj()
dist = icalib.dir_Dist()
pp2i = icalib.mapPProj2Im()

p=(0.754045221758523621, 0.496427193197002514, 0.431416432371162006)
print("Point terrain: ",p)

pi=pp2i.value(dist.value(proj.value(pose.inverse(p))))
print("Point image 2D: ",cam.ground2Image(p), "=", pi)

pid=(pi[0],pi[1],pose.inverse(p)[2])
print("Point image 3D: ",cam.ground2ImageAndDepth(p), "=", pid)

i2pp=pp2i.mapInverse()
inv_proj=icalib.inv_Proj()
inv_dist=MMVII.DataInvertOfMapping2D(icalib.dir_DistInvertible())
pCam = inv_proj.value(inv_dist.value(i2pp.value((pid[0], pid[1]))))
pg = pose.value(pCam * (pid[2] / pCam[2]))

pCam2 = inv_proj.value(icalib.dir_DistInvertible().inverse(pp2i.inverse((pid[0], pid[1]))))
pg2 = pose.value(pCam2 * (pid[2] / pCam2[2]))

print ("Point terrain 1: ",pg, "=", p)
print ("Point terrain 2: ",pg2, "= ", p)


# Liste de points
pid1=(0,0,0.5)
pid2=(10,-10,10)
pid3=(100,100,15)
pids=[pid1,pid2,pid3]

# points image => points terrain
pgs = list(map(cam.imageAndDepth2Ground,pids))

# points terrains  => retour en points images
camPGs = list(map(cam.pose.inverse, pgs))
pi_calcs = pp2i.values(dist.values(proj.values(camPGs)))
pid_calcs = [ (p1[0],p1[1],p2[2]) for p1,p2 in zip(pi_calcs,camPGs)]

print (pids, "=>" )
print (pid_calcs)

