from MMVII import *

dirData = '../../MMVII-TestDir/Input/Saisies-MMV1/'
cam = SensorCamPC.fromFile(dirData + 'Ori-Ground-MMVII/Ori-PerspCentral-IMGP4168.JPG.xml')

pose = cam.pose
icalib = cam.internalCalib
proj = icalib.dir_Proj()
dist = icalib.dir_Dist()
sten = icalib.calibStenPerfect() 

p=(0.754045221758523621, 0.496427193197002514, 0.431416432371162006)
print("Point terrain: ",p)

pi=sten.value(dist.value(proj.value(pose.inverse(p))))
print("Point image 2D: ",cam.ground2Image(p), "=", pi)

pid=Pt3dr(pi.x,pi.y,pose.inverse(p).z)
print("Point image 3D: ",cam.ground2ImageAndDepth(p), "=", pid)

inv_sten=sten.mapInverse()
inv_proj=icalib.inv_Proj()
inv_dist=DataInvertOfMapping2D(icalib.dir_DistInvertible())
pCam = inv_proj.value(inv_dist.value(inv_sten.value((pid.x, pid.y))))
pg = pose.value(pCam * (pid.z / pCam.z))

pCam2 = inv_proj.value(icalib.dir_DistInvertible().inverse(sten.inverse((pid.x, pid.y))))
pg2 = pose.value(pCam2 * (pid.z / pCam2.z))

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
pi_calcs = sten.values(dist.values(proj.values(camPGs)))
pid_calcs = [ (p1.x,p1.y,p2.z) for p1,p2 in zip(pi_calcs,camPGs)]

print (pids, "=>" )
print (pid_calcs)
