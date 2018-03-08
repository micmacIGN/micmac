#ifndef CREGTIRVIS_H
#define CREGTIRVIS_H

#include "StdAfx.h"
#include <fstream>
#include "Image.h"
#include "msd.h"
#include "Keypoint.h"
#include "lab_header.h"
#include "../../uti_image/Digeo/Digeo.h"
#include "DescriptorExtractor.h"
#include "Arbre.h"

#include "../../uti_image/Digeo/DigeoPoint.h"

#define distMax 0.75
#define rayMax 0.5



class cAppliRegTIRVIS
{
public:
 cAppliRegTIRVIS(int argc, char** argv);

 void computeHomogWithMSD();
 void EnrichKps(std::vector< KeyPoint > Kpsfrom, ArbreKD * Tree, cElHomographie &Homog, int NbIter, int ImPairKey);
 void initMSD();

 void mkDirPastis(std::string aSH, std::string aImName, std::string aDir="./");
 //std::vector< KeyPoint> drunkKps(std::vector< KeyPoint>  * Kps,CamStenope * mCalib);

private:
 cInterfChantierNameManipulateur * mICNM;
 std::vector< std::string > ThermalImages, VisualImages;
 std::string mDirTest, mPatImTest;
 bool mDebug;
 cElHomographie * mH;
 MsdDetector msd;
 std::size_t found;
 std::size_t found_Vis;
 std::size_t found_Tir;
 std::size_t found_commonPattern;
 string ext;
 FPRIMTp Pt_of_Point;
 Box2dr box;
 ElSTDNS set<pair<int,Pt2dr> > Voisins ; // where to put nearest neighbours

};


//===============================================================================//
/*                           OrientedImage class                                 */
//===============================================================================//
class Orient_Image
{
 public:
 Orient_Image
 (
  std::string aOriIn,
  std::string aName,
  cInterfChantierNameManipulateur * aICNM
 );
 std::string getName(){return mName;}
 CamStenope * getCam(){return mCam;}
 std::string getOrifileName(){return mOriFileName;}
 CamStenope   * mCam;

 protected:

 std::string  mName;
 std::string mOriFileName;
 };

Orient_Image::Orient_Image
 ( std::string aOriIn,
 std::string aName,
 cInterfChantierNameManipulateur * aICNM):
 mName(aName),mOriFileName(aOriIn+"Orientation-"+mName+".xml")
{
 mCam=CamOrientGenFromFile(mOriFileName,aICNM);
}


#endif // CREGTIRVIS_H

