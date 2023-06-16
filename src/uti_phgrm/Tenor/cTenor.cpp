#include "StdAfx.h"


const bool ReadPtsXYZ2(const std::string & ptsXYZ_adress, std::vector<Pt3dr> & ptsXYZ){
  ptsXYZ=std::vector<Pt3dr>();
  std::ifstream fin(ptsXYZ_adress.c_str(),std::ios::in);

  std::string line;
  while (std::getline(fin, line)){
    std::vector<double> values;
    std::string value;
    std::stringstream ss(line);
    while (ss>>value){
      values.push_back(std::stod(value));
    }
    ptsXYZ.push_back(Pt3dr(values[0],values[1],values[2]));

  }
  return true;
}


void Direct(ElCamera * aCam,Pt3dr aPG, Pt2dr & coordim_avant_corr_dist, Pt2dr & coordim_apres_corr_dist){
  //Compute image coordinates before distorsion correction
  coordim_avant_corr_dist=aCam->R3toC2(aPG);

  //Compute image coordinates after distorsion correction
  coordim_apres_corr_dist=aCam->R3toF2(aPG);
}



extern void TestCamCHC(ElCamera & aCam);

void ReprojPtsXYZInShot(std::string orixmlmicmac_adress, std::string ptsXYZ, std::string output){
  
  std::string aNameCam;
  std::string aNameDir;
  std::string aNameTag = "OrientationConique";
  Pt2dr TDINV;
  
  bool aModeGrid = false;
  std::string Out;
  
  SplitDirAndFile(aNameDir,aNameCam,orixmlmicmac_adress);
  
  cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aNameDir);
  
  ElCamera * aCam  = Gen_Cam_Gen_From_File(aModeGrid,orixmlmicmac_adress,aNameTag,anICNM);
  

  TestCamCHC(*aCam);
  
  std::vector<Pt3dr> XYZ;
  ReadPtsXYZ2(ptsXYZ,XYZ);

  Pt2di cameraSize=aCam->Sz();
  int marge=0;
  int cmin=marge;
  int lmin=marge;
  int cmax=cameraSize.x-marge;
  int lmax=cameraSize.y-marge;

  std::ofstream fout(output.c_str(),std::ios::out);
  Pt2dr coordim_beforecorrdist, coordim_aftercorrdist;
  for(size_t n=0;n<XYZ.size();n++){

    Direct(aCam,XYZ[n],coordim_beforecorrdist,coordim_aftercorrdist);

    if(coordim_beforecorrdist.x<cmin){continue;}
    if(coordim_beforecorrdist.y<lmin){continue;}
    if(coordim_beforecorrdist.x>cmax){continue;}
    if(coordim_beforecorrdist.y>lmax){continue;}
 
    if(coordim_aftercorrdist.x<cmin){continue;}
    if(coordim_aftercorrdist.y<lmin){continue;}
    if(coordim_aftercorrdist.x>cmax){continue;}
    if(coordim_aftercorrdist.y>lmax){continue;}

    fout<<n<<" "<<std::fixed<<std::setprecision(5)<<coordim_aftercorrdist.x<<" "<<coordim_aftercorrdist.y<<std::endl;

  }

}


int Tenor_main(int argc,char ** argv)
{

    std::string orixmlmicmac_adress;
    std::string ptsXYZ;
    std::string output;
    
    ElInitArgMain
        (
        argc,argv,
        LArgMain()  << EAMC(orixmlmicmac_adress,"Orixmlmicmac adress", eSAM_IsPatFile)
        << EAMC(ptsXYZ, "File pts XYZ ", eSAM_IsPatFile)
        << EAMC(output, "Adresse export pts im", eSAM_IsPatFile),
        LArgMain()
        );
  
  
    ReprojPtsXYZInShot(orixmlmicmac_adress, ptsXYZ, output);

    return EXIT_SUCCESS;
}

