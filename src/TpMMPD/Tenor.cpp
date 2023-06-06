#include "StdAfx.h"
#include "Stringpp.hpp" //#include "/media/Data2/dev/lib/outilsstring/Stringpp.hpp"
#include "t3d.hpp" //#include "/media/Data2/dev/lib/t3d.hpp"

void HelloWorld(){ std::cout << "HELLO WORD \n";}


const bool LecturePtsXYZ(const std::string & adresse_ptsXYZ, std::vector<Pt3dr> & ptsXYZ){
  ptsXYZ=std::vector<Pt3dr>();
  std::ifstream fin(adresse_ptsXYZ.c_str(),std::ios::in);
  if(fin.fail()){return false;}
  while(!fin.eof()){
    stringpp ltmp;
    getline(fin,ltmp);
    if(ltmp.empty()){continue;}
    std::istringstream iss(ltmp.c_str());
    stringpp stmpX,stmpY,stmpZ;
    iss>>stmpX>>stmpY>>stmpZ;
    if(stmpX.empty()){continue;}
    if(stmpY.empty()){continue;}    
    if(stmpZ.empty()){continue;}
    ptsXYZ.push_back(Pt3dr(stmpX.atof(),stmpY.atof(),stmpZ.atof()));
  }
  return true;
}

void Direct(ElCamera * aCam,Pt3dr aPG, Pt2dr & coordim){
  coordim=aCam->R3toF2(aPG);
}

void Direct(ElCamera * aCam,Pt3dr aPG, Pt2dr & coordim_avant_corr_dist, Pt2dr & coordim_apres_corr_dist){
  //Calcule les coordonnees images avant correction de la distorsion
  coordim_avant_corr_dist=aCam->R3toC2(aPG);

  //Calcule les coordonnees images apres correction de toutes les distorsions
  coordim_apres_corr_dist=aCam->R3toF2(aPG);
}

void TestDirectbis(ElCamera * aCam,Pt3dr aPG)
{
    {
         std::cout.precision(10);

         std::cout << " ---PGround  = " << aPG << "\n";
         Pt3dr aPC = aCam->R3toL3(aPG);
         std::cout << " -0-CamCoord = " << aPC << "\n";
         Pt2dr aIm1 = aCam->R3toC2(aPG);

         std::cout << " -1-ImSsDist = " << aIm1 << "\n";
         Pt2dr aIm2 = aCam->DComplC2M(aCam->R3toF2(aPG));

         std::cout << " -2-ImDist 1 = " << aIm2 << "\n";

         Pt2dr aIm3 = aCam->OrGlbImaC2M(aCam->R3toF2(aPG));

         std::cout << " -3-ImDist N = " << aIm3 << "\n";

         Pt2dr aIm4 = aCam->R3toF2(aPG);
         std::cout << " -4-ImFinale = " << aIm4 << "\n";
    }
}

extern void TestCamCHC(ElCamera & aCam);

void ReprojPtsXYZDansCliche(int argc, char** argv){
  //Lire le fichier des points 3D 
  //Lire l'orientation des images
  
  std::string adresse_orixmlmicmac;
  std::string aNameCam;
  std::string aNameDir;
  std::string aNameTag = "OrientationConique";
  Pt2dr TDINV;
  

  std::string adresse_fichier_ptsXYZ, adresse_export_fichier_ptsim;
  bool aModeGrid = false;
  std::string Out;
  
  ElInitArgMain
    (
     argc,argv,
     LArgMain()  << EAMC(adresse_orixmlmicmac,"Adresse orixmlmicmac cliche", eSAM_IsPatFile)
     << EAMC(adresse_fichier_ptsXYZ, "Adresse pts XYZ ", eSAM_IsPatFile)
     << EAMC(adresse_export_fichier_ptsim, "Adresse export pts im", eSAM_IsPatFile),
    LArgMain()
     << EAM(aNameTag,"Tag",true,"inutile")
     << EAM(aModeGrid,"Grid",true,"inutile", eSAM_IsBool)
     << EAM(Out,"Out",true,"inutile",eSAM_NoInit)
     );
  
  
  SplitDirAndFile(aNameDir,aNameCam,adresse_orixmlmicmac);
  
  cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aNameDir);
  
  ElCamera * aCam  = Gen_Cam_Gen_From_File(aModeGrid,adresse_orixmlmicmac,aNameTag,anICNM);
  

  TestCamCHC(*aCam);
  
  std::vector<Pt3dr> XYZ;
  LecturePtsXYZ(adresse_fichier_ptsXYZ,XYZ);

  //Connaitre la taille de la camera...
  Pt2di taillecamera=aCam->Sz();
  int marge=0;
  int cmin=marge;
  int lmin=marge;
  int cmax=taillecamera.x-marge;
  int lmax=taillecamera.y-marge;

  //Reste a exporter en ne conservant que les points qui sont dans l'image
  std::ofstream fout(adresse_export_fichier_ptsim.c_str(),std::ios::out);
  Pt2dr coordim_avantcorrdist, coordim_aprescorrdist;
  for(size_t n=0;n<XYZ.size();n++){

    Direct(aCam,XYZ[n],coordim_avantcorrdist,coordim_aprescorrdist);

    //il est apparu que dans certains cas la correction de la distorsion faisait revenir dans l'image un point qui n'y etait pas. Il faut donc faire un premier test sur les coordonnees avant la correction de la distorsion
    if(coordim_avantcorrdist.x<cmin){continue;}
    if(coordim_avantcorrdist.y<lmin){continue;}
    if(coordim_avantcorrdist.x>cmax){continue;}
    if(coordim_avantcorrdist.y>lmax){continue;}
 
    if(coordim_aprescorrdist.x<cmin){continue;}
    if(coordim_aprescorrdist.y<lmin){continue;}
    if(coordim_aprescorrdist.x>cmax){continue;}
    if(coordim_aprescorrdist.y>lmax){continue;}

    fout<<n<<" "<<std::fixed<<std::setprecision(5)<<coordim_aprescorrdist.x<<" "<<coordim_aprescorrdist.y<<std::endl;

  }

}


#include "TpPPMD.h"
#include "PseudoXML.h"
 //Conversion d'un MNS MICMAC Z_NUM a un MNS avec des valeurs en absolu en m...
void MNSMICMAC(int argc, char** argv){
  if(argc!=5 && argc!=6){
    std::cout<<"Passage d'un mns Z_Num?? a un MNS en absolu"<<std::endl;
    std::cout<<"./exe MNSMICMAC adresse_image_z_num.tif adresse_export_mns.tif gain offset"<<std::endl;
    std::cout<<"./exe MNSMICMAC adresse_image_z_num.tif adresse_export_mns.tif adresse_param_mns.xml"<<std::endl;
    return;
  }
  std::string adresse_image_z_num(argv[2]);
  std::string adresse_export_mns(argv[3]);
  double a,b;
  a=1.0; b=0.0;
  if(argc==6){
    a=atof(argv[4]);
    b=atof(argv[5]);
  }else if(argc==5){
    std::string adresse_xml(argv[4]);
    std::vector<stringpp> balises, contenus;
    PseudoXML::LectureXML(adresse_xml,balises,contenus);
    for(size_t n=0;n<balises.size();n++){
      if(balises[n]=="RESOLUTIONALTI"){a=contenus[n].atof();}
      else if(balises[n]=="ORIGINEALTI"){b=contenus[n].atof();}
    }
  }
  std::cout<<a<<" "<<b<<std::endl;
  
  cTD_Im im=cTD_Im::FromString(std::string(argv[2]));
  
  Pt2di taille_image=im.Sz();
  
  for(int i=0;i<taille_image.x;i++){
    for(int j=0;j<taille_image.y;j++){
      im.SetVal(i,j,im.GetVal(i,j)*a+b);
    }
  }

  im.Save(std::string(argv[3]));
  
}


void TfwFinaux(int argc, char** argv){
  if(argc!=3 && argc!=4){
    std::cout<<"Calcule les tfw pour les fichiers _Tile_x_x.tif"<<std::endl;
    std::cout<<"./exe TfwFinaux radical"<<std::endl;
    std::cout<<"./exe TfwFinaux radical adresse_tfw_ini"<<std::endl;
    return;
  }
  
  
  //Voici comment recuperer la taille d'une image sans la charger
  // Tiff_Im aTF = Tiff_Im::StdConvGen(aName,-1,true);
  // Pt2di aSzIm = aTF.sz();
  // cTD_Im aRes(aSzIm.x,aSzIm.y);
  
  
  std::string adresse_tfw_ini;
  std::string adresse_radical(argv[2]);
  if(argc==4){adresse_tfw_ini=std::string(argv[3]);}else{adresse_tfw_ini=adresse_radical+".tfw";}
  
  
  //Lecture du tfw correspondant a l'image globale
  double E0,N0,dE,dN,dtmp;
  std::ifstream ftfw(adresse_tfw_ini.c_str(),std::ios::in);
  if(ftfw.fail()){std::cerr<<"Lecture impossible de '"<<adresse_tfw_ini<<"'"<<std::endl; return;}
  ftfw>>dE>>dtmp>>dtmp>>dN>>E0>>N0;

  //dN doit (logiquement) etre negatif
  
  
  int i,j;
  double I,J;
  i=0;
  I=0;
  while(true){
    j=0;
    J=0;
    double DI=0;
    while(true){
      stringpp radicaltile(adresse_radical+stringpp("_Tile_")+stringpp(i)+stringpp("_")+stringpp(j));
      stringpp adressetif_tile(radicaltile+".tif");
      if(std::ifstream(adressetif_tile.c_str(),std::ios::in | std::ios::binary).fail()){break;}
      std::cout<<adressetif_tile<<std::endl;
      Tiff_Im aTF = Tiff_Im::StdConvGen(adressetif_tile,-1,true);
      Pt2di aSzIm = aTF.sz();
      DI=aSzIm.x;
      //On va ecrire le tfw correspondant
      stringpp adressetfw_tile(radicaltile+".tfw");
      std::ofstream ftfwtile(adressetfw_tile.c_str(),std::ios::out);
      double E,N;
      E=E0+I*dE;
      N=N0+J*dN;
      ftfwtile<<std::fixed<<std::setprecision(5)<<dE<<std::endl;
      ftfwtile<<"0"<<std::endl;
      ftfwtile<<"0"<<std::endl;
      ftfwtile<<std::fixed<<std::setprecision(5)<<dN<<std::endl;
      ftfwtile<<std::fixed<<std::setprecision(5)<<E<<std::endl;
      ftfwtile<<std::fixed<<std::setprecision(5)<<N<<std::endl;
      ftfwtile.close();
      j++;
      J=J+aSzIm.y;
    }
    if(j==0){break;}
    I=I+DI;
    i++;
  }
  
}

void TestPTL(int argc, char ** argv){//essais de lecture ecriture de points de liaison...
  std::string adresse_result(argv[2]);
  std::cout<<adresse_result<<std::endl;

}



const bool Result_binmicmac2txt(const std::string & adresse_micmac_dat, const std::string & adresse_export_txt_result, const int prec=5){
  //VERSION QUI MARCHE
  ELISE_fp aFile(adresse_micmac_dat.c_str(),ELISE_fp::READ);
  int aDim = aFile.read((int*)0);
  if(aDim!=2){std::cerr<<"Je ne sais pas gerer ce cas de figure..."<<std::endl; return false;}
  int nbpts=aFile.read_INT4();
  std::cout<<nbpts<<std::endl;
  std::list<cNupletPtsHomologues> cnu;
  while (nbpts>0){
    cnu.push_back(aFile.read((std::list<cNupletPtsHomologues>::value_type *)0));
    nbpts--;
  }

  std::ofstream fresult(adresse_export_txt_result.c_str(),std::ios::out);
  for(std::list<cNupletPtsHomologues>::const_iterator it=cnu.begin();it!=cnu.end();it++){

    if((*it).NbPts()!=2){std::cerr<<"Je ne sais pas gerer ce cas de figure..."<<std::endl; return false;}
    fresult<<std::fixed<<std::setprecision(prec)<<(*it).PK(0).x<<"\t"<<(*it).PK(0).y<<"\t"<<(*it).PK(1).x<<"\t"<<(*it).PK(1).y<<std::endl;
  }

  return true;
}

const bool Result_txt2binmicmac(const std::string & adresse_txt_result, const std::string & adresse_export_micmac_dat, const int prec=5){
  std::ifstream fresult(adresse_txt_result.c_str(),std::ios::in);
  if(!fresult.good())return false;

  std::list<cNupletPtsHomologues> cnu;
  while(true){
    stringpp sx1,sy1,sx2,sy2;
    fresult>>sx1;
    if(fresult.eof()){break;}//gestion du cas nouvelle ligne a la fin du fichier 
    if(sx1.empty()){continue;}
    fresult>>sy1>>sx2>>sy2;
    if(sy1.empty()){continue;}
    if(sx2.empty()){continue;}
    if(sy2.empty()){continue;}
    cNupletPtsHomologues cnphtmp(2,1);
    cnphtmp.Pds()=1;
    cnphtmp.P1().x=sx1.atof(); cnphtmp.P1().y=sy1.atof();
    cnphtmp.P2().x=sx2.atof(); cnphtmp.P2().y=sy2.atof();
    cnu.push_back(cnphtmp);
    if(fresult.eof()){break;}
  }


  //Et maintenant on va tenter de reecrire...
  int aDim=2;
  ELISE_fp aFileout(adresse_export_micmac_dat.c_str(),ELISE_fp::WRITE);
  aFileout.write_INT4(aDim);
  aFileout.write_INT4((int)cnu.size());
  for(std::list<cNupletPtsHomologues>::const_iterator it=cnu.begin();it!=cnu.end();it++){aFileout.write(*it);}
 

  return true;
}

void Result_binmicmac2txt(int argc, char** argv){
  if(argc!=4){
    std::cout<<"Conversion d'un fichier binaire micmac .dat en un fichier texte result "<<std::endl;
    std::cout<<"./exe Homol:resulttxt2micmac adresse_fichier_micmac.dat adresse_export_fichier_txt.result"<<std::endl;
    return;
  }
  Result_binmicmac2txt(std::string(argv[2]),std::string(argv[3]));
}

void Result_txt2binmicmac(int argc, char** argv){
  if(argc!=4){
    std::cout<<"Conversion d'un fichier texte result en un fichier binaire micmac .dat "<<std::endl;
    std::cout<<"./exe Homol:micmac2resulttxt adresse_export_fichier_txt.result adresse_fichier_micmac.dat"<<std::endl;
    return;
  }
  Result_txt2binmicmac(std::string(argv[2]),std::string(argv[3]));
}


void CamDesactivAltiSol(int argc, char** argv){   
  if(argc!=4){
    std::cout<<"Efface les champs AltiSol et Profondeur des fichiers d'orientation. Penser a refaire un Campari ensuite."<<std::endl;
    std::cout<<"./exe PbAltiSol adresse_orixmlmicmac_a_corriger adresse_export_orixmlmicmac_corrige"<<std::endl;
    return;
  }

  //Lire le fichier des points 3D 
  //std::vector<T3D<double> > ptsXYZ;
  //  if(!LecturePtsXYZ()
  //Lire l'orientation des images
  
  std::string adresse_orixmlmicmac(argv[2]);
  std::string adresse_orixlmicmcac_out(argv[3]);

  std::vector<stringpp> balises, contenus;
  if(!PseudoXML::LectureXML(adresse_orixmlmicmac,balises,contenus,false)){std::cout<<"Pb en lecture de '"<<adresse_orixmlmicmac<<"'"<<std::endl; return;}
  
  std::ofstream fout(adresse_orixlmicmcac_out.c_str(),std::ios::out);
  bool partieaeffacer=false;
  for(size_t n=0;n<balises.size();n++){
    if(balises[n].UpCase()==std::string("ALTISOL")){partieaeffacer=true;}
    if(balises[n].UpCase()==std::string("PROFONDEUR")){partieaeffacer=true;}
    if(!partieaeffacer){
      fout<<"<"<<balises[n]<<">";
      fout<<contenus[n];
    }
    if(balises[n].UpCase()==std::string("/ALTISOL")){partieaeffacer=false;}
    if(balises[n].UpCase()==std::string("/PROFONDEUR")){partieaeffacer=false;}
  }
  fout.close();
  return;
  
  
}


int Tenor_main(int argc,char ** argv)
{
  if(argc==1){std::cout<<"Actions possibles : MNSMICMAC / TfwFinaux / Homol:micmac2resulttxt / Homol:resulttxt2micmac / PbAltiSol / [rien] "<<std::endl;}
  if(std::string(argv[1])=="MNSMICMAC"){MNSMICMAC(argc,argv);return 0;}
  if(std::string(argv[1])=="TfwFinaux"){TfwFinaux(argc,argv);return 0;}
  if(std::string(argv[1])=="Homol:micmac2resulttxt"){Result_binmicmac2txt(argc,argv); return 0;}
  if(std::string(argv[1])=="Homol:resulttxt2micmac"){Result_txt2binmicmac(argc,argv); return 0;}
  if(std::string(argv[1])=="PbAltiSol"){CamDesactivAltiSol(argc,argv); return 0;}
  if(argc==2){std::cout<<"./exe adresse_orixmlmicmac_cliche adresse_pts_xyz adresse_export_pts_im"<<std::endl;}
  
  ReprojPtsXYZDansCliche(argc,argv);
    //TestCam_main_bis(argc,argv);
    //ReprojPtsXYZDansCliche(argc,argv);

    return EXIT_SUCCESS;
}

