/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
// #include "anag_all.h"

/*
void f()
{
    FILE * aFP = ElFopen(MMC,"w");
    ElFclose(aFP);
}

*/

#include "StdAfx.h"

#if (ELISE_X11)



using namespace NS_ParamChantierPhotogram;

#if (0)

void TestFiltreMNT(int argc,char** argv)
{
   std::string aNF = "/home/mpierrot/Data/Gargouilles/DB-LF-GeoI/Z_Num6_DeZoom1_Geom-0503.tif";
   //std::string aNM = "/home/mpierrot/Data/Gargouilles/DB-LF-GeoI/Z_Num6_DeZoom1_Geom-0503.tif";

   Tiff_Im aTF = Tiff_Im::StdConv(aNF);
   Pt2di aP0;
   Pt2di aSz = aTF.sz();


   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAM(aP0),
        LArgMain()  << EAM(aSz,"SZ",true)
   );

   Im2D_REAL4 aIm(aSz.x,aSz.y);
   TIm2D<REAL4,REAL8> aTIm(aIm);
   ELISE_COPY(aIm.all_pts(),trans(aTF.in(),aP0),aIm.out());



/*
   Im2D_REAL4  aDif(aSz.x,aSz.y);
   ELISE_COPY(aDif.all_pts(),f-EnvKLipshcitz_32(f,100),aDif.out());
*/


   Fonc_Num f = aIm.in_proj();// +1000;
   Tiff_Im::Create8BFromFonc
   (
      "toto.tif",
      aSz,
      Max(0,Min(255,128 + f-EnvKLipshcitz_32(f,100)))
   );


/*
   Im2D_U_INT1 aRes(aSz.x,aSz.y);
   TIm2D<U_INT1,INT> aTRes(aRes);
   int aSzF = 5;
   double aStepMax = 1.0;
   

   for (int anX = aSzF; anX <aSz.x-aSzF ; anX++)
   {
std::cout <<  aSz.x-anX << "\n";
      for (int anY = aSzF; anY <aSz.y-aSzF ; anY++)
      {
          double aNb = 0;
          double aSPlus = 0;
          double aSMoins = 0;
          Pt2dr aP(anX,anY);

          for (int aDx = -aSzF ; aDx <= aSzF ; aDx++)
          {
              for (int aDy = -aSzF ; aDy <= aSzF ; aDy++)
              {
                  Pt2dr aDP(aDx,aDy);
                  double aDist = euclid (aDP);

                
                  if ((aDist<(aSzF+1e-4)) && (aDist >1e-2))
                  {
                     aNb+= 0.5;
                     double anEc = aTIm.get(aP)-aTIm.get(aP+aDP);
                     double aStep = ElAbs(anEc/aDist);
                     double aPds = 1/(1+ElSquare(aStep/aStepMax));
                     if (anEc> 0) 
                        aSPlus += aPds;
                     else if (anEc < 0) 
                        aSMoins += aPds;
                     else
                     {
                        aSPlus += aPds/2;
                        aSMoins += aPds/2;
                     }
                  }
              }
          }
          double aVal = ElMax(aSPlus,aSMoins)/aNb;
                  
// std::cout << anX << " " << anY << " " << aSPlus << " " << aSMoins  << " " << aVal << "\n";
                  aTRes.oset
                  (
                      aP,
                      ElMax(0,ElMin(128,round_ni(aVal*128)))
                  );
      }
   }
   Tiff_Im::CreateFromIm(aRes,"toto.tif");
*/
}

void TestDir(int argc,char** argv)
{

   std::string aName,aTruc;

   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAM(aName),
        LArgMain()  << EAM(aTruc,"Truc",true)
   );
   ELISE_fp::MkDirRec(aName);
}


void TestMTD(int argc,char** argv)
{

   std::string aName,aTruc;

   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAM(aName),
        LArgMain()  << EAM(aTruc,"Truc",true)
   );
   cMetaDataPhoto aMTD =  cMetaDataPhoto::CreateExiv2(aName);

   std::cout << "Foc = " << aMTD.FocMm()
             << " Exp= " << aMTD.ExpTime()
             << " Diaph= " << aMTD.Diaph()
             << " Iso= " << aMTD.IsoSpeed()
             << "\n";
}

void TestNew()
{
     int aNb = 1000000000;
     std::vector<char *> aVC;
     for (int aK=0 ; aK< 1000000 ; aK++)
     {
        char * aTb = new char [aNb];
        aVC.push_back(aTb);
        if (1)
        {
            for (int i=0 ; i<aNb ; i++)
                aTb[i] = 0;
        }
        // STD_NEW_TAB_USER(1000000000,char);
        std::cout << "K=" << aK << "\n";
     }
}


void TestTime()
{
   int aNB=100000000;

   ElTimer aChrono;
   int aResI=0;

   for  (int aK=1 ; aK<aNB ; aK++)
   {
      aResI = aK /1023;
   }
   std::cout << "Time  div I   " << aChrono.uval() << " " << aResI << "\n";
   aChrono.reinit();

   for  (int aK=1 ; aK<aNB ; aK++)
   {
      aResI = aK << 10 ; 
   }
   std::cout << "Time  shif I I   " <<  aChrono.uval() << " " << aResI << "\n";
   aChrono.reinit();

   double aResR;
   for  (int aK=1 ; aK<aNB ; aK++)
   {
      aResR = aK /1023.0;
   }
   std::cout << "Time  div R   " << aChrono.uval() << " " << aResR <<  "\n";
   aChrono.reinit();
}

void TestFishEye2()
{
  BugFE=true;
  CamStenope * aCS = Std_Cal_From_File("../TMP/AutoCalDRad-Canon-015.xml");
   Pt2dr aP2(8.67583000000e+02,3.62618000000e+03);
   Pt3dr aQ0,aQ1;
   aCS->F2toRayonL3(aP2,aQ0,aQ1) ;
}


void TestEquiSolid()
{
    for (int anX = 0 ; anX < 10 ; anX++)
    {
       double aV = (anX+0.5) / 50.0;
       double aEps = 1e-4;

       double aV0 = aV - aEps;
       double aV1 = aV + aEps;
       double aDifF = (f2SAtRxS2SRx(aV1)-f2SAtRxS2SRx(aV0)) / (2*aEps);

       std::cout << aV << " " << ElAbs(Dl_f2SAtRxS2SRx(aV)-Std_f2SAtRxS2SRx(aV)) << "\n";

       std::cout  << aDifF << " " << Der2SAtRxS2SRx(aV) << " " << Dl_Der2SAtRxS2SRx(aV) << "\n";

       aDifF = (f4S2AtRxS2(aV1)-f4S2AtRxS2(aV0)) / (2*aEps);

       std::cout  << aDifF << " " << Der4S2AtRxS2(aV) << " " << Dl_Der4S2AtRxS2(aV) << "\n";

       std::cout << Std_Tg2AsRxS2SRx(aV) << " " << Dl_Tg2AsRxS2SRx(aV) << "\n";

       std::cout<< "\n";
    }
}

void TestCamOrtho()
{
   cCameraOrtho * aCam = cCameraOrtho::Alloc(Pt2di(3000,4000));
   aCam->SetOrientation
   (
        ElRotation3D
        (
           Pt3dr(1,2,3),
           0.01,0.01,0.01
        )
   );
   cOrientationConique  anOC = aCam->StdExportCalibGlob();
   MakeFileXML(anOC,"toto.xml","MicMacForAPERO");

   cTplValGesInit<std::string> aNoName;
   cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::StdAlloc(0,0,"./",aNoName);
   ElCamera * aC2 = Cam_Gen_From_File("toto.xml","OrientationConique",anICNM);
   anOC = aC2->StdExportCalibGlob();
   MakeFileXML(anOC,"tata.xml","MicMacForAPERO");
}

void TestMatCstr()
{
   Im2D_REAL8 aIA = Im2D_REAL8::FromFileBasic("../micmac_data/Ref-Apero/Test-Lion/MatAvecCstr.tif");
   Im2D_REAL8 aIS = Im2D_REAL8::FromFileBasic("../micmac_data/Ref-Apero/Test-Lion/MatSansCstr.tif");
   Pt2di aSzA = aIA.sz();
   Pt2di aSzS = aIS.sz();

   std::cout << aSzA << aSzS << "\n";

   for (int aKx=0 ; aKx<aSzA.x ; aKx++)
   {
      for (int aKy=0 ; aKy<aSzA.y ; aKy++)
      {
         double aD = aIA.data()[aKy][aKx] - aIS.data()[aKy][aKx];
         std::cout <<  aD << " ";
      }
      std::cout <<  "\n";
   }
      std::cout <<  "\n";
   for (int aKx=0 ; aKx<aSzA.x ; aKx++)
   {
      for (int aKy=0 ; aKy<aSzA.y ; aKy++)
      {
         double aD = aIA.data()[aKy][aKx] ;
         std::cout <<  aD << " ";
      }
      std::cout <<  "\n";
   }
}

void TestDonDenis(int aK,const std::string & aChantier,int aNB,bool doPx,bool doImg)
{
   Pt2di aP0 (2200,600);
   Pt2di aP1 (3000,1300);

   int aNum = aNB-aK;
   int aZ = 1<< aK;
   std::string aDD = "/data/DonDenis/Demo/";
   std::string aNamePx = aDD + std::string("Z_Num") + ToString(aNum) 
                             + std::string("_DeZoom") + ToString(aZ)  
                             +std::string("_") + aChantier + std::string(".tif");
   std::string aNameMask = aDD+ std::string("Masq_MultiScale_DeZoom") +ToString(aZ) +  std::string(".tif");

   std::cout << aNamePx  << "\n" << aNameMask << "\n";
   aP0 = aP0 / aZ;
   aP1 = aP1 / aZ;
   Pt2di aSz = aP1-aP0;
   Im2D_INT4 anIm(aSz.x,aSz.y);
   Tiff_Im aFilePx = Tiff_Im::StdConv(aNamePx);
   Tiff_Im aFileMask = Tiff_Im::StdConv(aNameMask);

   ELISE_COPY
   (
        anIm.all_pts(),
        trans
        (
           128 * (aFileMask.in()==0)
           + (aFileMask.in()!=0) * (128 + (470.0/aZ+aFilePx.in())*(aZ *3.0))
          ,
           aP0
        ),
        anIm.out()
        
   );


   std::string aNameResult = aDD+ std::string("Demo_") +aChantier + std::string("_") +ToString(aK) +  std::string(".tif");
   if (doPx)
   {
      Tiff_Im::Create8BFromFonc
      (
           aNameResult,
           aSz * aZ,
           Max(0,Min(255,anIm.in()[Virgule(FX,FY)/aZ]))
      );
   }
   if (doImg)
   {
          std::string  aNIm =  aDD + std::string("F120_SG1L7888_MpDcraw8B_GR.tifDeZoom")+ToString(aZ) +std::string (".tif");
          if (aZ==1)
          {
             aNIm = "/data/DonDenis/F120_SG1L7888_MpDcraw8B_GR.tif";
          }
          
          Tiff_Im aFileIm = Tiff_Im::StdConv(aNIm);
          ELISE_COPY
          (
               anIm.all_pts(),
               trans (aFileIm.in(),aP0),
               anIm.out()
          );
          std::string aNameResult = aDD+ std::string("DemoImage_") + std::string("_") +ToString(aZ) +  std::string(".tif");
          Tiff_Im::Create8BFromFonc
          (
               aNameResult,
               aSz * aZ,
               Max(0,Min(255,anIm.in()[Virgule(FX,FY)/aZ]))
          );
   }
}

void TestDonDenis()
{
   TestDonDenis(0,"MultiScale",6,false,true);
   TestDonDenis(1,"MultiScale",6,false,true);
   TestDonDenis(2,"MultiScale",6,false,true);
   TestDonDenis(3,"MultiScale",6,false,true);
   TestDonDenis(4,"MultiScale",6,false,true);
   TestDonDenis(5,"MultiScale",6,false,true);

   TestDonDenis(0,"Regul-0_1",6,false,false);
   TestDonDenis(0,"Regul-0_01",6,false,false);
   TestDonDenis(0,"Regul-0_5",6,false,false);
   
   TestDonDenis(0,"OneViewBasic",1,false,false);
   TestDonDenis(0,"Sz3-OneViewBasic",1,false,false);
   TestDonDenis(0,"Basic",1,false,false);

}

void TestAMD()
{
   // amd_demo_1();

   int aN= 10;
   cAMD_Interf anAI(aN);
   

   for (int aK=0 ; aK< aN ; aK++)
   {
     if (aK%2)
        anAI.AddArc(aK,1);
     else
        anAI.AddArc(aK,0);
     anAI.AddArc(aK,aK);
     anAI.AddArc(aK,ElMin(aN-1,aK+1));
   }

   anAI.DoRank(true);
}

void texture()
{
    Pt2di aSz(1000,700);
    Im2D_U_INT1 aIm(aSz.x,aSz.y);

    ELISE_COPY
    (
        aIm.all_pts(),
        Max(0,Min(255,255*(0.7*unif_noise_2(1) + 0.3*unif_noise_4(6)))),
        aIm.out()
    );
    Tiff_Im::CreateFromIm(aIm,"toto.tif");
}



void OneTestXML(const std::string& aName)
{
     cElXMLTree  aXTree(aName);
     aXTree.StdShow("../TMP/Test.xml");
     cAvionJauneDocument anAJD = StdGetObjFromFile<cAvionJauneDocument>
                                 (
                                     aName,
                                     StdGetFileXMLSpec("SuperposImage.xml"),
                                     "Document",
                                     "AvionJauneDocument"
                                 );

      std::cout << anAJD.roulisAvion().value() << "\n";
}

void TestXML()
{
    // OneTestXML("../micmac_data/Ref-Apero/Test-XMl/toto.xml");
    OneTestXML("../micmac_data/Ref-Apero/Test-XMl/00047.igi");
}


void Test2AM(const std::string & aStr)
{
   std::vector<char *>  aV = ToArgMain(aStr);

   printf("=====================================\n");
   std::cout << aStr << "\n\n";
   for (int aK=1 ; aK<(int)aV.size() ; aK++)
   {
        printf("[%s]\n",aV[aK]);
   }
   getchar();
}

void TestSC()
{
   const cSysCoord * aSC = cSysCoord::WGS84();
   while(1)
   {
        double L,l,h;
        cin >> L >> l >> h;
        Pt3dr aP = aSC->ToGeoC(Pt3dr(L,l,h));
        Pt3dr aQ = aSC->FromGeoC(aP);
        Pt3dr aW = aSC->ToGeoC(aQ);
        std::cout << aP << aQ <<  aW  <<  " D=" << euclid(aP-aW) << "\n";
   }
}

void TestAdEx()
{
   while (1)
   {
      std::string aT;
      cin >> aT ; 
      //aT="";
      // AddExtensionToSubdir(aT,"Toto");
      std::cout << aT << "\n";
   }
}

extern bool TransFormArgKey (
         std::string & aName ,
         bool AMMNoArg,  // Accept mismatch si DirExt vide
         const std::vector<std::string> & aDirExt
     );


void TestTransFormArgKey(
         std::string  aName ,
         bool AMMNoArg,  // Accept mismatch si DirExt vide
         const std::vector<std::string> & aDirExt
      )
{
   std::cout << "IN=" << aName << "\n";
   for (int aK=0 ; aK<int(aDirExt.size()) ; aK++)
      std::cout << "    ARG " << aDirExt[aK] << "\n";

   bool TrDone = TransFormArgKey(aName,AMMNoArg,aDirExt);

   std::cout << "OUT =" << aName << "\n";
   std::cout << "TRANFSORM=" << TrDone << "\n\n";
}

void TestTransFormArgKey()
{
   {
      std::vector<std::string> aV0;
      TestTransFormArgKey("akk",true,aV0);
   }
   {
      std::vector<std::string> aV0;
      TestTransFormArgKey("akk#",true,aV0);
   }
   if (0) // Plante legitiemmeent
   {
      std::vector<std::string> aV0;
      TestTransFormArgKey("akk#",false,aV0);
   }
   {
      std::vector<std::string> aV0;
      aV0.push_back("UN");
      aV0.push_back("DEUX");
      TestTransFormArgKey("akk#1--#2--#1",false,aV0);
   }
   if (0) // Plante legitiemmeent
   {
      std::vector<std::string> aV0;
      aV0.push_back("UN");
      aV0.push_back("DEUX");
      TestTransFormArgKey("akk#1--#2--#1#",false,aV0);
   }
   if (1) // Plante legitiemmeent
   {
      std::vector<std::string> aV0;
      aV0.push_back("UN");
      aV0.push_back("DEUX");
      TestTransFormArgKey("akk#1--#2--#1",false,aV0);
   }
}

void TestNSplit()
{
   while (1)
   {
       std::string aStr,aS0;
       std::cin >> aStr;
       std::vector<std::string> aVS;
       SplitInNArroundCar(aStr,'@',aS0,aVS);
       std::cout << aS0;
       for (int aK=0; aK<int(aVS.size()) ; aK++)
          std::cout << "[" << aVS[aK] << "]";
       std::cout << "\n";
   }
}

void TestTif(const char * anArg)
{

  std::string aName = NameFileStd(anArg,1,false);
  Tiff_Im aTF =  Tiff_Im::StdConv(aName);


}

void TestCam2003(const std::string & aN1,const std::string & aN2)
{
   CamStenope * aCam1 = CamCompatible_doublegrid(aN1);
   CamStenope * aCam2 = CamCompatible_doublegrid(aN2);

   std::cout << aCam1->Sz() << " " << aCam2->Sz() << "\n";
   while (1)
   {
      double Xph,Yph;

      cin >> Xph >> Yph;
      Pt2dr aP(Xph,Yph);

      Pt2dr aQ1 = aCam1->PtDirRayonL3toF2(aP) ;
      Pt2dr aQ2 = aCam2->PtDirRayonL3toF2(aP) ;

      std::cout << aQ1 << " " <<  aQ2 << " " << euclid(aQ1-aQ2) << "\n";
   }
}

void TestCam2003()
{
   TestCam2003
   (
      "/media/FreeAgentDrive/calibration_viroise/CAM34/GRID_NoGrid_DRad_6_09_1_34_r_MetaDonnees.xml",
      "/media/FreeAgentDrive/calibration_viroise/CAM34/GRID_NoGrid_DRad_6_09_1_34_r.xml"
   );

}

void TestCart()
{
   std::string aNF = "../micmac_data/ExempleDoc/Lamballe/ReperePlanGlobal.xml";
   cRepereCartesien aRC = StdGetObjFromFile<cRepereCartesien>
                          (
                               "../micmac_data/ExempleDoc/Lamballe/ReperePlanGlobal.xml",
                                StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                "RepereCartesien",
                                "RepereCartesien"
                          );
   cChCoCart aCC = cChCoCart::Xml2El(aRC);

   Pt3dr aP(1,2,3);
   Pt3dr aQ = aCC.FromLoc(aP);

   cChCoCart aCCI = aCC.Inv();
   Pt3dr aR = aCCI.FromLoc(aQ);

   std::cout << aP << aR << aQ << "\n";
}

void Damier()
{
    Tiff_Im aTF  = Tiff_Im::StdConv("/home/mpd/micmac_data/ExempleDoc/Boudha/Debug.tif");
    ELISE_COPY(aTF.all_pts(),255*((FX+FY)%2),aTF.out());
}


Pt2dr InvY(const Pt2dr & aP) {return Pt2dr(aP.x,1-aP.y);}
void Box()
{
    Box2dr BoxImage(Pt2dr(0,0),Pt2dr(2712,5609));
    Box2dr BoxTer(Pt2dr(-31,-196),Pt2dr(116,108));

    Pt2dr aP1(1014 ,4063);
    Pt2dr aP2(2236,4838);
   
    Pt2dr aQ1 = InvY(BoxImage.ToCoordLoc(aP1));
    Pt2dr aQ2 = InvY(BoxImage.ToCoordLoc(aP2));

    Pt2dr aU1 =  BoxTer.FromCoordLoc(aQ1);
    Pt2dr aU2 =  BoxTer.FromCoordLoc(aQ2);

    std::cout <<  aU1.x << " " << aU2.y  << "\n";
    std::cout <<  aU2.x << " " << aU1.y  << "\n";
}


void TestRel(int argc,char** argv)
{

   cTplValGesInit<std::string> aNoName;
   cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::StdAlloc(0,0,argv[1],aNoName);


    const  std::vector<cCpleString> * aR = anICNM->GetRel(argv[2]);

   std::cout << "SIZE " << aR->size() << "\n";

}

void TestGetStr()
{
   Video_Win aW = Video_Win::WStd(Pt2di(800,800),1.0);

   while (1)
   {
       Clik aClK = aW.clik_in();

       std::string aStr = aW.GetString(aClK._pt,aW.pdisc()(P8COL::red),aW.pdisc()(P8COL::green));
       std::cout << "GOT " << aStr << "\n";
   }
}
extern void TestXD(const std::string & aNameIm);


void TestM(double aTeta,int aK)
{
   ElMatrix<double> aM = ElMatrix<double>::Rotation3D(aTeta,aK);

   std::cout  << " K = " << aK << " teta = " << aTeta << "\n";
   const char * aMes = "Salut";
   ShowMatr(aMes,aM);
}

void TestIm()
{
   Tiff_Im aTF("IMGP7505_Out_Scaled.tif");
    
   Video_Win aW = Video_Win::WStd(Pt2di(800,800),1.0);
   Im2D_U_INT1 anIm(800,800);

   ELISE_COPY
   (
        aW.all_pts(),
        aTF.in(123),
	aW.ogray() | anIm.out()
   );

   U_INT1 ** aData = anIm.data();
   for (int anX=0; anX<800; anX++)
       for (int anY=0; anY<800; anY++)
            aData[anY][anX] = 255 - aData[anY][anX] ;
   // getchar();
   ELISE_COPY
   (
        aW.all_pts(),
        rect_min(rect_max(anIm.in(0),5),5),
	aW.ogray() | anIm.out()
   );


   getchar();



}


void TestVP3D()
{
   std::vector<Pt3dr> * aV = StdNuage3DFromFile("/home/marc/TMP/Delphes/12-Tetes-Inc-4341-6/Ori-Circle/Pts3d.dat");

   std::cout << aV->size() << (*aV)[0] << aV->back() << "\n";
}

void TestQuadr() 
{
    Pt2di aSz(5634,3753);

    Tiff_Im::Create8BFromFonc("Quadr.tif",aSz,255 * ((FX%10)>=2) * ((FY%20)>=6));
}


void TestLG(const std::string & aFullName)
{
    Tiff_Im aTF= Tiff_Im::StdConvGen(aFullName,3,false);

    Pt2di aSz = aTF.sz();

    Im2D_U_INT1  aImR(aSz.x,aSz.y);
    Im2D_U_INT1  aImG(aSz.x,aSz.y);
    Im2D_U_INT1  aImB(aSz.x,aSz.y);

    ELISE_COPY
    (
       aTF.all_pts(),
       aTF.in(),
       Virgule(aImR.out(),aImG.out(),aImB.out())
    ); 


    std::string aDir,aName;
    SplitDirAndFile(aDir,aName,aFullName);

    std::string aNameOut = aDir + "Out_" + aName;


    U_INT1 ** aDataR = aImR.data();
    U_INT1 ** aDataG = aImG.data();
    U_INT1 ** aDataB = aImB.data();

 
    for (int aY=0 ; aY<aSz.y  ; aY++)
    {
        for (int aX=0 ; aX<aSz.x  ; aX++)
        {
             aDataR[aY][aX] = 255 - aDataR[aY][aX];
             aDataG[aY][aX] = 255 - aDataG[aY][aX];
             aDataB[aY][aX] = 255 - aDataB[aY][aX];
        }
    }
    


    Tiff_Im  aTOut 
             (
                  aNameOut.c_str(),
                  aSz,
                  GenIm::u_int1,
                  Tiff_Im::No_Compr,
                  Tiff_Im::RGB
             );


     ELISE_COPY
     (
         aTOut.all_pts(),
         Virgule(aImR.in(),aImG.in(),aImB.in()),
         aTOut.out()
     );

}


#endif

void TestXML2()
{
  // cElXMLTree aTree("Test.xml",0,false);
  std::string aFileIn = "TestSubst.xml";
  cArgCreatXLMTree anArg(aFileIn,true,true);
  cElXMLTree aTree(aFileIn,&anArg);
  aTree.StdShow("TestOur.xml");
}

int MPDtest_main (int argc,char** argv)
{
   BanniereMM3D();
   double aNan = strtod("NAN(teta01)", NULL);
   std::cout << "Nan=" << aNan << "\n";
   // TestXML2();
   //  cout <<  PolonaiseInverse(argv[1]) << "\n" ;
/*
  TestLG(argv[1]);
 TestQuadr() ;
   testim();
{
    std::string astr="[a,bc,o$$]";

    stringstream ss(astr);
    std::vector<std::string>  ares;
    ElStdRead(SS,aRES,ElGramArgMain::StdGram);

for (int aK=0; aK<int(aRES.size()) ; aK++)
   std::cout << "== "<< aRES[aK] << "\n";
}
*/


    // TestM(PI/2.0,0);
    // TestM(PI/2.0,1);
    // TestM(PI/2.0,2);
    // TestXD(argv[1]);
     // TestGetStr();
     //TestRel(argc,argv);
     // Damier();
     // TestCart();
     // TestCam2003();
    // TestTif(argv[1]);
    // TestNSplit();
    // TestTransFormArgKey();
    // TestCubic();
    // TestSC();
    // texture();
   // TestDonDenis();
   // TestMatCstr();
   // TestCamOrtho();

	return EXIT_SUCCESS;
}

#endif

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
