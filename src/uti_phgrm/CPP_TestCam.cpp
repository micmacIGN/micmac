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
#include "StdAfx.h"


void TestOneCorner(ElCamera * aCam,const Pt2dr&  aP, const Pt2dr&  aG)
{
     Pt2dr aQ0 = aCam->DistDirecte(aP);
     Pt2dr aQ1 = aCam->DistDirecte(aP+aG);

     std::cout <<  " Grad " << (aQ1-aQ0 -aG) / euclid(aG) << " For " << aP << "\n";
}


void TestOneCorner(ElCamera * aCam,const Pt2dr&  aP)
{
    TestOneCorner(aCam,aP,Pt2dr(1,0));
    TestOneCorner(aCam,aP,Pt2dr(0,1));
    std::cout << "=======================================\n";
}

void TestOneCorner(ElCamera * aCam)
{
    Pt2dr aSz = Pt2dr(aCam->Sz());

    TestOneCorner(aCam,Pt2dr(0,0));
    TestOneCorner(aCam,Pt2dr(aSz.x,0));
    TestOneCorner(aCam,Pt2dr(0,aSz.y));
    TestOneCorner(aCam,Pt2dr(aSz.x,aSz.y));
    TestOneCorner(aCam,Pt2dr(aSz.x/2.0,aSz.y/2.0));

    TestOneCorner(aCam,Pt2dr(1072,712));
}


void TestDistInv(ElCamera * aCam,const Pt2dr & aP)
{
    std::cout << "Test Dis Inv , aP " << aP << "\n";
    std::cout << "Res =  " << aCam->DistInverse(aP) << "\n";

}

void TestDirect(ElCamera * aCam,Pt3dr aPG)
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


int TestCam_main(int argc,char ** argv)
{
    std::string aFullName;
    std::string aNameCam;
    std::string aNameDir;
    std::string aNameTag = "OrientationConique";
    bool ExtP = false;
    bool TOC = false;
    Pt2dr TDINV;

    double X,Y,Z;
    X = Y = Z = 0;
    bool aModeGrid = false;
    std::string Out;

    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aFullName,"File name", eSAM_IsPatFile)
                << EAMC(X,"x")
                << EAMC(Y,"y")
                << EAMC(Z,"z"),
    LArgMain()
                    << EAM(aNameTag,"Tag",true,"Tag to get cam")
                    << EAM(aModeGrid,"Grid",true,"Test Grid Mode", eSAM_IsBool)
                    << EAM(Out,"Out",true,"To Regenerate an orientation file",eSAM_NoInit)
                    << EAM(ExtP,"ExtP",true,"Detail on external parameter", eSAM_IsBool)
                    << EAM(TOC,"TOC",true,"Test corners", eSAM_IsBool)
                    << EAM(TDINV,"TDINV",true,"Test Dist Inv",eSAM_NoInit)
    );

    if (MMVisualMode) return EXIT_SUCCESS;

    SplitDirAndFile(aNameDir,aNameCam,aFullName);

    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aNameDir);
/*
    cTplValGesInit<std::string>  aTplFCND;
    cInterfChantierNameManipulateur * anICNM =
        cInterfChantierNameManipulateur::StdAlloc(0,0,aNameDir,aTplFCND);
*/


   ElCamera * aCam  = Gen_Cam_Gen_From_File(aModeGrid,aFullName,aNameTag,anICNM);

   CamStenope * aCS = aCam->CS();

   if (ExtP)
   {
       std::cout << "  ###########  EXTERNAL ##############\n";
       if (aCS)
       {
           std::cout << "Center " << aCS->VraiOpticalCenter() << "\n";
       }
       std::cout <<  "  I : " << aCS->L3toR3(Pt3dr(1,0,0)) - aCS->L3toR3(Pt3dr(0,0,0)) << "\n";
       std::cout <<  "  J : " << aCS->L3toR3(Pt3dr(0,1,0)) - aCS->L3toR3(Pt3dr(0,0,0))<< "\n";
       std::cout <<  "  K : " << aCS->L3toR3(Pt3dr(0,0,1)) - aCS->L3toR3(Pt3dr(0,0,0))<< "\n";
       std::cout << "\n";
   }

    if (TOC)
       TestOneCorner(aCam);

    if (EAMIsInit(&TDINV))
       TestDistInv(aCam,TDINV);



   if (aModeGrid)
   {
       std::cout << "Camera is grid " << aCam->IsGrid() << " " << aCam->Dist().Type() << "\n";
   }


   TestCamCHC(*aCam);

   TestDirect(aCam,Pt3dr(X,Y,Z));

   if (Out!="")
   {
         cOrientationConique aCO = aCam->StdExportCalibGlob();
         MakeFileXML(aCO,Out);
   }

    return EXIT_SUCCESS;
}


// ========================================

class cAppliTestARCam 
{
     public :
        cAppliTestARCam(int argc,char ** argv);
        std::string mName;

        cBasicGeomCap3D *mCam;
        Pt2di           mSz;
        double          mZ0;

        void TestAR();
        void TestAR(double aZ);
        double TestAR(const Pt2dr & aP,const double & aZ);
};


double cAppliTestARCam::TestAR(const Pt2dr & aPI0,const double & aZ)
{
     Pt3dr aPTer = mCam->ImEtZ2Terrain(aPI0,aZ);
     Pt2dr aPI1 = mCam->Ter2Capteur(aPTer);

     return euclid(aPI0-aPI1) + ElAbs(aZ-aPTer.z);
}

void cAppliTestARCam::TestAR(double aZ)
{
    Pt2di aP;
    double aMaxD = 0;
    double aMoyD = 0;
    Pt2di  aPMax;
    int aNb1=0;
    for (aP.x=0 ; aP.x<=mSz.x ; aP.x++)
    {
        for (aP.y=0 ; aP.y<=mSz.y ; aP.y++)
        {
             double aD = TestAR(Pt2dr(aP),aZ);
             aMoyD += aD;
             if (aD>aMaxD)
             {
                 aMaxD = aD;
                 aPMax = aP;
             }
             if (aD>1) aNb1++;
        }
    }
    aMoyD /= double(mSz.x*mSz.y);
    std::cout << "MaxD " << aMaxD <<" ; MoyD " << aMoyD << " ; PMax " << aPMax  << " ; Nb>1 " << aNb1 << "\n";
}

void cAppliTestARCam::TestAR()
{
    TestAR(mZ0);
}



cAppliTestARCam::cAppliTestARCam(int argc,char ** argv)  :
    mZ0(0.0)
{
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mName,"Name Ori"),
        LArgMain()  << EAM(mZ0,"Z0")
    );

    int aType = eTIGB_Unknown;

    mCam = cBasicGeomCap3D::StdGetFromFile(mName,aType);

    mSz  = mCam->SzBasicCapt3D();
    std::cout << "Sz " << mSz << " Z0 " << mZ0 << "\n";

}

int TestARCam_main(int argc,char ** argv)
{
   cAppliTestARCam anAppli(argc,argv);
   anAppli.TestAR();

   return EXIT_SUCCESS;
}

int TestDistM2C_main(int argc,char ** argv)
{
    std::string aNameCam;
    Pt2dr aP0;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  <<  EAMC(aNameCam,"Name Cam")
                    <<  EAMC(aP0,"Pt"),
        LArgMain()  
    );

    cElemAppliSetFile anEASF(aNameCam);

    CamStenope * aCam =  CamOrientGenFromFile(aNameCam,anEASF.mICNM);

    std::cout << "SZ " << aCam->Sz() << "\n";
    std::cout   << " Delta=" << aCam->DistDirecte(aP0) -aP0 << "\n";
    std::cout   << " Dx=" << aCam->DistDirecte(aP0) -aCam->DistDirecte(aP0+Pt2dr(-1,0)) << "\n";
    std::cout   << " Dy=" << aCam->DistDirecte(aP0) -aCam->DistDirecte(aP0+Pt2dr(0,-1)) << "\n";
    std::cout   << " Delta=" << aCam->DistInverse(aP0) -aP0 << "\n";

   return EXIT_SUCCESS;
}



int TestDistortion_main(int argc,char ** argv)
{
    std::string aNameCalib;
    Pt2dr aPt2d;
    Pt3dr aPt3d;
    REAL prof=1;
    bool showAngles=false;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  <<  EAMC(aNameCalib,"Calibration Name"),
        LArgMain()  <<  EAM(aPt2d, "p2d", true, "Point in picture coordinates")
                    <<  EAM(aPt3d, "p3d", true, "Point in world coordinates")
                    <<  EAM(prof, "prof", true, "prof for p2d (default=1)")
                    <<  EAM(showAngles, "showAngles", true, "show angles in/out (default=false)")
    );

    cElemAppliSetFile anEASF(aNameCalib);
    std::cout << aNameCalib << std::endl;

    CamStenope * aCam =  CamOrientGenFromFile(aNameCalib,anEASF.mICNM);

    std::cout << "//   R3 : \"reel\" coordonnee initiale\n";
    std::cout << "//   L3 : \"Locale\", apres rotation\n";
    std::cout << "//   C2 :  camera, avant distortion\n";
    std::cout << "//   F2 : finale apres Distortion\n";
    std::cout << "//   M2 : coordonnees scannees \n";
    std::cout << "//\n";
    std::cout << "//       Orientation      Projection      Distortion      Interne mm\n";
    std::cout << "//   R3 -------------> L3------------>C2------------->F2------------>M2\n";
    std::cout << "//\n";

    std::cout << "Focale " << aCam->Focale() << "\n";

    if (EAMIsInit(&aPt2d))
    {
        std::cout << "M2 " << aPt2d << " ---> F2 " << aCam->NormM2C(aPt2d)  << " --->C2 " <<
                     aCam->F2toC2(aCam->NormM2C(aPt2d)) << " ---> R3 "<< aCam->ImEtProf2Terrain(aCam->NormM2C(aPt2d),prof) << "\n";
    }

    if (EAMIsInit(&aPt3d))
    {
        std::cout << "prof: "<< aCam->ProfondeurDeChamps(aPt3d) << "\n";
        std::cout << "R3 "<< aPt3d;
        std::cout << " ---> C2 "<< aCam->DistInverse(aCam->NormC2M(aCam->Ter2Capteur(aPt3d)));
        std::cout << " ---> F2 "<< aCam->Ter2Capteur(aPt3d);
        std::cout << " ---> M2 "<< aCam->NormC2M(aCam->Ter2Capteur(aPt3d)) << "\n";
    }

    /*std::cout << "NormM2C "<< aCam->NormM2C(aP0) <<"\n";
    Pt3dr aPtTer(0.976092906909062696, -0.797308401978953252, -8.10257300823790594);
    std::cout << "Ter2Capteur "<< aCam->Ter2Capteur(aPtTer) << " " << aCam->NormC2M(aCam->Ter2Capteur(aPtTer)) << "\n";*/

    if (showAngles)
    {
        //table of input angles VS ouput angles
        std::cout<<"\ninAngle outAngle outX/f\n";
        double inAngle,outAngle;
        Pt3dr inPt3d;
        Pt2dr outPt2d;
        for (int i=-90;i<=90;i++)
        {
            inAngle=i*PI/180;
            inPt3d=Pt3dr(sin(inAngle),0,cos(inAngle));
            outPt2d=aCam->L3toF2(inPt3d);
            if ((outPt2d.x>0)&&(outPt2d.x<aCam->Sz().x))
            {
                outAngle=atan((outPt2d.x-aCam->PP().x)/aCam->Focale());
                std::cout<<inAngle<<" "<<outAngle<<" "<<(outPt2d.x-aCam->PP().x)/aCam->Focale()<<"\n";
            }
        }
    }

    return EXIT_SUCCESS;
}


class cStrPtCamForInter
{
    public :
       cStrPtCamForInter(const std::vector<std::string> & aVS)
       {
            int aType= eTIGB_Unknown;
            ELISE_ASSERT(aVS.size()==3,"cStrPtCamForInter");
            mCam = cBasicGeomCap3D::StdGetFromFile(aVS[0],aType);
       }
       ElSeg3D           Seg() {return mCam->Capteur2RayTer(mPt);};
       cBasicGeomCap3D * mCam;
       Pt2dr             mPt;
};


class cAppli_TestBundleInter_main
{
   public :

       void AddPtCam(std::vector<std::string> & aVS)
       {
            if (! EAMIsInit(&aVS)) return;
            int aType= eTIGB_Unknown;
            ELISE_ASSERT(aVS.size()==3,"cStrPtCamForInter");

            cBasicGeomCap3D * aCam = cBasicGeomCap3D::StdGetFromFile(aVS[0],aType);
            mVCam.push_back(aCam);

            Pt2dr aP;
            FromString(aP.x,aVS[1]);
            FromString(aP.y,aVS[2]);
            mVPt.push_back(aP);
            mVSeg.push_back(aCam->Capteur2RayTer(aP));
            std::cout << "cAppli_TestBundleInter_main " << aP << "\n";
       }

       std::vector<cBasicGeomCap3D *> mVCam;
       std::vector<Pt2dr>             mVPt;
       std::vector<ElSeg3D>         mVSeg;
   
       cAppli_TestBundleInter_main (int argc,char ** argv)
       {
           std::vector<std::string> aVS1;
           std::vector<std::string> aVS2;
           std::vector<std::string> aVS3;

           ElInitArgMain
           (
             argc,argv,
             LArgMain()  << EAMC(aVS1,"[Cam1,x1,y1]")
                         << EAMC(aVS2,"[Cam2,x2,y2]"),
             LArgMain()
                        << EAM(aVS3,"Cam3",true,"[Cam3,x3,y3]")
           );


           AddPtCam(aVS1);
           AddPtCam(aVS2);

           bool Ok;
           Pt3dr  aPI = InterSeg(mVSeg,Ok);

           std::cout << "Inter= " << aPI << "\n";
           for (int aK=0 ; aK<int(mVSeg.size()) ; aK++)
           {
               Pt2dr aProj = mVCam[aK]->Ter2Capteur(aPI);
               std::cout << " D=" << euclid(aProj,mVPt[aK]) << "\n";
           }
       }
};

/*
mm3d TestBundleInter [Ori-C35/Orientation-F35_MG_0150.CR2.xml,3941.00320404687682,1863.818403035158] [Ori-C35/Orientation-F35_MG_0151.CR2.xml,4062.24841818221648,1829.81156999058044]

*/


int TestBundleInter_main(int argc,char ** argv)
{
    cAppli_TestBundleInter_main anAppli(argc,argv);

    return EXIT_SUCCESS;
}



int GenerateBorderCam_main(int argc,char **argv)
{
    std::string aNameCam;
    int  aNb;
    double aInt=1.0;
    std::string aNameOut;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  <<  EAMC(aNameCam,"Name Cam")
                    <<  EAMC(aNb,"Nb Pts / size"),
        LArgMain()  <<  EAM(aInt,"Int",true,"Interior to limit size with bad distorsion, Def=1.0")
                    <<  EAM(aNameOut,"Out",true,"File for result, Def= Border-Input")
    );

    cElemAppliSetFile anEASF(aNameCam);
    CamStenope * aCam =  CamOrientGenFromFile(aNameCam,anEASF.mICNM);
    if (! EAMIsInit(&aNameOut))
    {
          aNameOut  = anEASF.mDir + "Border-"+ StdPrefix(NameWithoutDir(aNameCam)) + ".xml";
    }



    Pt2dr aP0 = aCam->Sz() * (1-aInt);
    Pt2dr aP1 = aCam->Sz() * (aInt);
    std::cout << "SZ " << aP0 << aP1 << "\n";
    Box2dr aBox(aP0,aP1);
    Pt2dr aC[4];
    aBox.Corners(aC);


    cXmlOneContourCamera aRes;
    for (int aKC=0 ; aKC<4 ; aKC++)
    {
         Pt2dr aC1 = aC[aKC];
         Pt2dr aC2 = aC[(aKC+1)%4];
         for (int aKP=0 ; aKP<aNb ; aKP++)
         {
              double aPds = aKP/ double(aNb);
              Pt2dr aP = aC1 * (1-aPds) + aC2 * aPds;
              aP = aCam->F2toC2(aP);
              aRes.Pt().push_back(aP);
              // std::cout << aP << "\n";
         }
    }
    MakeFileXML(aRes,aNameOut);

	return 0;
}

/***************************************************************/
/*                                                             */
/*                                                             */
/*                                                             */
/***************************************************************/
/*
int CPP_RedressIm_Main()
{
}
*/


/* Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilite au code source et des droits de copie,
de modification et de redistribution accordes par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitee.  Pour les memes raisons,
seule une responsabilite restreinte pese sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concedants successifs.

A cet egard  l'attention de l'utilisateur est attiree sur les risques
associes au chargement,  a l'utilisation,  a la modification et/ou au
developpement et a la reproduction du logiciel par l'utilisateur etant
donne sa specificite de logiciel libre, qui peut le rendre complexe a
manipuler et qui le reserve donc a des developpeurs et des professionnels
avertis possedant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invites a charger  et  tester  l'adequation  du
logiciel a leurs besoins dans des conditions permettant d'assurer la
securite de leurs systemes et ou de leurs donnees et, plus generalement,
a l'utiliser et l'exploiter dans les memes conditions de securite.

Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
termes.
Footer-MicMac-eLiSe-25/06/2007/*/
