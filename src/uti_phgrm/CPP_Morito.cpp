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
#include <algorithm>

// ffmpeg -i MVI_0001.MOV  -ss 30 -t 20 Im%5d_Ok.png

// Im*_Ok => OK 
// Im*_Nl => Image Nulle (eliminee)


void BanniereMorito()
{
    std::cout <<  "\n";
    std::cout <<  " *********************************************\n";
    std::cout <<  " *     M-erge                                *\n";
    std::cout <<  " *     ORI-entations                         *\n";
    std::cout <<  " *     TO (? ?)                              *\n";
    std::cout <<  " *********************************************\n\n";

}

class cOriMorito;
class cAppliMorito;

//=================================================


class cOriMorito
{
    public :
        cOriMorito(); 
        CamStenope * mCam1;
        CamStenope * mCam2;
        std::string mNameFull;
        std::string mName;
};

class cAppliMorito
{
    public :
        cAppliMorito(int argc,char ** argv);
        void InitOneDir(const std::string & aDir,bool D1);
        const std::string & Dir() {return  mDir ;}
        

    private :
        void InitRotM2toM1();
        void InitScTr2to1();
        void ComputNewRot2();
        void Sauv();
        void SauvCalib(const std::string &);


        std::map<std::string,cOriMorito> mOrients;
        std::vector<cOriMorito*> mVDpl; 
        std::vector<Pt3dr>       mVP1;
        std::vector<Pt3dr>       mVP2;

        std::string       mOri1;
        std::string       mOri2;
        std::string       mOriOut;
        bool              mWithOutLayer;
        ElMatrix<double>  mRM2toM1;
        double            mSc2to1;
        Pt3dr             mTr2to1;
        bool              mShow;
        std::string       mDir;
        std::string       mDirOutLoc;
        std::string       mDirOutGlob;
};

// =============  cOriMorito   ===================================

cOriMorito::cOriMorito() :
   mCam1 (0),
   mCam2 (0)
{
}
// =============  cAppliMorito ===================================


void cAppliMorito::InitOneDir(const std::string & aPat,bool aD1)
{
   cElemAppliSetFile anEASF(aPat);

   const std::vector<std::string> * aVN = anEASF.SetIm();
   for (int aK=0 ; aK<int(aVN->size()) ; aK++)
   {
        std::string aNameOri = (*aVN)[aK];
        CamStenope * aCS = CamOrientGenFromFile(aNameOri,anEASF.mICNM);

        cOriMorito & anOri = mOrients[aNameOri];
        anOri.mName = aNameOri;
        anOri.mNameFull =  anEASF.mDir + aNameOri;
        if (aD1) 
           anOri.mCam1 = aCS;
        else  
           anOri.mCam2 = aCS;
   }
}


cAppliMorito::cAppliMorito(int argc,char ** argv)  :
    mWithOutLayer       (false),
    mRM2toM1            (3,3,0.0),
    mShow               (2),
    mDir                ("./")
{


    ElInitArgMain
     (
           argc,argv,
           LArgMain() << EAMC(mOri1,"First set of image", eSAM_IsPatFile) 
                      << EAMC(mOri2,"Second set of image")
                      << EAMC(mOriOut,"Orientation Dir"),
           LArgMain() << EAM(mWithOutLayer,"WithOutLayer",true,"Is robust estimation requires or simply L2 (Def=false, other not supported now)")
                      << EAM(mDir,"Dir",true,"Global directory, Def=true")
                     
    );
    InitOneDir(mOri1,true);
    std::cout << "====================================================\n";
    InitOneDir(mOri2,false);
    mDirOutLoc =  "Ori-"  + mOriOut + "/";
    mDirOutGlob = Dir() + mDirOutLoc;

    for 
    (
        std::map<std::string,cOriMorito>::iterator itO = mOrients.begin();
        itO != mOrients.end();
        itO++
    )
    {
         if (itO->second.mCam1 && itO->second.mCam2)
         {
             mVDpl.push_back(&(itO->second));
         }
    }
    ELISE_ASSERT(mVDpl.size()>=2,"Not Enough common orientation in Morito");


    InitRotM2toM1();
    InitScTr2to1();
    ComputNewRot2();
    Sauv();
}

/*
===============================================
   !!!!   C'est la meme camera, mais pas le meme monde, donc in faut ecrire :
      PM1 = aM2toM1 (PM2)

     M1toC PM1 = PCam   et  M2toC PM2 = PCam

     PCam =  M1toC PM1 = M2toC PM2 = M1toC aM2toM1 PM2
      
     =>    aM2toM1 = M1toC-1 M2toC   ; aM1toM2 = M2toC-1 M1toC
    => M1toC = M2toC * aM2toM1-1

*/

void cAppliMorito::InitRotM2toM1()
{
   // Monde -> Cam 
   for (int aK = 0 ; aK<int(mVDpl.size()) ; aK++)
   {
       ElRotation3D aRM1toCam =  mVDpl[aK]->mCam1->Orient();
       ElRotation3D aRM2toCam =  mVDpl[aK]->mCam2->Orient();

       ElRotation3D aLocM2toM1 =  aRM1toCam.inv() * aRM2toCam;
       mRM2toM1  = mRM2toM1 + aLocM2toM1.Mat();

       if (mShow>=2)
       {
          std::cout << "TETA : " << aLocM2toM1.teta01() << " " 
                                 << aLocM2toM1.teta02() << " " 
                                 << aLocM2toM1.teta12()  << "\n";
       }
   }
   mRM2toM1 = mRM2toM1 * (1.0/double(mVDpl.size()));
   mRM2toM1  = NearestRotation(mRM2toM1);
}

void cAppliMorito::InitScTr2to1()
{
   Pt3dr aCdg1(0,0,0);
   Pt3dr aCdg2(0,0,0);
   for (int aK = 0 ; aK<int(mVDpl.size()) ; aK++)
   {
       Pt3dr aP1 =  mVDpl[aK]->mCam1->PseudoOpticalCenter();
       Pt3dr aP2 =  mRM2toM1 * mVDpl[aK]->mCam2->PseudoOpticalCenter();

       mVP1.push_back(aP1);
       mVP2.push_back(aP2);

       aCdg1 = aCdg1 + aP1;
       aCdg2 = aCdg2 + aP2;
   }

   aCdg1  = aCdg1 / double(mVDpl.size());
   aCdg2  = aCdg2 / double(mVDpl.size());

   double aSomD1 = 0;
   double aSomD2 = 0;

   for (int aK = 0 ; aK<int(mVDpl.size()) ; aK++)
   {
       double aD1 = euclid(mVP1[aK]-aCdg1);
       double aD2 = euclid(mVP2[aK]-aCdg2);
       if (mShow>=2)
       {
           std::cout << "Ratio = " << aD1 / aD2 << " D1 " << aD1 << "\n";
       }

       aSomD1 += aD1;
       aSomD2 += aD2;
   }
   aSomD1 /= mVDpl.size();
   aSomD2 /= mVDpl.size();

   mSc2to1 = aSomD1 / aSomD2;
   if (mShow>=2)
   {
      std::cout << "RGLOB " << mSc2to1 << "\n";
      std::cout << "RRGLOBBBb = " << aSomD1 << " " << aSomD2 << "\n";
   }
   mTr2to1 = aCdg1  - aCdg2 * mSc2to1;
}


void cAppliMorito::ComputNewRot2()
{

    //   (P1-Cd1) = (P2- Cdg2) *aRatio
    //  aTr + aP2 * aRatio
    //      aR2to1 * aRMtoC2.Mat()  =  aRMtoC1.Mat();

   
   for (int aK = 0 ; aK<int(mVDpl.size()) ; aK++)
   {
       ElRotation3D aRM1toCam =  mVDpl[aK]->mCam1->Orient();
       ElRotation3D aRM2toCam =  mVDpl[aK]->mCam2->Orient();
       Pt3dr  aC1 =  mVP1[aK];
       Pt3dr  aC2 =  mVP2[aK];//  Pt3dr aP2 =  aRM2toM1 * mVDpl[aK]->mCam2->PseudoOpticalCenter();

    //  => M1toC = M2toC * aM2toM1-1
       ElMatrix<double> aDifM =  aRM2toCam.Mat() * mRM2toM1.transpose() - aRM1toCam.Mat();
       Pt3dr aDifP =   aC1 - ( aC2*mSc2to1+mTr2to1);

       std::cout << "RESIDU R= " << aDifM.L2() << " " << euclid(aDifP) << "\n";
   }

   for 
   (
      std::map<std::string,cOriMorito>::iterator itO=mOrients.begin();
      itO != mOrients.end();
      itO++
   )
   {
        CamStenope * aCam2 = itO->second.mCam2;
        if (aCam2)
        {
            ElRotation3D aRM2toCam =  aCam2->Orient();
            ElMatrix<double>  aRM1toCam =  aRM2toCam.Mat() * mRM2toM1.transpose();
            Pt3dr aC2 = aRM2toCam.ImRecAff(Pt3dr(0,0,0));

            Pt3dr aC1 =  (mRM2toM1*aC2)*mSc2to1+mTr2to1;

            CamStenope * aCam1 = itO->second.mCam1;
            // Pour les doublons on fait une moyenne, sachant que c'est M2 qui sera sauve en priorite
            if (aCam1)
            {
                aC1 = (aC1+ aCam1->PseudoOpticalCenter()) / 2.0;
                aRM1toCam = NearestRotation((aRM1toCam + aCam1->Orient().Mat())*0.5);
            }
            ElRotation3D  aCamToM1 (aC1,aRM1toCam.transpose(),true);


            aCam2->SetOrientation(aCamToM1.inv());
            if (aCam1 && (mShow>=1))
            {
                ElRotation3D aRM2toCam =  aCam2->Orient();
                ElRotation3D aRM1toCam =  aCam1->Orient();

                ElMatrix<double> aDifM =  aRM2toCam.Mat() - aRM1toCam.Mat();
                double aDMatr = aDifM.L2();
                double aDistP = euclid(aCam2->PseudoOpticalCenter()-aCam1->PseudoOpticalCenter());
                std::cout << itO->second.mName << " DMatr "  << sqrt(aDMatr) << " DPt " << aDistP << "\n";
            }
        }
   }
}

void cAppliMorito::SauvCalib(const std::string & anOri)
{
    ELISE_fp::CpFile
    (
          DirOfFile(anOri) + "AutoCal*.xml",
          mDirOutGlob
    );
}



void cAppliMorito::Sauv()
{
   ELISE_fp::MkDir(mDirOutGlob);
   for 
   (
      std::map<std::string,cOriMorito>::iterator itO=mOrients.begin();
      itO != mOrients.end();
      itO++
   )
   {
      cOriMorito & anOM =  itO->second;
      CamStenope * aCam = anOM.mCam2 ? anOM.mCam2 : anOM.mCam1;
      cOrientationConique  anOC = aCam->StdExportCalibGlob();
      cOrientationConique  anOCInit = StdGetFromPCP(anOM.mNameFull,OrientationConique);

      anOCInit.Externe() =  anOC.Externe();
      anOCInit.Verif() =  anOC.Verif();

      if (anOCInit.FileInterne().IsInit())
      {
          anOCInit.FileInterne().SetVal(mDirOutLoc + NameWithoutDir(anOCInit.FileInterne().Val()));
      }

      std::string  aName = mDirOutGlob+anOM.mName;
      MakeFileXML(anOCInit,aName);
   }

   SauvCalib(mOri1);
   SauvCalib(mOri2);
}



// =============  ::  ===================================

int Morito_main(int argc,char ** argv)
{
    cAppliMorito anAppli(argc,argv);

    BanniereMorito();
    return EXIT_SUCCESS;
}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
