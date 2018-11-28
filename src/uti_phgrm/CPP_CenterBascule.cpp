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
#include "math.h"

#define DEF_OFSET -12349876

int CentreBascule_main(int argc,char ** argv)
{
    NoInit = "NoP1P2";
    aNoPt = Pt2dr(123456,-8765432);

    // MemoArg(argc,argv);
    MMD_InitArgcArgv(argc,argv);
    std::string  aDir,aPat,aFullDir;


    std::string AeroOut;
    std::string AeroIn;
    //std::string DicoPts;
    std::string BDC;
    bool ModeL1 = false;
    bool CalcV   = false;


    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aFullDir,"Full name (Dir+Pat)", eSAM_IsPatFile )
                    << EAMC(AeroIn,"Orientation in", eSAM_IsExistDirOri)
                    << EAMC(BDC,"Localization of Information on Centers", eSAM_IsExistDirOri)
                    << EAMC(AeroOut,"Orientation out", eSAM_IsOutputDirOri),
    LArgMain()
                    <<  EAM(ModeL1,"L1",true,"L1 minimization vs L2; (Def=false)", eSAM_IsBool)
                    <<  EAM(CalcV,"CalcV",true,"Use speed to estimate time delay (Def=false)", eSAM_IsBool)
    );

    if (!MMVisualMode)
    {
#if (ELISE_windows)
        replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
#endif
        SplitDirAndFile(aDir,aPat,aFullDir);
        StdCorrecNameOrient(AeroIn,aDir);
        StdCorrecNameOrient(BDC,aDir);



        std::string aCom =   MMDir() + std::string("bin/mm3d Apero ")
                + XML_MM_File("Apero-Center-Bascule.xml")
                + std::string(" DirectoryChantier=") +aDir +  std::string(" ")
                + std::string(" +PatternAllIm=") + QUOTE(aPat) + std::string(" ")
                + std::string(" +AeroIn=") + AeroIn
                + std::string(" +AeroOut=") +  AeroOut
                + std::string(" +BDDC=") +  BDC
                ;

        if (ModeL1)
        {
            aCom = aCom+ std::string(" +L2Basc=") + ToString(!ModeL1);
        }

        if (CalcV)
        {
            aCom = aCom+ std::string(" +CalcV=") + ToString(CalcV);
        }


        std::cout << "Com = " << aCom << "\n";
        int aRes = system_call(aCom.c_str());

        return aRes;
    }
    else return EXIT_SUCCESS;
}


/*********************************************************************/
/*                                                                   */
/*                                                                   */
/*                                                                   */
/*********************************************************************/

class cAppli_CmpOriCam : public cAppliWithSetImage
{
    public :

        cAppli_CmpOriCam(int argc, char** argv);

        std::string mPat,mOri1,mOri2;
        std::string mDirOri2;
        std::string mXmlG;
        std::string mCSV = "CSVEachPose.csv";
        std::string mPly;
        cInterfChantierNameManipulateur * mICNM2;
};

cAppli_CmpOriCam::cAppli_CmpOriCam(int argc, char** argv) :
    cAppliWithSetImage(argc-1,argv+1,0)
{
   Pt3di aColXY(255,0,0);
   Pt3di aColZ(0,0,255);
   Pt3di aColOri(255,255,0);
   double aScaleC;
   double aScaleO;
   double aF;

   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(mPat,"Full Name (Dir+Pattern)",eSAM_IsPatFile)
                    << EAMC(mOri1,"Orientation 1", eSAM_IsExistDirOri)
                    << EAMC(mOri2,"Orientation 2"),
        LArgMain()  << EAM(mDirOri2,"DirOri2", true,"Orientation 2")
					<< EAM(mXmlG,"XmlG",true,"Generate Xml")
                    << EAM(mCSV,"CSV",true,"Generate detail CSV (excel compatible) for each image")
                    << EAM(mPly,"Ply",true,"Generate .ply File")
                    << EAM(aColXY,"ColXY", true, "color for XY component of .ply")
                    << EAM(aColZ,"ColZ", true, "color for Z component of .ply")
                    << EAM(aColOri,"ColOri",true,"color for orientation component of .ply")
                    << EAM(aScaleC,"ScaleC",true,"Scale for camera center difference, the center diff is displayed when this option is activated")
                    << EAM(aScaleO,"ScaleO",true,"Scale for camera orientation difference, the ori diff is displayed when this option is activated")
                    << EAM(aF,"F",true,"approximate value of focal length in (m), Def=0.03875m for Camlight")

   );

   mICNM2 = mEASF.mICNM;
   if (EAMIsInit(&mDirOri2))
   {
       mICNM2 = cInterfChantierNameManipulateur::BasicAlloc(mDirOri2);
   }
/*
*/


   mICNM2->CorrecNameOrient(mOri2);


   double aSomDC = 0;
   double aSomDM = 0;

   bool isCSV = false;
   ofstream mCSVContent;
   if (EAMIsInit(&mCSV))
   {
     mCSVContent.open(mCSV);
     isCSV = true;
     mCSVContent<< "Img,X1,Y1,Z1,dX,dY,dZ,dXY,dXYZ\n";
   }
   cPlyCloud aPlyC, aPlyO;

   for (int aK=0 ; aK<int(mVSoms.size()) ; aK++)
   {
       cImaMM * anIm = mVSoms[aK]->attr().mIma;
       CamStenope * aCam1 =  anIm->CamSNN();
       CamStenope * aCam2 = mICNM2->StdCamStenOfNames(anIm->mNameIm,mOri2);

       Pt3dr aC1 = aCam1->VraiOpticalCenter();
       Pt3dr aC2 = aCam2->VraiOpticalCenter();

       if (EAMIsInit(&aScaleO))
       {
           ElSeg3D aRay2 = aCam2->Capteur2RayTer(aCam2->PP());
           double prof = - aCam1->Focale()/5120*32.7/1000;

           Pt2dr aP = aCam1->Ter2Capteur(aRay2.P1()-(aC2-aC1));

           Pt3dr aPP3D = aCam1->ImEtProf2Terrain(aCam1->PP(),prof);

           Pt3dr aP3D = aCam1->ImEtProf2Terrain(aP,prof);

           aPlyO.AddSeg(aColOri,aPP3D,aP3D+(aPP3D-aP3D)*1000*aScaleO,10000);
       }
       ElRotation3D aR1= aCam1->Orient();
       ElRotation3D aR2= aCam2->Orient();

       double aDC = euclid(aC1-aC2);
       double aDCXY = euclid(Pt2dr(aC1.x,aC1.y)-Pt2dr(aC2.x,aC2.y));
       double aDM = aR1.Mat().L2(aR2.Mat());
       aSomDC += aDC;
       aSomDM += aDM;
       std::cout << anIm->mNameIm << "\n";

       if (isCSV)
       {
           mCSVContent << anIm->mNameIm <<","<< ToString(aC1.x) << "," << ToString(aC1.y) << "," << ToString(aC1.z) << "," << ToString(abs(aC1.x - aC2.x)) << "," << ToString(abs(aC1.y - aC2.y)) << "," <<ToString(aC1.z - aC2.z) << "," <<  ToString(aDCXY) << "," << ToString(aDC);
           mCSVContent << "\n";
       }

       if(EAMIsInit(&aScaleC))
       {
           aPlyC.AddSeg(aColXY,aC1,Pt3dr(aC2.x+(aC2.x-aC1.x)*100000*aScaleC,aC2.y+(aC2.y-aC1.y)*100000*aScaleC,aC1.z),10000);
           aPlyC.AddSeg(aColZ,aC1,Pt3dr(aC1.x,aC1.y,aC2.z+(aC2.z-aC1.z)*100000*aScaleC),10000);
       }
   }
	
   std::cout << "Aver;  DistCenter= " << aSomDC/mVSoms.size()
             << " DistMatrix= " << aSomDM/mVSoms.size()
             << "\n";
   if(mXmlG!="")
   {
	   cXmlTNR_TestOriReport aCmpOri;
	   aCmpOri.OriName() = mOri2;
	   aCmpOri.DistCenter() = aSomDC/mVSoms.size();
	   aCmpOri.DistMatrix() = aSomDM/mVSoms.size();
	   if(aSomDC/mVSoms.size()==0&&aSomDM/mVSoms.size()==0)
	   {
		   aCmpOri.TestOriDiff() = true;
	   }
	   else{aCmpOri.TestOriDiff() = false;}
	   MakeFileXML(aCmpOri, mXmlG);
   }

   if (isCSV)
   {
       mCSVContent.close();
   }

   if(EAMIsInit(&mPly))
   {
       aPlyC.PutFile(mPly.substr(0,mPly.size()-4)+"_Center.ply");
       aPlyO.PutFile(mPly.substr(0,mPly.size()-4)+"_Orientation.ply");
   }
}

int CPP_CmpOriCam_main(int argc, char** argv)
{
    cAppli_CmpOriCam anApplu(argc,argv);

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
