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

#include "general/all.h"
#include "private/all.h"
#include <algorithm>
#include "XML_GEN/all.h"
using namespace NS_ParamChantierPhotogram;

/*
No Param
EVAL = 2.32248

0.899008
Focale =649.065 PP [171.475,130.802]
Foc & PP 
EVAL = 1.1905

0.290786
Focale =663.233 PP [171.876,113.518]
DR1 



EVAL = 0.384176
0.276721
Focale =660.594 PP [174.726,111.283]
DR2 


EVAL = 0.392687
0.26562
Focale =661.39 PP [166.191,118.431]
Cdist/PP 

EVAL = 0.42369
*/


/*
     Hypothese pour utiliser la calibration croisee :

        - l'image de REF a  ete mise a la meme echelle,
	que l'image a calibrer, soit aRatioRef la valeur;

	- image de test a echelle 1 ;

*/

/****************************************************/
/*                                                  */
/*        cAppliCalibCroisee                        */
/*                                                  */
/****************************************************/

struct  cHomCC
{
     Pt3dr mRefDir;
     Pt2dr mPImTest;
     int   mNum;
};


class cAppliCalibCroisee
{
     public :
        cAppliCalibCroisee
	(
            double aRatioRef,
            const std::string & aNameCalibRef,
	    bool  aRefIsFirst,
            const std::string & aNameHoms,
	    Pt2dr aSzTest,
	    bool  aModeL1,
	    bool  SepCDPP
	);
     private :
	 void Sauv(const std::string);
         void Show();


	 double Eval(int aNbItere);

         void GetRefAndTest(const ElCplePtsHomologues &,Pt2dr &aPRef,Pt2dr & aPTest);
	 double OneItere(int aNumOut=-1);
	 double N_Itere(int aNbItere,int aNumOut=-1);
         cDbleGrid * mGr;
	 ElPackHomologue mPackH;
	 cSetEqFormelles mSet;

	 Pt2dr   mPPInit;
	 double   mFocaleInit;
	 // Par "homogeneisation" on met tout en pointeur
	 cParamIFDistStdPhgr *     mPIF;
	 cEqCalibCroisee *         mEqCC;
	 cRotationFormelle *       mRotF;
	 bool                      mRefIsFirst;
	 std::list<cHomCC>         mLHCC;
	 double                    mRatioRef;
	 Pt2di                     mSzTest;
	 bool                      mSimulation;
	 double                    mEcartGlob;
	 std::string               mDirH;
	 bool                      mSepCDPP;
};


void cAppliCalibCroisee::Sauv(const std::string aName)
{
     CamStenope * aCS = mPIF->CurPIF();
     cCalibrationInternConique anOC = aCS->ExportCalibInterne2XmlStruct(mSzTest);
     MakeFileXML(anOC,mDirH+aName);
}


void cAppliCalibCroisee::GetRefAndTest
     (
          const ElCplePtsHomologues & aCple,
          Pt2dr &aPRef,
          Pt2dr & aPTest
     )
{
   aPRef = aCple.P1();
   aPTest = aCple.P2();

   if (! mRefIsFirst)
      ElSwap(aPRef,aPTest);
}

double cAppliCalibCroisee::N_Itere(int aNbItere,int aNumOut)
{
   double aRes= 1e30;
   for (int aK= 0 ; aK<aNbItere ; aK++)
       aRes = OneItere(aNumOut);

  return aRes;
}


double cAppliCalibCroisee::Eval(int aNbItere)
{
    double aSE = 0;
    for (int aNum =0 ; aNum<int(mLHCC.size()) ; aNum++)
    {
	double anE=1e20;
	for (int iT=0 ;iT<aNbItere ; iT++)
	{
	    anE = OneItere(aNum); //  * mFocaleInit;
	    // std::cout <<  "                 " << anE << "\n";
	}
	aSE += anE;
    }
    double aRes =  aSE / mLHCC.size();
    std::cout << "EVAL = " << aRes << "\n";
    return aRes;
}


double cAppliCalibCroisee::OneItere(int aNumOut)
{

   double aS1=0, aSD=0;
   double aS1_o=0, aSD_o=0;



   mSet.AddContrainte(mPIF->StdContraintes());
   mSet.AddContrainte(mRotF->StdContraintes());
   for
   (
       std::list<cHomCC>::const_iterator itH =mLHCC.begin();
       itH !=mLHCC.end();
       itH++
   )
   {
      bool isOut = (itH->mNum == aNumOut );
      double aPds= isOut ? 0.0 : 1.0;
      const std::vector<REAL> & anER = 
           mEqCC->AddObservation(itH->mPImTest,itH->mRefDir,aPds);

      double aNEr  = sqrt(ElSquare(anER[0])+ElSquare(anER[1])+ElSquare(anER[2]));

      aS1 ++;
      aSD  += aNEr;
      if (isOut)
      {
         aS1_o ++;
         aSD_o  += aNEr;
      }
   }

   mEcartGlob = (aSD / aS1) ;
   mSet.SolveResetUpdate();

   return (aS1_o !=0) ? aSD_o / aS1_o : 1e30;
}

void cAppliCalibCroisee::Show()
{
   std::cout << mEcartGlob  << "\n";
   std::cout << "Focale =" << mPIF->CurFocale() << " PP " << mPIF->CurPP() << "\n";
}


cAppliCalibCroisee::cAppliCalibCroisee
(
      double                aRatioRef,
      const std::string &   aNameCalibRef,
      bool                  aRefIsFirst,
      const std::string &   aNameHoms,
      Pt2dr                 aSzTest,
      bool                  aModeL1,
      bool                  aSepCDPP
) :
  mSet (aModeL1 ? cNameSpaceEqF::eSysL1Barrodale : cNameSpaceEqF::eSysPlein,1000),
  mRefIsFirst (aRefIsFirst),
  mRatioRef   (aRatioRef),
  mSzTest     (aSzTest),
  mSepCDPP    (aSepCDPP)
{
    std::string aTmp;
    SplitDirAndFile(mDirH,aTmp,aNameHoms);
    mSimulation = false ;

    // Lecture de la calibration de reference
    {
       std::string aDirCal,aXMLCam;
       SplitDirAndFile(aDirCal,aXMLCam,aNameCalibRef);
       cDbleGrid::cXMLMode  aXmlMode;
       mGr = new cDbleGrid(aXmlMode, aDirCal,aXMLCam);
    }

    // Lecture des points homologues
    {
	cElXMLTree aTree(aNameHoms);
        mPackH = aTree.GetPackHomologues();
        std::cout << "NbPts =" << mPackH.size() << "\n";
    }


    // Initialisation des parametres intrinseques
    {
        mPPInit  = aSzTest / 2.0;
        mFocaleInit = mGr->Focale() /  mRatioRef ;

	ElDistRadiale_PolynImpair aDistRad(euclid(mPPInit),mPPInit);
	cCamStenopeModStdPhpgr aCam
	                       (
			            false,
				    mFocaleInit,
				    mPPInit,
				    cDistModStdPhpgr(aDistRad)
			       );
        mPIF = mSet.NewIntrDistStdPhgr(false,&aCam,0);
        mPIF->SetFocFree(false);
        mPIF->SetLibertePPAndCDist(false,false);
    }


    // Allocation de l'equation de calibration croisee
    {
        mEqCC = mSet.NewEqCalibCroisee(false,*mPIF);
    }

    // Allocation de la rotation
    {
        mRotF =  &(mEqCC->RotF());
    }

    mSet.SetClosed();


    Pt2dr aPPSim = mPPInit + Pt2dr(20,10);
    double aFSim = mFocaleInit * 1.05;
    ElRotation3D aRSim(Pt3dr(0,0,0),0.05,0.02,0.03);


    int aK=0;
    for
    (
       ElPackHomologue::const_iterator itP=mPackH.begin();
       itP!=mPackH.end();
       itP++
    )
    {
       Pt2dr aPRef;
       cHomCC  aH;
       aH.mNum = aK++;

       GetRefAndTest(itP->ToCple(),aPRef,aH.mPImTest);

       Pt2dr aRefPhgr = mGr->Direct(aPRef * mRatioRef);
       aH.mRefDir = PointNorm1(PZ1(aRefPhgr));

       if (mSimulation)
       {
           Pt3dr aP3Test = aRSim.ImVect(aH.mRefDir);
	   Pt2dr  aP2T = ProjStenope(aP3Test);
	   aP2T = (aP2T *aFSim) + aPPSim;
           aH.mPImTest  = aP2T;
       }

       std::cout << aPRef << " => " << aH.mRefDir << " !! "<< aRefPhgr << "\n";
       mLHCC.push_back(aH);
    }



   N_Itere(10); Show(); 
   std::cout << "No Param\n";
   Eval(5);
   getchar();

   mPIF->SetFocFree(true);
   mPIF->SetCDistPPLie();
   N_Itere(10); Show(); 
   std::cout << "Foc & PP \n";
   Eval(5);
   getchar();

   mPIF->SetDRFDegreFige(1);
   N_Itere(10); Show(); 
   Sauv("CCroisDRad1.xml");
   std::cout << "DR1 \n";
   Eval(5);
   getchar();

   if (mSepCDPP)
   {
        mPIF->SetLibertePPAndCDist(true,true);
        N_Itere(10); Show(); 
        Sauv("CCroisPCDSepDRad1.xml");
        std::cout << "PP CD LIBREEEEEEEEEEEEEEEEEE !!!!!!!!!!!!!! \n";
        Eval(5);
        getchar();
   }
   if (0)
   {
      // mPIF->SetParamAffineFree();
      mPIF->SetParam_Dec_Free();
      N_Itere(10); Show(); 
      std::cout << "Affine \n";
      Eval(5);
      getchar();
   }


   mPIF->SetDRFDegreFige(2);
   N_Itere(10); Show(); 
   Sauv("CCroisDRad2.xml");
   std::cout << "DR2 \n";
   Eval(5);
   getchar();

   mPIF->SetLibertePPAndCDist(true,true);
   N_Itere(10); Show(); 
   std::cout << "Cdist/PP \n";
   Eval(5);
}

/****************************************************/
/*                                                  */
/*            main                                  */
/*                                                  */
/****************************************************/


int main(int argc,char ** argv)
{
   double aRatio = 5.5; // Ratio dont a ete affecte l'image de reference

   /*
   std::string aNameGridRef = "/DATA2/Calibration/Pentax28mm/Polygone/"
                               "GRID_NoGrid_DRadPentax28mm_MetaDonnees.xml";
   std::string aNameHoms = "/DATA2/Calibration/Cross/MTD/"
                           "OK_Liaison_th780141_8BitsNEG3427.xml";
   */

   std::string aNameGridRef = "/mnt/data/Calib/Pentax-28mm/PolygIGN/"
                              "GRID_NoGrid_DRadPentax28mm_MetaDonnees.xml";

    std::string aNameHoms = "/mnt/data/ISPRS/NEW/Antigone/Homol/"
                            "PastisGth780029_8Bits-Gsg1l3472_Scaled_NEG.xml";
/*
    std::string aNameHoms = "/mnt/data/ISPRS/NEW/MTD/Homol/"
                            "Pastisth780141_8Bits-sg1l3427_Scaled_Neg.xml";
*/


   Pt2dr aSzTest(320,240);
   bool aModeL1 = true;
   bool aRefIsFirst= false;
   bool aSepCDPP = true;

   cAppliCalibCroisee anAppli
                      (
                          aRatio,
                          aNameGridRef,
			  aRefIsFirst,
                          aNameHoms,
                          aSzTest,
                          aModeL1, 
			  aSepCDPP
                      );
}




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
