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

#include "Apero.h"

#define DEBUG 0

namespace NS_ParamApero
{

class cLiaisORPO
{
     public :
       cLiaisORPO(const cPowPointLiaisons &,cPoseCam * aCam1,cPoseCam *aCam2);
       ElPackHomologue& Pack();
       double Residu();

       void SetOrCam();
       void SetProf();
       
     private :
       ElPackHomologue       mPack;
       std::vector<double>   mProfs;
       cPoseCam *      mPose1;
       cPoseCam *      mPose2;
       const cPowPointLiaisons * mPPL;
       CamStenopeIdeale  mCam1;
       CamStenopeIdeale  mCam2;
       bool              mProfCalc;
};

class cOneRotPowelOptimize
{
     public :
         cOneRotPowelOptimize
         (
             cAppliApero &             anAppli,
             const cOptimizationPowel& anOpt,
             cPoseCam * aCam,
             int        I0,
             eTypeContraintePoseCamera aContr
         );
         void InitPts();

         ElRotation3D  Param2Rot(const double * aP);  // Parametre globaux, il faut decaler I0
         void AvantEval(const double * aP);

         double  Residu();
         int     NbMes() const;
     private :
         double  ResiduLiaison();
         cAppliApero *              mAppli;
         const cOptimizationPowel * mOP;
         cPoseCam * mCam;
         int        mI0;
         eTypeContraintePoseCamera  mContr;
         ElRotation3D mRot0;
         // ElRotation3D mCurRot;
         Pt3dr mU;
         double mNorm;
         Pt3dr mV;
         Pt3dr mW;
         ElMatrix<double> mMatr;
         std::list<cLiaisORPO>   mLIais;
         int                     mNbMes;
        
};


class cPowelOptimize : public FoncNVarND<double>
{
     public :
          cPowelOptimize (int aNbTot);
          ~cPowelOptimize();
          std::list<cOneRotPowelOptimize> & Rots() ;
          void AvantEval(const double *);
     private :
          double ValFNV(const double *);


          std::list<cOneRotPowelOptimize>  mRots;
};


/*******************************************/
/*                                         */
/*            cLiaisORPO                   */
/*                                         */
/*******************************************/

static tParamAFocal aNoPAF;
cLiaisORPO::cLiaisORPO
(
    const cPowPointLiaisons  &aPPL,
    cPoseCam * aPose1,
    cPoseCam * aPose2
) :
   mPose1 (aPose1),
   mPose2 (aPose2),
   mPPL   (&aPPL),
   mCam1  (true,1.0,Pt2dr(0.0,0.0),aNoPAF),
   mCam2  (true,1.0,Pt2dr(0.0,0.0),aNoPAF),
   mProfCalc (false)
{
   SetOrCam();
}


ElPackHomologue& cLiaisORPO::Pack()
{
   return mPack;
}

void cLiaisORPO::SetOrCam()
{
    mCam1.SetOrientation(mPose1->CurRot().inv());
    mCam2.SetOrientation(mPose2->CurRot().inv());
}

void cLiaisORPO::SetProf()
{
   mProfs.clear();
   Pt3dr aC1 = mCam1.PseudoOpticalCenter();
   for(ElPackHomologue::iterator itP=mPack.begin();itP!=mPack.end();itP++)
   {
       Pt3dr aPTer = mCam1.PseudoInter(itP->P1(),mCam2,itP->P2());
       mProfs.push_back
       (
           scal(aPTer-aC1,mCam1.F2toDirRayonR3(itP->P1()))
       );
   }
}

double cLiaisORPO::Residu()
{
    SetOrCam();
    if (! mProfCalc)
    {
       mProfCalc = true;
       SetProf();
    }

    // ElRotation3D aC1toM = mCam1.Orientation().inv();
    // ElRotation3D aMtoC2 = mCam2.Orientation();

    ElRotation3D  aC1toC2 =  mCam2.Orient() *  mCam1.Orient().inv();

    Pt3dr aBaseInC2 = aC1toC2.ImAff(Pt3dr(0,0,0));

    ElMatrix<double> aMat =  aC1toC2.Mat() ;

    Pt3dr aC0,aC1,aC2;
    aMat.GetCol(0,aC0);
    aMat.GetCol(1,aC1);
    aMat.GetCol(2,aC2);


    double sDist=0.0;
    int aK = 0;

    for (ElPackHomologue::const_iterator itP =mPack.begin() ; itP!=mPack.end(); itP++)
    {
       double aPds = itP->Pds();
       double aLambda =  mProfs.at(aK);

        Pt2dr aPIm1 = itP->P1();
        Pt3dr aRay1 = aC0 * aPIm1.x + aC1 * aPIm1.y + aC2;


        Pt3dr aPTerA = aBaseInC2 + aRay1 * (aLambda *0.99);
        Pt3dr aPTerB = aBaseInC2 + aRay1 * (aLambda *1.01);

        Pt2dr aProjA(aPTerA.x/aPTerA.z,aPTerA.y/aPTerA.z);
        Pt2dr aProjB(aPTerB.x/aPTerB.z,aPTerB.y/aPTerB.z);

         Pt2dr aVAB  (aProjB.y-aProjA.y,aProjA.x-aProjB.x);

         double aD2 =  ElAbs(scal(aVAB,aProjA-itP->P2()) / euclid(aVAB));
         sDist += aPds*aD2;

         aK++;
    }

    if (DEBUG)
    {
        std::cout << sDist/mPack.size() << "\n"; // getchar();
    }

    return sDist * mPPL->Pds().Val();
}



/*******************************************/
/*                                         */
/*       cOneRotPowelOptimize              */
/*                                         */
/*******************************************/

cOneRotPowelOptimize::cOneRotPowelOptimize
(
    cAppliApero &               anAppli,
    const cOptimizationPowel&   anOpt,
    cPoseCam *                  aCam,
    int                         anI0,
    eTypeContraintePoseCamera   aContr
) :
   mAppli (&anAppli),
   mOP    (&anOpt),
   mCam   (aCam),
   mI0    (anI0),
   mContr (aContr),
   mRot0  (aCam->CurRot()),
   // mCurRot (mRot0),
   mU     (mRot0.tr()),
   mNorm  (euclid(mU)),
   mMatr  (mRot0.Mat()),
   mNbMes (0)
{
    mV  =  OneDirOrtho(mU);  // mV est unitaire a ce stade
    mW =  mU ^ mV;
    mV = mV * mNorm;
}



int   cOneRotPowelOptimize::NbMes() const
{
   return mNbMes;
}



void cOneRotPowelOptimize::InitPts()
{
    for 
    (
       std::list<cPowPointLiaisons>::const_iterator itP=mOP->PowPointLiaisons().begin();
       itP!=mOP->PowPointLiaisons().end();
       itP++
    )
    {
        std::list<cPoseCam *> aLCam = mAppli->ListCamInitAvecPtCom(itP->Id(),mCam);
    
    // Pour l'insntant la selec se fait par simple tirage aleatoire
        int aNbPtsTot = 0;
        for (std::list<cPoseCam *>::iterator itC=aLCam.begin();itC!=aLCam.end();itC++)
        {

             ElPackHomologue aPack ;
             mAppli->InitPack(itP->Id(),aPack,mCam->Name(),(*itC)->Name());
             aNbPtsTot += aPack.size();
        }

        int aNbCible = itP->NbTot();
        for (std::list<cPoseCam *>::iterator itC=aLCam.begin();itC!=aLCam.end();itC++)
        {
             mLIais.push_back(cLiaisORPO(*itP,mCam,*itC));
             ElPackHomologue aPack ;
             mAppli->InitPackPhgrm
             (
                  itP->Id(),aPack,
                  mCam->Name(),&mCam->Calib()->CamInit(),
                  (*itC)->Name(),&(*itC)->Calib()->CamInit()
             );
             for (ElPackHomologue::iterator itPt=aPack.begin(); itPt!=aPack.end() ; itPt++)
             {
                 if (NRrandom3() < double(aNbCible)/aNbPtsTot)
                 {
                     mLIais.back().Pack().Cple_Add(itPt->ToCple());
                     mNbMes++;
                     aNbCible--;
                 }
                 aNbPtsTot--;
             }
        }
    }
}


ElRotation3D  cOneRotPowelOptimize::Param2Rot(const double * aP)
{
if (DEBUG)
{
for (int aK=0 ; aK<5 ; aK++)
    std::cout << aP[aK] << " ";
std::cout <<  "\n";
}
   if (mContr==ePoseFigee)
      return ElRotation3D(mU,mMatr);

   aP+= mI0;
   double aF = 0.05;

   Pt3dr aTr =    mU*cos(sqrt(ElSquare(aP[0]*aF)+ElSquare(aP[1]*aF)))
                 + mV*sin(aP[0]*aF)
                 + mW*sin(aP[1]*aF);

    aTr =  vunit(aTr) ;

   double aN = mNorm;
   if (mContr==ePoseLibre)
      aN *= 1+aF*aP[5];

   return ElRotation3D
          (
              vunit(aTr) * aN,
              mMatr * ElMatrix<double>::Rotation(aP[2]*aF,aP[3]*aF,aP[4]*aF)
          );
}

void cOneRotPowelOptimize::AvantEval(const double * aP)
{
   mCam->SetCurRot(Param2Rot(aP));
}

double   cOneRotPowelOptimize::ResiduLiaison()
{
    double aRes=0;
    for 
    (
       std::list<cLiaisORPO>::iterator itL=mLIais.begin();
       itL!=mLIais.end();
       itL++
    )
    {
       aRes+=itL->Residu();
    }

    return aRes;
}

double   cOneRotPowelOptimize::Residu()
{
    return 
            ResiduLiaison()
    ;
}

/*******************************************/
/*                                         */
/*            cPowelOptimize               */
/*                                         */
/*******************************************/

cPowelOptimize::cPowelOptimize(int aNbTot) :
    FoncNVarND<double>(aNbTot)
{
}

cPowelOptimize::~cPowelOptimize()
{
}

std::list<cOneRotPowelOptimize> & cPowelOptimize::Rots()
{
   return mRots;
}

void cPowelOptimize::AvantEval(const double * aP)
{
   for 
   (
      std::list<cOneRotPowelOptimize>::iterator itORPO=mRots.begin();
      itORPO!=mRots.end();
      itORPO++
   )
   {
       itORPO->AvantEval(aP);
   }
}

double cPowelOptimize::ValFNV(double const* aP)
{
   AvantEval(aP);
   double aRes = 0;

   for 
   (
      std::list<cOneRotPowelOptimize>::iterator itORPO=mRots.begin();
      itORPO!=mRots.end();
      itORPO++
   )
   {
       aRes += itORPO->Residu();
   }
   return aRes;
}


/*******************************************/
/*                                         */
/*          cAppliApero                    */
/*                                         */
/*******************************************/

int  NbDegOfLib(eTypeContraintePoseCamera aContr)
{
     switch(aContr)
     {
         case ePoseLibre :      return 6;
         case ePoseFigee :      return 0;
         case ePoseBaseNormee : return 5;
         case ePoseVraieBaseNormee : return 5;
         case eCentreFige : return 3;
     }
     ELISE_ASSERT(false,"NbDegOfLib");
     return 0;
}

void cAppliApero::PowelOptimize
     (
           const cOptimizationPowel& anOpt,
           const std::vector<cPoseCam *>&  aCams,
           const std::vector<eTypeContraintePoseCamera>&   aDegOfLib
     )
{
   ELISE_ASSERT
   (
     aCams.size()==aDegOfLib.size(),
     "Incoh in cAppliApero::PowelOptimize"
   );

   std::vector<double> aParam00;
   int aNbDegTot = 0;
   for (int aK=0; aK<int(aCams.size()) ; aK++)
   {
       int aDeg = NbDegOfLib(aDegOfLib[aK]);
       aNbDegTot += aDeg;
       for (int aD=0 ; aD<aDeg ; aD++)
           if (DEBUG)
             aParam00.push_back(0.5-aD%2);
           else
             aParam00.push_back(0);
   }

   cPowelOptimize aPOpt(aNbDegTot);
   aNbDegTot = 0;
   int aNbMes = 0;
   for (int aK=0; aK<int(aCams.size()) ; aK++)
   {
       aPOpt.Rots().push_back
       (
          cOneRotPowelOptimize
          (
               *this,anOpt,aCams[aK],aNbDegTot,aDegOfLib[aK]
          )
       );
       aPOpt.Rots().back().InitPts();
       aNbMes += aPOpt.Rots().back().NbMes();
       aNbDegTot += NbDegOfLib(aDegOfLib[aK]);
   }
   aPOpt.powel(&aParam00[0],1e-7*aNbMes,200);
   aPOpt.AvantEval(&aParam00[0]);

   if (DEBUG)
      getchar();

}


};

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
