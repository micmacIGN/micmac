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
#include "Apero.h"


cArg_UPL::cArg_UPL(const cXmlSLM_RappelOnPt * aRop) :
    mRop (aRop)
{
}


    /************************************************/
    /*                                              */
    /*                                              */
    /*                                              */
    /************************************************/

void cAppliApero::AddRappelOnAngle(const cRappelOnAngles & aRAO,double aMult,cStatObs & aSO)
{
    const cParamForceRappel & aPFR = aRAO.ParamF();
     // std::cout << "---------LVM::INC---- " << aPFR.Incertitude() / aMult << "\n";
    for (int aKP=0;aKP<int(mVecPose.size()) ; aKP++)
    {
        cPoseCam & aPC = *(mVecPose[aKP]);
        if (aRAO.ParamF().PatternNameApply()->Match(aPC.Name()))
        {
            const std::vector<double>&  aVI = aPFR.Incertitude();
            const std::vector<int> & aVA = aRAO.TetaApply();
            ELISE_ASSERT(aVI.size()<=aVA.size(),"To many Incertitude in RappelOnAngles");
            double aSomPond = 0.0;
            for (int aKA=0 ; aKA<int(aVA.size()) ; aKA++)
            {
               int aKI = ElMin(aKA,int(aVI.size()-1));
               aSomPond += aPC.RF().AddRappelOnAngles
                           (
                               aPFR.OnCur().ValWithDef(true),
                               aVA[aKA],
                               aPFR.Incertitude()[aKI] / aMult,
                               aSO.AddEq()
                           );
            }
            aSO.AddSEP(aSomPond);
        }
    }
}

void cAppliApero::AddRappelOnCentre(const cRappelOnCentres & aRAC,double aMultInit,cStatObs & aSO)
{
    const cParamForceRappel & aPFR = aRAC.ParamF();
     // std::cout << "---------LVM::INC---- " << aPFR.Incertitude() / aMult << "\n";
    for (int aKP=0;aKP<int(mVecPose.size()) ; aKP++)
    {
        cPoseCam & aPC = *(mVecPose[aKP]);
        if (    (aRAC.ParamF().PatternNameApply()->Match(aPC.Name()))
             && ((! aRAC.OnlyWhenNoCentreInit().Val()) || (!aPC.LastItereHasUsedObsOnCentre()))
           )
        {
            const std::vector<double>&  aVI = aPFR.Incertitude();
            ELISE_ASSERT(aVI.size()<=3,"Bas size Incertitude in cAppliApero::AddRappelOnCentre");
            Pt3dr anI;
            if (aVI.size()==1) anI = Pt3dr(aVI[0],aVI[0],aVI[0]);
            else if (aVI.size()==2) anI = Pt3dr(aVI[0],aVI[0],aVI[1]);
            else if (aVI.size()==3) anI = Pt3dr(aVI[0],aVI[1],aVI[2]);
            else
            {
                ELISE_ASSERT(false,"Size in cAppliApero::AddRappelOnCentre");
            }

            double  aMult = aMultInit;
            double aProf;
            int OkProf;
            aProf = aPC.GetProfDyn(OkProf);
            if (OkProf)
            {
               aMult *= (10 / aProf);
            }


            aPC.RF().AddRappelOnCentre(aPFR.OnCur().ValWithDef(true),anI/aMult,aSO.AddEq());
        }
    }
}

void cAppliApero::AddRappelOnIntrinseque(const cRappelOnIntrinseque & aROI,double aMultInit,cStatObs & aSO)
{
    for (tDiCal::iterator itC=mDicoCalib.begin(); itC!=mDicoCalib.end() ; itC++)
    {
        if (  aROI.ParamF().PatternNameApply()->Match(itC->first))
        {
             itC->second->AddViscosite(aROI.ParamF().Incertitude());
        }
    }
}

void cAppliApero::AddOneLevenbergMarkard
     (
        const cSectionLevenbergMarkard * aSLM,
        double                           aMult,
        cStatObs & aSO
     )
{
   if (! aSLM) return;
   if (aMult<=0 ) return;


   for 
   (
        std::list<cRappelOnAngles>::const_iterator itR=aSLM->RappelOnAngles().begin();
        itR!=aSLM->RappelOnAngles().end();
        itR++
   )
   {
         AddRappelOnAngle(*itR,aMult,aSO);
   }

   for 
   (
        std::list<cRappelOnCentres>::const_iterator itR=aSLM->RappelOnCentres().begin();
        itR!=aSLM->RappelOnCentres().end();
        itR++
   )
   {
         AddRappelOnCentre(*itR,aMult,aSO);
   }

   for 
   (
        std::list<cRappelOnIntrinseque>::const_iterator itR=aSLM->RappelOnIntrinseque().begin();
        itR!=aSLM->RappelOnIntrinseque().end();
        itR++
   )
   {
         AddRappelOnIntrinseque(*itR,aMult,aSO);
   }



}
void cAppliApero::AddLevenbergMarkard(cStatObs & aSO)
{
    AddOneLevenbergMarkard(mCurSLMGlob,mMulSLMGlob,aSO);
    AddOneLevenbergMarkard(mCurSLMEtape,mMulSLMEtape,aSO);
    AddOneLevenbergMarkard(mCurSLMIter,mMulSLMIter,aSO);
}

void  cAppliApero::InitLVM
      (
           const cSectionLevenbergMarkard*&  aPtr,
           const cTplValGesInit<cSectionLevenbergMarkard> & anOpt,
           double & aMult,
           const cTplValGesInit<double> & aOpMul

       )
{
   if (anOpt.IsInit())
   {
     aPtr = anOpt.PtrVal();
     const cXmlSLM_RappelOnPt * aRop = aPtr->XmlSLM_RappelOnPt().PtrVal();
     if (aRop)
     {
        mXmlSMLRop  = aRop;
     }
   }

   if (aOpMul.IsInit())
     aMult = aOpMul.Val();
}


void cAppliApero:: UpdateMul(double & aMult,double aNewV,bool aModeMin)
{
   if (aModeMin) 
      ElSetMin(aMult,aNewV);
   else
       aMult = aNewV;
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
