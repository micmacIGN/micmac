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

#include "Vino.h"


#if (ELISE_X11)


/****************************************/
/*                                      */
/*          Grab Geom                   */
/*                                      */
/****************************************/

double VerifInt(const int * anInput,int aNb)
{
   return 0;
}

double  VerifInt(const double * anInput,int aNb)
{
   double aSom = 0.0;

   for (int aK=0 ; aK<aNb ; aK++)
   {
        aSom += ElAbs(anInput[aK]-round_ni(anInput[aK]));
   }
   return (aNb==0) ? 0.0 : (aSom/aNb);
}

template <class Type> void  cAppli_Vino_TplChgDyn<Type>::SetDyn(cAppli_Vino & anAppli,int * anOut,const Type * anInput,int aNb)
{
    // std::cout << " VerifInt== " << VerifInt(anInput,aNb) << "\n"; getchar();
    if (anAppli.mTabulDynIsInit)
    {
       int aMaxInd = anAppli.mTabulDyn.size() - 1;
       int * aTD = & (anAppli.mTabulDyn[0]);
       double aV0 = anAppli.mV0TabulDyn;
       double aStep = anAppli.mStepTabulDyn;
       for (int aK=0 ; aK<aNb ; aK++)
       {
           // int aInd = round_ni((anInput[aK]-mV0TabulDyn)/mStepTabulDyn);
           int aInd = round_ni((anInput[aK]-aV0)/aStep);
           aInd = ElMax(0,ElMin(aMaxInd,aInd));
           anOut[aK] = aTD[aInd];
       }
       return;
    }

   const cXml_StatVino & aStats = *(anAppli.mCurStats);

    switch (aStats.Type())
    {
          case eDynVinoModulo :
          {
              for (int aK=0 ; aK<aNb ; aK++)
                   anOut[aK] =  int(anInput[aK]) % 256;
              return;
          }

          case eDynVinoMaxMin :
          {
              
              int aV0 = aStats.IntervDyn().x; 
              int anEcart = aStats.IntervDyn().y -aV0; 
              for (int aK=0 ; aK<aNb ; aK++)
              {
                   anOut[aK] = ElMax(0,ElMin(255, int(((anInput[aK] -aV0) * 255) / anEcart)));
              }
              return;
          }

          case eDynVinoStat2 :
          {
              
              double aMoy   = aStats.Soms()[0];
              double anECT  = aStats.ECT()[0] / aStats.MulDyn();
              for (int aK=0 ; aK<aNb ; aK++)
              {
                  float aVal = (anInput[aK]-aMoy)/ anECT;
                   // anOut[aK] = 128 * (1+ aVal / (ElAbs(aVal) +0.5));
                   anOut[aK] = ElMax(0,ElMin(255,round_ni(256 * erfcc (aVal))));
              }
              return;
          }

          default :
          {
              for (int aK=0 ; aK<aNb ; aK++)
                   anOut[aK] =  anInput[aK] ;
              return;
          }
    }
}

void cAppli_Vino::ChgDyn(int * anOut,const double * anInput,int aNb) 
{
    cAppli_Vino_TplChgDyn<double>::SetDyn(*this,anOut,anInput,aNb);
    // TplChgDyn(*mCurStats,anOut,anInput,aNb);
}

void cAppli_Vino::ChgDyn(int * anOut,const int * anInput,int aNb) 
{
    cAppli_Vino_TplChgDyn<int>::SetDyn(*this,anOut,anInput,aNb);
    // TplChgDyn(*mCurStats,anOut,anInput,aNb);
}


void cAppli_Vino::InitTabulDyn()
{
   if (mCurStats==0) return;
   if (!mCurStats->IsInit()) return;


   mTabulDynIsInit = false;
   double aMoy   = mCurStats->Soms()[0];
   double anECT  = mCurStats->ECT()[0] ;

   double  anEcart = anECT * 10 ; // 10 Sigma

   mV0TabulDyn = aMoy - anEcart;
   mStepTabulDyn = anECT / (255.0 * 5);

   int aNbTabul = round_ni((anEcart/mStepTabulDyn) * 2);

   if (mCurStats->Type() == eDynVinoStat2)
   {
       mTabulDyn.clear();
       double aDiv  = anECT / mCurStats->MulDyn();
       mTabulDynIsInit = true;
       for (int aK=0 ; aK<= aNbTabul ; aK++)
       {
           double  aVal = mV0TabulDyn + aK * mStepTabulDyn;
           aVal = (aVal-aMoy)/aDiv;
           mTabulDyn.push_back(ElMax(0,ElMin(255,round_ni(256 * erfcc (aVal)))));
       }

       std::cout << "mTabulDyn.push_backmTabulDyn.push_back \n";
   }
}

void cAppli_Vino::HistoSetDyn()
{
    std::string aMes = "Clik  for polygone ; Shift Clik  to finish ; Enter 2 point for rectangle";
    ElList<Pt2di> aL = GetPtsImage(false,false,aMes);
    if (aL.card() >= 2)
    {
        if (aL.card()== 2)
           FillStat(*mCurStats,rectangle(aL.car(),aL.cdr().car()),mScr->CurScale()->in());
        else
           FillStat(*mCurStats,polygone(aL),mScr->CurScale()->in());

        if (mCaseCur==mCaseHStat)
           mCurStats->Type() =  eDynVinoStat2;

        if (mCaseCur==mCaseHMinMax)
           mCurStats->Type() =  eDynVinoMaxMin;

        mCurStats->IsInit() = true;
        InitTabulDyn();
        SaveState();
    }
    Refresh();
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
