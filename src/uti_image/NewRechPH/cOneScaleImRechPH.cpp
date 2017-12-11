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


#include "NewRechPH.h"

void FilterGaussProgr(tImNRPH anIm,double  aSTarget,double  aSInit,int aNbIter)
{
    aSTarget = ElSquare(aSTarget);
    aSInit = ElSquare(aSInit);
    if (aSTarget > aSInit)
    {
        FilterGauss(anIm,sqrt(aSTarget-aSInit),aNbIter);
    }
    else if (aSTarget==aSInit)
    {
    }
    else
    {
      ELISE_ASSERT(false,"FilterGaussProgr");
    }
}
/*****************************************************/
/*                                                   */
/*            Constructeur                           */
/*                                                   */
/*****************************************************/


cOneScaleImRechPH::cOneScaleImRechPH(cAppli_NewRechPH &anAppli,const Pt2di & aSz,const double & aScale,const int & aNiv) :
   mAppli (anAppli),
   mSz    (aSz),
   mIm    (aSz.x,aSz.y),
   mTIm   (mIm),
   mScale (aScale),
   mNiv   (aNiv)
{
}

cOneScaleImRechPH* cOneScaleImRechPH::FromFile
(
     cAppli_NewRechPH & anAppli,
     const double & aS0,
     const std::string & aName,
     const Pt2di & aP0,
     const Pt2di & aP1Init
)
{
   Tiff_Im aTifF = Tiff_Im::StdConvGen(aName,1,true);
   Pt2di aP1 = (aP1Init.x > 0) ? aP1Init : aTifF.sz();
   cOneScaleImRechPH * aRes = new cOneScaleImRechPH(anAppli,aP1-aP0,aS0,0);

   ELISE_COPY ( aRes->mIm.all_pts(),trans(aTifF.in_proj(),aP0),aRes->mIm.out());

   FilterGaussProgr(aRes->mIm,aS0,1.0,4);

   return aRes;
}


cOneScaleImRechPH* cOneScaleImRechPH::FromScale(cAppli_NewRechPH & anAppli,cOneScaleImRechPH & anIm,const double & aSigma)
{
     cOneScaleImRechPH * aRes = new cOneScaleImRechPH(anAppli,anIm.mSz,aSigma,anIm.mNiv+1);

     // Pour reduire le temps de calcul, si deja plusieurs iters la fon de convol est le resultat
     // de plusieurs iters ...
     int aNbIter = 4;
     if (aRes->mNiv==1) aNbIter = 3;
     else if (aRes->mNiv>=2) aNbIter = 2;

     // anIm.mIm.dup(aRes->mIm);
     aRes->mIm.dup(anIm.mIm);
     // Passe au filtrage gaussien le sigma cible et le sigma actuel, il se debrouille ensuite
     FilterGaussProgr(aRes->mIm,aSigma,anIm.mScale,aNbIter);


     return aRes;
}

tImNRPH cOneScaleImRechPH::Im() {return mIm;}


// Indique si tous les voisins se compare a la valeur cible aValCmp
// autrement dit est un max ou min local

bool   cOneScaleImRechPH::SelectVois(const Pt2di & aP,const std::vector<Pt2di> & aVVois,int aValCmp)
{
    tElNewRechPH aV0 =  mTIm.get(aP);
    for (int aKV=0 ; aKV<int(aVVois.size()) ; aKV++)
    {
        const Pt2di & aVois = aVVois[aKV];
        tElNewRechPH aV1 =  mTIm.get(aP+aVois,aV0);
        if (CmpValAndDec(aV0,aV1,aVois) != aValCmp)
           return false;
    }
    return true;
}


// Recherche tous les points topologiquement interessant

void cOneScaleImRechPH::CalcPtsCarac()
{
   // voisin tries excluant le pixel central, le tri permet normalement de
   // beneficier le plus rapidement possible d'une "coupe"
   std::vector<Pt2di> aVoisMinMax  = SortedVoisinDisk(0.5,mAppli.DistMinMax(),true);


   bool DoMin = mAppli.DoMin();
   bool DoMax = mAppli.DoMax();

   Im2D_U_INT1 aIFlag = MakeFlagMontant(mIm);
   TIm2D<U_INT1,INT> aTF(aIFlag);
   Pt2di aP ;
   for (aP.x = 1 ; aP.x <mSz.x-1 ; aP.x++)
   {
       for (aP.y = 1 ; aP.y <mSz.y-1 ; aP.y++)
       {
           int aFlag = aTF.get(aP);
           eTypePtRemark aLab = eTPR_NoLabel;
           
           if (DoMax &&  (aFlag == 0)  && SelectVois(aP,aVoisMinMax,1))
           {
              // std::cout << "DAxx "<< DoMax << " " << aFlag << "\n";
               aLab = eTPR_Max;
           }
           if (DoMin &&  (aFlag == 255) && SelectVois(aP,aVoisMinMax,-1))
           {
               // std::cout << "DInnn "<< DoMin << " " << aFlag << "\n";
               aLab = eTPR_Min;
           }

           if (aLab!=eTPR_NoLabel)
           {
              mLIPM.push_back(new cPtRemark(Pt2dr(aP),aLab));
           }
        }
   }

}

Pt3dr cOneScaleImRechPH::PtPly(const cPtRemark & aP,int aNiv)
{
   return Pt3dr(aP.Pt().x,aP.Pt().y,aNiv*mAppli.DZPlyLay());
}

void cOneScaleImRechPH::Export(cOneScaleImRechPH * aHR, cPlyCloud *  aPlyC)
{
   mNbExLR = 0;
   mNbExHR = 0;
   for (std::list<cPtRemark*>::const_iterator itIPM=mLIPM.begin(); itIPM!=mLIPM.end() ; itIPM++)
   {
       cPtRemark & aP = **itIPM;
       double aDistZ = mAppli.DZPlyLay();

       if (!aP.HR())
       {
          cBrinPtRemark * aBr = new cBrinPtRemark(&aP,mNiv);
          mAppli.AddBrin(aBr);

          if (aPlyC)
          {
             int aN0 = aBr->Niv0();
             int aL  = aBr->Long();
             Pt3di aCol = CoulOfType(aP.Type(),aN0,aL);
             Pt3di aColP0 = (aN0!=0)  ? Pt3di(0,255,0)  : aCol;
             aPlyC->AddSphere(aColP0,PtPly(*(aBr->P0()),aN0),aDistZ/6.0,3);

             aPlyC->AddSphere(aCol,PtPly(*(aBr->PLast()),aN0+aL),aDistZ/6.0,3);

             int aNiv = aN0;
             for (cPtRemark * aP = aBr->P0() ; aP->LR() ; aP = aP->LR())
             {
                 aPlyC->AddSeg(aCol,PtPly(*aP,aNiv),PtPly(*(aP->LR()),aNiv+1),20);
                 aNiv++;
             }
          }
       }
 
       if (!aP.HR()) mNbExHR ++;
       if (!aP.LR()) mNbExLR ++;
   }

   // std::cout << "NIV=" << mNiv << " HR " << mNbExHR << " LR " << mNbExLR << "\n";
}

void cOneScaleImRechPH::Show(Video_Win* aW)
{
   if (! aW) return;

   Im2D_U_INT1 aIR(mSz.x,mSz.y);
   Im2D_U_INT1 aIV(mSz.x,mSz.y);
   Im2D_U_INT1 aIB(mSz.x,mSz.y);


   ELISE_COPY(mIm.all_pts(),Max(0,Min(255,mIm.in())),aIR.out()|aIV.out()|aIB.out());

   for (std::list<cPtRemark*>::const_iterator itIPM=mLIPM.begin(); itIPM!=mLIPM.end() ; itIPM++)
   {
       Pt3di aC = CoulOfType((*itIPM)->Type(),0,1000);

       ELISE_COPY
       (
          disc((*itIPM)->Pt(),2.0),
          Virgule(aC.x,aC.y,aC.z),
          Virgule(aIR.oclip(),aIV.oclip(),aIB.oclip())
       );
   }

   ELISE_COPY
   (
      mIm.all_pts(),
      Virgule(aIR.in(),aIV.in(),aIB.in()),
      aW->orgb()
   );
}

// Initialise la matrice des pt remarquable, en Init les met, sinon les supprime
void cOneScaleImRechPH::InitBuf(const eTypePtRemark & aType, bool Init)
{
   for (std::list<cPtRemark *>::iterator itP=mLIPM.begin() ; itP!=mLIPM.end() ; itP++)
   {
      if ((*itP)->Type()==aType)
      {
         Pt2di aPi = round_ni((*itP)->Pt());
         if (mAppli.Inside(aPi))
         {
            mAppli.PtOfBuf(aPi) = Init ? *itP : 0;
         }
         else
         {
         }
      }
   }
}


void cOneScaleImRechPH::CreateLink(cOneScaleImRechPH & aHR,const eTypePtRemark & aType)
{
   // Initialise la matrice haute resolution
   aHR.InitBuf(aType,true);
   double aDist = mScale * 1.5 + 4;

  
   for (std::list<cPtRemark *>::iterator itP=mLIPM.begin() ; itP!=mLIPM.end() ; itP++)
   {
       if ((*itP)->Type()==aType)
       {
           // Pt2di aPi = round_ni((*itP)->Pt());
           tPtrPtRemark aNearest  =  mAppli.NearestPoint(round_ni((*itP)->Pt()),aDist);
           if (aNearest)
           {
              // std::cout << "LEVS " << mNiv << " " << aHR.mNiv << "\n";
              (*itP)->MakeLink(aNearest);
           }
           else
           {
           }
       }
   }
   // Desinitalise 
   aHR.InitBuf(aType,false);
}

void cOneScaleImRechPH::CreateLink(cOneScaleImRechPH & aLR)
{
    for (int aK=0 ; aK<eTPR_NoLabel ; aK++)
    {
       CreateLink(aLR,eTypePtRemark(aK));
    }

   std::cout <<  "CREATE LNK " << mNiv << "\n";
}


const int &  cOneScaleImRechPH::NbExLR() const {return mNbExLR;}
const int &  cOneScaleImRechPH::NbExHR() const {return mNbExHR;}
const double &  cOneScaleImRechPH::Scale() const {return mScale;}



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
aooter-MicMac-eLiSe-25/06/2007*/
