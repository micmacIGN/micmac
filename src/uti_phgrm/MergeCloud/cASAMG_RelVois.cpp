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


#include "MergeCloud.h"


void cResumNuage::Reset(int aReserve)
{
  mNbSom = 0;
  mVX.reserve(aReserve);
  mVY.reserve(aReserve);
  mVNb.reserve(aReserve);
}


void cASAMG::ComputeSubset(int aNbPts,cResumNuage & aRN)
{
   Video_Win * aW = mAppli->Param().VisuLowPts().Val() ?   TheWinIm() : 0;

   double aSzCel = sqrt(double(mSz.x*mSz.y)/aNbPts);
   Pt2di aNb2C = round_up(Pt2dr(mSz)/aSzCel);
   Pt2di aPK;

   Im2D_U_INT1 anImDist(mSz.x,mSz.y);
   TIm2D<U_INT1,INT> aTDist(anImDist);
   ELISE_COPY(mMasqN.all_pts(),mMasqN.in(),anImDist.out());

   Chamfer::d32.im_dist(anImDist);


   if (aW)
   {
      ELISE_COPY(anImDist.all_pts(),mStdN->ImDef().in()!=0,aW->odisc());
      ELISE_COPY(select(anImDist.all_pts(),anImDist.in()>0),P8COL::blue,aW->odisc());
      ELISE_COPY(select(anImDist.all_pts(),anImDist.in()>10),P8COL::yellow,aW->odisc());
   }

   int aNbSomTot  = 0;
   std::vector<Pt2di> aVPts;
   std::vector<int>   aVNb;
   for (aPK.x=0 ; aPK.x<aNb2C.x ; aPK.x++)
   {
       Pt2di aP0,aP1;
       aP0.x = (aPK.x*mSz.x)     / aNb2C.x;
       aP1.x = ((aPK.x+1)*mSz.x) / aNb2C.x;
       for (aPK.y=0 ; aPK.y<aNb2C.y ; aPK.y++)
       {
           aP0.y = (aPK.y*mSz.y)     / aNb2C.y;
           aP1.y = ((aPK.y+1)*mSz.y) / aNb2C.y;

           Pt2di aP;

           Pt2dr aPMoy(0,0);
           int aNbSom=0;
           if (aW)
           {
               aW->draw_rect(Pt2dr(aP0),Pt2dr(aP1),Line_St(aW->pdisc()(P8COL::green)));
           }
           for (aP.x=aP0.x; aP.x<aP1.x ; aP.x++)
           {
               for (aP.y=aP0.y; aP.y<aP1.y ; aP.y++)
               {
                  if (mTMasqN.get(aP))
                  {
                     aPMoy = aPMoy + Pt2dr(aP);
                     aNbSom++;
                  }
               }
           }
           if (aNbSom>0)
           {
              aPMoy = aPMoy/double(aNbSom);
              double aDBest = 1e10;
              Pt2di aPBest(0,0);
              for (aP.x=aP0.x; aP.x<aP1.x ; aP.x++)
              {
                  for (aP.y=aP0.y; aP.y<aP1.y ; aP.y++)
                  {
                     if (mTMasqN.get(aP))
                     {
                        double aDist = euclid(Pt2dr(aP),aPMoy) - ElMin(6,aTDist.get(aP))/1.8;

                        if (aDist<aDBest)
                        {
                           aDBest = aDist;
                           aPBest = aP;
                        }
                     }
                  }
              }
              aVPts.push_back(aPBest);
              aVNb.push_back(aNbSom);
/*
              cLinkPtFuNu aPF(aPBest.x,aPBest.y,aNbSom);
              aVPTR.push_back(aPF);
*/
              aNbSomTot += aNbSom;
              if (aW)
              {
                  aW->draw_circle_abs(Pt2dr(aPBest),3.0,aW->pdisc()(P8COL::red));
              }
           }
       }
   }

   aRN.Reset((int)aVPts.size());
   aRN.mNbSom = aNbSomTot;
   for (int aK=0 ; aK<int(aVPts.size()); aK++)
   {
      aRN.mVX.push_back(aVPts[aK].x);
      aRN.mVY.push_back(aVPts[aK].y);
      aRN.mVNb.push_back(aVNb[aK]);
   }

   if (aW)   
   {
       //  aW->clik_in();
   }
}


// ===============================================

// Pt2dr aPTest(93,85);
Pt2dr aPTest(96,117);

double cASAMG::SignedDifProf(const Pt3dr & aPE) const
{
   Pt3dr aQ = mStdN->Euclid2ProfPixelAndIndex(aPE);
   Pt2dr aQ2(aQ.x,aQ.y);
   if (mStdN->IndexHasContenuForInterpol(aQ2))
   {
      // double aProfIm = mStdN->ProfOfIndexInterpol(aQ2);
      double aProfIm = mStdN->ProfInterpEnPixel(aQ2);
      return aQ.z-aProfIm;
   }
   return 1000;
}

double cASAMG::DifProf2Gain(double aDif) const
{
    aDif = ElAbs(aDif/mAppli->Param().RecSeuilDistProf().Val());
    return ElMin(1.0,ElMax(0.0,2.0-aDif));
}

double cASAMG::QualityProjOnMe(const Pt3dr & aPE) const
{
    return DifProf2Gain(SignedDifProf(aPE));
/*
    double aDif = ElAbs(SignedDifProf(aPE)) / mAppli->Param().RecSeuilDistProf().Val();
    return ElMin(1.0,ElMax(0.0,2.0-aDif));
*/
}

void cASAMG::MakeVec3D(std::vector<Pt3dr> & aVPts,const cResumNuage & aRN) const
{
   for (int aK=0 ; aK<int(aRN.mVX.size()) ; aK++)
   {
       aVPts.push_back(mStdN->PtOfIndex(aRN.PK(aK)));
   }
}

double cASAMG::Recouvrt(const cASAMG & anA2,const cResumNuage & aRN,const std::vector<Pt3dr> & aVPts) const
{
   double aRes = 0.0;
   for (int aK=0 ; aK<int(aRN.mVX.size()) ; aK++)
   {
       aRes += aRN.mVNb[aK] * anA2.QualityProjOnMe(aVPts[aK]);
   }
   return aRes / aRN.mNbSom;
}

double cASAMG::Recouvrt(const cASAMG & anA2,const cResumNuage & aRN) const
{
   std::vector<Pt3dr> aVP3;
   MakeVec3D(aVP3,aRN);
   return Recouvrt(anA2,aRN,aVP3);
}

double cASAMG::LowRecouvrt(const cASAMG & anA2) const
{
   return Recouvrt(anA2,mLowRN);
}

void cASAMG::TestDifProf(const cASAMG & aNE) const
{
    Im2D_REAL4 aImDif(mSz.x,mSz.y,1000);
    TIm2D<REAL4,REAL8> aTDif(aImDif);
    Pt2di anIndex;

    for (anIndex.x=0 ; anIndex.x <mSz.x ; anIndex.x++)
    {
        for (anIndex.y=0 ; anIndex.y <mSz.y ; anIndex.y++)
        {
             if (mTMasqN.get(anIndex))
             {
                 Pt3dr aPE = mStdN->PtOfIndex(anIndex);
                 aTDif.oset(anIndex,aNE.SignedDifProf(aPE));
             }
        }
    }
    Video_Win * aW = TheWinIm();
    if (aW)
    {
       ELISE_COPY
       (
           aImDif.all_pts(),
           Min(255,Abs(aImDif.in()*300)),
           aW->ogray()
       );
       aW->clik_in();
    }

}

void cASAMG::TestImCoher() 
{
    ElTimer aChrono;
    const   std::vector<cASAMG *> & aVN = mCloseNeigh;

    int aNbIm = (int)aVN.size();
    Im2D_REAL4 aImDif(mSz.x,mSz.y,1000);
    TIm2D<REAL4,REAL8> aTDif(aImDif);
    Pt2di aP;

    for (aP.x=0 ; aP.x<mSz.x ; aP.x++)
    {
        for (aP.y=0 ; aP.y<mSz.y ; aP.y++)
        {
             if (mTMasqN.get(aP))
             {
                 Pt3dr aPE = mStdN->PtOfIndex(aP);
                 double aSom=0;
                 for (int aK=0 ; aK<aNbIm ; aK++)
                 {
                     aSom += aVN[aK]->QualityProjOnMe(aPE);
                 }
                 aTDif.oset(aP,aSom);
             }
        }
    }

    for (int aNbRec = 3 ; aNbRec>=0 ; aNbRec --)
    {
        double aSeuil = aNbRec-0.25;
        Fonc_Num aFInside = aImDif.in(-1) > aSeuil;
        if (aNbRec>0)
        {
            eQualCloud aQual = eQC_Coh1;
            if (aNbRec==2) aQual = eQC_Coh2;
            if (aNbRec==3) aQual = eQC_Coh3;
            // Pour ceux qui ont ete valide (aNbRec> 0.75) et non bord ou autre chose ont leur met la bonne valeur
            ELISE_COPY
            (
                 select(mImQuality.all_pts(),aFInside && (mImQuality.in()==eQC_NonAff)),
                 aQual,
                 mImQuality.out()
            );

            ELISE_COPY
            (
                 select
                 (
                      mImQuality.all_pts(),
                         aFInside 
                      && (mImQuality.in()==eQC_GradFaibleC1) 
                      && (erod_32(mMasqN.in_proj(),2*mPrm.DilateBord().Val()))
                 ),
                 eQC_GradFaibleC2,
                 mImQuality.out()
            );
        }
        else
        {
            aFInside =  aImDif.in(-1) <= (aSeuil+1);
            Im2D_Bits<1>   aImLQ(mSz.x,mSz.y);
            ELISE_COPY(aImLQ.all_pts(),aFInside,aImLQ.out());

            ELISE_COPY
            (
                 select(mImQuality.all_pts(), aImLQ.in() && (mImQuality.in()==eQC_NonAff)),
                 eQC_ZeroCohImMul,
                 mImQuality.out()
            );
            ELISE_COPY
            (
                 select
                 (
                     mImQuality.all_pts(),
                     aImLQ.in()&&( mImQuality.in()>=eQC_GradFort) && ( mImQuality.in()<=eQC_Bord)
                 ),
                 eQC_ZeroCohBrd,
                 mImQuality.out()
            );
/*
*/
        }
    }
    ELISE_COPY(mImQuality.border(1),eQC_Out,mImQuality.out());
    InitGlobHisto();

    for (int aK=0 ; aK<mHisto.tx() ; aK++)
    {
        if (mDH[aK])
        {
           mMaxNivH = aK;
        }
    }

   
    Video_Win * aW =  mAppli->Param().VisuImageCoh().Val() ? TheWinIm() : 0 ;
    if (aW)
    {
        aW->set_title(mIma->mNameIm.c_str());
        std::cout << "For " << mIma->mNameIm << " time " << aChrono.uval() << " NbIm " << aNbIm << "\n";
        Fonc_Num fGray = Min(255,aImDif.in() * (255.0/aNbIm));

        for (int aK=0 ; aK<mHisto.tx() ; aK++)
            std::cout << "H[" << aK << "]= " << mHisto.data()[aK] << "\n";
        InspectQual(true);
        
    }
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
