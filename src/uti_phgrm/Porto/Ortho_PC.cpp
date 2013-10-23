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

/*************************************************/
/*                                               */
/*          cCC_Appli                            */
/*                                               */
/*************************************************/


void cAppli_Ortho::OrthoRedr()
{
   DoIndexNadir();

   if (mCO.BoucheTrou().IsInit())
   {
      const cBoucheTrou  & aBT = mCO.BoucheTrou().Val();
      InitInvisibilite(aBT);
      InitBoucheTrou(aBT);
   }

   MakeOrthoOfIndex();
   VisuLabel();
   // getchar();
   SauvAll();
}

void cAppli_Ortho::InitInvisibilite(const cBoucheTrou  & aBT)
{
   int aSV = mCO.SeuilVisib().Val();
   Pt2di aP;
   for  (aP.x =0 ; aP.x<mSzCur.x  ; aP.x++)
   {
      for  (aP.y =0 ; aP.y<mSzCur.y  ; aP.y++)
      {
          int aInd = mTImIndex.get(aP);
          if (( aInd >=0) && (mTImEtiqTmp.get(aP)!=10))
          {
              int aValPC = mVLI[aInd]->ValeurPC(aP);
              if (aValPC>aSV)
              {
                 mTImIndex.oset(aP,-1);
                 mTImEtiqTmp.oset(aP,1);
              }
          }
      }
   }
}

Liste_Pts_INT2 cAppli_Ortho::CompConx
               (
                   Pt2di aGerm,
                   int aValSet
               )
{
     Liste_Pts_INT2 cc(2);
     Neighbourhood V4 = Neighbourhood::v4();




     ELISE_COPY
     (
         conc
         (
             aGerm,
             mImEtiqTmp.neigh_test_and_set
             (
                 V4,
                 mTImEtiqTmp.get(aGerm),
                 aValSet,
                 100
             )
         ),
         0,
         cc 
     );

     return cc;
}


double cAppli_Ortho::ScoreOneHypBoucheTrou
       (
          const cBoucheTrou  & aBT,
          Liste_Pts_INT2 aL,
          cLoadedIm * aLI
       )
{
    double aRes;
    ELISE_COPY
    (
        select
        (
             aL.all_pts(),
             (aLI->ImPCBrute().in() <= aBT.SeuilVisibBT().Val())
        ),
        1.0/(aBT.CoeffPondAngul().Val()+aLI->FoncInc()),
        sigma(aRes)
    );
    return aRes;
}


void cAppli_Ortho::InitBoucheTrou(const Pt2di & aP,const cBoucheTrou  & aBT)
{
   Liste_Pts_INT2 aL = CompConx(aP,2);
   ELISE_COPY(aL.all_pts(),1,mImEtiqTmp.out());


   double aScoreMax = 1e-5;
   cLoadedIm * mBestLI = 0;
   // Im2D_INT2 aL.image();
   for (int aK=0 ; aK<int(mVLI.size()) ; aK++)
   {
        double aScore = ScoreOneHypBoucheTrou(aBT,aL,mVLI[aK]);
        if (aScoreMax<aScore)
        {
            aScoreMax = aScore;
            mBestLI = mVLI[aK];
        }
        // std::cout << "SCORE =" << aScore << "\n";
   }
   if (!mBestLI)
   {
       ELISE_COPY(aL.all_pts(),2,mImEtiqTmp.out());
       return;
   }

   ELISE_COPY
   (
        select
        (
             aL.all_pts(),
             (mBestLI->ImPCBrute().in() <= aBT.SeuilVisibBT().Val())
        ),
        2,
        mImEtiqTmp.out() | (mImIndex.out() << mBestLI->Ind()) 
        // | (mW->odisc() << P8COL::red)
    );


// Fermeture morphologique

   Liste_Pts_INT2 aLFront(2);
   Neighbourhood V4 = Neighbourhood::v4();
           // On dilate 3 fois de suite , a partir des point valant 2,
           // sur les points valante 1, on les colorie en 3
   ELISE_COPY
   (
        conc
        (
            select(aL.all_pts(),mImEtiqTmp.in()==2),
            mImEtiqTmp.neigh_test_and_set(V4,1,3,100),
            2
        ),
        0,
        aLFront
   );
      // On dilate dansl'autre sens (3 ->1) a partir des points ayant
      // au moins un voisin 1
   ELISE_COPY
   (
        conc
        (
            select
            (
                aLFront.all_pts(),
                Neigh_Rel(V4).red_sum(mImEtiqTmp.in()==1)
            ),
            mImEtiqTmp.neigh_test_and_set(V4,3,1,100),
            2
        ),
        0,
        Output::onul()
   );


   int aCpt;
   ELISE_COPY
   (
       select(aL.all_pts(),mImEtiqTmp.in()==3),
       Virgule(1,2, mBestLI->Ind()),
       Virgule(sigma(aCpt),mImEtiqTmp.out(),mImIndex.out())
   );

   // std::cout << "CPT = " << aCpt << "\n";





// getchar();

    Im2D_INT2 aIm = aL.image();
    int aNb = aIm.tx();
    INT2 * aTabX = aIm.data()[0];
    INT2 * aTabY = aIm.data()[1];
    for (int aK=0 ; aK<aNb ; aK++)
    {
        Pt2di aP(aTabX[aK],aTabY[aK]);
        if (mTImEtiqTmp.get(aP)==1)
           InitBoucheTrou(aP,aBT);
    }

}


void cAppli_Ortho::InitBoucheTrou(const cBoucheTrou  & aBT)
{

   Neighbourhood V8 = Neighbourhood::v8();

   Pt2di aP;
   for  (aP.x =0 ; aP.x<mSzCur.x  ; aP.x++)
   {
      for  (aP.y =0 ; aP.y<mSzCur.y  ; aP.y++)
      {
           if (mTImEtiqTmp.get(aP)==1)
           {
              InitBoucheTrou(aP,aBT);
           }
      }
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
