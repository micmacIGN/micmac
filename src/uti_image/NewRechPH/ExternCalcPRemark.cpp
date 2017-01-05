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


// Visualise une conversion de flux en vecteur de point
void  TestFlux2StdCont()
{
    std::vector<Pt2di> aVp;
    Flux2StdCont(aVp,circle(Pt2dr(60,60),50));
    Im2D_U_INT1 aIm(120,120,0);
    for (int aK=0; aK<int(aVp.size()) ; aK++)
    {
        Pt2di aP = aVp[aK];
        aIm.data()[aP.y][aP.x] = 1;
    }
    Video_Win  aW =  Video_Win::WStd(Pt2di(300,200),3);
    ELISE_COPY(aIm.all_pts(),aIm.in(),aW.odisc());
    getchar();

}

/*
template <class TypeIm> class  cAutoCorrelDir
{
    public :
        typedef typename TypeIm::tValueElem tElem;
        typedef typename TypeIm::OutputFonc tBase;
        cAutoCorrelDir(TypeIm anIm,const Pt2di & aP0,double aRho,int aSzW) :
           mTIm   (anIm),
           mP0    (aP0),
           mRho   (aRho),
           mSzW   (aSzW)
        {
        }

        void DoIt()
        {
           double aStep0 = 1/mRho;
           int aNb = round_up(PI/aStep0);
           Pt2dr aRes0 = DoItOneStep(0.0,aNb,aStep0);
           Pt2dr aRes1 = DoItOneStep(aRes0.x,3,aStep0/4.0);
           Pt2dr aRes2 = DoItOneStep(aRes1.x,2,aStep0/10.0);
        }

    private :
        Pt2dr  DoItOneStep(double aTeta0,int aNb,double aStep)
        {
           double aScMax = -1e10;
           double aTetaMax = 0;
           for (int aK=-aNb; aK<aNb ; aK++)
           {
               double aTeta =  aTeta0 + aK * aStep;
               double aVal =  CorrelTeta(aTeta) ;
               if (aVal >aScMax)
               {
                  aScMax = aVal;
                  aTetaMax = aTeta;
               }
           }
           return Pt2dr(aTetaMax,aScMax);
        }


        double  CorrelOneOffset(const Pt2di & aP0,const Pt2dr & anOffset,int aSzW)
        {
            tBase aDef =   El_CTypeTraits<tBase>::MaxValue();
            RMat_Inertie aMat;
            for (int aDx=-aSzW ; aDx<=aSzW ; aDx++)
            {
                for (int aDy=-aSzW ; aDy<=aSzW ; aDy++)
                {
                    Pt2di aP1 = aP0 + Pt2di(aDx,aDy);
                    tBase aV1 = mTIm.get(aP1,aDef);
                    if (aV1==aDef) return -1;
                    tBase aV2 = mTIm.getr(Pt2dr(aP1)+anOffset,aDef);
                    if (aV2==aDef) return -1;
                    aMat.add_pt_en_place(aV1,aV2);
                }
            }
            return aMat.correlation();
        }

        double  CorrelTeta(double aTeta)
        {
            return CorrelOneOffset(mP0,Pt2dr::FromPolar(mRho,aTeta),mSzW);
        }


        TypeIm  mTIm;
        Pt2di   mP0;
        double  mRho;
        int     mSzW;
};

void TestcAutoCorrelDir(TIm2D<double,double> aTIm,const Pt2di & aP0)
{
    cAutoCorrelDir<TIm2D<double,double> >  aACD(aTIm,aP0,3.0,3);
    aACD.DoIt();
}
*/


/*****************************************************/
/*                                                   */
/*                 ::                                */
/*                                                   */
/*****************************************************/
class cCmpPt2diOnEuclid
{
   public : 
       bool operator () (const Pt2di & aP1, const Pt2di & aP2)
       {
                   return square_euclid(aP1) < square_euclid(aP2) ;
       }
};

std::vector<Pt2di> SortedVoisinDisk(double aDistMin,double aDistMax,bool Sort)
{
   std::vector<Pt2di> aResult;
   int aDE = round_up(aDistMax);
   Pt2di aP;
   for (aP.x=-aDE ; aP.x <= aDE ; aP.x++)
   {
       for (aP.y=-aDE ; aP.y <= aDE ; aP.y++)
       {
            double aD = euclid(aP);
            if ((aD <= aDistMax) && (aD>aDistMin))
               aResult.push_back(aP);
       }
   }
   if (Sort)
   {
      cCmpPt2diOnEuclid aCmp;
      std::sort(aResult.begin(),aResult.end(),aCmp);
   }

   return aResult;
}

Pt3di CoulOfType(eTypePtRemark aType)
{
    switch(aType)
    {
         case eTPR_Max : return Pt3di(255,0,0);
         case eTPR_Min : return Pt3di(0,0,255);

         default :;
    }

    return  Pt3di(128,128,128);
}

/*****************************************************/
/*                                                   */
/*                  cPtRemark                        */
/*                                                   */
/*****************************************************/

cPtRemark::cPtRemark(const Pt2dr & aPt,eTypePtRemark aType) :
           mPtR   (aPt),
           mType  (aType),
           mHR    (0),
           mLR    (0)
{
}

/*
  mLR      this
    \
    aHR    
*/

void cPtRemark::MakeLink(cPtRemark * aHR)
{
   if (aHR->mLR)
   {
        if (euclid(aHR->mLR->mPtR-aHR->mPtR) < euclid(mPtR-aHR->mPtR))
           return;

         aHR->mLR->mHR=0;
         aHR->mLR=0;
   }
   mHR = aHR;
   aHR->mLR = this;
}

/*****************************************************/
/*                                                   */
/*                  cBrinPtRemark                    */
/*                                                   */
/*****************************************************/

cBrinPtRemark::cBrinPtRemark(cPtRemark * aP0,int aNiv0) :
    mP0    (aP0),
    mPLast (mP0),
    mNiv0  (aNiv0),
    mLong  (0)
{
   ELISE_ASSERT(mP0->HR()==0,"Incoh in cBrinPtRemark");
   while (mPLast->LR())
   {
       mPLast  = mPLast->LR();
       mLong++;
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
aooter-MicMac-eLiSe-25/06/2007*/
