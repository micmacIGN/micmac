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
#include "hough_include.h"
#include <iterator>

#define NoTemplateOperatorVirgule





template <class InputIterator, class BinaryPredicate, class MergeOperation>
InputIterator unique_merge
                (
                      InputIterator first, 
                      InputIterator last,
                      BinaryPredicate binary_pred,
                      MergeOperation  merger
                 ) 
{
   first = std::adjacent_find(first, last, binary_pred);
   if (first == last) return first; 
   InputIterator result = first;
   while (++first != last)
   {
     if (binary_pred(*result, *first)) 
        merger(*result,*first);
     else
        *++result = *first;
   }
   return ++result;
}
                     



class ElHoughSubPixellaire : public ElHoughImplem
{
    public :
        ElHoughSubPixellaire(Pt2di ,REAL,REAL,REAL RabRho,REAL RabTeta);

		void PostInit();

        class ElemXYRT
        {
            public :
               ElemXYRT(Pt2di aXY,Pt2di aRhotTeta,REAL aPds);
               bool operator < (const ElemXYRT &) const;
               friend bool GeomEq (const ElemXYRT &, const ElemXYRT &);
               friend bool XYCmp (const ElemXYRT &, const ElemXYRT &);
               friend void Merge (ElemXYRT &, const ElemXYRT &);

               REAL Pds() const {return mPds/theFactPds;}
               Pt2di  RhoTeta() const {return Pt2di(mRho,mTeta);}

            private : 
               INT2 mX,mY,mRho,mTeta;
               U_INT2 mPds;
               static const REAL theFactPds; 
        };

    protected :
        void AddElem(Pt2di XY,Pt2di RhoTeta,REAL Pds);
        void compute_contrib(INT iTeta,REAL);

        PolygoneClipBandeVert  mPCBV;

        Pt2dr  InTetaCoord(Pt2dr p) {return p*mRot;}
        Pt2dr  mRot;

        typedef enum {MatriceMap,MatriceVect,VectXYRT} Mode;
        Mode  mode;

        typedef ElSTDNS map<Pt2di,REAL>         tMapCelAccum;
        typedef ElSTDNS vector<tMapCelAccum>    tMapLineAccum;
        typedef ElSTDNS vector<tMapLineAccum>   tMapAccum;
        tMapAccum *                     mMapAccum;
   
        typedef ElSTDNS pair <Pt2di,REAL>  tPCel;

        typedef ElSTDNS vector<tPCel> tVecCelAccum;
        typedef ElSTDNS vector<tVecCelAccum>    tVecLineAccum;
        typedef ElSTDNS vector<tVecLineAccum>   tVecAccum;
        tVecAccum *                     mVecAccum;



        typedef ElSTDNS vector<ElemXYRT>  tVxyrt;
        typedef tVxyrt::iterator  tItVxyrt;
        tVxyrt *                  mVxyrt;

        INT mNbTot;

        void clean();
        void ElemPixel(tLCel &,Pt2di);

};

const REAL ElHoughSubPixellaire::ElemXYRT::theFactPds = 3e3;

ElHoughSubPixellaire::ElemXYRT::ElemXYRT
(
   Pt2di aXY,
   Pt2di aRhoTeta,
   REAL aPds
)  :
     mX    ((INT2) aXY.x),
     mY    ((INT2) aXY.y),
     mRho  ((INT2) aRhoTeta.x),
     mTeta ((INT2) aRhoTeta.y),
     mPds  ((INT2) (theFactPds * aPds))
{
   ELISE_ASSERT
   (
       ((theFactPds * aPds) < ((1<<16)-1)),
       "Inc in ElHoughSubPixellaire"
   );
}


bool ElHoughSubPixellaire::ElemXYRT::operator < 
     (const ElHoughSubPixellaire::ElemXYRT & El) const
{
   if (mX < El.mX) return true;
   if (mX > El.mX) return false;

   if (mY < El.mY) return true;
   if (mY > El.mY) return false;

   if (mRho < El.mRho) return true;
   if (mRho > El.mRho) return false;
   
   if (mTeta < El.mTeta) return true;
   if (mTeta > El.mTeta) return false;

   if (mPds < El.mPds) return true;
   if (mPds > El.mPds) return false;

   return false;
}

bool XYCmp
     (
         const ElHoughSubPixellaire::ElemXYRT & El1,
         const ElHoughSubPixellaire::ElemXYRT & El2
     )
{
   if (El1.mX < El2.mX) return true;
   if (El1.mX > El2.mX) return false;

   if (El1.mY < El2.mY) return true;
   if (El1.mY > El2.mY) return false;

   return false;
}

bool GeomEq
     (
         const ElHoughSubPixellaire::ElemXYRT & El1,
         const ElHoughSubPixellaire::ElemXYRT & El2
     )
{
   return     (El1.mX    == El2.mX)
          &&  (El1.mY    == El2.mY)
          &&  (El1.mRho  == El2.mRho)
          &&  (El1.mTeta == El2.mTeta);
}
void Merge
     (
         ElHoughSubPixellaire::ElemXYRT & El1,
         const ElHoughSubPixellaire::ElemXYRT & El2
     )
{
   El1.mPds += El2.mPds;
}


/***********************************************************************/
/***********************************************************************/
/***********************************************************************/

void ElHoughSubPixellaire::clean()
{
    delete mMapAccum;
    delete mVecAccum;
    delete mVxyrt;
}



void ElHoughSubPixellaire::ElemPixel(tLCel & v,Pt2di Pxy)
{

    if (mMapAccum)
    {
       tMapCelAccum::iterator itBegin = (*mMapAccum)[Pxy.y][Pxy.x].begin();
       tMapCelAccum::iterator itEnd = (*mMapAccum)[Pxy.y][Pxy.x].end();

       for (tMapCelAccum::iterator it = itBegin; it!= itEnd ; it++)
           v.push_back(tCel(it->first,it->second));
/*
       ElSTDNS copy
       (
          (*mMapAccum)[Pxy.y][Pxy.x].begin(),
          (*mMapAccum)[Pxy.y][Pxy.x].end(),
          ElSTDNS back_inserter(v)
       );
*/
    }
    if (mVecAccum)
       ElSTDNS copy
       (
          (*mVecAccum)[Pxy.y][Pxy.x].begin(),
          (*mVecAccum)[Pxy.y][Pxy.x].end(),
          ElSTDNS back_inserter(v)
       );

    if (mVxyrt)
    {
       ElSTDNS pair<tItVxyrt,tItVxyrt>  aPair =
                                ElSTDNS equal_range
                                (
                                     mVxyrt->begin(),
                                     mVxyrt->end(),
                                     ElemXYRT(Pxy,Pt2di(0,0),0),
                                     XYCmp
                                );
       for (tItVxyrt anIt=aPair.first;anIt!=aPair.second;anIt++)
       {
          v.push_back(tPCel(anIt->RhoTeta(),anIt->Pds()));
       }
    }
}



void ElHoughSubPixellaire::AddElem(Pt2di Pxy,Pt2di RhoTeta,REAL Pds)
{
    if (mMapAccum)
        (*mMapAccum)[Pxy.y][Pxy.x][RhoTeta] += Pds ;
    if (mVecAccum)
        (*mVecAccum)[Pxy.y][Pxy.x].push_back(tPCel(RhoTeta,Pds));
     if (mVxyrt)
        mVxyrt->push_back(ElemXYRT(Pxy,RhoTeta,Pds));
}


void ElHoughSubPixellaire::compute_contrib(INT IG_Teta,REAL G_Teta)
{
   REAL aTeta = G2S_Teta(G_Teta);
   mRot = Pt2dr::FromPolar(1.0,-aTeta);


   for (INT iX =0; iX<NbX() ; iX++)
   {
       for (INT iY =0; iY<NbY() ; iY++)
       {
            Pt2di G_Pt(iX,iY);
            Pt2dr S_Pt = G2S_XY(Pt2dr(G_Pt));

          // pt1-pt2-pt3-pt4 defini le contour du pixel dans le
          // repere lies a teta

            Pt2dr pt1 =  InTetaCoord(S_Pt+Pt2dr(-0.5,-0.5));
            Pt2dr pt2 =  InTetaCoord(S_Pt+Pt2dr(+0.5,-0.5));
            Pt2dr pt3 =  InTetaCoord(S_Pt+Pt2dr(+0.5,+0.5));
            Pt2dr pt4 =  InTetaCoord(S_Pt+Pt2dr(-0.5,+0.5));

            REAL S_RhoMax = ElMax4(pt1.x,pt2.x,pt3.x,pt4.x);
            REAL S_RhoMin = ElMin4(pt1.x,pt2.x,pt3.x,pt4.x);

            REAL  G_RhoMax = CentS2G_Rho(S_RhoMax,IG_Teta);
            REAL  G_RhoMin = CentS2G_Rho(S_RhoMin,IG_Teta);
            set_min_max(G_RhoMin,G_RhoMax);


            INT IG_RhoMin = round_down(G_RhoMin+0.5);
            INT IG_RhoMax = round_down(G_RhoMax+0.5);

            for (INT IG_Rho = IG_RhoMin; IG_Rho<=IG_RhoMax; IG_Rho++)
            {
                 REAL S_rho1 = CentG2S_Rho(IG_Rho-0.5,IG_Teta);
                 REAL S_rho2 = CentG2S_Rho(IG_Rho+0.5,IG_Teta);

                 REAL surf = mPCBV.SquareSurClipPts(S_rho1,S_rho2,pt1,pt2);

                 if (surf > 1e-5)
                 {
                    AddElem(G_Pt,Pt2di(IG_Rho,IG_Teta),surf);
                    mNbTot++;
                 }
            }
       }
   }

}

ElHoughSubPixellaire::ElHoughSubPixellaire
(
   Pt2di SzXY,
   REAL StepRho,
   REAL StepTeta,
   REAL RabRho,
   REAL RabTeta
) :
   ElHoughImplem(SzXY,StepRho,StepTeta,RabRho,RabTeta),
   mode  (VectXYRT), 
   mMapAccum (0),
   mVecAccum (0),
   mVxyrt    (0)
{
}

 
void ElHoughSubPixellaire::PostInit()
{
	ElHoughImplem::PostInit();
   
    switch (mode)
    {
        case MatriceMap :
             mMapAccum =  new tMapAccum (NbY(),tMapLineAccum(NbX()));
        break;

        case MatriceVect :
             mVecAccum =  new tVecAccum (NbY(),tVecLineAccum(NbX()));
        break;

        case VectXYRT :
             mVxyrt   = new  ElSTDNS vector<ElemXYRT>;
             mVxyrt->reserve(round_ni(NbY()*NbX()*NbTeta() * 2.35));
        break;
    }

    mNbTot = 0;
    for (INT iTeta=0; iTeta < NbTeta(); iTeta++)
    {
       compute_contrib(iTeta,iTeta);
    }

    if (mVxyrt)
    {
       std::sort(mVxyrt->begin(),mVxyrt->end());
       tItVxyrt it = unique_merge(mVxyrt->begin(),mVxyrt->end(),GeomEq,Merge);
      
       while (it != mVxyrt->end()) 
            mVxyrt->pop_back();
    }
}

/***********************************************************************/
/***********************************************************************/
/***********************************************************************/


ElHoughImplem * ElHoughImplem::SubPixellaire
(
     Pt2di SzXY,
     REAL  StepRho ,
     REAL  StepTeta ,
     REAL  RabRho,
     REAL  RabTeta
)
{
   ElHoughSubPixellaire * aRes = new ElHoughSubPixellaire(SzXY,StepRho,StepTeta,RabRho,RabTeta);
   aRes->PostInit();
   return aRes;
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
