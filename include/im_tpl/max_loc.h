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

#ifndef _ELISE_IMTPL_MAX_LOC_H_
#define _ELISE_IMTPL_MAX_LOC_H_

#include <vector>
#include <functional>


/*
    Algo basique de max-loc
*/


template <class Type,class TypeBase> class cMaxLocBicub : public Optim2DParam
{
     private  :
           TIm2D<Type,TypeBase> mTIm;
           cCubicInterpKernel   mKer;
           double               mDef;

           REAL Op2DParam_ComputeScore(REAL anX,REAL anY) 
           {
                return mTIm.getr(mKer,Pt2dr(anX,anY),(TypeBase)mDef,true);
           }
     public :
           cMaxLocBicub
           (
                   Im2D<Type,TypeBase> aIm,
                   bool                aMax
           ) :
               Optim2DParam
               (
                    1e-2,
                    (aMax ? -1e20 : 1e20),
                    1e-7,
                    aMax
               ),
               mTIm  (aIm),
               mKer  (-0.5),
               mDef  (aMax ? -1e20 : 1e20)
           {
           }
};

template <class Type,class TypeBase> Pt2dr MaxLocBicub 
                                           ( 
                                               Im2D<Type,TypeBase> aIm,
                                               Pt2dr               aP0,
                                               bool                aMax
                                           )
{
     cMaxLocBicub<Type,TypeBase> aCML(aIm,aMax);
     aCML.optim(aP0);
     return aCML.param();
}


template <class Type,class TypeBase> Pt2di MaxLocEntier
                                           (
                                                Im2D<Type,TypeBase> aIm,
                                                Pt2di               aP0,
                                                bool                aMax,
                                                double              aRay
                                           )
{
      TIm2D<Type,TypeBase> aTIm(aIm);
      Pt2di aPExtre = aP0;
      aP0 = aPExtre + Pt2di(1,1);
      int aRInt = round_up(aRay);
      double aR2 = ElSquare(aRay);

      while (aP0 != aPExtre)
      {
          aP0 = aPExtre;
          TypeBase aValExtre = aTIm.get(aP0);
          Pt2di aDP;
          for (aDP.x=-aRInt; aDP.x<=aRInt; aDP.x++)
          {
              for (aDP.y=-aRInt; aDP.y<=aRInt; aDP.y++)
              {
                   if (square_euclid(aDP)<aR2)
                   {
                       Pt2di aP = aP0 + aDP;
                       if (aTIm.inside(aP))
                       {
                           TypeBase aVal = aTIm.get(aP);
                           if (aMax ? (aVal>aValExtre) : (aVal<aValExtre))
                           {
                                aValExtre = aVal;
                                aPExtre = aP;
                           }
                       }
                   }
              }
          }
      }

      return aP0;
}

/*
    Contient des algorithmes utiles pour Hough (et developpes
   dans ce cadre) mais dont la portee est a vocation plus generale.
*/




class GenMaxLoc
{
    public :
    protected :
      // z=1 => pts tq y<0 ou y=0 et x<0
      // ie les pts qui sont favorise en cas d'eglite
      std::vector<Pt3di> & OrdVois(Pt2di);
      GenMaxLoc();
    private :
       void                  MajOvois();
       U_INT1 &              Marq(Pt2di);

       Pt2di                 mLastVois;
       std::vector<Pt3di>    mOVois;
       Im2D_U_INT1           mMarq;
       U_INT1 **             mDataMarq;
};

template <class Type,class TypeBase,class Compare = ElSTDNS less<Type> >  
         class CalcMaxLoc  : private GenMaxLoc
{
    public :

         void  AllMaxLoc  
               (
                  std::vector<Pt2di> &res,
                  Im2D<Type,TypeBase> Im,
                  Pt2di    vois,
                  Pt2di    p0, Pt2di    p1,
                  TypeBase  VMin // supprime les max <= VMin
                );


          CalcMaxLoc() : mCmp (Compare()) {}

		  bool BandeConnectedVsup
			   (
				     Pt2di p1,
					 Pt2di p2,
					 Im2D<Type,TypeBase> Im,
					 Type  VInf,
					 REAL Tol,
					 Im2D_U_INT1 Marq
			   );

		  void FiltrMaxLoc_BCVS
			   (
				        ElSTDNS vector<Pt2di> & Pts,
					Im2D<Type,TypeBase> Im,
					REAL  FactInf,
					REAL  TolGeom,
                    Pt2di SzVois,
					Im2D_U_INT1 Marq
			   );

    private :

          Compare  mCmp;
		  std::vector<Pt2di>  mBufCC;
		  std::vector<Pt2di>  mBufFiltr;

		  bool CmpTot(Type v1,Type v2,Pt2di p1,Pt2di p2);

};





#endif //  _ELISE_IMTPL_MAX_LOC_H_

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
