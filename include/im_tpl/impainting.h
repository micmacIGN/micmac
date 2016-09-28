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



#ifndef _ELISE_IMTPL_IMPAINTING_H_
#define _ELISE_IMTPL_IMPAINTING_H_

/*
    Impainting est un peu prétentieux, il s'agit de combler des trous dans une image (mais sans idee de realisme).
    Trois methode :

       1- L2 qui minimise la courbure ?
       2- Au plus proche voisin
       3- Completion "par le bas" (adapte au bouchage de MNT si les zones a combler sont des parties cachees)

*/

template  <class Type,class Type_Base>  class cImpainting
{
      public :



         cImpainting
         ( 
              Im2D_Bits<1>          aMaskInit,
              Im2D_Bits<1>          aMaskFinale,
              Im2D<Type,Type_Base>  anImVal
         );

         void ComplKLips(double aDynZ);  // aDynZ=0 => Approx au  PPV ; aDynZ infini  => au  min du void composante conexe

         TIm2DBits<1>             mTMaskInit;
         TIm2DBits<1>             mTMaskFinal;
         TIm2D <Type,Type_Base>   mTImVal;
         Pt2di                    mSz;
         Im2D<REAL4,REAL8>        mImBuf;
         TIm2D<REAL4,REAL8>       mTImBuf;

         std::vector<Pt2di>       mVPts;
         int                      mNbPts;

      private :
};

template  <class Type,class Type_Base>  
void  ComplKLipsParLBas
       (
              Im2D_Bits<1>          aMaskInit,
              Im2D_Bits<1>          aMaskFinale,
              Im2D<Type,Type_Base>  anImVal,
              double                aDynZ
       )
{
    cImpainting<Type,Type_Base> aIP(aMaskInit,aMaskFinale,anImVal);
    aIP.ComplKLips(aDynZ);
}




template  <class Type,class Type_Base>  
    void cImpainting<Type,Type_Base>::ComplKLips(double aDynZ)
{
   Pt2di aP;
   for (aP.y =0 ; aP.y<mSz.y ; aP.y++)
   {
      for (aP.x =0 ; aP.x<mSz.x ; aP.x++)
      {
           double aVal = mTMaskInit.get(aP) ? (mTImVal.get(aP) *aDynZ) : 1e20;
           mTImBuf.oset(aP,aVal);
      }
   }

    for (int aKIter=0 ; aKIter< 6 ; aKIter++)
   {
       int aNbUpdate = 0;
       bool Pair= ((aKIter%2)==0);

       int IndDeb = Pair ? 0       : (mNbPts-1);
       int IndOut = Pair ? mNbPts  : (-1)      ;
       int Incr   = Pair ? 1       : (-1)      ;

       for (int Ind=IndDeb ; Ind!=IndOut ; Ind+=Incr)
       {
            Pt2di aP2Cur = mVPts[Ind];
            Type aValOfZMin = (Type)mTImVal.get(aP2Cur);
            double aZMin = mTImBuf.get(aP2Cur);
            for (int aKV = 0 ; aKV<8 ; aKV++)
            {
                Pt2di aPVois = aP2Cur + TAB_8_NEIGH[aKV];
                if (mTMaskFinal.get(aPVois) || mTMaskInit.get(aPVois))
                {
                     double aZAugm = mTImBuf.get(aPVois) + ((aKV%2) ? 3 : 2);
                     if (aZAugm < aZMin)
                     {
                          aZMin = aZAugm;
                          aValOfZMin = (Type)mTImVal.get(aPVois);
                          aNbUpdate++;
                     }
                }
            }
            mTImVal.oset(aP2Cur,aValOfZMin);
            mTImBuf.oset(aP2Cur,aZMin);
       }
       // std::cout << "NNNbUodta " << aNbUpdate << "\n";
   }
}

template  <class Type,class Type_Base>  
cImpainting<Type,Type_Base>::cImpainting
(
      Im2D_Bits<1>          aMaskInit,
      Im2D_Bits<1>          aMaskFinal,
      Im2D<Type,Type_Base>  anImVal
)  :
     mTMaskInit    (aMaskInit),
     mTMaskFinal   (aMaskFinal),
     mTImVal       (anImVal),
     mSz           (anImVal.sz()),
     mImBuf        (mSz.x,mSz.y) ,
     mTImBuf       (mImBuf)
{
    ELISE_ASSERT
    (
          (aMaskInit.sz()==mSz)&&(aMaskFinal.sz()==mSz),
          "cImpainting  Sz incoherent"
    );


   Pt2di aP;
   for (aP.y =1 ; aP.y<(mSz.y-1) ; aP.y++)
   {
      for (aP.x =1 ; aP.x<(mSz.x-1) ; aP.x++)
      {
           if (mTMaskFinal.get(aP) && (!mTMaskInit.get(aP)))
           {
               mVPts.push_back(aP);
           }
      }
   }
   mNbPts = (int)mVPts.size();
}
Im2D_REAL4 RecursiveImpaint
     (
          Im2D_REAL4 aFlMaskInit,
          Im2D_REAL4 aFlMaskFinal,
          Im2D_REAL4 aFlIm,
          int        aDeZoom,
          int        aZoomCible
     );

template <class TypeIn,class TypeOut>
Im2D<TypeIn,TypeOut> ImpaintL2
     (
         Im2D_Bits<1>           aB1MaskInit,
         Im2D_Bits<1>           aB1MaskFinal,
         Im2D<TypeIn,TypeOut>   anIn,
         int                    aDezoom = 16
     )
{
   Im2D_REAL4  aFlRes = RecursiveImpaint
                         (
                             Conv2Type(aB1MaskInit,(Im2D_REAL4*)0),
                             Conv2Type(aB1MaskFinal,(Im2D_REAL4*)0),
                             Conv2Type(anIn,(Im2D_REAL4*)0),
                             1,
                             aDezoom
                         );


   return Conv2Type(aFlRes,(Im2D<TypeIn,TypeOut>*)0);
}

template <class Type,class TypeBase> Im2D<Type,TypeBase>  BouchePPV(Im2D<Type,TypeBase> anIm,Fonc_Num aFoncIn)
{
   Pt2di aSz = anIm.sz();
   Im2D<Type,TypeBase> aRes(aSz.x,aSz.y);

   cResProj32 aP32 = Projection32(aFoncIn,aSz);

   ELISE_COPY
   (
        aRes.all_pts(),
        anIm.in()[Virgule(aP32.PX().in(),aP32.PY().in())],
        aRes.out()
   );

   return aRes;
}



    



#endif  //  _ELISE_IMTPL_IMPAINTING_H_











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
