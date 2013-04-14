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
#include "private/all.h"
#include "Digeo.h"

namespace NS_ParamDigeo
{

/****************************************/
/*                                      */
/*             cTplImInMem               */
/*                                      */
/****************************************/

 
template <class Type> template <class TMere> void  cTplImInMem<Type>::MakeReduce_121(const cTplImInMem<TMere> & aMere)
{
   Resize(aMere.Sz()/2);
   TFlux_BordRect2d aRect(Pt2di(0,0),mSz);

   Pt2di aPt = aRect.PtsInit();
   typename cTplImInMem<TMere>::tTIm  aImM(aMere.TIm());
   while(aRect.next(aPt))
   {
       Pt2di aP2 = aPt *2;
       mTIm.oset
       (
         aPt,
         (        aImM.getproj(Pt2di(aP2.x-1,aP2.y-1))
           +  2 * aImM.getproj(Pt2di(aP2.x  ,aP2.y-1))
           +      aImM.getproj(Pt2di(aP2.x+1,aP2.y-1))
           +  2 * aImM.getproj(Pt2di(aP2.x-1,aP2.y  ))
           +  4 * aImM.getproj(Pt2di(aP2.x  ,aP2.y  ))
           +  2 * aImM.getproj(Pt2di(aP2.x+1,aP2.y  ))
           +      aImM.getproj(Pt2di(aP2.x-1,aP2.y+1))
           +  2 * aImM.getproj(Pt2di(aP2.x  ,aP2.y+1))
           +      aImM.getproj(Pt2di(aP2.x+1,aP2.y+1))
         ) / 16
       );
   }

   for (int anY=1 ; anY<mSz.y-1; anY++)
   {
      Type * aDOut = mIm.data()[anY] + 1;
      const TMere * aDInM1 = aMere.TIm().data()[2*anY-1] + 2;
      const TMere * aDIn0  = aMere.TIm().data()[2*anY  ] + 2;
      const TMere * aDInP1 = aMere.TIm().data()[2*anY+1] + 2;
      for (int anX=1 ; anX<mSz.x-1; anX++)
      {
          *aDOut = (
                       aDInM1[-1] + 2*aDInM1[0] +  aDInM1[1]
                    + 2*aDIn0[-1] + 4*aDIn0[0]  + 2*aDIn0[1]
                    +  aDInP1[-1] + 2*aDInP1[0] +  aDInP1[1]
                    ) / 16;
                       
            aDOut++;
            aDInM1 +=2;
            aDIn0  +=2;
            aDInP1 +=2;
      }
   }
}

template <class Type> void  cTplImInMem<Type>::VMakeReduce_121(cImInMem & aMere)
{
    if (aMere.TypeEl()==GenIm::u_int1)
    {
        MakeReduce_121(static_cast<cTplImInMem<U_INT1> &> (aMere));
        return;
    }

    if (aMere.TypeEl()==GenIm::u_int2)
    {
        MakeReduce_121(static_cast<cTplImInMem<U_INT2> &> (aMere));
        return;
    }

    if (aMere.TypeEl()==GenIm::real4)
    {
        MakeReduce_121(static_cast<cTplImInMem<REAL4> &> (aMere));
        return;
    }

    ELISE_ASSERT(false,"::VMakeReduce");
}






     //  ======================================================================

template <class Type> template <class TMere> void  cTplImInMem<Type>::MakeReduce_010(const cTplImInMem<TMere> & aMere)
{
   Resize(aMere.Sz()/2);
   TFlux_BordRect2d aRect(Pt2di(0,0),mSz);

   Pt2di aPt = aRect.PtsInit();
   typename cTplImInMem<TMere>::tTIm  aImM(aMere.TIm());
   while(aRect.next(aPt))
   {
       Pt2di aP2 = aPt *2;
       mTIm.oset ( aPt, aImM.getproj(Pt2di(aP2.x,aP2.y)));
   }

   for (int anY=1 ; anY<mSz.y-1; anY++)
   {
      Type * aDOut = mIm.data()[anY] + 1;
      const TMere * aDIn0  = aMere.TIm().data()[2*anY  ] + 2;
      for (int anX=1 ; anX<mSz.x-1; anX++)
      {
          *aDOut = aDIn0[0];
          aDOut++;
          aDIn0  +=2;
      }
   }
}



template <class Type> void  cTplImInMem<Type>::VMakeReduce_010(cImInMem & aMere)
{
    if (aMere.TypeEl()==GenIm::u_int1)
    {
        MakeReduce_010(static_cast<cTplImInMem<U_INT1> &> (aMere));
        return;
    }

    if (aMere.TypeEl()==GenIm::u_int2)
    {
        MakeReduce_010(static_cast<cTplImInMem<U_INT2> &> (aMere));
        return;
    }

    if (aMere.TypeEl()==GenIm::real4)
    {
        MakeReduce_010(static_cast<cTplImInMem<REAL4> &> (aMere));
        return;
    }

    ELISE_ASSERT(false,"::VMakeReduce");
}

template <class Type> template <class TMere> void  cTplImInMem<Type>::MakeReduce_11(const cTplImInMem<TMere> & aMere)
{
   Resize(aMere.Sz()/2);
   TFlux_BordRect2d aRect(Pt2di(0,0),mSz);

   Pt2di aPt = aRect.PtsInit();
   typename cTplImInMem<TMere>::tTIm  aImM(aMere.TIm());
   while(aRect.next(aPt))
   {
       Pt2di aP2 = aPt *2;
       mTIm.oset
       (
         aPt,
           (      aImM.getproj(Pt2di(aP2.x,aP2.y))
           +      aImM.getproj(Pt2di(aP2.x+1,aP2.y))
           +      aImM.getproj(Pt2di(aP2.x,aP2.y+1))
           +      aImM.getproj(Pt2di(aP2.x+1,aP2.y+1))
         ) / 4
       );
   }

   for (int anY=1 ; anY<mSz.y-1; anY++)
   {
      Type * aDOut = mIm.data()[anY] + 1;
      const TMere * aDIn0  = aMere.TIm().data()[2*anY  ] + 2;
      const TMere * aDInP1 = aMere.TIm().data()[2*anY+1] + 2;
      for (int anX=1 ; anX<mSz.x-1; anX++)
      {
          *aDOut = ( aDIn0[0]  + aDIn0[1] +   aDInP1[0] +  aDInP1[1]) / 4;
                       
           aDOut++;
           aDIn0  +=2;
           aDInP1 +=2;
      }
   }
}

template <class Type> void  cTplImInMem<Type>::VMakeReduce_11(cImInMem & aMere)
{
    if (aMere.TypeEl()==GenIm::u_int1)
    {
        MakeReduce_11(static_cast<cTplImInMem<U_INT1> &> (aMere));
        return;
    }

    if (aMere.TypeEl()==GenIm::u_int2)
    {
        MakeReduce_11(static_cast<cTplImInMem<U_INT2> &> (aMere));
        return;
    }

    if (aMere.TypeEl()==GenIm::real4)
    {
        MakeReduce_11(static_cast<cTplImInMem<REAL4> &> (aMere));
        return;
    }

    ELISE_ASSERT(false,"::VMakeReduce");
}




template  class cTplImInMem<U_INT1>;
template  class cTplImInMem<U_INT2>;
template  class cTplImInMem<INT>;
template  class cTplImInMem<float>;

/****************************************/
/*                                      */
/*             cImInMem                 */
/*                                      */
/****************************************/

void  cImInMem::MakeReduce(cImInMem & aMere,eReducDemiImage aMode)
{
   switch (aMode)
   {
        case eRDI_121 :
             VMakeReduce_121(aMere);
        return;

        case eRDI_010 :
             VMakeReduce_010(aMere);
        return;

        case eRDI_11 :
             VMakeReduce_11(aMere);
        return;
   }
   ELISE_ASSERT(false,"Bad Value in cImInMem::MakeReduce");
}

/*
void  cImInMem::MakeReduce()
{
    VMakeReduce(*mMere);
}
*/

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
