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


template <class Traits> class PackBitsGen
{

   public :

        static void  UCompr (
                                   Packed_Flux_Of_Byte * pfob,
                                   U_INT1 * res,
                                   INT nb_tot
                            );

       INT Compr (
                       Packed_Flux_Of_Byte * pfob,
                       INT nb_tot
                 );
        PackBitsGen (const U_INT1 * line) : mLine (line) {}

   private :



         bool EqInd(INT ind1,INT ind2) const
         {
              return Traits::EqualsRaw2Val(mLine+ind1*Traits::eSizeOf,mLine+ind2*Traits::eSizeOf);
         }

        const U_INT1 * mLine;
};







template <class Traits> 
void PackBitsGen<Traits>::UCompr 
     (
          Packed_Flux_Of_Byte * pfob,
          U_INT1 * RawRes,
          INT nb_tot
     ) 
{

    INT i;

    for (i=0; i<nb_tot; )
    {
        INT1 v;
        pfob->Read((U_INT1 *) &v,1);
        if (v!= -128)
        {
           INT nb = (v>=0) ? (v+1) : (1-v);
           El_Internal.ElAssert
           (
                (i+nb) <= nb_tot,
                EEM0 << "Incoherent chck_sum in Tiff_Tiles_PckBits::uncomp_line"
           );
           if (RawRes)
           {
               if (v>=0)
               {
                   pfob->Read(RawRes+i*Traits::eSizeOf,nb*Traits::eSizeOf);
               }
               else
               {
                  U_INT1 vrep[Traits::eSizeOf] ;
                  pfob->Read(vrep,Traits::eSizeOf);
                  for (INT iCh=0 ; iCh<Traits::eSizeOf ; iCh++)
                  {
                      for (INT k=0 ; k< nb; k++)
                          RawRes[(i+k)*Traits::eSizeOf+iCh] = vrep[iCh];
                  }
               }
           }
           else
           {
              pfob->Rseek(((v>=0) ? nb : 1)*Traits::eSizeOf);
           }
           i+= nb;
        }
        else
        {
        }
    }
}

void PackBitsUCompr
     (
          Packed_Flux_Of_Byte * pfob,
          U_INT1 * res,
          INT nb_tot
     )
{
    PackBitsGen<El_CTypeTraits<U_INT1> >::UCompr(pfob,res,nb_tot);
}

void PackBitsUCompr_B2
     (
          Packed_Flux_Of_Byte * pfob,
          U_INT1 * res,
          INT nb_tot
     )
{
    PackBitsGen<El_CTypeTraits<U_INT2> >::UCompr(pfob,res,nb_tot);
}






template <class Traits> 
INT PackBitsGen<Traits>::Compr 
    (
          Packed_Flux_Of_Byte * pfob,
          INT nb_tot
    )
{

     INT nb_byte = 0;


     for (INT i =0; i<nb_tot;)
     {
         INT i0 = i;
         INT lim = ElMin(nb_tot,i+128);

         /*
             Si au moins 2 element et 2 premier egaux : RLE
         */
         if ((i+1<lim) && EqInd(i,i+1))
         {
             while((i<lim) && EqInd(i,i0) )    i++;
             INT1 n = 1- (i-i0);
             pfob->Write((U_INT1 *) &n,1);
             pfob->Write(mLine+i0*Traits::eSizeOf,Traits::eSizeOf);
             nb_byte += 1+Traits::eSizeOf;
         }
         /*
              Sinon "run" litteralle
         */
         else
         {
             i++;
             bool cont = true;
             while(cont && (i<lim))
             {
                 // 2 elet dif : on continue le run
                 if (!EqInd(i,i-1)) 
                    i++;
                 // 3 elt egaux au -, on arete le run; le prochain sera rle
                 else if ((i+1 <lim) && EqInd(i,i+1))
                 {
                    cont = false;
                    i--;
                 }
                 // si run de 2, entre deux run litteraux, on le saute
                 else if
                 (
                         (i+2<lim)
                      && EqInd(i,i+1)
                      && EqInd(i+1,i+2)
                 )
                      i+=3;
                 // si run de 2, juste avant 1 dernier element, on le saute
                 else if ( i+2 == lim)
                      i+=2;
                 // sinon, run de 2= a coder en RLE
                 else
                 {
                    cont = false;
                    i--;
                 }
             }
             INT1 n =  (i-i0)-1;
             pfob->Write((U_INT1 *) &n,1);
             pfob->Write(mLine+i0*Traits::eSizeOf,(i-i0)*Traits::eSizeOf);
             nb_byte += 1+(i-i0)*Traits::eSizeOf;
         }
     }

     return nb_byte;
}

INT PackBitsCompr
     (
          Packed_Flux_Of_Byte * pfob,
          const U_INT1 * line,
          INT nb_tot
     )               
{
    PackBitsGen<El_CTypeTraits<U_INT1> >   PBG(line);
    return PBG.Compr(pfob,nb_tot);
}

INT PackBitsCompr_B2
     (
          Packed_Flux_Of_Byte * pfob,
          const U_INT1 * line,
          INT nb_tot
     )               
{

    PackBitsGen<El_CTypeTraits<U_INT2> >   PBG(line);
    return PBG.Compr(pfob,nb_tot);
}




/***********************************************************/
/*                                                         */
/*            Pack_Bits_Flow                               */
/*                                                         */
/***********************************************************/
Pack_Bits_Flow::Pack_Bits_Flow
(
     Packed_Flux_Of_Byte * flx,          // compressed flow
     bool read,
     INT  tx
)   :
    Packed_Flux_Of_Byte(1),
    _flx  (flx),
    _read (read),
    _buf  (NEW_VECTEUR(0,tx,U_INT1)),
    _tx   (tx),
    _n    (0)
{
    El_Internal.ElAssert
    (
       (!_read),
       EEM0 << "do not handle Pack_Bits_Flow::Read"
    );
}

Pack_Bits_Flow::~Pack_Bits_Flow()
{
    DELETE_VECTOR(_buf,0);
    delete _flx;
}


bool Pack_Bits_Flow::compressed() const
{
    return true;
}

tFileOffset Pack_Bits_Flow::Read(U_INT1 * ,tFileOffset nb)
{
    El_Internal.ElAssert
    (
       _read,
       EEM0 << "bad call to Pack_Bits_Flow::Read"
    );
    return nb;
}


tFileOffset Pack_Bits_Flow::Write(const U_INT1 * vals ,tFileOffset nbo)
{
    int nb = nbo.CKK_IntBasicLLO();
    El_Internal.ElAssert
    (
       ! _read,
       EEM0 << "bad call to Pack_Bits_Flow::Write"
    );

    for (int i = 0; i< nb ; i++)
    {
        _buf[_n++] = vals[i];
        if (_n == _tx)
           reset();
    }
    return nb;
}

void Pack_Bits_Flow::reset()
{
   if (_n)
      PackBitsCompr(_flx,_buf,_n);
   _n = 0;
}

tFileOffset Pack_Bits_Flow::tell()
{
    El_Internal.ElAssert
    (
       false,
       EEM0 << "no Pack_Bits_Flow::tell"
    );
    return 0;
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
