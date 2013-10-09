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


/***********************************************************************/
/*                                                                     */
/*       ImOutNotComp                                                  */
/*                                                                     */
/***********************************************************************/



Output_Computed * ImOutNotComp::compute(const Arg_Output_Comp & arg)
{
     Tjs_El_User.ElAssert
     (
          arg.flux()->dim() == _gi->dim(),
          EEM0 << " Bitmap writing : dim of flux should equal dim of bitmap\n"
               << "|    dim flux = " <<  arg.flux()->dim()
               << " , dim bitmap = " <<  _gi->dim()
     );

     if (arg.flux()->type() ==  Pack_Of_Pts::rle)
     {
          Output_Computed * res;
          if (arg.fonc()->integral())
             res = new ImOutRLE_Comp<INT>(_gi);
          else
             res = new ImOutRLE_Comp<REAL>(_gi);

           if (_auto_clip_rle)
              res = clip_out_put(res,arg,_gi->p0(),_gi->p1());

           return res;
     }
   
     if (arg.flux()->type() ==  Pack_Of_Pts::integer)
     {
          Output_Computed * res;

          if (_gi->integral_type())
             res = new ImOutInteger<INT> (arg,_gi,_auto_clip_int) ;
          else
	     res = new ImOutInteger<REAL>(arg,_gi,_auto_clip_int) ;
          res =
              out_adapt_type_fonc
              (
                  arg,
                  res,
                 _gi->integral_type()         ? 
                        Pack_Of_Pts::integer  : 
                        Pack_Of_Pts::real
              );

          if (_auto_clip_int)
             res = clip_out_put(res,arg,_gi->p0(),_gi->p1());


         return res;
     }


     elise_fatal_error("cannot handle REAL Pts in bitmap output",__FILE__,__LINE__);

     return 0;
}

ImOutNotComp::ImOutNotComp(DataGenIm * gi,GenIm pgi,bool auto_clip_rle,bool auto_clip_int) :
   _gi (gi),
   _pgi (pgi),
   _auto_clip_rle (auto_clip_rle),
   _auto_clip_int (auto_clip_int)
{
}



/***********************************************************************/
/*                                                                     */
/*       GenImOutRleComp                                               */
/*                                                                     */
/***********************************************************************/

GenImOutRLE_Comp::~GenImOutRLE_Comp(){}

GenImOutRLE_Comp::GenImOutRLE_Comp(DataGenIm *gi)  :
     Output_Computed(1),
     _gi (gi)
{
}


/***********************************************************************/
/*                                                                     */
/*       ImInNotComp                                                   */
/*                                                                     */
/***********************************************************************/


bool ImInNotComp::integral_fonc(bool iflux) const
{
   return    iflux && _gi->integral_type();
}

INT  ImInNotComp::dimf_out() const
{
   return    1;
}

void ImInNotComp::VarDerNN(ElGrowingSetInd &)const 
{
    ELISE_ASSERT(false,"No VarDerNN");
}



ImInNotComp::ImInNotComp(DataGenIm * gi,GenIm pgi,bool with_def,REAL def) :
    _with_def_value  (with_def),
    _def_value      (def),
    _gi               (gi) ,
    _pgi               (pgi) 
{
}



Fonc_Num_Computed * ImInNotComp::compute(const Arg_Fonc_Num_Comp & arg)
{
     Tjs_El_User.ElAssert
     (
          arg.flux()->dim() == _gi->dim(),
          EEM0 << " Bitmap reading : dim of flux should equal dim of bitmap\n"
               << "|    dim flux = " <<  arg.flux()->dim()
               << " , dim bitmap = " <<  _gi->dim()
     );




     switch (arg.flux()->type())
     {
         case Pack_Of_Pts::rle:
         {
           Fonc_Num_Computed * res; 
		   
           if (_gi->integral_type())
              res = new ImInRLE_Comp<INT>(arg,arg.flux(),_gi,_with_def_value);
           else 
              res = new ImInRLE_Comp<REAL>(arg,arg.flux(),_gi,_with_def_value);


           if( _with_def_value)
              res =  clip_fonc_num_def_val
                     (
                         arg,
                         res,
                         arg.flux(),
                         _gi->p0(),
                         _gi->p1(),
                         _def_value
                     );
           return  res ;
           break;
         }

 

        case Pack_Of_Pts::integer:
        {
            Fonc_Num_Computed * res;

            if (_gi->integral_type())
                res = new ImInInteger<INT>(arg,_gi,_with_def_value);
            else 
                res = new ImInInteger<REAL>(arg,_gi,_with_def_value);

            if (_with_def_value)
               res = clip_fonc_num_def_val
                     (
                          arg,
                          res,
                          arg.flux(),
                          _gi->p0(),
                          _gi->p1(),
                          _def_value
                     );
           return res;
           break;
        }

        case Pack_Of_Pts::real:
        {
           Fonc_Num_Computed * res = new ImInReal(arg,_gi,_with_def_value);

           if (_with_def_value)
              res = clip_fonc_num_def_val
                    (
                       arg,
                       res,
                       arg.flux(),
                       _gi->p0(),
                       _gi->p1(),
                       _def_value,
                       0.0,
                       1.0
                    );
           return res;
           break;
        }

        default :
            elise_fatal_error( "unknown point mode ImInNotComp",__FILE__,__LINE__);
            return 0;
     };
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
