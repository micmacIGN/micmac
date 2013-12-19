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



/*********************************************************************/
/*                                                                   */
/*         Neighbourhood                                             */
/*                                                                   */
/*********************************************************************/


Neighbourhood Neighbourhood::v4()
{
   return Neighbourhood(TAB_4_NEIGH,4);
}
Neighbourhood Neighbourhood::v8()
{
    return Neighbourhood(TAB_8_NEIGH,8);
}


Neighbourhood::Neighbourhood(Im2D<INT4,INT> I) :
   PRC0(new Data_Neighbourood(I))
{
}

Neighbourhood::Neighbourhood(Pt2di * pts,INT nb) :
   PRC0(new Data_Neighbourood(pts,nb))
{
}


Data_Neighbourood * Neighbourhood::data_n()
{
    return SAFE_DYNC(Data_Neighbourood *,_ptr);
}

/*********************************************************************/
/*                                                                   */
/*         Data_Neighbourood                                         */
/*                                                                   */
/*********************************************************************/


void Data_Neighbourood::init(INT dim,INT nb) 
{
    _dim       = dim;
    _nb_neigh  = nb;
    _coord     = NEW_MATRICE_ORI(nb+1,dim,INT);
    _tr_coord  = NEW_MATRICE_ORI(dim,nb+1,INT);
}

void Data_Neighbourood::init_tr()
{
	INT d;
    for (d=0 ; d < _dim ; d++)
       _coord[d][_nb_neigh] = 0;

    for (d=0 ; d < _dim ; d++)
        for (int n=0 ; n <= _nb_neigh ; n++)
            _tr_coord[n][d] = _coord[d][n];
}


Data_Neighbourood::Data_Neighbourood(Im2D<INT4,INT> I)
{
    init(I.ty(),I.tx());
    
    for (int d=0 ; d < _dim ; d++)
        convert(_coord[d],I.data()[d],_nb_neigh);
    init_tr();
}

Data_Neighbourood::Data_Neighbourood(Pt2di * pts,INT nb)
{
    init(2,nb);

    for (int n=0 ; n < _nb_neigh ; n++)
    {
        _coord[0][n] = pts[n].x;
        _coord[1][n] = pts[n].y;
    }
    init_tr();
}

Data_Neighbourood::~Data_Neighbourood()
{
      DELETE_MATRICE_ORI(_coord,_nb_neigh+1,_dim);
      DELETE_MATRICE_ORI(_tr_coord,_dim,_nb_neigh+1);
}

/*********************************************************************/
/*                                                                   */
/*         Arg_Neigh_Rel_Comp                                        */
/*                                                                   */
/*********************************************************************/

Arg_Neigh_Rel_Comp::Arg_Neigh_Rel_Comp(Flux_Pts_Computed * flux,bool reflex) :
      _flux     (flux) ,
      _reflexif (reflex)
{
}

/*********************************************************************/
/*                                                                   */
/*         Neigh_Rel_Compute                                         */
/*                                                                   */
/*********************************************************************/

Neigh_Rel_Compute::Neigh_Rel_Compute
         (  const Arg_Neigh_Rel_Comp & arg,
            Data_Neighbourood * neigh,
            Pack_Of_Pts::type_pack  _type_pack,
            INT Sz_buf) :

          _pack (Pack_Of_Pts::new_pck
                      (  arg.flux()->dim(),
                         Sz_buf,
                         _type_pack
                      )
                ),
          _neigh (neigh),
          _type_pack (_type_pack),
          _sz_buf    (Sz_buf)
{
     ASSERT_TJS_USER
     (
            (neigh->_dim == arg.flux()->dim()),
            "incoherence between flux and neighbourood  dimensions"
     );
}


Neigh_Rel_Compute::~Neigh_Rel_Compute()
{
   delete _pack;
}


/*********************************************************************/
/*                                                                   */
/*         Simple_Neigh_Rel_Comp                                     */
/*                                                                   */
/*********************************************************************/

class Simple_Neigh_Rel_Comp : public Neigh_Rel_Compute
{
  public :

      void set_reflexif(bool){};

      Simple_Neigh_Rel_Comp(const Arg_Neigh_Rel_Comp &,Data_Neighbourood *);

      virtual const Pack_Of_Pts * neigh_in_num_dir
                           ( const Pack_Of_Pts * pack_0,
                             char ** is_neigh,
                             INT & num_dir)
      {
            if (is_neigh)
               set_cste(is_neigh[num_dir],(char)1,(INT)pack_0->nb());
            _pack->trans(pack_0,_neigh->_tr_coord[num_dir++]);
            return _pack;
      }


};


Simple_Neigh_Rel_Comp::Simple_Neigh_Rel_Comp
         (const Arg_Neigh_Rel_Comp & arg,Data_Neighbourood * neigh) :
              Neigh_Rel_Compute(arg,neigh,arg.flux()->type(),arg.flux()->sz_buf())
{
}



/*********************************************************************/
/*                                                                   */
/*         Simple_Neigh_Rel_Not_Comp                                 */
/*                                                                   */
/*********************************************************************/

class Simple_Neigh_Rel_Not_Comp : public Neigh_Rel_Not_Comp
{
   public :
       Simple_Neigh_Rel_Not_Comp(Neighbourhood,Data_Neighbourood *);

   private :


      Neigh_Rel_Compute * compute 
             (const Arg_Neigh_Rel_Comp & arg)
      {
            return new Simple_Neigh_Rel_Comp(arg,_neigh);
      };

   // -----------------------------data----------------------
      Neighbourhood _neighood;
      Data_Neighbourood *_neigh;
};


Simple_Neigh_Rel_Not_Comp::Simple_Neigh_Rel_Not_Comp
        (Neighbourhood neighood,Data_Neighbourood * neigh) :
            _neighood (neighood),
            _neigh    (neigh)
{
}


/*********************************************************************/
/*                                                                   */
/*         Neigh_Rel                                                 */
/*                                                                   */
/*********************************************************************/

Neigh_Rel::Neigh_Rel(Neigh_Rel_Not_Comp * NRNC) :
    PRC0(NRNC)
{
}

Neigh_Rel::Neigh_Rel(Neighbourhood neigh) :
     PRC0(new Simple_Neigh_Rel_Not_Comp(neigh,neigh.data_n()))
{
}

Neigh_Rel_Compute * Neigh_Rel::compute(const class Arg_Neigh_Rel_Comp & arg)
{
    return (SAFE_DYNC(Neigh_Rel_Not_Comp *,_ptr))->compute(arg);
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
