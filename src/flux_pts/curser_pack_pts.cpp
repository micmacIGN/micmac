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
/*         STD_Curser_on_PoP                                         */
/*                                                                   */
/*********************************************************************/

class STD_Curser_on_PoP : public Curser_on_PoP
{
    public :
      STD_Curser_on_PoP(INT dim,INT sz_buf,Pack_Of_Pts::type_pack);
      virtual  ~STD_Curser_on_PoP()     
      {
           delete _curs;
      }

    private :

      virtual  Pack_Of_Pts * next()
      {
          _nb_rest -= _sz_buf;
          if (_nb_rest > 0)
          {
              for (int d=0 ; d<_dim ; d++)
                  _pts_curs[d] += _sz_buf_el;
              _curs->set_nb(ElMin(_sz_buf,_nb_rest));
              return _curs;
          }
          else
              return 0;
      }


      virtual  void re_start(const Pack_Of_Pts * pack) 
      {
          _pts_p = C_CAST(char **,pack->adr_coord());

          for (int i=0 ; i < _dim; i++)
              _pts_curs[i] = _pts_p[i]   - _sz_buf_el;
         _nb_rest = pack->nb()+_sz_buf;
      }



      char **       _pts_curs;
      char **       _pts_p;


      INT           _sz_buf_el;
      Pack_Of_Pts * _curs;

};


STD_Curser_on_PoP::STD_Curser_on_PoP(INT dim,INT sz_buf,Pack_Of_Pts::type_pack type) :
     Curser_on_PoP(dim,sz_buf,type)
{
     if (type == Pack_Of_Pts::integer)
     {
          _curs = Std_Pack_Of_Pts<INT>::new_pck(dim,0);
          _sz_buf_el = sizeof(INT);
     }
     else
     {
          _curs = Std_Pack_Of_Pts<REAL>::new_pck(dim,0);
          _sz_buf_el = sizeof(REAL) ;
     }
    _sz_buf_el *= _sz_buf;
     _pts_curs = C_CAST(char **,_curs->adr_coord());
}


/*********************************************************************/
/*                                                                   */
/*         RLE_Curser_on_PoP                                         */
/*                                                                   */
/*********************************************************************/

class RLE_Curser_on_PoP : public Curser_on_PoP
{
    public :
      RLE_Curser_on_PoP(INT dim,INT sz_buf);
      virtual  ~RLE_Curser_on_PoP()     
      {
           delete _curs;
      }

    private :

      virtual  Pack_Of_Pts * next()
      {
          _nb_rest -= _sz_buf;
          if (_nb_rest > 0)
          {
              _curs->_pt0[0] += _sz_buf;
              _curs->_nb = ElMin(_sz_buf,_nb_rest);
              return _curs;
          }
          else
              return 0;
      }


      virtual  void re_start(const Pack_Of_Pts * pack) 
      {
          _pack = SAFE_DYNC(RLE_Pack_Of_Pts *,const_cast<Pack_Of_Pts *>(pack));
          for (int i=1 ; i < _dim; i++)
              _curs->_pt0[i] = _pack->_pt0[i];
         _curs->_pt0[0] =  _pack->_pt0[0] - _sz_buf;
         _nb_rest = _pack->nb()+_sz_buf;
      }



      RLE_Pack_Of_Pts * _curs;
      RLE_Pack_Of_Pts * _pack;

};


RLE_Curser_on_PoP::RLE_Curser_on_PoP(INT dim,INT sz_buf) :
     Curser_on_PoP(dim,sz_buf,Pack_Of_Pts::rle),
     _curs        (RLE_Pack_Of_Pts::new_pck(dim,sz_buf))
{
}

/*********************************************************************/
/*                                                                   */
/*         Curser_on_PoP                                             */
/*                                                                   */
/*********************************************************************/

Curser_on_PoP::~Curser_on_PoP(){}

Curser_on_PoP::Curser_on_PoP
(
        INT dim,
        INT sz_buf,
        Pack_Of_Pts::type_pack type
) :
    _dim     (dim),
    _sz_buf  (sz_buf),
    _type    (type),
    _nb_rest (0)
{
}

Curser_on_PoP *  Curser_on_PoP::new_curs
(    
     INT dim,    
     INT sz_buf,    
     Pack_Of_Pts::type_pack type
)
{
    switch(type)
    {
         case Pack_Of_Pts::rle :
              return new RLE_Curser_on_PoP(dim,sz_buf);

         default :
              return new STD_Curser_on_PoP(dim,sz_buf,type);
    }
}

/**************************************************/
/*        Split_To_Max_Buf                        */
/**************************************************/


class Split_to_max_buf : public Flux_Pts_Computed
{
      public :
           Split_to_max_buf
           (      Flux_Pts_Computed *,
                  const Arg_Flux_Pts_Comp & arg
           );

         
           virtual ~Split_to_max_buf()
           {
              delete _flx;
              delete _curs;
           }

           virtual const Pack_Of_Pts * next(void)
           {
               for (;;)
               {
                   const Pack_Of_Pts * r;

                   r = _curs->next();
                   if (r)
                      return r;

                   r = _flx->next();
                   if (r)
                      _curs->re_start(r);
                   else
                      return 0;
               }
           }


      private :
           Flux_Pts_Computed * _flx;
           Curser_on_PoP * _curs;

           virtual  bool   is_rect_2d(Box2di & box)
           {
                    return _flx->is_rect_2d(box);
           }

           REAL average_dist()
           {
                return _flx->average_dist();
           }
};

Split_to_max_buf::Split_to_max_buf
(      Flux_Pts_Computed * flx,
       const Arg_Flux_Pts_Comp & arg
) :
     Flux_Pts_Computed(flx->dim(),flx->type(),arg.sz_buf()) ,
     _flx (flx),
     _curs (Curser_on_PoP::new_curs(flx->dim(),arg.sz_buf(),flx->type()))
{
}

Flux_Pts_Computed *  split_to_max_buf
                         (
                                Flux_Pts_Computed * flx,
                                const  Arg_Flux_Pts_Comp &   arg
                         )
{
    return new Split_to_max_buf(flx,arg);
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
