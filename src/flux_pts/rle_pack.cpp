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
/*         RLE_Pack_Of_Pts                                           */
/*                                                                   */
/*********************************************************************/

RLE_Pack_Of_Pts::RLE_Pack_Of_Pts(INT dim,INT sz_buf) :
      Pack_Of_Pts(dim,sz_buf,Pack_Of_Pts::rle)
{
    _tpr = alloc_tab_prov_int_small(dim);
    _pt0 = _tpr->coord();
}

RLE_Pack_Of_Pts::~RLE_Pack_Of_Pts(void)
{
      delete _tpr;
}


void RLE_Pack_Of_Pts::show(void) const
{

   cout << "{[" << _pt0[0] << " ; " << _pt0[0]+_nb << "[ " ;
   for (int i=1 ; i<_dim ; i++)
       cout << _pt0[i] << " ";
   cout << "}\n";
}

void RLE_Pack_Of_Pts::show_kth(INT k) const
{

   cout << "[" << _pt0[0]+k;
   for (int i=1 ; i<_dim ; i++)
       cout << " " << _pt0[i];
   cout << "]";
}

RLE_Pack_Of_Pts * RLE_Pack_Of_Pts::new_pck(INT dim,INT sz_buf) 
{
   return new RLE_Pack_Of_Pts(dim,sz_buf);
}


void RLE_Pack_Of_Pts::set_pt0(Pt2di p)
{
     _pt0[0] = p.x ;
     _pt0[1] = p.y ;
}

void RLE_Pack_Of_Pts::set_pt0(const INT4 * p)
{
    for (int i=0 ; i<_dim ; i++)
        _pt0[i] = p[i];
}




INT RLE_Pack_Of_Pts::clip(const RLE_Pack_Of_Pts * pck,
                          const INT * p0,
                          const INT * p1)
{
    for(int i = 1; i <_dim ; i++)
    {
       _pt0[i] = pck->_pt0[i];
       if (     (_pt0[i] < p0[i])
            ||  (_pt0[i] >= p1[i])
          )
        {
           _nb = 0;
           return 0; // why not, anyway, segment is empty
        }
    }

    _pt0[0] = ElMax(p0[0],pck->_pt0[0]);  // begining of segment
    INT x1  = ElMin(p1[0],pck->_pt0[0]+ pck->_nb);  // end of segment
    _nb = ElMax(0,x1-_pt0[0]);

    return _nb ? _pt0[0]-pck->_pt0[0] : 0;
}

INT RLE_Pack_Of_Pts::proj_brd
        (
              const Pack_Of_Pts * pck_gen,
              const INT * p0,
              const INT * p1,
              INT       // rab
        )
{
    const RLE_Pack_Of_Pts * pck = pck_gen->rle_cast();
    ASSERT_INTERNAL(pck->_nb,"proj_brd with empty pack");

    for(int i = 1; i <_dim ; i++)
    {
       _pt0[i] = ElMax(pck->_pt0[i],p0[i]);
       _pt0[i] = ElMin(_pt0[i],p1[i]-1);
    }

    INT xp0 = pck->_pt0[0];
    INT X0 = p0[0];
    INT X1 = p1[0];

    INT x0 =  ElMax(X0,xp0);
    if (x0 >= X1)
    {
         _pt0[0] = X1-1;
         _nb = 1;
         return 0;
    }
    INT x1 =  ElMin(X1,xp0+pck->_nb);
    if (x1 <= X0)
    {
         _pt0[0] = X0;
         _nb = 1;
         return pck->_nb-1;
    }

    _nb = x1-x0;
    ASSERT_INTERNAL(_nb>0,"incoherence in proj_brd");
    _pt0[0] = x0;

    return x0-xp0;
}

INT RLE_Pack_Of_Pts::clip(const RLE_Pack_Of_Pts * pck,Pt2di p0,Pt2di p1)
{
    INT t0[2],t1[2];
    p0.to_tab(t0);
    p1.to_tab(t1);

    return clip(pck,t0,t1);
}


bool RLE_Pack_Of_Pts::same_line (const RLE_Pack_Of_Pts * pck) const
{
     ASSERT_INTERNAL
     (
          pck->_dim == _dim,
          "different dim in RLE_Pack_Of_Pts::same_line"
     );
     ASSERT_INTERNAL
     (
          pck->_nb && _nb,
          "empty pack in RLE_Pack_Of_Pts::same_line"
     );

     for (int i=1; i<_dim ; i++)
         if (_pt0[i] != pck->_pt0[i])
            return false;

     return true; 
}



void RLE_Pack_Of_Pts::trans(const Pack_Of_Pts * pack, const INT * tr)
{
     RLE_Pack_Of_Pts * rle_pack = SAFE_DYNC(RLE_Pack_Of_Pts *,const_cast<Pack_Of_Pts *>(pack));
     ASSERT_INTERNAL
     (
          (rle_pack->_dim == _dim),
          "incoherence in RLE_Pack_Of_Pts::trans" 
     );

     _nb = rle_pack->_nb;
     for (int i =0; i <_dim ; i++)
         _pt0[i] = rle_pack->_pt0[i] + tr[i];
}

void  RLE_Pack_Of_Pts::Show_Outside(const INT * p0, const INT * p1) const
{
     std::cout << "D" << 0 << ": ";
     std::cout << p0[0] << " [" << _pt0[0] << ", "  << _pt0[0] + _nb << "[ " <<  p1[1] ;

     if (_nb && (! ((_pt0[0] >= p0[0]) &&  (_pt0[0] + _nb <= p1[0]))))
        std::cout << "   --XXXXXX";
     std::cout << "\n";

    for(int i = 1; i <_dim ; i++)
    {
       std::cout << "D" << i << ": ";
       std::cout << p0[i] << " " << _pt0[i] << " " <<  p1[i] ;
       if ((_pt0[i] < p0[i]) ||  (_pt0[i]  >= p1[i]))
           std::cout << "   --XXXXXX";
       std::cout << "\n";
    }

}

bool  RLE_Pack_Of_Pts::inside( const INT * p0, const INT * p1) const
{

    if (! _nb)
       return true;

    for(int i = 1; i <_dim ; i++)
       if ((_pt0[i] < p0[i]) ||  (_pt0[i]  >= p1[i]))
           return false;

    return (_pt0[0] >= p0[0]) &&  (_pt0[0] + _nb <= p1[0]);
}

INT  RLE_Pack_Of_Pts::ind_outside( const INT * p0, const INT * p1) const
{

    if (nb()==0)
       El_Internal.ElAssert
       (
            false,
            EEM0 << "Bad call to  RLE_Pack_Of_Pts::ind_outside, _nb =0"
       );

    for(int i = 1; i <_dim ; i++)
    {
       if (_pt0[i] < p0[i]) 
          return 0;
       if (_pt0[i]  >= p1[i])
           return 0;
    }

    if (_pt0[0] < p0[0]) 
       return 0;

    if  (_pt0[0] + _nb > p1[0])
       return _nb-1;

     El_Internal.ElAssert
     (
        false,
        EEM0 << "Bad call to  RLE_Pack_Of_Pts::ind_outside, incoherence"
     );
     return -1;
}




void RLE_Pack_Of_Pts::select_tab
                     (   Pack_Of_Pts * pack_gen,
                         const INT * tab_sel
                      ) const
{
  Std_Pack_Of_Pts<INT> * pck =
             SAFE_DYNC( Std_Pack_Of_Pts<INT> *,pack_gen);
  INT ** res = pck->_pts;
  INT * r0 = res[0];
  INT nb =pck->nb();
  INT nb0 = nb;
  INT x0 = _pt0[0];

  for (INT i=0; i< _nb; i++)
      if (tab_sel[i])
          r0[nb++] = x0 + i;

  for (int j = 1 ; j < _dim ; j++)
       set_cste(res[j]+nb0,_pt0[j],nb-nb0);
   pack_gen->set_nb(nb);
}


Pack_Of_Pts * RLE_Pack_Of_Pts::dup(INT) const
{
    elise_internal_error("call to  RLE_Pack_Of_Pts::dup",__FILE__,__LINE__);
    return 0;
}

void RLE_Pack_Of_Pts::copy (const Pack_Of_Pts * gen)
{
    const RLE_Pack_Of_Pts * p = gen->rle_cast();

    _nb = p->_nb;
     for(INT d=0 ; d<_dim ; d++)
        _pt0[d] = p->_pt0[d];
}


const INT RLE_Pack_Of_Pts::_sz_buf_infinite = 1000000000;

void RLE_Pack_Of_Pts::kth_pts(INT *p,INT k) const
{
     ASSERT_INTERNAL
     (
         (k>=0) && (k<_nb),
         "RLE_Pack_Of_Pts::kth_pts"
     );
     p[0] = _pt0[0] + k;
     for(INT d=1 ; d<_dim ; d++)
        p[d] = _pt0[d];
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
