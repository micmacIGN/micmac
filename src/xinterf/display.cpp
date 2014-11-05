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



/*****************************************************************/
/*                                                               */
/*                  Data_Disp_Pallete                            */
/*                                                               */
/*****************************************************************/


Data_Disp_Pallete::Data_Disp_Pallete
(
      Data_Elise_Raster_D * ded,
      Elise_Palette p,
      unsigned long * lut
)
{
    _dep = p.dep();
    _i1  = _dep->i1();
    _i2  = _dep->i2();

    switch (ded->_cmod)
    {
        case Indexed_Colour:
        {
             _lut_compr = NEW_VECTEUR(_i1,_i2,U_INT2);

              for (INT i=_i1; i<_i2 ; i++)
                 _lut_compr[i] = (U_INT2) lut[_dep->compr_indice(i)];
        }
        break;

        case True_16_Colour:
        {

             if (_dep->dim_pal() == 1)
             {
                _lut_compr = NEW_VECTEUR(_i1,_i2,U_INT2);
                 for (INT i=_i1; i<_i2 ; i++)
                 {
                      Elise_colour c = _dep->rkth_col(i,(i-_i1)/(_i2-_i1-1.0));
                      INT Rouge = round_ni(c.r()*255);
					// Pourquoi une formule != ??? Apparemment un bug du
					// compilo
                      INT Vert = ElMax(0,ElMin(255,(INT)(c.g()*255)));
                      INT Bleu = round_ni(c.b()*255);
                      if (_dep->is_gray_pal())
                      {
                          INT mm =ElMin3(ded->_r_mult,ded->_g_mult,ded->_b_mult);
                          mm = 256/ mm;
                          INT gr = (Rouge/mm) * mm;
                         _lut_compr[i] =  ded->rgb_to_16(gr,gr,gr);
                      }
                      else
					  {
                         _lut_compr[i] = ded->rgb_to_16(Rouge,Vert,Bleu);
					  }
                 }
             }
             else
             {
                _lut_compr = 0;
                _dep->init_true();
             }
        }
        break;


        case True_24_Colour:
        {
             _lut_compr = 0;
             _dep->init_true();
        }
        break;

    }
}

Data_Disp_Pallete::~Data_Disp_Pallete()
{
     if (_lut_compr)
        DELETE_VECTOR(_lut_compr,_i1);
}

/*****************************************************************/
/*                                                               */
/*                  Data_Disp_Set_Of_Pal                         */
/*                                                               */
/*****************************************************************/

Data_Disp_Set_Of_Pal::Data_Disp_Set_Of_Pal
       (Data_Elise_Raster_D * ded,Elise_Set_Of_Palette esop,unsigned long * pix_ind) :
    _esop         (esop),
    _last_ddp     (0),
    _last_dep     (0),
    _derd         (ded)
{
    _nb   = esop.lp().card();
    _ddp = NEW_VECTEUR(0,_nb,Data_Disp_Pallete);

    L_El_Palette l = esop.lp();
    INT          nb_tot = 0;
    for
    (
         INT i= 0; 
         i < _nb                                 ;
         i++
    )
   {
       new (_ddp+i) Data_Disp_Pallete (ded,l.car(),pix_ind+nb_tot);
       nb_tot += l.car().nb();
       l = l.cdr();
   }
}



Data_Disp_Set_Of_Pal::~Data_Disp_Set_Of_Pal()
{
    for (INT i=0 ; i<_nb ; i++)
        (_ddp+i)->~Data_Disp_Pallete();
    DELETE_VECTOR(_ddp,0);
}

Data_Disp_Pallete * Data_Disp_Set_Of_Pal::_priv_ddp_of_dep
                    (Data_Elise_Palette * dep,bool svp) 
{
   for (INT i=0 ; i<_nb ;i++)
       if (dep == _ddp[i]._dep)
       {
          _last_dep = dep;
          return (_last_ddp = _ddp+i);
       }

   if (! svp)
       elise_internal_error("Data_Disp_Set_Of_Pal::ddp",__FILE__,__LINE__);
   return 0;
}


INT Data_Disp_Set_Of_Pal::get_tab_col(Elise_colour * col,INT nb_max)
{
    INT res =0;
    for (INT kp=0; kp<_nb ; kp++)
    {
         Data_Disp_Pallete *  ddp  = kth_ddp(kp);
         Data_Elise_Palette * dep  = ddp->dep_of_ddp(); 
         for (INT kc=0 ; kc<dep->nb() ; kc++)
         {
              ELISE_ASSERT(res<nb_max,"Data_Disp_Set_Of_Pal::get_tab_col");
              col[res++] = dep->kth_col(kc);
         }
    } 

    return res;
}

/*****************************************************************/
/*                                                               */
/*                  Disp_Set_Of_Pal                              */
/*                                                               */
/*****************************************************************/

Disp_Set_Of_Pal::Disp_Set_Of_Pal(Data_Disp_Set_Of_Pal * DDSOP) :
     PRC0(DDSOP)
{
     
}


Data_Disp_Set_Of_Pal * Disp_Set_Of_Pal::ddsop()
{
    return SAFE_DYNC(Data_Disp_Set_Of_Pal *,_ptr);
}




/*****************************************************************/
/*                                                               */
/*                  Data_Elise_Gra_Disp                          */
/*                                                               */
/*****************************************************************/


Data_Elise_Gra_Disp::Data_Elise_Gra_Disp() :
    _last_cp           (Data_Col_Pal()),
    _last_line_width   (-1e8),
    _auto_flush        (true),
    _nb_graw           (0),
    _nb_geow           (0)
{
}

void Data_Elise_Gra_Disp::reinit_cp()
{
    _last_line_width = -1e8;
    _last_cp = Data_Col_Pal();
}

void Data_Elise_Gra_Disp::degd_flush()
{
    reinit_cp();
}
/*****************************************************************/
/*                                                               */
/*                  Data_Elise_Raster_D                          */
/*                                                               */
/*****************************************************************/

Data_Disp_Set_Of_Pal * 
       Data_Elise_Raster_D::get_comp_pal(Elise_Set_Of_Palette sop)
{
       for 
       ( 
            L_Disp_SOP l = _ldsop  ;
            !(l.empty())           ;
            l = l.cdr()
       )
       {
            Data_Disp_Set_Of_Pal * res = l.car().ddsop();
            if (res->esop() == sop)
               return res;
       }

       augmente_pixel_index(sop.som_nb_col());
       
       Data_Disp_Set_Of_Pal * res = new Data_Disp_Set_Of_Pal(this,sop,_pix_ind);
       _ldsop = _ldsop + Disp_Set_Of_Pal(res);

       return res;
}

Data_Elise_Raster_D::Data_Elise_Raster_D(const char *,INT nb_pix_ind_max) :
    Data_Elise_Gra_Disp(),
   _ldsop (L_Disp_SOP()),
   _nb_pix_ind_max (nb_pix_ind_max),
   _pix_ind (NEW_VECTEUR(0,nb_pix_ind_max,unsigned long)),
   _nb_pix_ind (0),
   _cur_coul   (-19923456)
{
}

Data_Elise_Raster_D::~Data_Elise_Raster_D()
{
     degd_flush();
     DELETE_VECTOR(_pix_ind,0);
}

U_INT1 * Data_Elise_Raster_D::alloc_line_buf(INT length)
{
        switch(_cmod)
        {
              case Indexed_Colour :
                   return NEW_VECTEUR(0,length,U_INT1);

              case True_16_Colour :
                   return (U_INT1 *) NEW_VECTEUR(0,length,U_INT2);

              case True_24_Colour :
                   return  NEW_VECTEUR(0,length*_byte_pp,U_INT1);
        }
        return 0;
}

void Data_Elise_Raster_D::init_mode(INT depth)
{
    _depth = depth;
     switch (_depth)
    {
           case 8:
           {
              _cmod     =   Indexed_Colour;
              _byte_pp  =   1;
           }
           break;

           case 16 :
           {
               _cmod    =   True_16_Colour;
              _byte_pp  =   2;
           }
           break;

           case 24 :
           {
               _cmod    =   True_24_Colour;
              _byte_pp  =   3;
           }
           break;

           default :
           elise_fatal_error
           (
               " Elise Display : handle only 8, 16 or 24 bit depth display",
               __FILE__,__LINE__
           );
    }
}



void Data_Elise_Raster_D::read_rgb_line
     (
           U_INT1 * aR,
           U_INT1 * aG,
           U_INT1 * aB,
           INT aNb,
           U_INT1 * anXim
     )
{
     switch(_cmod)
     {
         case True_24_Colour:
			 {
              for (INT x=0; x<aNb ; x++)
              {
                  aR[x] =  anXim[_r_ind];
                  aG[x] =  anXim[_g_ind];
                  aB[x] =  anXim[_b_ind];
                  anXim += _byte_pp;
              }
			 }
         break;

         case True_16_Colour:
			 {
              for (INT x=0; x<aNb ; x++)
              {
		  U_INT2  aS = *((U_INT2 *)anXim);
                  aR[x] =  (((aS &_r_mask) >> _r_shift) * 256) / _r_mult;
                  aG[x] =  (((aS &_g_mask) >> _g_shift) * 256) / _g_mult;
                  aB[x] =  (((aS &_b_mask) >> _b_shift) * 256) / _b_mult;
                  anXim += _byte_pp;
              }
			 }
         break;


         default :
              ELISE_ASSERT
              (
                 false,
                 "unHandled color mode in Data_Elise_Raster_D::read_rgb_line"
              );
         break;
        
     }
}                       


void Data_Elise_Raster_D::write_rgb_line
     (
           U_INT1 * anXim,
           INT aNb,
           U_INT1 * aR,
           U_INT1 * aG,
           U_INT1 * aB
     )
{
     switch(_cmod)
     {
         case True_24_Colour:
			 {
              for (INT x=0; x<aNb ; x++)
              {
                  anXim[_r_ind] = aR[x];
                  anXim[_g_ind] = aG[x];
                  anXim[_b_ind] = aB[x];
                  anXim += _byte_pp;
              }
			 }
         break;

         default :
	      U_INT2 * aUI2 = (U_INT2 *) anXim;
              for (INT x=0; x<aNb ; x++)
              {
		  aUI2[x] =  rgb_to_16(aR[x],aG[x], aB[x]);
	      }
	      /*
              ELISE_ASSERT
              (
                 false,
                 "unHandled color mode in Data_Elise_Raster_D::write_rgb_line"
              );
	      */
         break;
        
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
Footer-MicMac-eLiSe-25/06/2007*/
