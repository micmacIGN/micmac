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


   /*************************************************************************/
   /*                                                                       */
   /*         Linear_Order_Comp                                             */
   /*                                                                       */
   /*************************************************************************/


class Linear_Order_Comp : public  Fonc_Num_Comp_TPL<INT>
{
      public :


          Linear_Order_Comp
          (
               const Arg_Fonc_Num_Comp & arg,
               Fonc_Num_Computed *      f,
               INT                      x0,
               INT                      x1,
			   INT                      kth,
			   Histo_Kieme::mode_h      mode,
			   INT                      max_vals,
       		   Histo_Kieme::mode_res    mode_res      
          ) :
            	Fonc_Num_Comp_TPL<INT>(arg,f->idim_out(),arg.flux()),
                _f          (f),
                _x0         (x0),
                _x1         (x1),
                _kth        (kth),
                _hk         (Histo_Kieme::New_HK(mode,max_vals)),
                _mode_res   (mode_res),
				_v		    (0),
				_nb 		(-1)
		  {
          }

          virtual ~Linear_Order_Comp(){ delete  _hk;}

      private :

		  inline INT v(INT x)
          {
			if (x<0)
				return _v[0];
			if (x>= _nb)
				return _v[_nb-1];
			return _v[x];
          }
          const Pack_Of_Pts * 	values(const Pack_Of_Pts * pts);



          Fonc_Num_Computed * 	_f;
          INT    			 	_x0;
          INT     				_x1;
		  INT                 	_kth;
		  Histo_Kieme *        	_hk;
		  Histo_Kieme::mode_res _mode_res;

          INT *  _v;
		  INT	  _nb;
};


const Pack_Of_Pts * Linear_Order_Comp::values(const Pack_Of_Pts * pts)
{
  	Std_Pack_Of_Pts<INT> * vals = const_cast<Std_Pack_Of_Pts<INT> *>(_f->values(pts)->int_cast());
    _nb = pts->nb();
    

    _pack_out->set_nb(_nb);

    for (int d=0 ; d<_dim_out ; d++)
    {
		_v = vals->_pts[d];
		_hk->verif_vals(_v,_nb);
		INT * o = _pack_out->_pts[d];
		_hk->raz();

         for (INT x = _x0; x<_x1; x++)
             _hk->add(v(x));

		 {
         for (INT x=0; x<_nb ; x++)
		 {
             _hk->add(v(x+_x1));
  			 switch(_mode_res)
             {
                  case Histo_Kieme::KTH :
                        o[x] = _hk->kth_val(_kth);
                  break;
 
                  case Histo_Kieme::RANK :
                        o[x] = _hk->rank(v(x));
                  break;
             }                         
             _hk->sub(v(x+_x0));
	     }
		 }
    }

    return _pack_out;
}

class Linear_Order_Not_Comp : public Fonc_Num_Not_Comp
{
    public :
        Linear_Order_Not_Comp
        (
               Fonc_Num                 f,
               INT                      x0,
               INT                      x1,
			   INT                      kth,
			   INT                      max_vals,
       		   Histo_Kieme::mode_res    mode_res      
        ) :
			_f        (Iconv(f)),
            _x0       (x0),
            _x1       (x1),
            _kth      (kth),
            _max_vals (max_vals),
            _mode_res (mode_res)
        {
        }

    private :

          Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
          {
              Fonc_Num_Computed * fc = _f.compute(arg);
			  Histo_Kieme::mode_h      mode= Histo_Kieme::Opt_HK(1,_max_vals);
              return new Linear_Order_Comp
                         (arg,fc,_x0,_x1,_kth,mode,_max_vals,_mode_res);
          }

		   virtual bool  integral_fonc (bool iflx) const
          {return _f.integral_fonc(iflx);}
 
          virtual INT dimf_out() const {return _f.dimf_out();}
          void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}

 
          virtual Fonc_Num deriv(INT k) const
          {
                ELISE_ASSERT(false,"No derivation for formal filters");
                return 0;
          }
          virtual void  show(ostream & os) const
          {
                os << "[Linear Filter]";
          }
          REAL ValFonc(const PtsKD &) const
          {
                ELISE_ASSERT(false,"No ValFonc for linear Filter");
                return 0;
          }                                                        


         Fonc_Num   _f;
         INT       _x0;
         INT       _x1;
         INT       _kth;
         INT       _max_vals;
         Histo_Kieme::mode_res    _mode_res;
};


Fonc_Num linear_Kth(Fonc_Num f,INT x0,INT x1,INT kth,INT max_vals)
{
	return r2d_adapt_filtr_lin
           (
               new Linear_Order_Not_Comp(f,x0,x1,kth,max_vals,Histo_Kieme::KTH),
				"linear_Kth"
           );
}

Fonc_Num linear_median(Fonc_Num f,INT sz,INT max_vals)
{
   return linear_Kth(f,-sz,sz,sz,max_vals);
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
