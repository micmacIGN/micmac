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
   /*         Symb_FNum_Set_Comp                                            */
   /*                                                                       */
   /*************************************************************************/

class   Symb_FNum_Not_Comp ;

class Symb_FNum_Set_Comp : public Fonc_Num_Computed
{
    public :
       Symb_FNum_Set_Comp 
       (
            const Arg_Fonc_Num_Comp & arg,
            Fonc_Num_Computed       * f,
            class Symb_FNum_Not_Comp *
       );

       inline const Pack_Of_Pts * pack_mem() {return _pack;}
       inline const Fonc_Num_Computed * fc() {return _f;}
        ~Symb_FNum_Set_Comp();
      
    private :

      const Pack_Of_Pts * values(const Pack_Of_Pts * pts)
      {
           _pack   = _f->values(pts);
           return _pack;
      };

      const Pack_Of_Pts *             _pack;
      Fonc_Num_Computed *             _f;
      class Symb_FNum_Not_Comp *  _symb_fn;
      
};


Symb_FNum_Set_Comp::Symb_FNum_Set_Comp
(
       const Arg_Fonc_Num_Comp & arg,
       Fonc_Num_Computed       * f,
       class Symb_FNum_Not_Comp  * symb_fn
) :
      Fonc_Num_Computed (arg,f->idim_out(),f->type_out()),
      _f                (f),
      _symb_fn          (symb_fn)
{
}


   /*************************************************************************/
   /*                                                                       */
   /*         Symb_FNum_Use_Comp                                            */
   /*                                                                       */
   /*************************************************************************/



class Symb_FNum_Use_Comp : public Fonc_Num_Computed
{
     public :
        Symb_FNum_Use_Comp
        (
           const Arg_Fonc_Num_Comp & arg,
           Symb_FNum_Set_Comp *
        );

     private :
         
        const Pack_Of_Pts * values(const Pack_Of_Pts *) { return _sc->pack_mem();}
        Symb_FNum_Set_Comp  *_sc;
};


Symb_FNum_Use_Comp::Symb_FNum_Use_Comp
(
     const Arg_Fonc_Num_Comp & arg,
     Symb_FNum_Set_Comp      * sc
)    :
     Fonc_Num_Computed(arg,sc->fc()->idim_out(),sc->fc()->type_out()),
     _sc (sc)
{
};



   /*************************************************************************/
   /*                                                                       */
   /*         Symb_FNum_Not_Comp                                            */
   /*                                                                       */
   /*************************************************************************/

class Symb_FNum_Not_Comp : public Fonc_Num_Not_Comp
{
      
    public :

       Symb_FNum_Not_Comp(Fonc_Num f);

       void set_no_init()
       {
            _ssc    = 0;
            _fc     = 0;
            _flx_c  = 0;
       }

    private :


       Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
       {
            if (! _ssc)
            {
                _flx_c = arg.flux();
                _fc    =  _f.compute(arg);
                _ssc = new Symb_FNum_Set_Comp(arg,_fc,this);
                return _ssc;
             }
             Tjs_El_User.ElAssert
             (    arg.flux() == _flx_c,
                 EEM0 << "a Symb_FNum is used in a context "
                      << " different from its initialization"
             );

             return new Symb_FNum_Use_Comp(arg,_ssc);
       }


        Fonc_Num   _f;

        virtual bool  integral_fonc (bool iflx) const 
        {return _f.integral_fonc(iflx);}

        virtual INT dimf_out() const
        {
                return _f.dimf_out();
        }
        
        void VarDerNN(ElGrowingSetInd & aSet)const {_f.VarDerNN(aSet);}

        Symb_FNum_Set_Comp * _ssc;

        Fonc_Num_Computed *  _fc;
        Flux_Pts_Computed *  _flx_c;

        
         virtual Fonc_Num deriv(INT k)  const
         {
              return _f.deriv(k);
         }
         virtual void  show(ostream & os) const
         {
             _f.show(os);
         }
         virtual bool  is0() const
         {
             return _f.is0();
         }
         virtual bool  is1() const
         {
             return _f.is1();
         }

         REAL ValFonc(const PtsKD & pts) const
         {
             return _f.ValFonc(pts);
         }

};


Symb_FNum_Not_Comp::Symb_FNum_Not_Comp
(
    Fonc_Num f
)   :
    _f   (f)
{
    set_no_init();
}


Symb_FNum_Set_Comp::~Symb_FNum_Set_Comp()
{
       delete _f;
       _symb_fn->set_no_init();
};

   /*************************************************************************/
   /*                                                                       */
   /*         Symb_FNum                                                     */
   /*                                                                       */
   /*************************************************************************/


Symb_FNum::Symb_FNum(Fonc_Num f) :
   Fonc_Num(new Symb_FNum_Not_Comp(f))
{
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
