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
/*********************************************************************/
/*********************************************************************/
/*                                                                   */
/*         Som_Rvar_Comp                                             */
/*                                                                   */
/*********************************************************************/
/*********************************************************************/
/*********************************************************************/
/*********************************************************************/


            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */
            /*                                          */
            /*           Som_Rvar_Comp                  */
            /*                                          */
            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */



template <class Type> class Som_Rvar_Comp :
                                   public Fonc_Num_OPB_TPL<Type>
{
         public :

            Som_Rvar_Comp
            (
                 const Arg_Fonc_Num_Comp & arg,
                 INT                     dim_out,
                 Fonc_Num                f0,
                 Fonc_Num                fside,
                 Box2di                  side_0
            );

         private :

            virtual void post_new_line(bool);
            void integr(INT yloc,bool very_first);


            INT _y_cpt;


};

template <class Type> void Som_Rvar_Comp<Type>::integr(INT yloc,bool very_first)
{
        for (INT d =0; d < this->_dim_out ; d++)
        {
            Type ** lines = this->kth_buf((Type *)0,0)[d];
            Type * l0  = lines[yloc];

            for (INT x = this->_x0_buf+1; x < this->_x1_buf; x++)
                l0[x] += l0[x-1];
            if (! very_first)
            {
               Type * lm1  = lines[yloc-1];
               for (INT x = this->_x0_buf; x < this->_x1_buf; x++)
                    l0[x] += lm1[x];
            }
            
        }

}


template <class Type> void Som_Rvar_Comp<Type>::post_new_line(bool first)
{
        if (first)
        {
           integr(this->_y0_side,true);
           for (INT y = this->_y0_side +1; y < this->_y1_side ; y++)
               integr(y,false);
        }
        integr(this->_y1_side,false);

        INT * X0  = this->kth_buf((INT *)0,1)[0][0];
        INT * Y0  = this->kth_buf((INT *)0,1)[1][0];
        INT * X1  = this->kth_buf((INT *)0,1)[2][0];
        INT * Y1  = this->kth_buf((INT *)0,1)[3][0];

        _y_cpt ++;

        for (INT d =0; d < this->_dim_out ; d++)
        {
            Type ** lines = this->kth_buf((Type *)0,0)[d];
            Type *  res = this->_buf_res[d];

            for (INT x = this->_x0; x < this->_x1; x++)
            {
                INT x0 = ElMax(this->_x0_side,X0[x]-1);
                INT y0 = ElMax(this->_y0_side,Y0[x]-1);
                INT x1 = ElMin(this->_x1_side,X1[x]);
                INT y1 = ElMin(this->_y1_side,Y1[x]);
                res[x] =  lines[y0][x+x0] + lines[y1][x+x1]
                        - lines[y0][x+x1] - lines[y1][x+x0];
            }
     
            // Strictement inutile sur une machine ideale;
            // mais evite les pbs d'overflow en pratique
            
            if (_y_cpt % (1+this->_y1_side-this->_y0_side) == 0)
            {
               Type * l0 = lines[this->_y0_side];
               for (INT y = this->_y0_side; y <= this->_y1_side ; y++)
               {
                   Type * l = lines[y];
                   for (INT x = this->_x0_buf; x < this->_x1_buf; x++)
                       l[x] -= l0[x];
               }
            }
        }
}


template <class Type> Som_Rvar_Comp<Type>::Som_Rvar_Comp
(
          const Arg_Fonc_Num_Comp &   arg,
          INT                         dim_out,
          Fonc_Num                    f0,
          Fonc_Num                    fside,
          Box2di                      side_0
)    :
       Fonc_Num_OPB_TPL<Type>
       (
              arg,
              dim_out,
              Arg_FNOPB(f0,side_0),
              Arg_FNOPB(fside,Box2di(0))
        )
{
      _y_cpt   = 0;
}


            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */
            /*                                          */
            /*           Som_RVar_Not_Comp              */
            /*                                          */
            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */

class Som_RVar_Not_Comp : public Fonc_Num_Not_Comp
{
      public :
          Som_RVar_Not_Comp
          (
               Fonc_Num                    f0,
               Fonc_Num                    fside,
               Box2di                      side_0
          );

      private :

          virtual bool  integral_fonc (bool iflx) const
          {
               return _f.integral_fonc(iflx);
          }

          virtual INT dimf_out() const { return _f.dimf_out();}
          void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}


          Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg);

          Fonc_Num                  _f;
          Fonc_Num                  _fs;
          Box2di                    _side;
};


Som_RVar_Not_Comp::Som_RVar_Not_Comp
(
        Fonc_Num                    f0,
        Fonc_Num                    fs,
        Box2di                      side_0
)  :
   _f     (f0),
   _fs    (fs),
   _side  (Box2di(side_0._p0+Pt2di(-1,-1),side_0._p1))
{
}

Fonc_Num_Computed * Som_RVar_Not_Comp::compute
                    (const Arg_Fonc_Num_Comp & arg)
{
/*
   INT d = _fs.dimf_out();
   INT dout = dimf_out();

   if (d == 1)
       _fs = Symb_FNum((-_fs,-_fs,_fs,_fs));
   else if (d == 2)
       _fs = Symb_FNum((-_fs,_fs));
   else 
   {
       Tjs_El_User.ElAssert
       (
           d==4,
           EEM0 << "in som _vect_var, coordinate of box must be func of dim 1,2 or 4"
       );
   }
   _fs = Symb_FNum(round_ni(_fs));
*/
   INT dout = dimf_out();

   if (_f.integral_fonc(true))
       return new Som_Rvar_Comp<INT>(arg,dout,_f,_fs,_side);
   else 
       return new Som_Rvar_Comp<REAL>(arg,dout,_f,_fs,_side);
}


Fonc_Num rect_var_som(Fonc_Num f,Fonc_Num fs,Box2di b)
{

   INT d = fs.dimf_out();

   fs = Symb_FNum(round_ni(fs));

   if (d == 1)
       fs = Symb_FNum(Virgule(-fs,-fs,fs,fs));
   else if (d == 2)
       fs = Symb_FNum(Virgule(-fs,fs));
   else 
   {
       Tjs_El_User.ElAssert
       (
           d==4,
           EEM0 << "in som _vect_var, coordinate of box must be func of dim 1,2 or 4"
       );
   }
        
   return new Som_RVar_Not_Comp(f,fs,b);
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
