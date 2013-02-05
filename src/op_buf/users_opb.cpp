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

INT adjust_nb_pack_y(INT nb_pack_y,INT nby_tot)
{
    while (
             (nb_pack_y-1) 
          && (  (nby_tot/(nb_pack_y-1)) == (nby_tot/nb_pack_y)   )
          )
              nb_pack_y--;

   return nb_pack_y;
}


/*********************************************************************/
/*                                                                   */
/*              Simple_OPBuf_Gen                                     */
/*                                                                   */
/*********************************************************************/

Fonc_Num Simple_OPBuf_Gen::adapt_box(Fonc_Num f,Box2di)
{
     return f;
}


/*********************************************************************/
/*                                                                   */
/*              Simple_OPBuf1                                        */
/*                                                                   */
/*********************************************************************/


template <class  Tout,class Tin> Simple_OPBuf1<Tout,Tin>::~Simple_OPBuf1(){}


template <class  Tout,class Tin> Simple_OPBuf1<Tout,Tin> *
                       Simple_OPBuf1<Tout,Tin>::dup_comp()
{
    return this;
}

template <class  Tout,class Tin> void
     Simple_OPBuf1<Tout,Tin>::calc_buf (Tout **,Tin ***)
{
    ELISE_ASSERT(false,"Simple_OPBuf1::calc_buf");
}


template <class  Tout,class Tin>  
          Im2D<Tin,typename  El_CTypeTraits<Tin>::tBase>
           Simple_OPBuf1<Tout,Tin>::AllocImageBufIn()
{
   return Im2D<Tin,tBaseIn>(Im2D_NoDataLin(),SzXBuf(),SzYBuf());
}

// Methode transferee dans include/general/users_op_buf.h
/*
template  <class  Tout,class Tin> void Simple_OPBuf1<Tout,Tin>::SetImageOnBufEntry
                                       (
                                            Im2D<Tin,typename  El_CTypeTraits<Tin>::tBase> anIm,
                                            Tin**                                         aData
                                       )
{
   Tin ** aDataIm = anIm.data();

   for (INT y=0; y<SzYBuf() ; y++)
        aDataIm[y] = aData[y+y0Buf()]+x0Buf();
}
*/

template class Simple_OPBuf1<INT,U_INT2>;
template class Simple_OPBuf1<INT,U_INT1>;
template class Simple_OPBuf1<INT,INT1>;
template class Simple_OPBuf1<INT,INT>;
template class Simple_OPBuf1<REAL,REAL>;

/***************************************************************/
/*                                                             */
/*          PRC_Simple_OPBuf1<Type>                            */
/*                                                             */
/***************************************************************/


template <class  Tout,class Tin> class PRC_Simple_OPBuf1 : public PRC0
{
     public :
          PRC_Simple_OPBuf1(Simple_OPBuf1<Tout,Tin> * op) : PRC0 (op) {}
};

/*********************************************************************/
/*                                                                   */
/*              Hyper_Simple_OPBuf1                                  */
/*                                                                   */
/*********************************************************************/

template <class Tout,class Tin>  class Hyper_Simple_OPBuf1 : public Simple_OPBuf1<Tout,Tin>
{
    public :
       typedef void  (* FC)(   Tout ** out_put,
                               Tin *** in_put,
                               const Simple_OPBuf_Gen&
                           );


       static Hyper_Simple_OPBuf1<Tout,Tin> *  new1(FC f)
       {
             if (f) return new  Hyper_Simple_OPBuf1<Tout,Tin> (f);
             return 0; 
       }
    private :
        virtual void  calc_buf(Tout ** output,Tin *** input)
        {
            _f(output,input,*this);
        }

        Hyper_Simple_OPBuf1(FC f) :
             _f (f)
        {
        }


        FC  _f;
};


/*********************************************************************/
/*                                                                   */
/*              Simple_Buffered_Op_Comp<Type>                        */
/*                                                                   */
/*********************************************************************/


template <class Tout,class Tin> class Simple_Buffered_Op_Comp :  
                                   public Fonc_Num_OPB_TPL<Tout>
{
         public :

               Simple_Buffered_Op_Comp
               (
                    const Arg_Fonc_Num_Comp       & arg,
                    INT                           dim_out,
                    Fonc_Num                      f0,
                    Box2di                        side_0,
                    Simple_OPBuf1<Tout,Tin>       *   calc,
                    INT                           nb_pack_y,
                    bool                          aCatFoncInit 

               );
               ~Simple_Buffered_Op_Comp(){}


         private :
            virtual void post_new_line(bool);

            Simple_OPBuf1<Tout,Tin> *        _calc;
            PRC_Simple_OPBuf1<Tout,Tin>     _pcalc; 

};

template <class Tout,class Tin>  
         Simple_Buffered_Op_Comp<Tout,Tin>::Simple_Buffered_Op_Comp
(
          const Arg_Fonc_Num_Comp &   arg,
          INT                         dim_out,
          Fonc_Num                    f0,
          Box2di                      side_0,
          Simple_OPBuf1<Tout,Tin> *   calc,
          INT                         nb_pack_y,
          bool                        aCatFoncInit
)   :
          Fonc_Num_OPB_TPL<Tout>
          (
                arg,
                dim_out,
                Arg_FNOPB
                (
                   f0,
                   Box2di
                   (
                      side_0._p0,
                      side_0._p1+Pt2di(0,nb_pack_y-1)
                   ),
                   type_of_ptr((const Tin *)0)
                 ),
                 Arg_FNOPB::def,
                 Arg_FNOPB::def,
                 aCatFoncInit
          ),
         _calc  (0),
         _pcalc (0)
{
    calc->_first_line = true;
    calc->_ycur = this->_y0;
    calc->_nb_pack_y = nb_pack_y;
    calc->_y_in_pack = 0;

    calc->_x0 = this->_x0;
    calc->_x1 = this->_x1;
    calc->_y0 = this->_y0;
    calc->_y1 = this->_y1;

    calc->_dx0 = side_0._p0.x;
    calc->_dy0 = side_0._p0.y;
    calc->_dx1 = side_0._p1.x;
    calc->_dy1 = side_0._p1.y;

    calc->_x0Buf =  calc->_x0 + calc->_dx0;
    calc->_x1Buf =  calc->_x1 + calc->_dx1;
    calc->_y0Buf =   calc->_dy0;
    calc->_y1Buf =   calc->_dy1 + nb_pack_y;


    calc->_dim_in   = f0.dimf_out();
    calc->_dim_out  = this->idim_out();
    calc->_integral = this->integral();


   _calc  = calc->dup_comp();

   _calc->_first_line =calc->_first_line;
   _calc->_ycur = calc->_ycur;

   _calc->_x0 = calc->_x0;
   _calc->_x1 = calc->_x1;
   _calc->_y0 = calc->_y0;
   _calc->_y1 = calc->_y1;
   _calc->_dx0 = calc->_dx0;
   _calc->_dx1 = calc->_dx1;
   _calc->_dy0 = calc->_dy0;
   _calc->_dy1 = calc->_dy1;
   _calc->_x0Buf =  calc->_x0Buf;
   _calc->_x1Buf =  calc->_x1Buf;
   _calc->_y0Buf =  calc->_y0Buf;
   _calc->_y1Buf =  calc->_y1Buf;

   _calc->_nb_pack_y = calc->_nb_pack_y;
   _calc->_y_in_pack = calc->_y_in_pack;


    _calc->_dim_in   = calc->_dim_in  ;
    _calc->_dim_out  = calc->_dim_out ;
    _calc->_integral = calc->_integral;

   _pcalc = PRC_Simple_OPBuf1<Tout,Tin>(_calc);
}

template <class Tout,class Tin>  void Simple_Buffered_Op_Comp<Tout,Tin>::post_new_line(bool )
{
    _calc->_out = this->_buf_res;
    _calc->_in = this->kth_buf((Tin *) 0,0);

    _calc->calc_buf(_calc->_out, _calc->_in);
    _calc->_ycur++;
    _calc->_first_line = false;
    _calc->_y_in_pack  = (_calc->_y_in_pack +1) % _calc->_nb_pack_y;
}

template class Simple_Buffered_Op_Comp<INT,U_INT2>;
template class Simple_Buffered_Op_Comp<INT,U_INT1>;
template class Simple_Buffered_Op_Comp<INT,INT1>;
template class Simple_Buffered_Op_Comp<INT,INT>;
template class Simple_Buffered_Op_Comp<REAL,REAL>;

/*********************************************************************/
/*                                                                   */
/*              Simple_op_Buf_Not_Comp                               */
/*                                                                   */
/*********************************************************************/

template <class Itin,class Rtin> class Simple_op_Buf_Not_Comp : public Fonc_Num_Not_Comp
{

      public :


          virtual bool  integral_fonc (bool iflx) const
          {
               return _f.integral_fonc(iflx);
          }


          Fonc_Num                          _f;
          Box2di                         _side;
          Simple_OPBuf1<INT,Itin> *      _calc_I;
          Simple_OPBuf1<REAL,Rtin> *     _calc_R;
          PRC_Simple_OPBuf1<INT,Itin>    _pcalc_I; 
          PRC_Simple_OPBuf1<REAL,Rtin>   _pcalc_R; 
          INT                            _dim_out;
          INT                            _nb_pack_y;
          bool                           mOptimizeNbPackY;
          bool                           mCatFoncInit;

      

          virtual INT dimf_out() const 
          { 
              return _dim_out + (mCatFoncInit?_f.dimf_out() : 0);
          }
         void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}


        


          Simple_op_Buf_Not_Comp
          (
                 Simple_OPBuf1<INT,Itin> *    calc_I,
                 Simple_OPBuf1<REAL,Rtin> *   calc_R,
                 Fonc_Num                f,
                 INT                     dim_out,
                 Box2di                  side,
                 INT                     nb_pack_y,
                 bool                    OptimizeNbPackY,
                 bool                    aCatFoncInit
          ) :
            _f                (f)      ,
            _side             (side)   ,
            _calc_I           (calc_I),
            _calc_R           (calc_R),
            _pcalc_I          (calc_I),
            _pcalc_R          (calc_R),
            _dim_out          (dim_out),
            _nb_pack_y        (nb_pack_y),
            mOptimizeNbPackY  (OptimizeNbPackY),
            mCatFoncInit      (aCatFoncInit)
          {
              Tjs_El_User.ElAssert
              (
                   calc_I || calc_R,
                   EEM0 << "One of two pointer must be != 0 in create_op_buf_simple_tpl"
              );
          
              if (! _calc_I) _f = Rconv(_f);
              if (! _calc_R) _f = Iconv(_f);
          }



          Fonc_Num_Computed * compute (const Arg_Fonc_Num_Comp & arg)
          {
    
              Box2di b(0);
              Tjs_El_User.ElAssert
              (
                   arg.flux()->is_rect_2d(b),
                   EEM0 << "Must use operator created  by `create_op_buf_simple_tpl'"
                        << " with 2D rectangle flux"
              );


              INT nbpcky = mOptimizeNbPackY                               ?
                           adjust_nb_pack_y(_nb_pack_y,b._p1.y-b._p0.y)   :
                           _nb_pack_y                                     ;

              Simple_OPBuf_Gen * calc = 0;

              if (_f.integral_fonc(true))
                 calc = _calc_I;
              else
                 calc = _calc_R;

              Fonc_Num f =  calc->adapt_box(_f,b);

              if (f.integral_fonc(true))
                 return new Simple_Buffered_Op_Comp<INT,Itin> 
                            (arg,_dim_out,f,_side,_calc_I,nbpcky,mCatFoncInit);
              else
                 return new Simple_Buffered_Op_Comp<REAL,Rtin>
                            (arg,_dim_out,f,_side,_calc_R,nbpcky,mCatFoncInit);
          }
};

Fonc_Num  create_op_buf_simple_tpl
(
              Simple_OPBuf1<INT,INT> *   calc_I,
              Simple_OPBuf1<REAL,REAL> *  calc_R,
              Fonc_Num               f,
              INT                    dim_out,
              Box2di                 side,
              INT                    nb_pack_y,
              bool                   OptimizeNbPackY,
              bool                   aCatFoncInit
)
{


     return new Simple_op_Buf_Not_Comp<INT,REAL>
                (calc_I,calc_R,f,dim_out,side,nb_pack_y,OptimizeNbPackY,aCatFoncInit);
}


Fonc_Num  create_op_buf_simple_tpl
(
              Simple_OPBuf1<INT,U_INT1> *   calc_I,
              Fonc_Num               f,
              INT                    dim_out,
              Box2di                 side,
              INT                    nb_pack_y,
              bool                   OptimizeNbPackY,
              bool                   aCatFoncInit
)
{
     return new Simple_op_Buf_Not_Comp<U_INT1,REAL>
                (calc_I,0,f,dim_out,side,nb_pack_y,OptimizeNbPackY,aCatFoncInit);
}

Fonc_Num  create_op_buf_simple_tpl
(
              Simple_OPBuf1<INT,U_INT2> *   calc_I,
              Fonc_Num               f,
              INT                    dim_out,
              Box2di                 side,
              INT                    nb_pack_y,
              bool                   OptimizeNbPackY,
              bool                   aCatFoncInit
)
{
     return new Simple_op_Buf_Not_Comp<U_INT2,REAL>
                (calc_I,0,f,dim_out,side,nb_pack_y,OptimizeNbPackY,aCatFoncInit);
}

Fonc_Num  create_op_buf_simple_tpl
(
              Simple_OPBuf1<INT,INT1> *   calc_I,
              Fonc_Num               f,
              INT                    dim_out,
              Box2di                 side,
              INT                    nb_pack_y,
              bool                   OptimizeNbPackY,
              bool                   aCatFoncInit
)
{
     return new Simple_op_Buf_Not_Comp<INT1,REAL>
                (calc_I,0,f,dim_out,side,nb_pack_y,OptimizeNbPackY,aCatFoncInit);
}







Fonc_Num  create_op_buf_simple_tpl
(
              Simple_OPBuf1_I_calc fi,
              Simple_OPBuf1_R_calc fr,
              Fonc_Num            f,
              INT                 dim_out,
              Box2di              side,
              bool                aCatFoncInit
)
{
     return new Simple_op_Buf_Not_Comp<INT,REAL>
                (
                      Hyper_Simple_OPBuf1<INT,INT>::new1 (fi),
                      Hyper_Simple_OPBuf1<REAL,REAL>::new1(fr),
                      f,
                      dim_out,
                      side,
                      1,
                      true,
                      aCatFoncInit
                );
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
