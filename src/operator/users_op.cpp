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


/***************************************************************/
/*                                                             */
/*          Arg_Comp_Simple_OP_UN                              */
/*          Arg_Comp_Simple_OP_BIN                             */
/*                                                             */
/***************************************************************/


Arg_Comp_Simple_OP_UN::Arg_Comp_Simple_OP_UN
(
   bool        integer_fonc,
   int         dim_in,
   int         dim_out,
   int         sz_buf
)  :
         //_integer    (integer_fonc),
         _dim_in     (dim_in),
         _dim_out    (dim_out),
         _sz_buf     (sz_buf)
{
}

Arg_Comp_Simple_OP_BIN::Arg_Comp_Simple_OP_BIN
(
   bool        integer_fonc,
   int         dim_in1,
   int         dim_in2,
   int         dim_out,
   int         sz_buf
)  :
         //_integer    (integer_fonc),
         _dim_in1    (dim_in1),
         _dim_in2    (dim_in2),
         _dim_out    (dim_out),
         _sz_buf     (sz_buf)
{
}

/***************************************************************/
/*                                                             */
/*          Simple_OP_UN<Type>                                 */
/*          Simple_OP_BIN<Type>                                */
/*                                                             */
/***************************************************************/

template <class  Type> Simple_OP_UN<Type>::~Simple_OP_UN(){}


template <class  Type> Simple_OP_UN<Type> * 
                       Simple_OP_UN<Type>::dup_comp(const Arg_Comp_Simple_OP_UN&)
{
    return this;
}

template <class  Type> void Simple_OP_UN<Type>::calc_buf
						(
                           Type ** ,
                           Type ** ,
                           INT     ,
                           const Arg_Comp_Simple_OP_UN  &
                        )
{
   ELISE_ASSERT(false,"Simple_OP_UN<Type>::calc_buf");

}


template <class  Type> Simple_OP_BIN<Type>::~Simple_OP_BIN(){}


template <class  Type> Simple_OP_BIN<Type> * 
                       Simple_OP_BIN<Type>::dup_comp(const Arg_Comp_Simple_OP_BIN&)
{
    return this;
}

template <class  Type> void Simple_OP_BIN<Type>::calc_buf
						(
                           Type ** ,
                           Type ** ,
						   Type **,
                           INT     ,
                           const Arg_Comp_Simple_OP_BIN  &
                        )
{
   ELISE_ASSERT(false,"Simple_OP_BIN<Type>::calc_buf");

}


template class Simple_OP_UN<INT>;
template class Simple_OP_UN<REAL>;
template class Simple_OP_BIN<INT>;
template class Simple_OP_BIN<REAL>;

/***************************************************************/
/*                                                             */
/*          Hyper_Simple_OP_UN<Type>                           */
/*          Hyper_Simple_OP_BIN<Type>                          */
/*                                                             */
/***************************************************************/

template <class Type> class Hyper_Simple_OP_UN : public Simple_OP_UN<Type>
{
   public :

   typedef void (*FC)(Type **,Type **,int,const Arg_Comp_Simple_OP_UN&);


            static Hyper_Simple_OP_UN * new1(FC fc)
            {
                 if (fc) return new Hyper_Simple_OP_UN (fc);
                 return 0;
            }

   private :
           Hyper_Simple_OP_UN(FC fc) :
               _fcalc (fc)
           {
           }
           

           FC     _fcalc;

           virtual void  calc_buf
                        (
                             Type ** out,
                             Type ** in,
                             int nb,
                             const Arg_Comp_Simple_OP_UN& arg
                        )
           {
                _fcalc(out,in,nb,arg); 
           }
};

template <class Type> class Hyper_Simple_OP_BIN : public Simple_OP_BIN<Type>
{
   public :

   typedef void (*FC)(Type **,Type **,Type **,int,const Arg_Comp_Simple_OP_BIN&);


            static Hyper_Simple_OP_BIN * new1(FC fc)
            {
                 if (fc) return new Hyper_Simple_OP_BIN (fc);
                 return 0;
            }

   private :
           Hyper_Simple_OP_BIN(FC fc) :
               _fcalc (fc)
           {
           }
           

           FC     _fcalc;

           virtual void  calc_buf
                        (
                             Type ** out,
                             Type ** in1,
                             Type ** in2,
                             int nb,
                             const Arg_Comp_Simple_OP_BIN& arg
                        )
           {
                _fcalc(out,in1,in2,nb,arg); 
           }
};




/***************************************************************/
/*                                                             */
/*          PRC_Simple_OP_UN<Type>                             */
/*                                                             */
/***************************************************************/


template <class Type> class PRC_Simple_OP_UN : public PRC0
{
     public :
          PRC_Simple_OP_UN(Simple_OP_UN<Type> * op) : PRC0 (op) {}
};


template <class Type> class PRC_Simple_OP_BIN : public PRC0
{
     public :
          PRC_Simple_OP_BIN(Simple_OP_BIN<Type> * op) : PRC0 (op) {}
};

/***************************************************************/
/*                                                             */
/*                     Simpl_OpUn_Comp<Type>                   */
/*                     Simpl_OpBin_Comp<Type>                  */
/*                                                             */
/***************************************************************/

template <class Type> class Simpl_OpUn_Comp : public Fonc_Num_Comp_TPL<Type>
{
      public :

           virtual ~Simpl_OpUn_Comp()
           {
               delete _f;
           }

           Simpl_OpUn_Comp
           (
              const Arg_Fonc_Num_Comp &      arg,
              Fonc_Num_Computed       *        f,
              INT                         dim_out,
              Simple_OP_UN<Type>            * calc
           )  :
                 Fonc_Num_Comp_TPL<Type>(arg,dim_out,arg.flux()),
                 _f (f),
				 _argf(_f->integral(),_f->idim_out(),Simpl_OpUn_Comp<Type>::idim_out(),arg.flux()->sz_buf()),
                 _calc  (calc->dup_comp(_argf)),
                 _pcalc (_calc)
           {
           }

            const Pack_Of_Pts * values(const Pack_Of_Pts * pts)
            {
                const Std_Pack_Of_Pts<Type> *v0 = 
                       _f->values(pts)->std_cast((Type *)0);

                _calc->calc_buf(this->_pack_out->_pts,v0->_pts,pts->nb(),_argf);
                this->_pack_out->set_nb(pts->nb());
                 return this->_pack_out;
            }


     private :
         Fonc_Num_Computed *        _f;
         Arg_Comp_Simple_OP_UN      _argf;
         Simple_OP_UN<Type> *       _calc;
         PRC_Simple_OP_UN<Type>     _pcalc;
};


template <class Type> class Simpl_OpBin_Comp : public Fonc_Num_Comp_TPL<Type>
{
      public :

           virtual ~Simpl_OpBin_Comp()
           {
               delete _f1;
               delete _f2;
           }

           Simpl_OpBin_Comp
           (
              const Arg_Fonc_Num_Comp &      arg,
              Fonc_Num_Computed       *        f1,
              Fonc_Num_Computed       *        f2,
              INT                         dim_out,
              Simple_OP_BIN<Type>          * calc
           )  :
                 Fonc_Num_Comp_TPL<Type>(arg,dim_out,arg.flux()),
                 _f1 (f1),
                 _f2 (f2),
				 _argf(_f1->integral(),_f1->idim_out(),_f2->idim_out(),Simpl_OpBin_Comp<Type>::idim_out(),arg.flux()->sz_buf()),
                 _calc  (calc->dup_comp(_argf)),
                 _pcalc (_calc)
           {
           }

            const Pack_Of_Pts * values(const Pack_Of_Pts * pts)
            {
                const Std_Pack_Of_Pts<Type> *v1 = 
                       _f1->values(pts)->std_cast((Type *)0);
                const Std_Pack_Of_Pts<Type> *v2 = 
                       _f2->values(pts)->std_cast((Type *)0);

                _calc->calc_buf(this->_pack_out->_pts,v1->_pts,v2->_pts,pts->nb(),_argf);
                this->_pack_out->set_nb(pts->nb());
                 return this->_pack_out;
            }


     private :
         Fonc_Num_Computed *        _f1;
         Fonc_Num_Computed *        _f2;
         Arg_Comp_Simple_OP_BIN    _argf;
         Simple_OP_BIN<Type> *       _calc;
         PRC_Simple_OP_BIN<Type>     _pcalc;
};


/***************************************************************/
/*                                                             */
/*           Simple_OpUn_NotComp                               */
/*                                                             */
/***************************************************************/


class  Simple_OpUn_NotComp      : public  Fonc_Num_Not_Comp
{
     public :

        Simple_OpUn_NotComp
        (
             Simple_OP_UN<INT>  * calc_i,
             Simple_OP_UN<REAL> * calc_r,
             Fonc_Num                  f,
             INT                  dim_out

        ) :
             _f         (f),
             _dim_out   (dim_out),
             _calc_i    (calc_i ),
             _calc_r    (calc_r ),
             _pcalc_i   (calc_i ),
             _pcalc_r   (calc_r )
        {
               Tjs_El_User.ElAssert
               (
                     _calc_i || _calc_r,
                     EEM0 << "Requires one No Null object for create_users_oper"
               );
               if (! _calc_i)   _f = Rconv(_f);
               if (! _calc_r)   _f = Iconv(_f);
        }

        Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
        {
               Fonc_Num_Computed * f = _f.compute(arg); 

               if (f->integral())
                   return new Simpl_OpUn_Comp<INT>(arg,f,_dim_out,_calc_i);
               else
                   return new Simpl_OpUn_Comp<REAL>(arg,f,_dim_out,_calc_r);
        }




     private :

         virtual bool  integral_fonc (bool iflx) const {return _f.integral_fonc(iflx);}
         virtual INT dimf_out() const {return _dim_out;}
         void VarDerNN(ElGrowingSetInd & aSet)const {_f.VarDerNN(aSet);}

         Fonc_Num   _f;
         INT        _dim_out;
         Simple_OP_UN <INT>    * _calc_i;
         Simple_OP_UN <REAL>   * _calc_r;
         PRC_Simple_OP_UN<INT>     _pcalc_i;
         PRC_Simple_OP_UN<REAL>     _pcalc_r;

};

class  Simple_OpBin_NotComp      : public  Fonc_Num_Not_Comp
{
     public :

        Simple_OpBin_NotComp
        (
             Simple_OP_BIN<INT>  * calc_i,
             Simple_OP_BIN<REAL> * calc_r,
             Fonc_Num                  f1,
             Fonc_Num                  f2,
             INT                  dim_out

        ) :
             _f1        (f1),
             _f2        (f2),
             _dim_out   (dim_out),
             _calc_i    (calc_i ),
             _calc_r    (calc_r ),
             _pcalc_i   (calc_i ),
             _pcalc_r   (calc_r )
        {
               Tjs_El_User.ElAssert
               (
                     _calc_i || _calc_r,
                     EEM0 << "Requires one No Null object for create_users_oper"
               );
               if (! _calc_i)
               {
                     _f1 = Rconv(_f1);
                     _f2 = Rconv(_f2);
               }
               if (! _calc_r)
               {
                   _f1 = Iconv(_f1);
                   _f2 = Iconv(_f2);
               }
        }

        Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
        {
               Fonc_Num_Computed * f1 = _f1.compute(arg); 
               Fonc_Num_Computed * f2 = _f2.compute(arg); 

               if (f1->integral() && f2->integral())
                   return new Simpl_OpBin_Comp<INT>(arg,f1,f2,_dim_out,_calc_i);
               else
                   return new Simpl_OpBin_Comp<REAL>
                          (
                              arg,
                              convert_fonc_num(arg,f1,arg.flux(),Pack_Of_Pts::real),
                              convert_fonc_num(arg,f2,arg.flux(),Pack_Of_Pts::real),
                              _dim_out,
                              _calc_r
                          );
        }




     private :

         virtual bool  integral_fonc (bool iflx) const 
                {return _f1.integral_fonc(iflx) && _f2.integral_fonc(iflx);}
         virtual INT dimf_out() const {return _dim_out;}

         void VarDerNN(ElGrowingSetInd & aSet)const 
         {
              _f1.VarDerNN(aSet);
              _f2.VarDerNN(aSet);
         }

         Fonc_Num   _f1;
         Fonc_Num   _f2;
         INT        _dim_out;
         Simple_OP_BIN <INT>    * _calc_i;
         Simple_OP_BIN <REAL>   * _calc_r;
         PRC_Simple_OP_BIN<INT>     _pcalc_i;
         PRC_Simple_OP_BIN<REAL>     _pcalc_r;

};




/***************************************************************/
/*                                                             */
/*          create_users_oper                                  */
/*                                                             */
/***************************************************************/

Fonc_Num  create_users_oper
          (
              Simple_OP_UN<INT> *     calc_i,
              Simple_OP_UN<REAL> *    calc_r,
              Fonc_Num                     f,
              INT                     dim_out
          )
{
    return new Simple_OpUn_NotComp(calc_i,calc_r,f,dim_out);
}



Fonc_Num  create_users_oper
          (
              Simple_OPUn_I_calc fi,
              Simple_OPUn_R_calc fr,
              Fonc_Num           f,
              INT   dim_out
          )
{
     return create_users_oper
            (
                  Hyper_Simple_OP_UN<INT>::new1(fi),
                  Hyper_Simple_OP_UN<REAL>::new1(fr),
                  f,
                  dim_out
            );
}



Fonc_Num  create_users_oper
          (
              Simple_OP_BIN<INT> *     calc_i,
              Simple_OP_BIN<REAL> *    calc_r,
              Fonc_Num                     f1,
              Fonc_Num                     f2,
              INT                     dim_out
          )
{
    return new Simple_OpBin_NotComp(calc_i,calc_r,f1,f2,dim_out);
}

Fonc_Num  create_users_oper
          (
              Simple_OPBin_I_calc fi,
              Simple_OPBin_R_calc fr,
              Fonc_Num             f1,
              Fonc_Num             f2,
              INT   dim_out
          )
{
     return create_users_oper
            (
                  Hyper_Simple_OP_BIN<INT>::new1(fi),
                  Hyper_Simple_OP_BIN<REAL>::new1(fr),
                  f1,
                  f2,
                  dim_out
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
