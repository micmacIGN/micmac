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
/*         Red assoc                                                 */
/*                                                                   */
/*********************************************************************/
/*********************************************************************/
/*********************************************************************/
/*********************************************************************/


            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */
            /*                                          */
            /*           Red_Ass_OPB_Comp               */
            /*                                          */
            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */

 

template <class Type> class Red_Ass_OPB_Comp :
                                   public Fonc_Num_OPB_TPL<Type>
{
         public :

               Red_Ass_OPB_Comp
               (
                    const Arg_Fonc_Num_Comp & arg,
                    INT                     dim_out,
                    Fonc_Num                f0,
                    Box2di                  side_0,
                    const OperAssocMixte &  op,
                    bool                    CatFInit
               );
            virtual ~Red_Ass_OPB_Comp();

         private :

            virtual void post_new_line(bool);
            virtual void pre_new_line(bool);

            void reduce_line(INT yloc);



            const OperAssocMixte & _op;

            INT _y_cpt;
            bool _grp_op;


            Type * _buf_redx_av;
            Type * _buf_redx_ar;

            Type *** cumul_line_arr;
            Type **  cumul_line_av;

};

template <class Type> Red_Ass_OPB_Comp<Type>::Red_Ass_OPB_Comp
(
          const Arg_Fonc_Num_Comp &   arg,
          INT                         dim_out,
          Fonc_Num                    f0,
          Box2di                      side_0,
          const OperAssocMixte &      op,
          bool                        aCatFInit
)    :
       Fonc_Num_OPB_TPL<Type>
       (
              arg,
              dim_out,
              Arg_FNOPB(f0,side_0),
              Arg_FNOPB::def,
              Arg_FNOPB::def,
              aCatFInit
       ),
       _op (op)
{
      _y_cpt   = 0;
      _grp_op  =   _op.grp_oper();

      _buf_redx_av = NEW_VECTEUR(this->_x0_buf,this->_x1_buf,Type);
      _buf_redx_ar = NEW_VECTEUR(this->_x0_buf,this->_x1_buf,Type);
      cumul_line_arr = NEW_TAB_MATRICE
                      (    this->mDimOutSpec,
                           Pt2di(this->_x0,this->_y0_buf),
                           Pt2di(this->_x1,this->_y1_buf),
                           Type
                      );
      cumul_line_av = NEW_MATRICE(Pt2di(this->_x0,0),Pt2di(this->_x1,this->mDimOutSpec),Type);
}

template <class Type> Red_Ass_OPB_Comp<Type>::~Red_Ass_OPB_Comp()
{
     DELETE_VECTOR(_buf_redx_av,this->_x0_buf);
     DELETE_VECTOR(_buf_redx_ar,this->_x0_buf);
     DELETE_TAB_MATRICE(cumul_line_arr,this->mDimOutSpec,Pt2di(this->_x0,this->_y0_buf),Pt2di(this->_x1,this->_y1_buf));
     DELETE_MATRICE(cumul_line_av,Pt2di(this->_x0,0),Pt2di(this->_x1,this->mDimOutSpec));
}

template <class Type> void  Red_Ass_OPB_Comp<Type>::reduce_line(INT yloc)
{
      for (INT d=0; d<this->mDimOutSpec; d++)
      {
           Type * line = this->kth_buf((Type *)0,0)[d][yloc]; 
           _op.reduce_seg
           (
                  line,
                  line,
                  _buf_redx_ar,
                  _buf_redx_av,
                  this->_x0,this->_x1,this->_x0_side,this->_x1_side
           );
      }
}

template <class Type> void Red_Ass_OPB_Comp<Type>::pre_new_line(bool first)
{
     if ((! _grp_op) || first || (this->_y1_buf-this->_y0_buf <=2))
        return;

     for (INT d =0; d < this->mDimOutSpec ; d++)
     {
         Type ** lines = this->kth_buf((Type *)0,0)[d];

         _op.t0_opinveg_t1
         (
             this->_buf_res[d]+this->_x0,
             lines[this->_y0_buf]+this->_x0,
             this->_x1-this->_x0
         );
     }
}



template <class Type> void Red_Ass_OPB_Comp<Type>::post_new_line(bool first)
{
        if (first)
        {
           for (INT y = this->_y0_buf ; y < this->_y1_buf -1; y++)
               reduce_line(y);
        }
        reduce_line(this->_y1_buf-1);

        if ( _grp_op)
        {
           for (INT d =0; d < this->mDimOutSpec ; d++)
           {
               Type ** lines = this->kth_buf((Type *)0,0)[d];
            
               switch(this->_y1_buf-this->_y0_buf)
               {
                      case 1 :
                           convert(this->_buf_res[d]+this->_x0,lines[this->_y0_buf]+this->_x0,this->_x1-this->_x0);
                      break;

                      case 2 :
                           _op.t0_eg_t1_op_t2
                           (
                              this->_buf_res[d]+this->_x0,
                              lines[this->_y0_buf]+this->_x0,
                              lines[this->_y0_buf+1]+this->_x0,
                              this->_x1-this->_x0
                           );
                      break;

                      default :
                           if (first)
                           {
                              convert(this->_buf_res[d]+this->_x0,lines[this->_y0_buf]+this->_x0,this->_x1-this->_x0);
                              for (INT y = this->_y0_buf+1; y <this->_y1_buf-1; y++)
                                   _op.t0_opeg_t1
                                   (
                                       this->_buf_res[d]+this->_x0,
                                       lines[y]+this->_x0,
                                       this->_x1-this->_x0
                                   );
                           }
                           _op.t0_opeg_t1
                           (
                               this->_buf_res[d]+this->_x0,
                               lines[this->_y1_buf-1]+this->_x0,
                               this->_x1-this->_x0
                           );
                      break;
               }
           }
           return;
        }

        for (INT d =0; d < this->mDimOutSpec ; d++)
        {
            Type ** lines = this->kth_buf((Type *)0,0)[d];
            
            switch(this->_y1_buf-this->_y0_buf)
            {
                   case 1 :
                        convert(this->_buf_res[d]+this->_x0,lines[this->_y0_buf]+this->_x0,this->_x1-this->_x0);
                   break;

                   case 2 :
                   case 3 :
                        _op.t0_eg_t1_op_t2
                        (
                              this->_buf_res[d]+this->_x0,
                              lines[this->_y0_buf]+this->_x0,
                              lines[this->_y0_buf+1]+this->_x0,
                              this->_x1-this->_x0
                        );
                        if (this->_y1_buf-this->_y0_buf > 2)
                           _op.t0_opeg_t1
                           (
                                 this->_buf_res[d]+this->_x0,
                                 lines[this->_y0_buf+2]+this->_x0,
                                 this->_x1-this->_x0
                           );
                   break;

                   default :

                       if (_y_cpt)
                       {
                           rotate_plus_data(cumul_line_arr[d],this->_y0_buf,this->_y1_buf);
                           _op.t0_opeg_t1
                           (
                                  cumul_line_av[d]+this->_x0,
                                  lines[this->_y1_buf-1]+this->_x0,
                                  this->_x1-this->_x0
                           );
                       }
                       else
                       {
                           convert
                           (
                                  cumul_line_arr[d][this->_y1_buf-1]+this->_x0,
                                  lines[this->_y1_buf-1]+this->_x0,
                                  this->_x1-this->_x0
                           );
                           for (INT y = this->_y1_buf-2; y>= this->_y0_buf; y--)
                               _op.t0_eg_t1_op_t2
                               (
                                      cumul_line_arr[d][ y ]+this->_x0,
                                      cumul_line_arr[d][y+1]+this->_x0,
                                      lines[y]+this->_x0,
                                      this->_x1-this->_x0
                               );

                           Type vneutre;
                           _op.set_neutre(vneutre);
                           set_cste(cumul_line_av[d]+this->_x0,vneutre,this->_x1-this->_x0);
                       }

                       _op.t0_eg_t1_op_t2
                       (
                             this->_buf_res[d] + this->_x0,
                             cumul_line_arr[d][this->_y0_buf]+this->_x0,
                             cumul_line_av[d]+this->_x0,
                             this->_x1-this->_x0
                       );
                   break;
            }
       }

        _y_cpt  = (_y_cpt + 1) % (this->_y1_buf-this->_y0_buf);
}

            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */
            /*                                          */
            /*           Red_Ass_OPB_No_Comp            */
            /*                                          */
            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */

class Red_Ass_OPB_No_Comp : public Fonc_Num_Not_Comp
{
      public :
          Red_Ass_OPB_No_Comp
          (
               Fonc_Num                    f0,
               Box2di                      side_0,
               const OperAssocMixte &      op,
               bool                        aCatFinit 
          );

      private :


          virtual bool  integral_fonc (bool iflx) const
          {
               return _f.integral_fonc(iflx);
          }

          virtual INT dimf_out() const {return _f.dimf_out()*(mCatFinit?2:1);}
          void VarDerNN(ElGrowingSetInd &) const {ELISE_ASSERT(false,"No VarDerNN");}


          Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg);

          Fonc_Num                  _f;
          Box2di                    _side;
          const OperAssocMixte &    _op;
          bool                      mCatFinit;
};

Red_Ass_OPB_No_Comp::Red_Ass_OPB_No_Comp
(
        Fonc_Num                    f0,
        Box2di                      side_0,
        const OperAssocMixte &      op,
        bool                        aCatFinit
)  :
   _f        (f0),
   _side     (side_0),
   _op       (op),
   mCatFinit (aCatFinit)
{
/*
   // Les operateur Buf etant implantes de manieres telle qu'il modifient
   // l'entre pour calculer la sortie, on ne peut pas utiliser pour
   // l'instant l'option de concatenation de l'entree
   //
   //  Pour l'instant on inhibe, on verra + tard

   Tjs_El_User.Assert
   (
       (! mCatFinit),
        EEM0<< "Do Not Use Cat option with reduc oper"
   );
*/
}

Fonc_Num_Computed * Red_Ass_OPB_No_Comp::compute
                    (const Arg_Fonc_Num_Comp & arg)
{
       if (_f.integral_fonc(true))
          return new Red_Ass_OPB_Comp<INT>(arg,_f.dimf_out(),_f,_side,_op,mCatFinit);
       else 
          return new Red_Ass_OPB_Comp<REAL>(arg,_f.dimf_out(),_f,_side,_op,mCatFinit);
}


       // rect_red  rect_red  rect_red  rect_red

Fonc_Num rect_red(const OperAssocMixte & op,Fonc_Num f,Box2di b,bool aCatFinit)
{
    return new Red_Ass_OPB_No_Comp(f,b,op,aCatFinit);
}

       //============================================
       // rect_som  rect_som  rect_som  rect_som
       //============================================

Fonc_Num rect_som(Fonc_Num f,Box2di b,bool aCatFinit) 
{
   return rect_red(OpSum,f,b,aCatFinit);
}

Fonc_Num rect_som(Fonc_Num f,Pt2di p,bool aCatFinit)  
{
   return rect_som(f,Box2di(-p,p),aCatFinit);
}


Fonc_Num rect_som(Fonc_Num f,INT   x,bool aCatFinit)  
{
   return rect_som(f,Pt2di(x,x),aCatFinit);
}

Fonc_Num FoncMoy(Fonc_Num aF,int aK)
{
   if (aK<=0) return aF;
   return rect_som(aF,aK) / ElSquare(1+2.0*aK);
}


std::vector<int>  DecompSigmaEnInt(double aSigma,int aNb)
{
    std::vector<int> aVI;
    double aSom2 = 0;
    for (int aK = 0 ; aK < aNb ; aK++)
    {
         double  aRes = ElSquare(aSigma)-aSom2;
         aRes /= aNb - aK;
         aRes = sqrt(ElMax(0.0,aRes));
         int aIRes = round_ni(aRes);
         aSom2 += ElSquare(aIRes);
         aVI.push_back(aIRes);
    }
    return aVI;
}

Fonc_Num MoyByIterSquare(Fonc_Num aF,double aSigmaCible,int aNbIter)
{
     std::vector<int>  aVI =  DecompSigmaEnInt(aSigmaCible,aNbIter);
     for (int aK=0 ; aK<int(aVI.size()) ; aK++)
     {
         aF = FoncMoy(aF,aVI[aK]);
     }
     return aF;
}




       //============================================
       // rect_max  rect_max  rect_max  rect_max
       //============================================

Fonc_Num rect_max(Fonc_Num f,Box2di b,bool aCatFinit) 
{
   return rect_red(OpMax,f,b,aCatFinit);
}

Fonc_Num rect_max(Fonc_Num f,Pt2di p,bool aCatFinit)  
{
    return rect_max(f,Box2di(-p,p),aCatFinit);
}


Fonc_Num rect_max(Fonc_Num f,INT   x,bool aCatFinit)  
{
    return rect_max(f,Pt2di(x,x),aCatFinit);
}



       //============================================
       // rect_min  rect_min  rect_min  rect_min
       //============================================

Fonc_Num rect_min(Fonc_Num f,Box2di b,bool aCatFinit) 
{
     return rect_red(OpMin,f,b,aCatFinit);
}

Fonc_Num rect_min(Fonc_Num f,Pt2di p,bool aCatFinit)  
{
    return rect_min(f,Box2di(-p,p),aCatFinit);
}

Fonc_Num rect_min(Fonc_Num f,INT   x,bool aCatFinit)  
{
   return rect_min(f,Pt2di(x,x),aCatFinit);
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
