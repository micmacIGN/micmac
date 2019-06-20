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
/*         Operator ,                                                */
/*                                                                   */
/*********************************************************************/


template <class Type> class CatCoord_Out_Comp :  public Output_Computed
{
      public :


         CatCoord_Out_Comp
         (
              const Arg_Output_Comp &,
              Output_Computed       * o0,
              Output_Computed       * o1
         ) :
                Output_Computed
                   (o0->dim_consumed()+o1->dim_consumed())
         {
                _to[0] = o0;
                _to[1] = o1;

                for (INT o=0 ; o <2 ; o++)
                    _tpck[o] =   Std_Pack_Of_Pts<Type>::new_pck
                                         (_to[o]->dim_consumed(),0);
         }

         virtual ~CatCoord_Out_Comp()
         {
             INT o;
             for (o=1 ; o>=0  ; o--)
                 delete _tpck[o];

             for (o=1 ; o>=0  ; o--)
                 delete _to[o];
         }

         virtual void update( const Pack_Of_Pts * pts,
                              const Pack_Of_Pts * vals_gen)
         {
                INT d = 0;
                const  Std_Pack_Of_Pts<Type> *val =
                       SAFE_DYNC(Std_Pack_Of_Pts<Type> *,const_cast<Pack_Of_Pts *>(vals_gen));

                for (INT o=0 ; o <2 ; o++)
                {
                     for(INT dim_o = 0; dim_o < _to[o]->dim_consumed(); dim_o++)
                        _tpck[o]->_pts[dim_o] = val->_pts[d++];
                      _tpck[o]->set_nb(pts->nb());
                      _to[o]->update(pts,_tpck[o]);
                }
         }

      private :

         Output_Computed *         _to[2];
         Std_Pack_Of_Pts<Type> *    _tpck[2];
};


class CatCoord_Out_Not_Comp :  public Output_Not_Comp
{

      public :

          CatCoord_Out_Not_Comp(Output o0,Output o1) :
              _o0 (o0),
              _o1 (o1)
          {
          }

          virtual  Output_Computed * compute(const Arg_Output_Comp & arg)
          {
                 Output_Computed *o0,*o1;

                 o0 = _o0.compute(arg);
                 o1 = _o1.compute(arg);

                 Tjs_El_User.ElAssert
                 (
                    arg.fonc()->idim_out() >= o0->dim_consumed()+o1->dim_consumed(),
                    EEM0 << "dimension of result of function insufficient in operator, (Output) "
                         <<  arg.fonc()->idim_out() << " "
                         << o0->dim_consumed() << " "
                         << o1->dim_consumed()
                 );

                 if (arg.fonc()->type_out() == Pack_Of_Pts::integer)
                    return new   CatCoord_Out_Comp<INT> (arg,o0,o1);
                 else
                    return new CatCoord_Out_Comp<REAL> (arg,o0,o1) ;
          }

      private :

            Output _o0;
            Output _o1;
};


Output Virgule  (Output o0,Output o1)
{
     return new CatCoord_Out_Not_Comp(o0,o1);
}


/*********************************************************************/
/*                                                                   */
/*         Operator |                                                */
/*                                                                   */
/*********************************************************************/


class Pipe_Out_Comp :  public Output_Computed
{
      public :


         Pipe_Out_Comp
         (
              const Arg_Output_Comp &,
              Output_Computed       * o0,
              Output_Computed       * o1
         ) :
                Output_Computed
                   (ElMax(o0->dim_consumed(),o1->dim_consumed()))
         {
                _to[0] = o0;
                _to[1] = o1;
         }

         virtual ~Pipe_Out_Comp()
         {
             for (INT o=1 ; o>=0  ; o--)
                  delete _to[o];
         }

         virtual void update( const Pack_Of_Pts * pts,
                              const Pack_Of_Pts * vals)
         {

                for (INT o=0 ; o <2 ; o++)
                     _to[o]->update(pts,vals);
         }

      private :
         Output_Computed *         _to[2];
};


class Pipe_Out_Not_Comp :  public Output_Not_Comp
{

      public :

          Pipe_Out_Not_Comp(Output o0,Output o1) :
                _o0 (o0),
                _o1 (o1)
          {
          }

          virtual  Output_Computed * compute(const Arg_Output_Comp & arg)
          {
                 Output_Computed *o0,*o1;

                 o0 = _o0.compute(arg);
                 o1 = _o1.compute(arg);

                 return new Pipe_Out_Comp(arg,o0,o1);
          }

      private :

            Output _o0;
            Output _o1;
};


Output operator | (Output o0,Output o1)
{
     return new Pipe_Out_Not_Comp(o0,o1);
}


/*********************************************************************/
/*                                                                   */
/*         Operator <<                                               */
/*                                                                   */
/*********************************************************************/

class Redir_Out_Comp :  public Output_Computed
{
      public :

         Redir_Out_Comp
         (
              const Arg_Output_Comp &,
              Output_Computed       * o,
              Fonc_Num_Computed     * f
         ) :
                Output_Computed (0), // o->dim_consumed()),
                _f              (f),
                _o              (o)
         {
         }

         virtual ~Redir_Out_Comp()
         {
             delete _o;
             delete _f;
         }

         virtual void update( const Pack_Of_Pts * pts,
                              const Pack_Of_Pts * )
         {
               _o->update(pts,_f->values(pts));
         }

      private :
         Fonc_Num_Computed *  _f;
         Output_Computed *    _o;
};

class Redir_Out_Not_Comp :  public Output_Not_Comp
{

      public :

          Redir_Out_Not_Comp(Output o,Fonc_Num f) :
              _f(f),
              _o(o)
          {
          }

          virtual  Output_Computed * compute(const Arg_Output_Comp & arg)
          {
                 Output_Computed *o;
                 Fonc_Num_Computed *f;

                 f = _f.compute(Arg_Fonc_Num_Comp(arg.flux()));
                 o = _o.compute(Arg_Output_Comp(arg.flux(),f));

                 return new Redir_Out_Comp(arg,o,f);
          }

      private :

             Fonc_Num _f;
             Output  _o;
};


Output operator << (Output o,Fonc_Num f)
{
       return new Redir_Out_Not_Comp(o,f);
}


/*********************************************************************/
/*                                                                   */
/*         o.chc()                                                   */
/*                                                                   */
/*********************************************************************/



        /*  Out_Chc_Comp  */

class Out_Chc_Comp :  public Output_Computed
{
   public :

     Out_Chc_Comp
     (
          Output_Computed *           o,
          Fonc_Num_Computed  *        fchc,
          Flux_Pts_Computed  *        flx_interf
     );


      ~Out_Chc_Comp()
      {
           delete _o;
           delete _flxi;
           delete _fchc;
      }

   private :
      virtual void update( const Pack_Of_Pts * pts,
                           const Pack_Of_Pts * vals_gen)
      {

            _o->update
            (
                 _fchc->values(pts),
                  vals_gen
            );
      }


      Output_Computed *   _o;
      Fonc_Num_Computed * _fchc;
      Flux_Pts_Computed * _flxi;

};


Out_Chc_Comp::Out_Chc_Comp
(
          Output_Computed *           o,
          Fonc_Num_Computed  *        fchc,
          Flux_Pts_Computed  *        flx_interf
)  :
        Output_Computed (o->dim_consumed()),
        _o              (o),
        _fchc           (fchc),
        _flxi           (flx_interf)

{
}


class Out_Chc_Not_Comp :  public Output_Not_Comp
{

      public :

          Out_Chc_Not_Comp(Output o,Fonc_Num fchc) :
              _o(o),
              _fchc(fchc)
          {
          }

          virtual  Output_Computed * compute(const Arg_Output_Comp & arg)
          {
                 Fonc_Num_Computed *fchc =
                      _fchc.compute(Arg_Fonc_Num_Comp(arg.flux()));

                 Flux_Pts_Computed * flx_int = interface_flx_chc(arg.flux(),fchc);

                 Output_Computed *o  =
                      _o.compute(Arg_Output_Comp(flx_int,arg.fonc()));

                 return new Out_Chc_Comp(o,fchc,flx_int);
          }

      private :

             Output     _o;
             Fonc_Num _fchc;
};


Output  Output::chc(Fonc_Num f)
{
        return new Out_Chc_Not_Comp(*this,f);
}


/*********************************************************************/
/*                                                                   */
/*              Filtre Reduc Bin                                     */
/*                                                                   */
/*********************************************************************/



template <class Type> class cFiltreReducBin :  public Output_Computed
{
      public :


         cFiltreReducBin
         (
              const Arg_Output_Comp & anArg,
              Output_Computed       * anOut,
              bool                    aFiltrX,
              bool                    aFiltrY
         ) :
                Output_Computed (anOut->dim_consumed()),
                mOut    (anOut),
                mFiltrX (aFiltrX),
                mFiltrY (aFiltrY)
         {
             INT aSzB = anArg.flux()->sz_buf();
             INT aDimF    = anArg.fonc()->idim_out();
             mPackVF =   Std_Pack_Of_Pts<Type>::new_pck(aDimF,aSzB);
             for (INT aK=0; aK<(aSzB+1) ; aK++)
                 aVFiltr.push_back(mFiltrX ? ((aK+1)%2) : 1 );
             mPackPF = RLE_Pack_Of_Pts::new_pck(2,aSzB);
         }

         virtual ~cFiltreReducBin()
         {
             delete mOut;
             delete mPackPF;
             delete mPackVF;
         }

         virtual void update( const Pack_Of_Pts * aPackPGen,
                              const Pack_Of_Pts * aPackValGen)
         {
                const RLE_Pack_Of_Pts * aPackRLE = aPackPGen->rle_cast();
                INT anY = aPackRLE->y();
                if (mFiltrY && (anY%2))
                    return;
                INT anX =  aPackRLE->vx0();

                INT * aVF = &(aVFiltr[0]) + (mFiltrX ? (anX %2 ): 0);
                mPackVF->set_nb(0);
                aPackValGen->select_tab(mPackVF,aVF);

                mPackPF->set_nb(mPackVF->nb());
                mPackPF->set_pt0(Pt2di(anX/(1+mFiltrX),anY/(1+mFiltrY)));

                mOut->update(mPackPF,mPackVF);

         }

      private :

         Output_Computed *       mOut;
         Std_Pack_Of_Pts<Type> * mPackVF;
         RLE_Pack_Of_Pts *       mPackPF;
         std::vector<int>        aVFiltr;
         bool                    mFiltrX;
         bool                    mFiltrY;
};

template class cFiltreReducBin<INT>;


class cFiltreReducBin_NotComp :  public Output_Not_Comp
{

      public :

          cFiltreReducBin_NotComp(Output anOut,bool aFiltrX,bool aFiltrY) :
              mOut    (anOut),
              mFiltrX (aFiltrX),
              mFiltrY (aFiltrY)
          {
          }

          virtual  Output_Computed * compute(const Arg_Output_Comp & arg)
          {
                 Output_Computed *anOut;

                 anOut = mOut.compute(arg);


                 if (arg.fonc()->type_out() == Pack_Of_Pts::integer)
                    return new cFiltreReducBin<INT> (arg,anOut,mFiltrX,mFiltrY);
                 else
                    return new cFiltreReducBin<REAL>(arg,anOut,mFiltrX,mFiltrY);
          }

      private :

            Output mOut;
            bool mFiltrX;
            bool mFiltrY;
};

Output Filtre_Out_RedBin_Gen  (Output anOut,bool aRedX,bool aRedY)
{
     if ((!aRedX) && (! aRedY))
        return anOut;
     return new cFiltreReducBin_NotComp(anOut,aRedX,aRedY);
}

Output Filtre_Out_RedBin  (Output anOut)
{
     return new cFiltreReducBin_NotComp(anOut,true,true);
}

Output Filtre_Out_RedBin_X  (Output anOut)
{
     return new cFiltreReducBin_NotComp(anOut,true,false);
}

Output Filtre_Out_RedBin_Y  (Output anOut)
{
     return new cFiltreReducBin_NotComp(anOut,false,true);
}





/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
