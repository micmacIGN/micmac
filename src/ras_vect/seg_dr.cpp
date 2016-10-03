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




        //==================================================







// static const INT NBB = 8;
#define NBB 8

/**********************************************************/
/*                                                        */
/*       SCORES                                           */
/*                                                        */
/**********************************************************/

/*
    ????????

    Ces lignes ont le pouvoir "magique" de supprimer un
    internal bug de G++ 
    (gcc version egcs-2.90.29 980515 (egcs-1.0.3 release)     )
*/

template <class Type,class TypeBase>  void f()
{
    Pt2dr p1;
    Pt2dr p2;
    ElPFixed<NBB> pf1 (p1);
    ElPFixed<NBB> pf2 (p2);    
    ElSegIter<NBB>  Seg (pf1,pf2,44); 
    Seg.next(pf1);
}
void g()
{
    f<U_INT1,INT>();
}

/* fin ???????? */




template <class Type,class TypeBase> 
REAL FixedSomSegDr
     (
             TIm2D<Type,TypeBase>& Tim,
             Pt2dr p1,
             Pt2dr p2,
             INT NBPts,
             REAL DefOut
     )
{
    ElPFixed<NBB> pf1 (p1);
    ElPFixed<NBB> pf2 (p2);    
    if ( 
               (! pf1.inside(Pt2di(0,0),Tim.sz()-Pt2di(1,1)))
           ||  (! pf2.inside(Pt2di(0,0),Tim.sz()-Pt2di(1,1)))
       )
       return DefOut;

    ElSegIter<NBB>  Seg (pf1,pf2,NBPts); 

    INT res = 0;
    ElPFixed<NBB> pcur; 
    while(Seg.next(pcur))
    {                      
		res += (TImGet<Type,TypeBase,NBB>::getb2(Tim,pcur)>>NBB);
    }
    return res / (REAL) (1<<NBB);
}


template <class Type,class TypeBase> 
REAL FixedSomSegDr
     (
         Im2D<Type,TypeBase> im,
         Pt2dr p1,
         Pt2dr p2,
         INT NBPts,
         REAL DefOut
    )
{
     TIm2D<Type,TypeBase> Tim(im);

     return FixedSomSegDr(Tim,p1,p2,NBPts,DefOut);
}


/*
template <class Type,class TypeBase> 
REAL FixedSomScalSegDr
     (
             TIm2D<Type,TypeBase>& TimX,
             TIm2D<Type,TypeBase>& TimY,
             Pt2dr p1,
             Pt2dr p2,
             INT NBPts,
             REAL DefOut
     )
{
    Pt2di aSzIm = Inf(TimX.sz(),TimY.sz());
    ElPFixed<NBB> pf1 (p1);
    ElPFixed<NBB> pf2 (p2);    
    if ( 
               (! pf1.inside(Pt2di(0,0),aSzIm-Pt2di(1,1)))
           ||  (! pf2.inside(Pt2di(0,0),aSzIm-Pt2di(1,1)))
       )
       return DefOut;

    ElSegIter<NBB>  Seg (pf1,pf2,NBPts); 

    INT res = 0;
    ElPFixed<NBB> pcur; 
    while(Seg.next(pcur))
    {                      
		res += (TImGet<Type,TypeBase,NBB>::getb2(Tim,pcur)>>NBB);
    }
    return res / (REAL) (1<<NBB);
}


template <class Type,class TypeBase> 
REAL FixedSomSegDr
     (
         Im2D<Type,TypeBase> im,
         Pt2dr p1,
         Pt2dr p2,
         INT NBPts,
         REAL DefOut
    )
{
     TIm2D<Type,TypeBase> Tim(im);

     return FixedSomSegDr(Tim,p1,p2,NBPts,DefOut);
}
*/

/**********************************************************/
/*                                                        */
/*        PRIMITIVES                                      */
/*                                                        */
/**********************************************************/

class SegTournant
{
      public :
         SegTournant(Pt2dr p1,Pt2dr p2,REAL step);

         Seg2d  SegParam(REAL Pabs,REAL Pteta) 
         {
              Pt2dr vec = _vec*Pt2dr::FromPolar(1.0,Pteta*_step_teta);
              Pt2dr milieu = _milieu + _norm * (Pabs*_step_abs);
              return Seg2d(milieu-vec,milieu+vec);
         }

      private :

          REAL  _step_teta;
          REAL  _step_abs;
          Pt2dr _milieu;
          Pt2dr _vec;
          Pt2dr _norm;
};

SegTournant::SegTournant(Pt2dr p1,Pt2dr p2,REAL step) :
     _step_teta  (step*2.0/ euclid(p1,p2)),
     _step_abs   (step),
     _milieu     ((p1+p2)/2.0),
     _vec        ((p2-p1)/2.0),
     _norm       (rot90(vunit(_vec)))
{
}


/**********************************************************/
/*                                                        */
/*        CLASSE D'OPTIMISATION GENERALE                  */
/*                                                        */
/**********************************************************/

class ElMEM_K_POSSIBLE
{
   public :
      friend class ElTAB_MEM_K_POSSIBLE;

      INT nbk() {return _nbk;}
      INT JiemeK(INT j) {return _kpos[j];}

   private :
      void  init(Pt2di p);
      INT   _nbk;
      INT   _kpos[9];
};

class  ElTAB_MEM_K_POSSIBLE
{
       public :
            static INT k0()   {return 9;}
            static INT no_k() {return -1;}
            static ElTAB_MEM_K_POSSIBLE  &  THE_ONE();

            static ElMEM_K_POSSIBLE & kpos(INT k)
            {return THE_ONE()._mem_kpos[k];}

       private :

             ElTAB_MEM_K_POSSIBLE();
             ElMEM_K_POSSIBLE _mem_kpos[10];
};





void ElMEM_K_POSSIBLE::init(Pt2di p)
{
    _nbk = 0;
    for (INT k =0; k<9; k++)
       if ( dist8(VOIS_9[k]+p) > 1)
       {
          _kpos[_nbk++] = k;
       }
}                         

ElTAB_MEM_K_POSSIBLE &  ElTAB_MEM_K_POSSIBLE::THE_ONE ()
{
   static ElTAB_MEM_K_POSSIBLE * aRes=0;
   if (aRes==0)
   {
       aRes = new ElTAB_MEM_K_POSSIBLE;
   }
   return *aRes;
}



ElTAB_MEM_K_POSSIBLE::ElTAB_MEM_K_POSSIBLE()
{
    for (INT k =0 ; k<9 ; k++)
        _mem_kpos[k].init(VOIS_9[k]);
      _mem_kpos[k0()].init(Pt2di(1000,3000));
}               

        //==================================================



Optim2DParam::Optim2DParam
(
       REAL     step_lim,
       REAL     def_out,
       REAL     epsilon,
       bool     Maxim,
       REAL     lambda,
       bool     optim_p1,
       bool     optim_p2
) :
    _step_lim     (step_lim),
    _step_cur     (1.0),
    _param        (0,0),
    _def_out      (def_out),
    _epsilon      (epsilon),
    _Maxim        (Maxim),
    _lambda       (lambda),
    _optim_p1     (optim_p1),
    _optim_p2     (optim_p2),
    mFreelyOpt    (true),
    mStepInit     (1.0),
    mSzVoisInit   (1)
{
}

void Optim2DParam::set(REAL aStepInit,INT aSzVoisInit) 
{
     mStepInit   = aStepInit;
     mSzVoisInit = aSzVoisInit;
}

void Optim2DParam::reset()
{
    _step_cur = mStepInit;
    _param  =  Pt2dr(0,0);
    mFreelyOpt =true;
    mScOpt = _def_out;
}


void  Optim2DParam::optim_step_fixed (int aNbMaxStep)
{
     mNbStep2Do  = aNbMaxStep;
     REAL score_opt = _def_out;
     INT last_k = ElTAB_MEM_K_POSSIBLE::k0();

     for(;;)
     {
          mNbStep2Do--;
          ElMEM_K_POSSIBLE & kpos = ElTAB_MEM_K_POSSIBLE::kpos(last_k);
          last_k                  = ElTAB_MEM_K_POSSIBLE::no_k();

          INT j,k;     REAL score_cur;
          for (j=0; j<kpos.nbk(); j++)
          {
              k = kpos.JiemeK(j);
              Pt2di v9 = VOIS_9[k];
              if (
                        (_optim_p1 || (v9.x == 0))
                    &&  (_optim_p2 || (v9.y == 0))
                 )
              {
                   Pt2dr p = _param+v9*_step_cur;
                   score_cur = Op2DParam_ComputeScore(p.x,p.y);
                   if (  _Maxim                               ?  
                         (score_cur > score_opt + _epsilon)   :
                         (score_cur < score_opt - _epsilon)
                      )
                   {
                        score_opt  = score_cur;
                        last_k = k;
                   }
                   if (score_cur == _def_out)
                      mFreelyOpt = false;
              }
          }
          if ((last_k == ElTAB_MEM_K_POSSIBLE::no_k()) || (mNbStep2Do==0))
          {
             mScOpt = score_opt;
             return ;
          }
          else
             _param += VOIS_9[last_k] * _step_cur;
    }                           
}

void  Optim2DParam::optim()
{
    optim(Pt2dr(0,0));
}

void  Optim2DParam::optim(Pt2dr aPInit)
{

     reset();
    _param = aPInit;

     if (mSzVoisInit != 1)
     {
         mScOpt = _def_out;
         Pt2dr pOpt =aPInit;
         for (INT kX=-mSzVoisInit ; kX<=mSzVoisInit ; kX++)
         {
             for (INT kY=-mSzVoisInit ; kY<=mSzVoisInit ; kY++)
             {
                   Pt2dr aP = aPInit + Pt2dr(kX,kY) * _step_cur;
                   REAL Score = Op2DParam_ComputeScore ( aP.x,aP.y);
                   if (Score == _def_out)
                      mFreelyOpt = false;
                   if (  _Maxim                               ?  
                         (Score > mScOpt + _epsilon)   :
                         (Score < mScOpt - _epsilon)
                      )
                   {
                        mScOpt  = Score;
                        pOpt = Pt2di(kX,kY)*_step_cur;
                   }
             }
         }
         _param = pOpt;
     }

     while (_step_cur > _step_lim)
     {
           optim_step_fixed ();
          _step_cur *=  _lambda;
     }
}

void   Optim2DParam::optim_step_fixed (Pt2dr aPInit,int aNbMaxStep)
{
    reset();
    _param = aPInit;
    optim_step_fixed (aNbMaxStep);
}


/**********************************************************/
/*                                                        */
/*        CLASSE DEDIEE AU SEGMENT DE DROITE              */
/*                                                        */
/**********************************************************/

template <class Type,class TypeBase> 
class OptimSegTournantSom : public Optim2DParam
{
      public :
          OptimSegTournantSom
          (
               Im2D<Type,TypeBase>    im,
               Seg2d        seg,
               INT          NbPts,
               REAL         step_init,
               REAL         step_limite,
               bool         optim_absc = true,
               bool         optim_teta = true
          )    :
               Optim2DParam(step_limite/step_init,-1e20,1e-2,true,0.5,optim_absc,optim_teta),
               _im   (im),
               _Tim  (im),
               _segT (seg.p0(),seg.p1(),step_init),
               _NbPts (NbPts)
          {
          }

          Seg2d seg_cur(REAL & Score);


      private :
           REAL Op2DParam_ComputeScore(REAL,REAL);

           Im2D<Type,TypeBase>      _im;
           TIm2D<Type,TypeBase>     _Tim;
           SegTournant              _segT;
           INT                      _NbPts;
           
};



template <class Type,class TypeBase>
REAL OptimSegTournantSom<Type,TypeBase>::Op2DParam_ComputeScore(REAL PAbs,REAL PTeta)
{
     Seg2d  seg = _segT.SegParam(PAbs,PTeta);
     return FixedSomSegDr(_Tim,seg.p0(),seg.p1(),_NbPts,def_out());
}


template <class Type,class TypeBase>
Seg2d OptimSegTournantSom<Type,TypeBase>::seg_cur(REAL & Score)
{
     Pt2dr p = param();
     Score = Op2DParam_ComputeScore(p.x,p.y);
     return _segT.SegParam(p.x,p.y);
}



template <class Type,class TypeBase>
Seg2d   OptimizeSegTournantSomIm
        (
             REAL &                  score,
             Im2D<Type,TypeBase>    im,
             Seg2d                  seg,
             INT                    NbPts,
             REAL                   step_init,
             REAL                   step_limite,
             bool                   optim_absc,
             bool                   optim_teta ,
             bool  *                FreelyOpt
        )
{
      OptimSegTournantSom<Type,TypeBase> 
               OSTSI1
               (
                  im,
                  seg,
                  NbPts,
                  step_init,
                  step_limite,
                  optim_absc,
                  optim_teta 
               );
      OSTSI1.optim();

      if (FreelyOpt)
         *FreelyOpt =  OSTSI1.FreelyOptimized();
      return OSTSI1.seg_cur(score);
}


#define INSTANTIATE_OptimizeSegTournantSomIm(Type,TypeBase)\
template Seg2d   OptimizeSegTournantSomIm\
                 (\
                      REAL &                 score,\
                      Im2D<Type,TypeBase>    im,\
                      Seg2d                  seg,\
                      INT                    NbPts,\
                      REAL                   step_init,\
                      REAL                   step_limite,\
                      bool                   optim_absc,\
                      bool                   optim_teta, \
					  bool *                  FreelyOpt\
                 );



INSTANTIATE_OptimizeSegTournantSomIm(INT1,INT)
INSTANTIATE_OptimizeSegTournantSomIm(U_INT1,INT)



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
