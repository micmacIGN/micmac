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



/******************************************************/
/*                                                    */
/*                 cSFN2D                             */
/*                                                    */
/******************************************************/

REAL cSFN2D::SFN2_Calc(const Pt2dr & aP) const
{
   return SFN2_Calc(Pt2di(round_ni(aP.x),round_ni(aP.y)));
}

REAL cSFN2D::SFN2_Calc(const Pt2di & aP) const
{
   return SFN2_Calc(Pt2dr(aP.x,aP.y));
}

REAL cSFN2D::SFN_Calc(const REAL * aV) const
{
   return SFN2_Calc(Pt2dr(aV[0],aV[1]));
}

REAL cSFN2D::SFN_Calc(const INT * aV) const
{
   return SFN2_Calc(Pt2di(aV[0],aV[1]));
}

cSFN2D::cSFN2D() :
   cSimpleFoncNum(2)
{
}


/******************************************************/
/*                                                    */
/*              cSimpleFoncNum                        */
/*                                                    */
/******************************************************/

cSimpleFoncNum::~cSimpleFoncNum() {}
cSimpleFoncNum::cSimpleFoncNum(INT aDim)  :
   mDim (aDim)
{
}

void cSimpleFoncNum::AcceptDim(INT aDim) const
{
   ELISE_ASSERT(aDim == mDim,"cSimpleFoncNum::AcceptDim");
}


REAL cSimpleFoncNum::SFN_Calc(const REAL * aVals) const
{
    static std::vector<INT> aVI;
    aVI.clear();
    for (INT aK=0 ; aK<mDim ; aK++)
        aVI.push_back(round_ni(aVals[aK]));
    return  SFN_Calc(&aVI[0]);
}

REAL cSimpleFoncNum::SFN_Calc(const INT * aVals) const
{
    static std::vector<REAL> aVR;
    aVR.clear();
    for (INT aK=0 ; aK<mDim ; aK++)
        aVR.push_back(aVals[aK]);
    return  SFN_Calc(&aVR[0]);
}


/******************************************************/
/*                                                    */
/*          RLE_cSimpleFoncNum                        */
/*                                                    */
/******************************************************/

class RLE_cSimpleFoncNum : public  Fonc_Num_Comp_TPL<REAL>
{
     public :
            const Pack_Of_Pts * values(const Pack_Of_Pts *);
            RLE_cSimpleFoncNum(const Arg_Fonc_Num_Comp & arg,cSimpleFoncNum & aSFN) :
                Fonc_Num_Comp_TPL<REAL>(arg,1,arg.flux()),
                mDim  (arg.flux()->dim()),
                mSFN  (aSFN)
            {
                 for (INT aK=0 ; aK<mDim ; aK++)
                     mPtsIn.push_back(0);
            }
     private :
        INT               mDim;
        std::vector<INT>  mPtsIn;
        cSimpleFoncNum &  mSFN;
};

const Pack_Of_Pts * RLE_cSimpleFoncNum::values(const Pack_Of_Pts * aPack)
{
   RLE_Pack_Of_Pts * rle_pack = SAFE_DYNC(RLE_Pack_Of_Pts *,const_cast<Pack_Of_Pts *>(aPack));
   INT nb = rle_pack->nb();
   REAL *  aValOut = _pack_out->_pts[0];
   _pack_out->set_nb(aPack->nb());

   for (INT aD=0 ; aD<nb ; aD++)
      mPtsIn[aD] = rle_pack->pt0()[aD];

   for (INT aK=0 ; aK<nb ; aK++)
   {
       aValOut[aK]=  mSFN.SFN_Calc(&mPtsIn[0]);
       mPtsIn[0] ++;
   }
   return _pack_out;
}

/******************************************************/
/*                                                    */
/*          cSimpleFoncNum                            */
/*                                                    */
/******************************************************/

template <class Type>
          class STD_cSimpleFoncNum : public  Fonc_Num_Comp_TPL<REAL>
{
       public :

            const Pack_Of_Pts * values(const Pack_Of_Pts * pts)
            {
                const Std_Pack_Of_Pts<Type> *  pack_in =
                      SAFE_DYNC(Std_Pack_Of_Pts<Type> *,const_cast<Pack_Of_Pts *>(pts));
                INT aNb = pts->nb();
                Type ** aVIn = pack_in->_pts;
                _pack_out->set_nb(pts->nb());

                for (INT aK=0; aK< aNb ; aK++)
                {
                    std::vector<Type> aV;
                    for (INT aD=0 ; aD<mDim; aD++)
                        aV.push_back(aVIn[aD][aK]);
                     _pack_out->_pts[0][aK] = mSFN.SFN_Calc(&aV[0]);
                }

                return _pack_out;
            }

            STD_cSimpleFoncNum(const Arg_Fonc_Num_Comp & arg,cSimpleFoncNum & aSFN) :
                  Fonc_Num_Comp_TPL<REAL>(arg,1,arg.flux()),
                  mSFN (aSFN),
                  mDim (arg.flux()->dim())
            {
            }

      private :
           cSimpleFoncNum & mSFN;
           INT              mDim;
};


/******************************************************/
/*                                                    */
/*          cSimpleFoncNum_NotComp                    */
/*                                                    */
/******************************************************/


class cSimpleFoncNum_NotComp : public Fonc_Num_Not_Comp
{
       public :

            cSimpleFoncNum_NotComp(cSimpleFoncNum & aSFN) :
                mSFN (aSFN)
            {
            }

            virtual Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp &);


       private :

           virtual bool integral_fonc (bool iflux) const { return false; }
           virtual INT dimf_out() const { return 1;}
           void VarDerNN(ElGrowingSetInd & aSet) const
           {
               ELISE_ASSERT(false,"cSimpleFoncNum_NotComp::VarDerNN");
           }

           cSimpleFoncNum & mSFN;

};

Fonc_Num_Computed * cSimpleFoncNum_NotComp::compute(const Arg_Fonc_Num_Comp & arg)
{
    mSFN.AcceptDim(arg.flux()->dim());

    switch (arg.flux()->type())
    {
         case Pack_Of_Pts::rle :
              return new RLE_cSimpleFoncNum(arg,mSFN);

         case Pack_Of_Pts::integer :
              return new STD_cSimpleFoncNum<INT>(arg,mSFN);

         case Pack_Of_Pts::real :
              return new STD_cSimpleFoncNum<REAL>(arg,mSFN);
    }

    return 0;
}

Fonc_Num cSimpleFoncNum::ToF()
{
    return new  cSimpleFoncNum_NotComp(*this);
}


/******************************************************/
/*                                                    */
/*       Fonc_Coord RLE / kth   coordinate            */
/*                                                    */
/******************************************************/


class RLE_Coord_Computed_kth : public  Fonc_Num_Comp_TPL<INT>
{
       public :

            const Pack_Of_Pts * values(const Pack_Of_Pts *);

            RLE_Coord_Computed_kth(const Arg_Fonc_Num_Comp & arg,INT k):
                  Fonc_Num_Comp_TPL<INT>(arg,1,arg.flux()),
                  _kth (k),
                  _last_nb (-1)
            {
            }

      private :
           INT _kth;
           INT _last_val;
           INT _last_nb;
};

const Pack_Of_Pts * RLE_Coord_Computed_kth::values(const Pack_Of_Pts * pack)
{
   RLE_Pack_Of_Pts * rle_pack = SAFE_DYNC(RLE_Pack_Of_Pts *,const_cast<Pack_Of_Pts *>(pack));
   INT nb = rle_pack->nb();
   INT val = rle_pack->pt0()[_kth];


   if (_kth)
   {
      if ((val != _last_val) || (nb > _last_nb))
         set_cste(_pack_out->_pts[0],val,nb);
      _last_val = val;
      _last_nb = nb;
   }
   else
   {
      set_fonc_x
      (
          _pack_out->_pts[0],
          val,
          val+nb
      );
   }

   _pack_out->set_nb(nb);
   return _pack_out;
}

/******************************************************/
/*                                                    */
/*       Fonc_Coord STD / kth   coordinate            */
/*                                                    */
/******************************************************/

template <class Type>
          class STD_Coord_Computed_kth : public  Fonc_Num_Comp_TPL<Type>
{
       public :

            const Pack_Of_Pts * values(const Pack_Of_Pts * pts)
            {
                const Std_Pack_Of_Pts<Type> *  pack_in =
                      SAFE_DYNC(Std_Pack_Of_Pts<Type> *,const_cast<Pack_Of_Pts *>(pts));

                this->_pack_out->_pts[0] = pack_in->_pts[_kth];
                this->_pack_out->set_nb(pts->nb());
                return this->_pack_out;
            }

            STD_Coord_Computed_kth(const Arg_Fonc_Num_Comp & arg,INT k):
                  Fonc_Num_Comp_TPL<Type>(arg,1,arg.flux()),
                  _kth (k)
            {
            }

      private :
           INT _kth;
};




/******************************************************/
/*                                                    */
/*       Fonc_Coord Not comp                          */
/*                                                    */
/******************************************************/

bool FnumCoorUseCsteVal = false;

class Fonc_Coord_Not_Comp : public Fonc_Num_Not_Comp
{
       public :

            Fonc_Coord_Not_Comp(INT k,bool HasAlwaysSameVal,double aVal) :
                _kth (k),
                mHASV (HasAlwaysSameVal),
                mVal  (aVal)
            {
            }
            void inspect() const
            {
                std::cout << "INSPECT " << mHASV << " " << FnumCoorUseCsteVal << " " << mVal << " " << _kth << "\n";
            }

            virtual bool  is0() const
            {
                 return mHASV && FnumCoorUseCsteVal && (mVal==0);
            }
            virtual bool  is1() const
            {
                 return mHASV && FnumCoorUseCsteVal && (mVal==1);
            }

            virtual Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp &);


       private :
           INT    _kth;
           bool   mHASV;
           double mVal;

           virtual bool integral_fonc (bool iflux) const
           {
                   return iflux;
           }

           virtual INT dimf_out() const { return 1;}
           void VarDerNN(ElGrowingSetInd & aSet) const {aSet.insert(_kth);}


           virtual Fonc_Num deriv(INT k) const
           {
                return Fonc_Num ((k==_kth) ? 1 : 0);
           }
           REAL ValDeriv(const  PtsKD &  pts,INT k) const
           {
                return (k==_kth) ? 1.0 : 0.0;
           }
           INT  NumCoord() const
           {
               return _kth;
           }

           virtual void show(ostream & os) const
           {
               os << "X" << _kth;
           }

            void compile (cElCompileFN &);

           virtual Fonc_Num::tKindOfExpr  KindOfExpr();
           virtual INT CmpFormelIfSameKind(Fonc_Num_Not_Comp *);




           REAL ValFonc(const PtsKD & pts) const
           {
             return  pts(_kth);
           }
       INT DegrePoly() const {return 1;}
};

Fonc_Num_Computed * Fonc_Coord_Not_Comp::compute(const Arg_Fonc_Num_Comp & arg)
{
    INT dim = arg.flux()->dim();

    Tjs_El_User.ElAssert
    (
       _kth< dim,
       EEM0 << " fonc coordinate (FX,FY ...) incompatible with flux\n"
            << "   (dim flx = " <<  arg.flux()->dim()
            << ", coord required " << _kth  << ")"

    );

    switch (arg.flux()->type())
    {
         case Pack_Of_Pts::rle :
              return new RLE_Coord_Computed_kth(arg,_kth);

         case Pack_Of_Pts::integer :
              return new STD_Coord_Computed_kth<INT>(arg,_kth);

         case Pack_Of_Pts::real :
              return new STD_Coord_Computed_kth<REAL>(arg,_kth);
    }

    return 0;
}

void  Fonc_Coord_Not_Comp::compile (cElCompileFN & anEnv)
{
// std::cout << "UuUcompile K: " << _kth << " Has "  << mHASV << " Use " << FnumCoorUseCsteVal << "\n";
    if (mHASV && FnumCoorUseCsteVal)
          anEnv << mVal ;
    else
       anEnv.PutVarNum(_kth);
}


Fonc_Num::tKindOfExpr  Fonc_Coord_Not_Comp::KindOfExpr()
{
   return Fonc_Num::eIsFCoord;
}

INT Fonc_Coord_Not_Comp::CmpFormelIfSameKind(Fonc_Num_Not_Comp * aF2)
{
   return CmpTertiare(_kth,((Fonc_Coord_Not_Comp *) aF2)->_kth);
}

/******************************************************/
/*                                                    */
/*       Interface functions                          */
/*                                                    */
/******************************************************/

Fonc_Num kth_coord(INT k,bool HasAlwaysSameValue,double InitialValue)
{
    ELISE_ASSERT(k>=0,"KthCoord<0");
    return new Fonc_Coord_Not_Comp(k,HasAlwaysSameValue,InitialValue);
}

const Fonc_Num FX = kth_coord(0);
const Fonc_Num FY = kth_coord(1);
const Fonc_Num FZ = kth_coord(2);


/*  "CONCATENATION OF COORDINATE OF FUNCTIONS"               */
/*    That is, let f1 and f2 be two function :               */
/*       f1 : p ->  (x1, ... ,xn1)                           */
/*       f2 : p ->  (x'0, ... ,x'n2)                         */
/*                                                           */
/*     we define (f1,f2) as the function (with a n1+n2       */
/*     dimensional results) as :                            */
/*                                                           */
/*       (f1,f2) : p ->  (x1, ... ,xn1,x'1, ... , x'n2)      */
/*                                                           */
/*************************************************************/

template <class Type> class CatCoord_Fonc_Compute
                        : public  Fonc_Num_Comp_TPL<Type>
{

          public :

             const Pack_Of_Pts * values(const Pack_Of_Pts * pts)
             {
                   INT d_out = 0;

                   for(int f=0 ; f<2 ; f++)
                   {
                       const Std_Pack_Of_Pts<Type> *  pack_in =
                             SAFE_DYNC(Std_Pack_Of_Pts<Type> *,const_cast<Pack_Of_Pts *>(_tf[f]->values(pts)));
                        for(int d_in =0; d_in <_tf[f]->idim_out(); d_in++)
                           this->_pack_out->_pts[d_out++] = pack_in->_pts[d_in];
                   }
                   this->_pack_out->set_nb(pts->nb());
                   return this->_pack_out;
             }

             virtual ~CatCoord_Fonc_Compute()
             {
                 for(int f = 1; f>=0 ; f--)
                     delete _tf[f];
             }

             CatCoord_Fonc_Compute
             (
                  const Arg_Fonc_Num_Comp & arg,
                  Fonc_Num_Computed       * f0,
                  Fonc_Num_Computed       * f1
             ):
                  Fonc_Num_Comp_TPL<Type>
                  (arg,f0->idim_out()+f1->idim_out(),arg.flux(),true)
             {
                    _tf[0] = f0;
                    _tf[1] = f1;
             }

             virtual bool  icste( INT * v)
             {
                return
                       _tf[0]->icste(v)
                    && _tf[1]->icste(v+_tf[0]->idim_out());
             }

          private :
             Fonc_Num_Computed * _tf[2];
};



class CatCoord_Fonc_Num_Not_Comp : public Fonc_Num_Not_Comp
{
        public:

            Fonc_Num_Computed * compute (const Arg_Fonc_Num_Comp & arg)
            {
                  Fonc_Num_Computed * tfc[2];
                  Pack_Of_Pts::type_pack type_f;

                  tfc[0] = _f0.compute(arg);
                  tfc[1] = _f1.compute(arg);

                  type_f = convert_fonc_num_to_com_type(arg,tfc,arg.flux(),2);

                  if (type_f == Pack_Of_Pts::integer)
                     return new CatCoord_Fonc_Compute<INT>(arg,tfc[0],tfc[1]);
                  else
                     return new CatCoord_Fonc_Compute<REAL>(arg,tfc[0],tfc[1]);
            }


            CatCoord_Fonc_Num_Not_Comp (Fonc_Num f0,Fonc_Num f1) :
               _f0 (f0),
               _f1 (f1)
            {
            }

       private :
           Fonc_Num _f0;
           Fonc_Num _f1;


           virtual bool integral_fonc (bool iflux) const
           {
                return
                              _f0.integral_fonc(iflux)
                         &&   _f1.integral_fonc(iflux);
           }

           virtual INT dimf_out() const
           {
                   return
                            _f0.dimf_out()
                          + _f1.dimf_out() ;
           }

           void VarDerNN(ElGrowingSetInd & aSet) const
           {
                _f0.VarDerNN(aSet);
                _f1.VarDerNN(aSet);
           }
           virtual Fonc_Num deriv(INT k) const
           {
                 return Virgule(_f0.deriv(k),_f1.deriv(k));
           }
           REAL ValDeriv(const  PtsKD &  pts,INT k) const
           {
                 ELISE_ASSERT(false,"No Val Deriv for Virgule");
                 return 0;
           }

           REAL ValFonc(const PtsKD & pts) const
           {
                ELISE_ASSERT(false,"No ValFonc for KDimOut Foncs");
                return 0;
           }

           virtual void show(ostream & os) const
           {
              _f0.show(os);
              os << ",";
              _f1.show(os);
           }

           virtual bool  is0() const
           {
                 return _f0.is0()&&_f1.is0();
           }
           virtual bool  is1() const
           {
                 return _f0.is1()&&_f1.is1();
           }
};


Fonc_Num Virgule(Fonc_Num f0,Fonc_Num f1)
{
         return new CatCoord_Fonc_Num_Not_Comp(f0,f1);
}


/******************************************************/
/*                                                    */
/*       Fonc_Inside                                  */
/*                                                    */
/******************************************************/


class RLE_Inside_Comp : public  Fonc_Num_Comp_TPL<INT>
{
      public :
           RLE_Inside_Comp(const Arg_Fonc_Num_Comp & arg,INT * p0,INT *p1);

           virtual ~RLE_Inside_Comp()
           {
               delete _cliped;
           }

      private :
           const Pack_Of_Pts * values(const Pack_Of_Pts *);



           INT _p0[Elise_Std_Max_Dim];
           INT _p1[Elise_Std_Max_Dim];
           RLE_Pack_Of_Pts * _cliped;

};

RLE_Inside_Comp::RLE_Inside_Comp
(
     const Arg_Fonc_Num_Comp & arg,
     INT * p0,
     INT * p1
) :
       Fonc_Num_Comp_TPL<INT> (arg,1,arg.flux())
{
      convert(_p0,p0,arg.flux()->dim());
      convert(_p1,p1,arg.flux()->dim());
      _cliped = RLE_Pack_Of_Pts::new_pck(arg.flux()->dim(),arg.flux()->sz_buf());
}

const Pack_Of_Pts * RLE_Inside_Comp::values(const Pack_Of_Pts * pts)
{
     RLE_Pack_Of_Pts * rpts = SAFE_DYNC(RLE_Pack_Of_Pts *,const_cast<Pack_Of_Pts *>(pts));
     INT nb = pts->nb();

     _pack_out->set_nb(nb);
     INT * v = _pack_out->_pts[0];
     INT i0 = _cliped->clip(rpts,_p0,_p1);
     INT nbc  = _cliped->nb();

     set_cste(v,0,i0);
     set_cste(v+i0,1,nbc);
     INT i1 = i0+nbc;
     set_cste(v+i1,0,nb-i1);
     return _pack_out;
}


template <class Type> class Std_Inside_Comp :
                             public  Fonc_Num_Comp_TPL<INT>
{
      public :
           Std_Inside_Comp(const Arg_Fonc_Num_Comp & arg,INT * p0,INT *p1);
      private :

           const Pack_Of_Pts * values(const Pack_Of_Pts *);



           Type _p0[Elise_Std_Max_Dim];
           Type _p1[Elise_Std_Max_Dim];

};


template <class Type> Std_Inside_Comp<Type>::Std_Inside_Comp
(
         const Arg_Fonc_Num_Comp & arg,
         INT * p0,
         INT * p1
)   :
       Fonc_Num_Comp_TPL<INT> (arg,1,arg.flux())
{
      convert(_p0,p0,arg.flux()->dim());
      convert(_p1,p1,arg.flux()->dim());
}

template <class Type> const Pack_Of_Pts *
        Std_Inside_Comp<Type>::values(const Pack_Of_Pts * gen_pts)
{
        const Std_Pack_Of_Pts<Type> *  pts =
              SAFE_DYNC(Std_Pack_Of_Pts<Type> *,const_cast<Pack_Of_Pts *>(gen_pts));

        compute_inside
        (
             _pack_out->_pts[0],
             pts->_pts,
             pts->nb(),
             pts->dim(),
             _p0,
             _p1

        );
        _pack_out->set_nb(gen_pts->nb());
        return _pack_out;
}



class Inside_Not_Comp : public Fonc_Num_Not_Comp
{
        public:

            Fonc_Num_Computed * compute (const Arg_Fonc_Num_Comp & arg);


            Inside_Not_Comp (const INT *,const INT *,INT dim);

       private :
           INT _p0[Elise_Std_Max_Dim];
           INT _p1[Elise_Std_Max_Dim];
           INT _dim;


           virtual bool integral_fonc (bool ) const
           {
                return true;
           }

           virtual INT dimf_out() const
           {
                   return 1;
           }
           void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}


           virtual Fonc_Num deriv(INT k) const
           {
               ELISE_ASSERT(false,"No Deriv for inside");
               return 0;
           }
           REAL ValFonc(const PtsKD & pts) const
           {
                ELISE_ASSERT(false,"No ValFonc for inside");
                return 0;
           }
           REAL ValDeriv(const  PtsKD &  pts,INT k) const
           {
                 ELISE_ASSERT(false,"No Val Deriv for inside");
                 return 0;
           }
           virtual void show(ostream & os) const
           {
               os << "inside()";
           }
};



Inside_Not_Comp::Inside_Not_Comp(const INT * p0,const INT * p1,INT dim)
{
     _dim = dim;
     convert(_p0,p0,dim);
     convert(_p1,p1,dim);
}

Fonc_Num_Computed * Inside_Not_Comp::compute (const Arg_Fonc_Num_Comp & arg)
{

      Tjs_El_User.ElAssert
      (
            (_dim == arg.flux()->dim()),
             EEM0 << " inside fonction : dim of object != dim of flux\n"
                  << "    ( dim obj = " <<  _dim
                  << ", dim flux = " <<   arg.flux()->dim() << ")"
      );
      switch(arg.flux()->type())
      {
            case Pack_Of_Pts::rle :
                 return new RLE_Inside_Comp(arg,_p0,_p1);

            case Pack_Of_Pts::integer :
                 return new Std_Inside_Comp<INT>(arg,_p0,_p1);

            case Pack_Of_Pts::real :
                 return new Std_Inside_Comp<REAL>(arg,_p0,_p1);
      }

      return 0;
}




Fonc_Num  inside(const INT * p0,const INT * p1,INT dim)
{
    return new Inside_Not_Comp(p0,p1,dim);
}

Fonc_Num  inside(Pt2di pt0,Pt2di pt1)
{
    INT t0[2],t1[2];

    pt0.to_tab(t0);
    pt1.to_tab(t1);
    return inside(t0,t1,2);
}

Fonc_Num  Identite(INT dim)
{
   Fonc_Num identite = FX;

   for (INT d = 1; d<dim ; d++)
       identite = Virgule(identite,kth_coord(d));

   return identite;
}

template <class Type> Fonc_Num TpleCsteNDim(const Type & aVal, int aDim)
{
   Fonc_Num aRes(aVal);
   for (INT d = 1; d<aDim ; d++)
       aRes = Virgule(aRes,Fonc_Num(aVal));

   return aRes;
}

Fonc_Num CsteNDim(double aVal,INT aDim) {return TpleCsteNDim<double>(aVal,aDim);}
Fonc_Num CsteNDim(int    aVal,INT aDim) {return TpleCsteNDim<int>(aVal,aDim);}


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
