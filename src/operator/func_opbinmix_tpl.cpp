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
/*             OP_Bin_TPL                                      */
/*                                                             */
/***************************************************************/



Fonc_Num Op_Bin_Not_Comp::Simplify() 
{
   std::string aStrName(_name);
   if (aStrName == "+") return _f0.Simplify() +  _f1.Simplify();
   if (aStrName == "*") return _f0.Simplify() *  _f1.Simplify();
   if (aStrName == "/") return _f0.Simplify() /  _f1.Simplify();
   if (aStrName == "-") return _f0.Simplify() -  _f1.Simplify();
   if (aStrName == "pow") return pow(_f0.Simplify(),_f1.Simplify());

   //if (aStrName == "+") return _f0.Simplify() +  _f1.Simplify();


   std::cout << "FOR operator = " << aStrName << "\n";
   ELISE_ASSERT(false,"Unhandled operator in Simplify");
   return 0;
}

Fonc_Num Op_Bin_Not_Comp::deriv(INT k) const
{
    return _OpBinDeriv(_f0,_f1,k);
}

REAL  Op_Bin_Not_Comp::ValFonc(const  PtsKD &  pts) const
{
     return _OpBinVal(_f0.ValFonc(pts),_f1.ValFonc(pts));
}

void  Op_Bin_Not_Comp::show(ostream & os) const
{
    os << _name << "(";
    _f0.show(os);
     os << ",";
    _f1.show(os);
    os << ")";
}


static const char * mTGauche = "+*-/*-*/";
static const char * mTCentre = "++++*---";
static const char * mTDroite = "+*-/* */";

void  Op_Bin_Not_Comp::PutFoncPar
      (
          Fonc_Num f,
	  cElCompileFN & anEnv,
	  const char * mSimpl
      )
{
    bool Par = ! f.IsAtom() ;
    if (f.IsOpUn())
       Par = false;
    if (f.IsOpbin())
    {
       Op_Bin_Not_Comp * op2 =  SAFE_DYNC(Op_Bin_Not_Comp *,f.ptr());
       char aC = * _name;
       char aC2 = *(op2->_name);

       for (int k=0 ; mTCentre[k]; k++)
          if ((mTCentre[k]==aC) && (mSimpl[k]==aC2))
             Par = false;
    }
    if (Par)
       anEnv << "(" << f << ")" ;
    else
       anEnv << f ;
}

void  Op_Bin_Not_Comp::compile (cElCompileFN & anEnv)
{
    if (mIsInfixe)
    {
         PutFoncPar(_f0,anEnv,mTGauche);
	 // anEnv << _name;
	 anEnv << " " <<  _name << " " ; // MPD Modif  sinon genere conflits  du genre -- 
	 PutFoncPar(_f1,anEnv,mTDroite);
    }
    else
      anEnv << _name << "(" << _f0 << "," << _f1 << ")" ;
}


Fonc_Num::tKindOfExpr  Op_Bin_Not_Comp::KindOfExpr()
{
    return Fonc_Num::eIsOpBin;
}

INT Op_Bin_Not_Comp::CmpFormelIfSameKind(Fonc_Num_Not_Comp * aF2)
{
   Op_Bin_Not_Comp * op2 = (Op_Bin_Not_Comp * ) aF2;

   INT res = CmpTertiare(_name,op2->_name);
   if (res) return res;
       
   res = _f0.CmpFormel(op2->_f0);
   if (res) return res;

    return _f1.CmpFormel(op2->_f1);
}



REAL Op_Bin_Not_Comp::ValDeriv(const  PtsKD &  pts,INT k) const
{
   return mOpBinValDeriv(_f0,_f1,pts,k);
}

static Fonc_Num NoDeriv(Fonc_Num,Fonc_Num,INT)
{
    ELISE_ASSERT(false,"No deriv for required Binary Operator ");
    return 0;
}                  

static double NoValDeriv(Fonc_Num,Fonc_Num,const  PtsKD &,INT k)
{
    ELISE_ASSERT(false,"No val deriv for required Binary Operator ");
    return 0;
}


/***************************************************************/
/*                                                             */
/*             OP_Bin_TPL                                      */
/*                                                             */
/***************************************************************/



template <class Type> OP_Bin_Comp_TPL<Type>::OP_Bin_Comp_TPL
                      (     const Arg_Fonc_Num_Comp & arg,
                            Fonc_Num_Computed * f0,
                            Fonc_Num_Computed * f1,
                            Flux_Pts_Computed * flx
                      ) :
         Fonc_Num_Comp_TPL<Type>(arg,ElMax(f0->idim_out(),f1->idim_out()),flx),
         _f0 (f0),
         _f1 (f1),
         _dim_0 (f0->idim_out()),
         _dim_1 (f1->idim_out())
{
}


template <class Type> OP_Bin_Comp_TPL<Type>::~OP_Bin_Comp_TPL()
{
    delete _f1;
    delete _f0;
}


/***************************************************************/
/*                                                             */
/*    OP_Mixte_Comp_TPL <class TOut,class T0In,class T1In>     */
/*                                                             */
/***************************************************************/

template <class Oper,class TOut,class T0In,class T1In>
          class OP_Mixte_Comp_TPL :
                  public  OP_Bin_Comp_TPL<TOut>
{
     public :
         virtual const Pack_Of_Pts * values(const Pack_Of_Pts *);
         OP_Mixte_Comp_TPL
         (       const Arg_Fonc_Num_Comp &,
                 Fonc_Num_Computed * f1,
                 Fonc_Num_Computed * f2,
                 Flux_Pts_Computed * flx,
                 const Oper &  op
         );

      private :
         const Oper & _op;

};



template <class Oper,class TOut,class T0In,class T1In>   
         OP_Mixte_Comp_TPL<Oper,TOut,T0In,T1In>::OP_Mixte_Comp_TPL
              (       const Arg_Fonc_Num_Comp & arg,
                      Fonc_Num_Computed * f0,
                      Fonc_Num_Computed * f1,
                      Flux_Pts_Computed * flx,
                      const Oper &  op
              ):
              OP_Bin_Comp_TPL<TOut>(arg,f0,f1,flx),
              _op(op)

{
}

template <class Oper,class TOut,class T0In,class T1In>   
         const Pack_Of_Pts * OP_Mixte_Comp_TPL<Oper,TOut,T0In,T1In>::values
                         (const Pack_Of_Pts * pts)
{
     Std_Pack_Of_Pts<T0In> * v0
            =  SAFE_DYNC (Std_Pack_Of_Pts<T0In> *,const_cast<Pack_Of_Pts *>(this->_f0->values(pts)));
     T0In **  t0 =  v0->_pts;
     OLD_BUG_CARD_VAL_FONC(v0,pts);

     Std_Pack_Of_Pts<T1In> * v1
            =  SAFE_DYNC (Std_Pack_Of_Pts<T1In> *,const_cast<Pack_Of_Pts *>(this->_f1->values(pts)));
     T1In **  t1 =  v1->_pts;
     OLD_BUG_CARD_VAL_FONC(v1,pts);


     TOut ** to = this->_pack_out->_pts;


     INT nb = pts->nb();

     for (int d = 0; d < this->_dim_out; d++)
         _op.t0_eg_t1_op_t2
         (
             to[d],
             t0[ElMin(d,this->_dim_0-1)],
             t1[ElMin(d,this->_dim_1-1)],
             nb
         );

   this->_pack_out->set_nb(nb);
   return this->_pack_out;
}

typedef OP_Mixte_Comp_TPL<OperBinMixte,INT,INT,INT>     OP_MC_III;
typedef OP_Mixte_Comp_TPL<OperBinMixte,REAL,REAL,INT>   OP_MC_RRI;
typedef OP_Mixte_Comp_TPL<OperBinMixte,REAL,INT,REAL>   OP_MC_RIR;
typedef OP_Mixte_Comp_TPL<OperBinMixte,REAL,REAL,REAL>  OP_MC_RRR;


/*****************************************************/
/*                                                   */
/*           Op_Bin_Mix_Not_Comp                     */
/*                                                   */
/*****************************************************/

static INT StdMixDegre(INT,INT)
{
   return -1;
}

class Op_Bin_Mix_Not_Comp : public Op_Bin_Not_Comp
{
       public :
          typedef INT (* TyMixDegre)(INT,INT);

          static Fonc_Num New
               ( Fonc_Num,
                 Fonc_Num,
                 const OperBinMixte & op,
		 bool  isInfixe,
                 const char *,
                 TyVal,
                 TyDeriv,
                 TyValDeriv,
                 TyMixDegre
               );

       private :
         
           virtual  Fonc_Num_Computed * op_bin_comp
                                      (const Arg_Fonc_Num_Comp &,
                                       Fonc_Num_Computed       * f1,
                                       Fonc_Num_Computed       * f2
                                      );

            Op_Bin_Mix_Not_Comp
               ( Fonc_Num,
                 Fonc_Num,
                 const OperBinMixte & op,
		 bool  isInfixe,
                 const char *,
                 TyVal,
                 TyDeriv,
                 TyValDeriv,
                 TyMixDegre
               );

            const OperBinMixte & _op;
            TyMixDegre           mMixDeg;

            virtual  bool integral_fonc(bool iflux) const
            {
                     return      _f0.integral_fonc(iflux) 
                             &&  _f1.integral_fonc(iflux);
            }
           virtual INT DegrePoly() const
           {
                INT d0 = _f0.DegrePoly();
                if (d0==-1) 
                   return -1;

                INT d1 = _f1.DegrePoly();
                if (d1==-1) 
                   return -1;

                return mMixDeg(d0,d1);
           }

};

Fonc_Num Op_Bin_Mix_Not_Comp::New
        (       Fonc_Num f0,
                Fonc_Num f1,
                const OperBinMixte & op,
		bool        isInfixe,
                const char * name,
                TyVal       Value,
                TyDeriv     Deriv,
                TyValDeriv  ValDeriv,
                TyMixDegre  MixDeg
        ) 
{
   REAL aCst0,aCst1;

    if (f0.IsCsteRealDim1(aCst0) && f1.IsCsteRealDim1(aCst1))
       return Value(aCst0,aCst1);

   return new Op_Bin_Mix_Not_Comp(f0,f1,op,isInfixe,name,Value,Deriv,ValDeriv,MixDeg);
}

Op_Bin_Mix_Not_Comp::Op_Bin_Mix_Not_Comp
        (       Fonc_Num f0,
                Fonc_Num f1,
                const OperBinMixte & op,
		bool       isInfixe,
                const char * name,
                TyVal       Value,
                TyDeriv     Deriv,
                TyValDeriv  ValDeriv,
                TyMixDegre   MixDeg
         ) :
          Op_Bin_Not_Comp   (f0,f1,isInfixe,name,Value,Deriv,ValDeriv)   ,
         _op                (op),
         mMixDeg            (MixDeg)
{
}


Fonc_Num_Computed * Op_Bin_Mix_Not_Comp::op_bin_comp
                    (      const Arg_Fonc_Num_Comp & arg,
                           Fonc_Num_Computed       * f0,
                           Fonc_Num_Computed       * f1
                    )
{


   bool i0 =  (f0->type_out() == Pack_Of_Pts::integer);
   bool i1 =  (f1->type_out() == Pack_Of_Pts::integer);

   if (i0 && i1)
      return  new OP_MC_III(arg,f0,f1,arg.flux(),_op);


   if (i0 && (! i1))
      return  new OP_MC_RIR(arg,f0,f1,arg.flux(),_op);


   if ((! i0) && i1)
      return  new OP_MC_RRI(arg,f0,f1,arg.flux(),_op);

   return  new OP_MC_RRR(arg,f0,f1,arg.flux(),_op);
}



        //===========================================
        //       interface functions                     
        //===========================================

static REAL VPlus(REAL v1,REAL v2) {return v1+v2;}
static Fonc_Num  DPlus(Fonc_Num f1,Fonc_Num f2,INT k)
{
   return f1.deriv(k) + f2.deriv(k);
}

static REAL  VDPlus(Fonc_Num f1,Fonc_Num f2,const PtsKD & aPts,INT k)
{
   return f1.ValDeriv(aPts,k) + f2.ValDeriv(aPts,k);
}

INT PlusMoinsMixDegre(INT D0,INT D1)
{
    return ElMax(D0,D1);
}

Fonc_Num operator + (Fonc_Num f1,Fonc_Num f2)
{

     if (f1.is0()) 
        return f2;
     if (f2.is0()) 
        return f1;

     return Op_Bin_Mix_Not_Comp::New(f1,f2,OpSum,true,"+",VPlus,DPlus,VDPlus,PlusMoinsMixDegre);
};




INT MulMixDegre(INT D0,INT D1)
{
    return D0+D1;
}
static REAL VMul(REAL v1,REAL v2) {return v1*v2;}
static Fonc_Num  DMul(Fonc_Num f1,Fonc_Num f2,INT k)
{
   return f1.deriv(k) * f2 + f2.deriv(k) * f1;
}

static REAL  VDMul(Fonc_Num f1,Fonc_Num f2,const PtsKD & aPts,INT k)
{
   return f1.ValDeriv(aPts,k)* f2.ValFonc(aPts) + f2.ValDeriv(aPts,k) * f1.ValFonc(aPts);
}

Fonc_Num operator * (Fonc_Num f1,Fonc_Num f2)
{
     if (f1.is0() || f2.is0()) 
        return Fonc_Num(0);

     if (f1.is1()) 
        return f2;
     if (f2.is1()) 
        return f1;
     return Op_Bin_Mix_Not_Comp::New(f1,f2,OpMul,true,"*",VMul,DMul,VDMul,MulMixDegre);
};


static REAL VMax(REAL v1,REAL v2) {return ElMax(v1,v2);}
Fonc_Num Max (Fonc_Num f1,Fonc_Num f2)
{
     return Op_Bin_Mix_Not_Comp::New(f1,f2,OpMax,false,"ElMax",VMax,NoDeriv,NoValDeriv,StdMixDegre);
};



static REAL VMin(REAL v1,REAL v2) {return ElMin(v1,v2);}
Fonc_Num Min (Fonc_Num f1,Fonc_Num f2)
{
     return Op_Bin_Mix_Not_Comp::New(f1,f2,OpMin,false,"ElMin",VMin,NoDeriv,NoValDeriv,PlusMoinsMixDegre);
};


INT DivMixDegre(INT D0,INT D1)
{
    if (D1 ==0) return D0;
    return -1;
}


static REAL VDiv(REAL v1,REAL v2) 
{
   ELISE_ASSERT(v2!=0,"/0 in VDiv");
   return v1 /v2;
}
static Fonc_Num  DDiv(Fonc_Num f1,Fonc_Num f2,INT k)
{
   return (f1.deriv(k) * f2 - f1 * f2.deriv(k)) /Square(f2);
}

static REAL  VDDiv(Fonc_Num f1,Fonc_Num f2,const PtsKD & aPts,INT k)
{
   REAL vF2 = f2.ValFonc(aPts);
   return (f1.ValDeriv(aPts,k)* vF2 - f2.ValDeriv(aPts,k) * f1.ValFonc(aPts))/ElSquare(vF2);
}

Fonc_Num operator / (Fonc_Num f1,Fonc_Num f2)
{
     if (f1.is0()) 
        return 0;
     if (f2.is1()) 
        return f1;
     return Op_Bin_Mix_Not_Comp::New(f1,f2,OpDiv,true,"/",VDiv,DDiv,VDDiv,DivMixDegre);
};



static REAL VMoins(REAL v1,REAL v2) {return v1-v2;}
static Fonc_Num  DMoins(Fonc_Num f1,Fonc_Num f2,INT k)
{
   return f1.deriv(k) - f2.deriv(k);
}
static REAL  VDMoins(Fonc_Num f1,Fonc_Num f2,const PtsKD & aPts,INT k)
{
   return f1.ValDeriv(aPts,k) - f2.ValDeriv(aPts,k);
}
Fonc_Num operator - (Fonc_Num f1,Fonc_Num f2)
{
     if (f1.is0()) 
        return -f2;
     if (f2.is0()) 
        return f1;
     return Op_Bin_Mix_Not_Comp::New(f1,f2,OpMinus2,true,"-",VMoins,DMoins,VDMoins,PlusMoinsMixDegre);
};




REAL CppPow(REAL x,REAL y) {return pow(x,y);}
Fonc_Num pow (Fonc_Num f1,Fonc_Num f2)
{
     if (f1.is0()) return 0;
     if (f1.is1()) return 1;
     if (f2.is0()) return 1;
     if (f2.is1()) return f1;



     return Op_Bin_Mix_Not_Comp::New(f1,f2,OpPow2,false,"pow",CppPow,NoDeriv,NoValDeriv,StdMixDegre);
};


       // Just utilitaries

Fonc_Num Max(Fonc_Num f1,Fonc_Num f2,Fonc_Num f3)
{
     return (Max(f1,Max(f2,f3)));
}

Fonc_Num Max(Fonc_Num f1,Fonc_Num f2,Fonc_Num f3,Fonc_Num f4)
{
     return (Max(f1,Max(f2,f3,f4)));
}

Fonc_Num Min(Fonc_Num f1,Fonc_Num f2,Fonc_Num f3)
{
     return (Min(f1,Min(f2,f3)));
}

Fonc_Num Min(Fonc_Num f1,Fonc_Num f2,Fonc_Num f3,Fonc_Num f4)
{
     return (Min(f1,Min(f2,f3,f4)));
}



double VF1OrF2IfBadNum(double aV1,double aV2) {return IsBadNum(aV1) ? aV2 : aV1;}

Fonc_Num   F1OrF2IfBadNum (Fonc_Num f1,Fonc_Num f2)
{
     return Op_Bin_Mix_Not_Comp::New(f1,f2,OpF1OrF2IfBadNum,true,"F1F2BN",VF1OrF2IfBadNum,NoDeriv,NoValDeriv,StdMixDegre);
};

/*****************************************************/
/*                                                   */
/*           Op_Compar_Not_Computed                  */
/*                                                   */
/*****************************************************/


typedef OP_Mixte_Comp_TPL<OperComp,INT,REAL,REAL>     OpComparRR;;
typedef OP_Mixte_Comp_TPL<OperComp,INT,REAL,INT>      OpComparRI;;
typedef OP_Mixte_Comp_TPL<OperComp,INT,INT,REAL>      OpComparIR;;
typedef OP_Mixte_Comp_TPL<OperComp,INT,INT,INT>       OpComparII;;



class Op_Compar_Not_Computed : public Op_Bin_Not_Comp
{
       public :
         
           virtual  Fonc_Num_Computed * op_bin_comp
                                      (const Arg_Fonc_Num_Comp &,
                                       Fonc_Num_Computed       * f1,
                                       Fonc_Num_Computed       * f2
                                      );

            Op_Compar_Not_Computed
               (Fonc_Num,Fonc_Num,const OperComp & op,bool isInfixe,const char *,TyVal);

       private :
           const OperComp & _op;
           virtual bool integral_fonc(bool) const { return true;}
          
};

Op_Compar_Not_Computed::Op_Compar_Not_Computed
        (       Fonc_Num f0,
                Fonc_Num f1,
                const OperComp & op,
		bool         isInfixe,
                const char * name,
                TyVal        Value
         ) :
          Op_Bin_Not_Comp(f0,f1,isInfixe,name,Value,NoDeriv,NoValDeriv)   ,
         _op (op)
{
}


Fonc_Num_Computed * Op_Compar_Not_Computed::op_bin_comp
                    (      const Arg_Fonc_Num_Comp & arg,
                           Fonc_Num_Computed       * f0,
                           Fonc_Num_Computed       * f1
                    )
{

   bool i0 =  (f0->type_out() == Pack_Of_Pts::integer);
   bool i1 =  (f1->type_out() == Pack_Of_Pts::integer);

   if (i0 && i1)
      return  new OpComparII (arg,f0,f1,arg.flux(),_op);


   if (i0 && (! i1))
      return  new OpComparIR(arg,f0,f1,arg.flux(),_op);


   if ((! i0) && i1)
      return  new OpComparRI(arg,f0,f1,arg.flux(),_op);

   return  new OpComparRR(arg,f0,f1,arg.flux(),_op);
}

        //===========================================
        //       interface functions                     
        //===========================================

static REAL VEgal(REAL v1,REAL v2) {return v1==v2;}
Fonc_Num operator == (Fonc_Num f1,Fonc_Num f2)
{
     return new Op_Compar_Not_Computed(f1,f2,OpEqual,true,"==",VEgal);
}

static REAL VNotEgal(REAL v1,REAL v2) {return v1!=v2;}
Fonc_Num operator != (Fonc_Num f1,Fonc_Num f2)
{
     return new Op_Compar_Not_Computed(f1,f2,OpNotEq,true,"!=",VNotEgal);
}


static REAL VInfStrict(REAL v1,REAL v2) {return v1 < v2;}
Fonc_Num operator < (Fonc_Num f1,Fonc_Num f2)
{
    return new Op_Compar_Not_Computed(f1,f2,OpInfStr,true,"<",VInfStrict);
}

static REAL VInfOuEgal(REAL v1,REAL v2) {return v1 <= v2;}
Fonc_Num operator <= (Fonc_Num f1,Fonc_Num f2)
{
    return new Op_Compar_Not_Computed(f1,f2,OpInfOrEq,true,"<=",VInfOuEgal);
}


static REAL VSupStrict(REAL v1,REAL v2) {return v1 > v2;}
Fonc_Num operator > (Fonc_Num f1,Fonc_Num f2)
{
    return new Op_Compar_Not_Computed(f1,f2,OpSupStr,true,">",VSupStrict);
}

static REAL VSupOuEgal(REAL v1,REAL v2) {return v1 >= v2;}
Fonc_Num operator >= (Fonc_Num f1,Fonc_Num f2)
{
    return new Op_Compar_Not_Computed(f1,f2,OpSupOrEq,true,">=",VSupOuEgal);
}




/*****************************************************/
/*                                                   */
/*           Op_Bin_Entier                           */
/*                                                   */
/*****************************************************/

typedef OP_Mixte_Comp_TPL<OperBinInt,INT,INT,INT>       OpBinEntCompute;;

class Op_Bin_Ent : public Op_Bin_Not_Comp
{
       public :
         
           virtual  Fonc_Num_Computed * op_bin_comp
                                      (const Arg_Fonc_Num_Comp &,
                                       Fonc_Num_Computed       * f1,
                                       Fonc_Num_Computed       * f2
                                      );

            Op_Bin_Ent
               (Fonc_Num,Fonc_Num,const OperBinInt & op,bool isInfixe,const char *,TyVal);

       private :
           const OperBinInt & _op;
          
           virtual bool integral_fonc(bool) const {return true;}
};

Op_Bin_Ent::Op_Bin_Ent
        (       Fonc_Num f0,
                Fonc_Num f1,
                const OperBinInt & op,
		bool         isInfixe,
                const char * name,
                TyVal        Value
         ) :
          Op_Bin_Not_Comp(f0,f1,isInfixe,name,Value,NoDeriv,NoValDeriv) ,
         _op (op)
{
}


Fonc_Num_Computed * Op_Bin_Ent::op_bin_comp
                    (      const Arg_Fonc_Num_Comp & arg,
                           Fonc_Num_Computed       * f0,
                           Fonc_Num_Computed       * f1
                    )
{
      f0 = convert_fonc_num(arg,f0,arg.flux(), Pack_Of_Pts::integer);
      f1 = convert_fonc_num(arg,f1,arg.flux(), Pack_Of_Pts::integer);

      return  new OpBinEntCompute (arg,f0,f1,arg.flux(),_op);
}


static REAL VEtBB(REAL v1,REAL v2) {return ((int) v1) & ((int)v2);}
Fonc_Num operator & (Fonc_Num f1,Fonc_Num f2)
{
     return new Op_Bin_Ent(f1,f2,OpAndBB,true,"&",VEtBB);
};




static REAL VEt(REAL v1,REAL v2) {return ((int) v1) && ((int)v2);}
Fonc_Num operator && (Fonc_Num f1,Fonc_Num f2)
{
     return new Op_Bin_Ent(f1,f2,OperAnd,true,"&&",VEt);
};

static REAL VOuBB(REAL v1,REAL v2) {return ((int) v1) | ((int)v2);}
Fonc_Num operator | (Fonc_Num f1,Fonc_Num f2)
{
     return new Op_Bin_Ent(f1,f2,OpOrBB,true,"|",VOuBB);
};

static REAL VOu(REAL v1,REAL v2) {return ((int) v1) || ((int)v2);}
Fonc_Num operator || (Fonc_Num f1,Fonc_Num f2)
{
     return new  Op_Bin_Ent(f1,f2,OperOr,true,"||",VOu);
};

static REAL VXOrBB(REAL v1,REAL v2) {return ((int) v1) ^ ((int)v2);}
Fonc_Num operator ^ (Fonc_Num f1,Fonc_Num f2)
{
     return new Op_Bin_Ent(f1,f2,OperXorBB,true,"^",VXOrBB);
};



static REAL VXor(REAL v1,REAL v2) {return (v1!=0) ^ (v2!=0);}
Fonc_Num ElXor (Fonc_Num f1,Fonc_Num f2)
{
     return new Op_Bin_Ent(f1,f2,OperXor,false,"ElXor",VXor);
};

static REAL VModC(REAL v1,REAL v2) {return ((int) v1) % ((int)v2);}
Fonc_Num operator % (Fonc_Num f1,Fonc_Num f2)
{
     return new Op_Bin_Ent(f1,f2,OperStdMod,true,"%",VModC);
};

static REAL VModElise(REAL v1,REAL v2) {return mod((int)v1,(int)v2);}
Fonc_Num mod        (Fonc_Num f1,Fonc_Num f2)
{
     return new Op_Bin_Ent(f1,f2,OperMod,false,"mod",VModElise);
};

static REAL VShiftDr(REAL v1,REAL v2) {return ((int) v1) >> ((int)v2);}
Fonc_Num operator >>(Fonc_Num f1,Fonc_Num f2)
{
     return new Op_Bin_Ent(f1,f2,OperRightShift,true,">>",VShiftDr);
};

static REAL VShiftGa(REAL v1,REAL v2) {return ((int) v1) << ((int)v2);}
Fonc_Num operator <<(Fonc_Num f1,Fonc_Num f2)
{
     return new Op_Bin_Ent(f1,f2,OperLeftShift,true,"<<",VShiftGa);
};


tOperFuncBin  OperFuncBinaireFromName(const std::string & aName)
{
   if (aName=="-") return operator -;
   if (aName=="/") return operator /;
   if (aName=="pow") return pow;

   if (aName==">=") return operator >=;
   if (aName==">")  return operator  >;
   if (aName=="<")  return operator  <;
   if (aName=="<=") return operator <=;
   if (aName=="!=") return operator !=;
   if (aName=="==") return operator ==;

   if (aName=="&") return operator &;
   if (aName=="&&") return operator &&;
   if (aName=="|") return operator   |;
   if (aName=="||") return operator ||;
   if (aName=="^") return operator ^;
   if (aName=="%") return operator %;
   if (aName=="mod") return mod;

   if (aName==">>") return operator >>;
   if (aName=="<<") return operator <<;

   if (aName=="F1F2BN") return F1OrF2IfBadNum;

   std::cout << "For name=" << aName << "\n";
   ELISE_ASSERT(false,"Cannot  get operator");
   return 0;
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
