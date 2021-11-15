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
/*             OP_Un_TPL                                       */
/*                                                             */
/***************************************************************/

Fonc_Num Op_Un_Not_Comp::Simplify()
{
   std::string aStrName = _name;
   if (aStrName=="-") return -(_f.Simplify());

   std::cout << "For operator = " << _name << "\n";
   ELISE_ASSERT(false,"Unhandled operator in Op_Un_Not_Comp::Simplify");
   return 0;
}

Fonc_Num Op_Un_Not_Comp::deriv(INT k) const 
{
    return _OpUnDeriv(_f,k);
}

REAL  Op_Un_Not_Comp::ValFonc(const  PtsKD &  pts) const
{
     return _OpUnVal(_f.ValFonc(pts))  ;
}

REAL Op_Un_Not_Comp::ValDeriv(const  PtsKD &  aPts,INT k) const
{
    return mOpUnValDeriv(_f,aPts,k);
}

void  Op_Un_Not_Comp::show(ostream & os) const 
{
    os << _name << "(";
    _f.show(os);
    os << ")";
}

void  Op_Un_Not_Comp::compile(cElCompileFN & anEnv)
{
     anEnv <<  _name << "(" << _f << ")";
}

Fonc_Num::tKindOfExpr  Op_Un_Not_Comp::KindOfExpr()
{
   return Fonc_Num::eIsOpUn;
}

INT Op_Un_Not_Comp::CmpFormelIfSameKind(Fonc_Num_Not_Comp * aF2)
{
  Op_Un_Not_Comp * op2 = (Op_Un_Not_Comp * ) aF2;

   INT res = CmpTertiare(_name,op2->_name);
   if (res) return res;

   return _f.CmpFormel(op2->_f);
}



static Fonc_Num NoDeriv(Fonc_Num,INT)
{
    ELISE_ASSERT(false,"No deriv for required Unary Operator "); 
    return 0;
}


static REAL NoValDeriv(Fonc_Num,const  PtsKD &,INT)
{
    ELISE_ASSERT(false,"No Val deriv for required Unary Operator "); 
    return 0;
}




/***************************************************************/
/***************************************************************/
/***************************************************************/

typedef  void (* FONC_UN_INT )(INT *  ,const INT *  , INT);
typedef  void (* FONC_UN_REAL)(REAL * ,const REAL * , INT);

typedef  void (* FONC_UN_IR )(INT *  ,const REAL * , INT);
typedef  void (* FONC_UN_RI )(REAL * ,const INT *  , INT);

/***************************************************************/
/*                                                             */
/*             OP_Un_TPL                                       */
/*                                                             */
/***************************************************************/


template <class Type> OP_Un_Comp_TPL_Gen<Type>::OP_Un_Comp_TPL_Gen
                      (     const Arg_Fonc_Num_Comp & arg,
                            Fonc_Num_Computed * f,
                            Flux_Pts_Computed * flx
                      ) :
         Fonc_Num_Comp_TPL<Type>(arg,f->idim_out(),flx),
         _f (f)
{
}


template <class Type> OP_Un_Comp_TPL_Gen<Type>::~OP_Un_Comp_TPL_Gen()
{
    delete _f;
}


/***************************************************************/
/*                                                             */
/*             OP_Un_TPL                                       */
/*                                                             */
/***************************************************************/

template <class TOut,class TIn> OP_Un_Comp<TOut,TIn>::OP_Un_Comp
                                (        const Arg_Fonc_Num_Comp & arg,
                                        Fonc_Num_Computed * f,
                                        Flux_Pts_Computed * flx,
                                        void  (* fonc) (TOut *,const TIn *,INT)
                                ):
            OP_Un_Comp_TPL_Gen<TOut>(arg,f,flx),
            _fonc                   (fonc)
{
}
 
template <class TOut,class TIn>  const Pack_Of_Pts * 
          OP_Un_Comp<TOut,TIn>::values(const Pack_Of_Pts * pts)
{
    TIn **  ti =  SAFE_DYNC
                  (   Std_Pack_Of_Pts<TIn> *,
                      const_cast<Pack_Of_Pts *>(this->_f->values(pts))
                  )->_pts;

    TOut ** to = this->_pack_out->_pts;

    INT nb = pts->nb();

    for (INT i=0 ; i<this->_dim_out ; i++)
        _fonc(to[i],ti[i],nb);

    this->_pack_out->set_nb(nb);
    return this->_pack_out;
}


typedef  OP_Un_Comp<INT,REAL>   OP_UC_IR;
typedef  OP_Un_Comp<REAL,INT>   OP_UC_RI;
typedef  OP_Un_Comp<INT,INT>    OP_UC_II;
typedef  OP_Un_Comp<REAL,REAL>  OP_UC_RR;


/***************************************************************/
/*                                                             */
/*             Convertion                                      */
/*                                                             */
/***************************************************************/


      //  Computed 

Fonc_Num_Computed * convert_fonc_num
                    (       const Arg_Fonc_Num_Comp & arg,
                            Fonc_Num_Computed * f,
                            Flux_Pts_Computed * flx,
                            Pack_Of_Pts::type_pack type_wished,
                            FONC_UN_IR fir
                    )
{
    if (f->type_out() == type_wished)
       return f;

    if (type_wished == Pack_Of_Pts::integer)
       return new OP_UC_IR(arg,f,flx,fir);

   return new OP_UC_RI(arg,f,flx,convert);
          
}

Fonc_Num_Computed * convert_fonc_num
                    (       const Arg_Fonc_Num_Comp & arg,
                            Fonc_Num_Computed * f,
                            Flux_Pts_Computed * flx,
                            Pack_Of_Pts::type_pack type_wished
                    )
{
    return convert_fonc_num(arg,f,flx,type_wished,convert);
}


      //  Not Computed 

class Oper_Conv_Not_Comp : public Op_Un_Not_Comp
{
      public :

          Oper_Conv_Not_Comp
          (
                 Fonc_Num f, 
                 Pack_Of_Pts::type_pack type_wished,
                 FONC_UN_IR  fir,
                 const char * name,
                 TyVal        Value,
                 TyDeriv      Deriv,
                 TyValDeriv   aValDeriv
           ) :
               Op_Un_Not_Comp(f,name,Value,Deriv,aValDeriv),
               _type_wished (type_wished),
               _fir         (fir)
          {
               ASSERT_INTERNAL
               (
                  type_wished != Pack_Of_Pts::rle,
                  "incoherent convertion type  in Oper_Conv_Not_Comp"
               );
          }

         
          virtual Fonc_Num_Computed * op_un_comp
                                      ( const Arg_Fonc_Num_Comp & arg,
                                        Fonc_Num_Computed *f
                                      )
          {
              return convert_fonc_num(arg,f,arg.flux(),_type_wished,_fir);
          }


          static Fonc_Num IdDeriv(Fonc_Num f,INT)
          {
               return f; 
          }
          static REAL IdValDeriv(Fonc_Num f,const PtsKD & aPts,INT k)
          {
               return f.ValDeriv(aPts,k); 
          }


      private :

            virtual bool integral_fonc(bool) const
            {
               return  Pack_Of_Pts::integer == _type_wished;
            }

            Pack_Of_Pts::type_pack _type_wished;
            FONC_UN_IR _fir;
};


      //======================
      //  Interface 
      //======================

static REAL VIconv(REAL v) {return (INT) v; }
Fonc_Num Iconv(Fonc_Num f)
{
    return new Oper_Conv_Not_Comp
               (  f,Pack_Of_Pts::integer,convert,"Iconv",VIconv,
                  NoDeriv,
                  NoValDeriv
               );
}


static REAL Vround_up(REAL v) {return round_up(v); }
Fonc_Num round_up(Fonc_Num f)
{
    return new Oper_Conv_Not_Comp
               (   f,Pack_Of_Pts::integer,round_up,"round_up",Vround_up,
                   NoDeriv,
                   NoValDeriv
                );
}




static REAL Vround_down(REAL v) {return round_down(v); }
Fonc_Num round_down(Fonc_Num f)
{
    return new Oper_Conv_Not_Comp
               (   f,Pack_Of_Pts::integer,round_down,"round_down",Vround_down,
                   NoDeriv,
                   NoValDeriv
                );
}



static REAL Vround_ni(REAL v) {return round_ni(v); }
Fonc_Num round_ni(Fonc_Num f)
{
    return new Oper_Conv_Not_Comp
               (   f,Pack_Of_Pts::integer,round_ni,"round_ni",Vround_ni,
                   NoDeriv,
                   NoValDeriv
                );
}



static REAL Vround_ni_inf(REAL v) {return round_ni_inf(v); }
Fonc_Num round_ni_inf(Fonc_Num f)
{
    return new Oper_Conv_Not_Comp
               (   f,Pack_Of_Pts::integer,round_ni_inf,
                   "round_ni_inf",Vround_ni_inf,
                   NoDeriv,
                   NoValDeriv
                );
}


static REAL VRConv(REAL v) {return v;}
Fonc_Num Rconv(Fonc_Num f)
{
    return new Oper_Conv_Not_Comp
               (   f,Pack_Of_Pts::real,0,"Rconv",VRConv,
                   Oper_Conv_Not_Comp::IdDeriv,
                   Oper_Conv_Not_Comp::IdValDeriv
               );
}


/***************************************************************/
/*                                                             */
/*             Mixte unary operator                            */
/*                                                             */
/*  eq operator with a result of the same type than operands   */
/*                                                             */
/*           Abs, -, square, cube                              */
/*                                                             */
/***************************************************************/

static INT StdDegreOpun(INT ) { return -1; }

class Op_Un_Mixte_Not_Comp : public Op_Un_Not_Comp
{
      public :

	  typedef INT (* tDegreOpun)(INT);

          Fonc_Num Simplify();

          Op_Un_Mixte_Not_Comp 
          (
                  Fonc_Num f, 
                  FONC_UN_INT fii,
                  FONC_UN_REAL frr,
                  const char * name,
                  TyVal        Value,
                  TyDeriv      Deriv,
                  TyValDeriv   ValDeriv,
		  tDegreOpun   aDegreOpun
          ) :

                   Op_Un_Not_Comp(f,name,Value,Deriv,ValDeriv),
                   _fii          (fii),
                   _frr          (frr),
		   mDegreOpun    (aDegreOpun)
          {
          }

         
          virtual Fonc_Num_Computed * op_un_comp
                                      ( const Arg_Fonc_Num_Comp & arg,
                                        Fonc_Num_Computed *f
                                      )
          {
              if (f->type_out() == Pack_Of_Pts::integer)
                  return new OP_UC_II(arg,f,arg.flux(),_fii);
              else
                  return new OP_UC_RR(arg,f,arg.flux(),_frr);
          }

      private :
              FONC_UN_INT  _fii;
              FONC_UN_REAL _frr;
	      tDegreOpun   mDegreOpun;

              virtual bool integral_fonc(bool iflux) const
              {
                 return _f.integral_fonc(iflux);
              }

	      virtual INT DegrePoly() const
	      {
		     INT aD = _f.DegrePoly();
		     if (aD==-1)
			 return -1;
		     return mDegreOpun(aD);
	      }

};


Fonc_Num Op_Un_Mixte_Not_Comp::Simplify()
{
   std::string aStrName = _name;
   if (aStrName=="-") return -(_f.Simplify());
   if (aStrName=="ElSquare") return Square(_f.Simplify());

   std::cout << "For operator = " << _name << "\n";
   ELISE_ASSERT(false,"Unhandled operator in Op_Un_Not_Comp::Simplify");
   return 0;
}
      //======================
      //  Interface 
      //======================

static Fonc_Num DMoins(Fonc_Num f,INT k)
{
    return - f.deriv(k);
}
static REAL VMoins(REAL v) {return -v;}
static REAL VDMoins(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   return - f.ValDeriv(aPts,aK);
}

static INT MinusDegreOpun(INT aD ) { return aD; }
Fonc_Num operator - (Fonc_Num f)
{
     if (f.is0())
        return 0;
     return new Op_Un_Mixte_Not_Comp
               (f,tab_minus1,tab_minus1,"-",VMoins,DMoins,VDMoins,MinusDegreOpun);
}


static REAL VAbs(REAL v) {return ElAbs(v);}
Fonc_Num Abs (Fonc_Num f)
{
     return new Op_Un_Mixte_Not_Comp
                (f,tab_Abs,tab_Abs,"Abs",VAbs,NoDeriv,NoValDeriv,StdDegreOpun);
}



static Fonc_Num DSquare(Fonc_Num f,INT k)
{
    return 2* f.deriv(k) * f;
}
static REAL VSquare(REAL v) {return ElSquare(v);}
static REAL VDSquare(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   return 2 * f.ValDeriv(aPts,aK) * f.ValFonc(aPts);
}
static INT SquareDegreOpun(INT aD ) { return 2* aD; }
Fonc_Num Square (Fonc_Num f)
{
     return new Op_Un_Mixte_Not_Comp
               (f,tab_square,tab_square,"ElSquare",VSquare,DSquare,VDSquare,SquareDegreOpun);
}


static Fonc_Num DCube(Fonc_Num f,INT k)
{
    return 3* f.deriv(k) * Square(f);
}
REAL VCube(REAL v) {return v*ElSquare(v);}
static REAL VDCube(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   return 3 * f.ValDeriv(aPts,aK) * ElSquare(f.ValFonc(aPts));
}
static INT CubeDegreOpun(INT aD ) { return 3* aD; }
Fonc_Num Cube (Fonc_Num f)
{
     return new Op_Un_Mixte_Not_Comp
               (f,tab_cube,tab_cube,"VCube",VCube,DCube,VDCube,CubeDegreOpun);
}


static Fonc_Num DPow4(Fonc_Num f,INT k)
{
    return 4* f.deriv(k) * Cube(f);
}
REAL VPow4(REAL v) {return ElSquare(ElSquare(v));}
static REAL VDPow4(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   return 4 * f.ValDeriv(aPts,aK) * VCube(f.ValFonc(aPts));
}
static INT Pow4DegreOpun(INT aD ) { return 4* aD; }
Fonc_Num Pow4 (Fonc_Num f)
{
     return new Op_Un_Mixte_Not_Comp
               (f,tab_pow4,tab_pow4,"VPow4",VPow4,DPow4,VDPow4,Pow4DegreOpun);
}



static Fonc_Num DPow5(Fonc_Num f,INT k)
{
    return 5* f.deriv(k) * Pow4(f);
}
REAL VPow5(REAL v) {return v * ElSquare(ElSquare(v));}
static REAL VDPow5(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   return 5 * f.ValDeriv(aPts,aK) * VPow4(f.ValFonc(aPts));
}
static INT Pow5DegreOpun(INT aD ) { return 5* aD; }
Fonc_Num Pow5 (Fonc_Num f)
{
     return new Op_Un_Mixte_Not_Comp
               (f,tab_pow5,tab_pow5,"VPow5",VPow5,DPow5,VDPow5,Pow5DegreOpun);
}





static Fonc_Num DPow6(Fonc_Num f,INT k)
{
    return 6* f.deriv(k) * Pow5(f);
}
REAL VPow6(REAL v) {return  ElSquare(VCube(v));}
static REAL VDPow6(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   return 6 * f.ValDeriv(aPts,aK) * VPow5(f.ValFonc(aPts));
}
static INT Pow6DegreOpun(INT aD ) { return 6* aD; }
Fonc_Num Pow6 (Fonc_Num f)
{
     return new Op_Un_Mixte_Not_Comp
               (f,tab_pow6,tab_pow6,"VPow6",VPow6,DPow6,VDPow6,Pow6DegreOpun);
}


static Fonc_Num DPow7(Fonc_Num f,INT k)
{
    return 7* f.deriv(k) * Pow6(f);
}
REAL VPow7(REAL v) {return  v *ElSquare(VCube(v));}
static REAL VDPow7(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   return 7 * f.ValDeriv(aPts,aK) * VPow6(f.ValFonc(aPts));
}
static INT Pow7DegreOpun(INT aD ) { return 7* aD; }
Fonc_Num Pow7 (Fonc_Num f)
{
     return new Op_Un_Mixte_Not_Comp
               (f,tab_pow7,tab_pow7,"VPow7",VPow7,DPow7,VDPow7,Pow7DegreOpun);
}



Fonc_Num PowI(Fonc_Num f,INT aDegre)
{
   switch (aDegre)
   {
       case -1 : return 1/f;
       case 0  : return 1;
       case 1  : return f;
       case 2  : return Square(f);
       case 3  : return Cube(f);
       case 4  : return Pow4(f);
       case 5  : return Pow5(f);
       case 6  : return Pow6(f);
       case 7  : return Pow7(f);
   }
   return pow(f,aDegre);

   ELISE_ASSERT(false,"Bad Degre in PowI");
   return 0;
}






/***************************************************************/
/*                                                             */
/*         Mathematical operators                              */
/*                                                             */
/***************************************************************/


class Op_Un_Math : public Op_Un_Not_Comp
{
      public :
         Fonc_Num Simplify();
         static Fonc_Num  New
                (
                     Fonc_Num f,
                     FONC_UN_REAL frr,
                     const char * name,
                     TyVal        Value,
                     TyDeriv      Deriv,
                     TyValDeriv   aValDeriv    
                );
      private :

          Op_Un_Math 
          (
                 Fonc_Num f, 
                 FONC_UN_REAL frr,
                 const char * name,
                 TyVal        Value,
                 TyDeriv      Deriv,
                 TyValDeriv   aValDeriv
          )  :
                 Op_Un_Not_Comp(f,name,Value,Deriv,aValDeriv),
                 _frr          (frr)
          {
          }

         
          virtual Fonc_Num_Computed * op_un_comp
                                      ( const Arg_Fonc_Num_Comp & arg,
                                        Fonc_Num_Computed *f
                                      )
          {
               f = convert_fonc_num(arg,f,arg.flux(),Pack_Of_Pts::real);
               return new OP_UC_RR(arg,f,arg.flux(),_frr);
          }

               FONC_UN_REAL _frr;
               virtual bool integral_fonc(bool) const { return false;}

};

Fonc_Num Op_Un_Math::Simplify()
{
   std::string aStrName = _name;
   // if (aStrName=="-") return -(_f.Simplify());
   // if (aStrName=="ElSquare") return Square(_f.Simplify());
   if (aStrName=="cos") return cos(_f.Simplify());
   if (aStrName=="sin") return sin(_f.Simplify());
   if (aStrName=="tan") return tan(_f.Simplify());
   if (aStrName=="exp")  return exp(_f.Simplify());
   if (aStrName=="log")  return log(_f.Simplify());
   if (aStrName=="log2") return log2(_f.Simplify());
   if (aStrName=="sqrt") return sqrt(_f.Simplify());
   if (aStrName=="atan") return atan(_f.Simplify());


   std::cout << "For operator = " << _name << "\n";
   ELISE_ASSERT(false,"Unhandled operator in Op_Un_Not_Comp::Simplify");
   return 0;
}
Fonc_Num Op_Un_Math::New
         (
            Fonc_Num f,
            FONC_UN_REAL frr,
            const char * name,
            TyVal        Value,
            TyDeriv      Deriv,
            TyValDeriv   aValDeriv
         )    
{
   REAL aCste;
   if (f.IsCsteRealDim1(aCste))
   {
      REAL aV = Value(aCste);
      if (f.integral_fonc(true))
          return Fonc_Num(round_ni(aV));
      else
          return Fonc_Num(aV);
   }
   return new Op_Un_Math(f,frr,name,Value,Deriv,aValDeriv);
}


      //======================
      //  Interface 
      //======================


           // cosinus

static Fonc_Num Dcos(Fonc_Num f,INT k)
{
    return - f.deriv(k) * sin(f);
}
static REAL  CppCos(REAL x){return cos(x);}
static REAL VDCos(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   return  - f.ValDeriv(aPts,aK) * sin(f.ValFonc(aPts));
}
Fonc_Num cos  (Fonc_Num f)
{
     return Op_Un_Math::New(f,tab_cos,"cos",CppCos,Dcos,VDCos);
}


           // sinus

static Fonc_Num Dsin(Fonc_Num f,INT k)
{
    return f.deriv(k) * cos(f);
}
static REAL  CppSin(REAL x){return sin(x);}
static REAL VDSin(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   return   f.ValDeriv(aPts,aK) * cos(f.ValFonc(aPts));
}
Fonc_Num sin  (Fonc_Num f)
{
     return Op_Un_Math::New(f,tab_sin,"sin",CppSin,Dsin,VDSin);
}


           // tangente
static Fonc_Num Dtan(Fonc_Num f,INT k)
{
    return f.deriv(k) * (1+ Square(tan(f)));
}
static REAL  CppTan(REAL x){return tan(x);}
static REAL VDTan(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   return   f.ValDeriv(aPts,aK) * (1+ElSquare(tan(f.ValFonc(aPts))));
}     
Fonc_Num tan  (Fonc_Num f)
{
     return Op_Un_Math::New(f,tab_tan,"tan",CppTan,Dtan,VDTan);
}



         // ==========  f4S2AtRxS2 ======
void tab_f4S2AtRxS2(REAL * out, const REAL * in,INT nb)
{
   for (INT i=0 ; i<nb ; i++)
         out[i] = f4S2AtRxS2(in[i]);
}
static Fonc_Num Df4S2AtRxS2(Fonc_Num f,INT k)
{
    return f.deriv(k) * Der4S2AtRxS2(f);
}
static double VDf4S2AtRxS2(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   return   f.ValDeriv(aPts,aK) *  Der4S2AtRxS2(f.ValFonc(aPts));
}
Fonc_Num f4S2AtRxS2  (Fonc_Num f)
{
     return Op_Un_Math::New
            (
                f,
                tab_f4S2AtRxS2,
                "f4S2AtRxS2",
                f4S2AtRxS2,
                Df4S2AtRxS2,
                VDf4S2AtRxS2
            );
}
          // ==========  Der-AtRxSRx ===============

void tab_Der4S2AtRxS2(REAL * out, const REAL * in,INT nb)
{
   for (INT i=0 ; i<nb ; i++)
         out[i] = Der4S2AtRxS2(in[i]);
}
Fonc_Num Der4S2AtRxS2  (Fonc_Num f)
{
     return Op_Un_Math::New
            (
                f,
                tab_Der4S2AtRxS2,
                "Der4S2AtRxS2",
                Der4S2AtRxS2,
                NoDeriv,
                NoValDeriv
            );
}

         // ==========  f2SAtRxS2SRx ======

void tab_f2SAtRxS2SRx(REAL * out, const REAL * in,INT nb)
{
   for (INT i=0 ; i<nb ; i++)
         out[i] = f2SAtRxS2SRx(in[i]);
}
static Fonc_Num Df2SAtRxS2SRx(Fonc_Num f,INT k)
{
    return f.deriv(k) * Der2SAtRxS2SRx(f);
}
static double VDf2SAtRxS2SRx(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   return   f.ValDeriv(aPts,aK) *  Der2SAtRxS2SRx(f.ValFonc(aPts));
}
Fonc_Num f2SAtRxS2SRx  (Fonc_Num f)
{
     return Op_Un_Math::New
            (
                f,
                tab_f2SAtRxS2SRx,
                "f2SAtRxS2SRx",
                f2SAtRxS2SRx,
                Df2SAtRxS2SRx,
                VDf2SAtRxS2SRx
            );
}


          // ==========  Der-AtRxSRx ===============

void tab_Der2SAtRxS2SRx(REAL * out, const REAL * in,INT nb)
{
   for (INT i=0 ; i<nb ; i++)
         out[i] = Der2SAtRxS2SRx(in[i]);
}
Fonc_Num Der2SAtRxS2SRx  (Fonc_Num f)
{
     return Op_Un_Math::New
            (
                f,
                tab_Der2SAtRxS2SRx,
                "Der2SAtRxS2SRx",
                Der2SAtRxS2SRx,
                NoDeriv,
                NoValDeriv
            );
}

/**********************************************************/
/*                                                        */
/*           ModeleStereographique                        */
/*                                                        */
/**********************************************************/

double PrecStereographique(double x);
double Der_PrecStereographique(double x);

double SqM2CRx_StereoG(double x);
double Der_SqM2CRx_StereoG(double x);

// double Inv_PrecStereographique(double x);

Fonc_Num Der_PrecStereographique(Fonc_Num f);
Fonc_Num Der_SqM2CRx_StereoG(Fonc_Num f);

         // ==========  PrecStereographique ======

void tab_PrecStereographique(REAL * out, const REAL * in,INT nb)
{
   ELISE_ASSERT(false,"tab_PrecStereographique");
/*
   for (INT i=0 ; i<nb ; i++)
         out[i] = PrecStereographique(in[i]);
*/
}
static Fonc_Num DPrecStereographique(Fonc_Num f,INT k)
{
    return f.deriv(k) * Der_PrecStereographique(f);
}
/*
static double VPrecStereographique(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   return   f.ValDeriv(aPts,aK) *  Der_PrecStereographique(f.ValFonc(aPts));
}
*/

Fonc_Num PrecStereographique  (Fonc_Num f)
{
     return Op_Un_Math::New
            (
                f,
                tab_PrecStereographique,
                "PrecStereographique",
                PrecStereographique,
                DPrecStereographique,
                NoValDeriv
            );
}


//=====================  Der_PrecStereographique =======

void tab_Der_PrecStereographique(REAL * out, const REAL * in,INT nb)
{
   ELISE_ASSERT(false,"tab_Der_PrecStereographique");
/*
   for (INT i=0 ; i<nb ; i++)
         out[i] = Der_PrecStereographique(in[i]);
*/
}
Fonc_Num Der_PrecStereographique(Fonc_Num f)
{
     return Op_Un_Math::New
            (
                f,
                tab_Der_PrecStereographique,
                "Der_PrecStereographique",
                Der_PrecStereographique,
                NoDeriv,
                NoValDeriv
            );
}
         // ==========  SqM2CRx_StereoG ======

void tab_SqM2CRx_StereoG(REAL * out, const REAL * in,INT nb)
{
   ELISE_ASSERT(false,"tab_SqM2CRx_StereoG");
/*
   for (INT i=0 ; i<nb ; i++)
         out[i] = SqM2CRx_StereoG(in[i]);
*/
}
static Fonc_Num DSqM2CRx_StereoG(Fonc_Num f,INT k)
{
    return f.deriv(k) * Der_SqM2CRx_StereoG(f);
}
/*
static double VSqM2CRx_StereoG(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   return   f.ValDeriv(aPts,aK) *  Der_SqM2CRx_StereoG(f.ValFonc(aPts));
}
*/

Fonc_Num SqM2CRx_StereoG  (Fonc_Num f)
{
     return Op_Un_Math::New
            (
                f,
                tab_SqM2CRx_StereoG,
                "SqM2CRx_StereoG",
                SqM2CRx_StereoG,
                DSqM2CRx_StereoG,
                NoValDeriv
            );
}

//=====================  Der_SqM2CRx_StereoG =======

void tab_Der_SqM2CRx_StereoG(REAL * out, const REAL * in,INT nb)
{
   ELISE_ASSERT(false,"tab_Der_SqM2CRx_StereoG");
/*
   for (INT i=0 ; i<nb ; i++)
         out[i] = Der_SqM2CRx_StereoG(in[i]);
*/
}
Fonc_Num Der_SqM2CRx_StereoG(Fonc_Num f)
{
     return Op_Un_Math::New
            (
                f,
                tab_Der_SqM2CRx_StereoG,
                "Der_SqM2CRx_StereoG",
                Der_SqM2CRx_StereoG,
                NoDeriv,
                NoValDeriv
            );
}


// double Der_SqM2CRx_StereoG(double x);



          // ==========  SinCardRx ===============
void  tab_SinCardRx(REAL * out, const REAL * in,INT nb)
{
   for (INT i=0 ; i<nb ; i++)
       out[i] = SinCardRx(in[i]);
}
Fonc_Num SinCardRx  (Fonc_Num f)
{
     return Op_Un_Math::New(f,tab_SinCardRx,"SinCardRx",SinCardRx,NoDeriv,NoValDeriv);
}
void tab_CosRx(REAL * out, const REAL * in,INT nb)
{
   for (INT i=0 ; i<nb ; i++)
         out[i] = CosRx(in[i]);
}
static Fonc_Num DerivCosRx(Fonc_Num f,INT k)
{
    return - 0.5 * f.deriv(k) * SinCardRx(f);
}
static REAL VDerivCosRx(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   return   -0.5 * f.ValDeriv(aPts,aK) *  SinCardRx(f.ValFonc(aPts));
}     
Fonc_Num CosRx  (Fonc_Num f)
{
     return Op_Un_Math::New(f,tab_CosRx,"CosRx",CosRx,DerivCosRx,VDerivCosRx);
}


          // ==========  BadNum  ===============
void  tab_IsBadNum(REAL * out, const REAL * in,INT nb)
{
   for (INT i=0 ; i<nb ; i++)
       out[i] = IsBadNum(in[i]);
}
Fonc_Num IsBadNum  (Fonc_Num f)
{
     return Op_Un_Math::New(f,tab_IsBadNum,"IsBadNum",IsBadNum,NoDeriv,NoValDeriv);
}


          // ==========  AtRxSRx ===============

void tab_AtRxSRx(REAL * out, const REAL * in,INT nb)
{
   for (INT i=0 ; i<nb ; i++)
         out[i] = AtRxSRx(in[i]);
}
static Fonc_Num DAtRxSRx(Fonc_Num f,INT k)
{
    return f.deriv(k) * DerAtRxSRx(f);
}
static REAL VDAtRxSRx(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   return   f.ValDeriv(aPts,aK) *  DerAtRxSRx(f.ValFonc(aPts));
}     
Fonc_Num AtRxSRx  (Fonc_Num f)
{
     return Op_Un_Math::New(f,tab_AtRxSRx,"AtRxSRx",AtRxSRx,DAtRxSRx,VDAtRxSRx);
}

          // ==========  Der-AtRxSRx ===============

void tab_DerAtRxSRx(REAL * out, const REAL * in,INT nb)
{
   for (INT i=0 ; i<nb ; i++)
         out[i] = DerAtRxSRx(in[i]);
}
Fonc_Num DerAtRxSRx  (Fonc_Num f)
{
     return Op_Un_Math::New(f,tab_DerAtRxSRx,"DerAtRxSRx",DerAtRxSRx,NoDeriv,NoValDeriv);
}


          // ==========  At2Rx ===============

void tab_At2Rx(REAL * out, const REAL * in,INT nb)
{
   for (INT i=0 ; i<nb ; i++)
         out[i] = At2Rx(in[i]);
}
static Fonc_Num DAt2Rx(Fonc_Num f,INT k)
{
    return f.deriv(k) * DerAt2Rx(f);
}
static REAL VDAt2Rx(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   return   f.ValDeriv(aPts,aK) *  DerAt2Rx(f.ValFonc(aPts));
}     
Fonc_Num At2Rx  (Fonc_Num f)
{
     return Op_Un_Math::New(f,tab_At2Rx,"At2Rx",At2Rx,DAt2Rx,VDAt2Rx);
}




          // ==========  At2Rx ===============

void tab_DerAt2Rx(REAL * out, const REAL * in,INT nb)
{
   for (INT i=0 ; i<nb ; i++)
         out[i] = DerAt2Rx(in[i]);
}
Fonc_Num DerAt2Rx  (Fonc_Num f)
{
     return Op_Un_Math::New(f,tab_DerAt2Rx,"DerAt2Rx",DerAt2Rx,NoDeriv,NoValDeriv);
}

// static REAL  CppAtRxSRx(REAL x){return AtRxSRx(x);}
// return Op_Un_Math::New(f,tab_erfcc,"erfcc",erfcc,NoDeriv,NoValDeriv);
           // sqrt


static Fonc_Num Dsqrt(Fonc_Num f,INT k)
{
    return 0.5 * f.deriv(k) / sqrt(f);
}
static REAL  CppSqrt(REAL x){return sqrt(x);}
static REAL VDSqrt(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   REAL vF = f.ValFonc(aPts);
   ELISE_ASSERT(vF>0,"Sqrt <0 in Val Deriv");
   return   0.5 * f.ValDeriv(aPts,aK) / sqrt(vF);
}     
Fonc_Num sqrt  (Fonc_Num f)
{
     return Op_Un_Math::New(f,tab_sqrt,"sqrt",CppSqrt,Dsqrt,VDSqrt);
}


           // arc-tangente
static Fonc_Num Datan(Fonc_Num f,INT k)
{
    return f.deriv(k) / (1+Square(f));
}
static REAL  CppAtan(REAL x){return atan(x);}
static REAL VDAtan(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   return   f.ValDeriv(aPts,aK) / (1+ElSquare(f.ValFonc(aPts)));
}     
Fonc_Num atan  (Fonc_Num f)
{
     return Op_Un_Math::New(f,tab_atan,"atan",CppAtan,Datan,VDAtan);
}

Fonc_Num erfcc  (Fonc_Num f)
{
     return Op_Un_Math::New(f,tab_erfcc,"erfcc",erfcc,NoDeriv,NoValDeriv);
}





Fonc_Num FoncNormalisee_S1S2 (Flux_Pts aFlux,Fonc_Num aFPds,Fonc_Num aF)
{
   double aS0,aS1,aS2;
   Symb_FNum  aSP (aFPds);
   Symb_FNum  aSF (aF);
   ELISE_COPY
   (
        aFlux,
	Virgule(aSP,aSP*aSF,aSP*Square(aSF)),
	Virgule
	(
	     sigma(aS0),
	     sigma(aS1),
	     sigma(aS2)
	)
   );
   aS1 /= aS0;
   aS2 /= aS0;
   aS2 -= ElSquare(aS1);
   return 255 * erfcc((aF-aS1)/sqrt(aS2));
}

Fonc_Num FoncNormalisee_S1S2 (Flux_Pts aFlux,Fonc_Num aF)
{
   return FoncNormalisee_S1S2(aFlux,1.0,aF);
}



static Fonc_Num Dlog(Fonc_Num f,INT k)
{
    return f.deriv(k)/f;
}
static REAL  CppLog(REAL x){return log(x);}
static REAL VDLog(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   REAL vF = f.ValFonc(aPts);
   ELISE_ASSERT(vF>0,"Log <0 in Val Deriv");
   return   f.ValDeriv(aPts,aK) / vF;
}     

Fonc_Num log  (Fonc_Num f)
{
     return Op_Un_Math::New(f,tab_log,"log",CppLog,Dlog,VDLog);
}

static Fonc_Num Dlog2(Fonc_Num f,INT k)
{
    return f.deriv(k)/(f*log(2.0));
}
static REAL  CppLog2(REAL x){return El_logDeux(x);}
static REAL VDLog2(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   REAL vF = f.ValFonc(aPts);
   ELISE_ASSERT(vF>0,"Log <0 in Val Deriv");
   return   f.ValDeriv(aPts,aK) / (vF*log(2.0));
}     
Fonc_Num log2  (Fonc_Num f)
{
     return Op_Un_Math::New(f,tab_log2,"log2",CppLog2,Dlog2,VDLog2);
}



static Fonc_Num Dexp(Fonc_Num f,INT k)
{
    return f.deriv(k) * exp(f);
}
static REAL  CppExp(REAL x){return exp(x);}
static REAL VDExp(Fonc_Num f,const PtsKD & aPts,INT aK)
{
   return   f.ValDeriv(aPts,aK) *exp(f.ValFonc(aPts));
}     
Fonc_Num exp  (Fonc_Num f)
{
     return Op_Un_Math::New(f,tab_exp,"exp",CppExp,Dexp,VDExp);
}



static REAL  VSF(REAL x){return signed_frac (x);}
static void tab_sfrac(REAL * out, const REAL * in,INT nb)
{
     for (INT i=0 ; i<nb ; i++)
         out[i] = signed_frac(in[i]);
}

Fonc_Num signed_frac  (Fonc_Num f)
{
     return Op_Un_Math::New(f,tab_sfrac,"signed_frac",VSF,NoDeriv,NoValDeriv);
}


static REAL  VEF(REAL x){return ecart_frac (x);}
static void tab_efrac(REAL * out, const REAL * in,INT nb)
{
     for (INT i=0 ; i<nb ; i++)
         out[i] = ecart_frac(in[i]);
}

Fonc_Num ecart_frac  (Fonc_Num f)
{
     return Op_Un_Math::New(f,tab_efrac,"ecart_frac",VEF,NoDeriv,NoValDeriv);
}







/***************************************************************/
/*                                                             */
/*         Integer      operators                              */
/*                                                             */
/***************************************************************/

class Op_Un_Integer : public Op_Un_Not_Comp
{
      public :

          Op_Un_Integer 
              (
                  Fonc_Num f, 
                  FONC_UN_INT fii,
                  const char * name,
                  TyVal        Value
              ) :
                   Op_Un_Not_Comp(f,name,Value,NoDeriv,NoValDeriv),
                   _fii          (fii)
          {
          }

         
          virtual Fonc_Num_Computed * op_un_comp
                                      ( const Arg_Fonc_Num_Comp & arg,
                                        Fonc_Num_Computed *f
                                      )
          {
               f = convert_fonc_num(arg,f,arg.flux(),Pack_Of_Pts::integer);
               return new OP_UC_II(arg,f,arg.flux(),_fii);
          }

      private :
               FONC_UN_INT _fii;

               virtual bool integral_fonc(bool) const {return true;}
};


      //======================
      //  Interface 
      //======================

static REAL VNot(REAL v){return ! ((INT) v);}
Fonc_Num operator !  (Fonc_Num f)
{
     return new Op_Un_Integer(f,tab_not_log,"!",VNot);
}

static REAL VNotBB(REAL v){return ~ ((INT) v);}
Fonc_Num operator ~  (Fonc_Num f)
{
     return new Op_Un_Integer(f,tab_not_bit_by_bit,"~",VNotBB);
}


tOperFuncUnaire  OperFuncUnaireFromName(const std::string & aName)
{
   if (aName=="u-") return operator -;
   if (aName=="~") return operator ~;
   if (aName=="!") return operator !;
   if (aName=="signed_frac") return signed_frac; // Partie fractionnaire entre -0.5 et 0.5
   if (aName=="ecart_frac") return ecart_frac;

   if (aName=="cos")     return cos;
   if (aName=="sin")     return sin;
   if (aName=="tan")     return tan;
   if (aName=="log")     return log;
   if (aName=="log2")    return log2;
   if (aName=="exp")     return exp;
   if (aName=="square")  return Square;
   if (aName=="cube")    return Cube;
   if (aName=="abs")     return Abs;
   if (aName=="atan")    return atan;
   if (aName=="sqrt")    return sqrt;
   if (aName=="erfcc")   return erfcc;
   if (aName=="IsBadNum")   return IsBadNum;

   std::cout << "For name =" << aName << "\n";
   ELISE_ASSERT(false,"Name is not a valid unary operator");

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
