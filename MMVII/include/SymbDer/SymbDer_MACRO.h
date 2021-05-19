#ifndef _SymbDer_MACRO_H_
#define _SymbDer_MACRO_H_

/** \file SymbDer_MACRO.h
    \brief This file contain the MACRO for fast definition of new operators

    Also macro are BAAAD from programming point of view, definition of new operators
is extremely repetitive . For operator like +, *, its justified to have special code
because we need to have special rule for each (like F+0 =0, F*1 = F ...),on the other
hand, for most user operator, when we add a new operator we just need to say do
the same thing but change the name. It can hardly (if olny possible) be done without Macro....

#define MACRO_SD_DECLARE_STD_BINARY_FUNC_OP(OPER)
#define MACRO_SD_DEFINE_STD_cBinaryF(NSPACE,OPER,OPERd1,OPERd2) 
#define MACRO_SD_DEFINE_STD_BINARY_FUNC_OP(OPER)
#define MACRO_DEFINE_STD_BINARY_FUNC_OP_DERIVABLE(NSPACE,OPER,OPERd1,OPERd2)

*/



namespace  NS_SymbolicDerivative
{
/*  *************************************************************** */
/*                                                                  */
/*          MACRO PART                                              */
/*                                                                  */
/*  *************************************************************** */

/* Given a binary operator Op on scals , this Macro declare  the existence  of  Op on formula
   and formula and scal
*/

#define MACRO_SD_DECLARE_STD_BINARY_FUNC_OP(OPER)\
template <class TypeElem> cFormula <TypeElem>\
         OPER(const cFormula <TypeElem> & aF1 ,const cFormula <TypeElem> & aF2);\
template <class TypeElem> cFormula <TypeElem>  OPER(const TypeElem & aV1,const cFormula <TypeElem> & aF2);\
template <class TypeElem> cFormula <TypeElem>  OPER(const cFormula <TypeElem> & aF1,const TypeElem & aV2);


/* Given a binary operator Op on scals , this Macro define the class cOp that implement
 this binary operator on formula. The 4 argument of the macro are :
 
   * NSPACE = namespace as Op on scal may be defined in any user namespace
   * OPER   = name of the operator itsef , a definition on scalar type must exist
   * OPERd1 = name of the operator corresponding to derivate of OPER at first variable
   * OPERd2 = name of the operator corresponding to derivate of OPER at second variable

Note : that if we dont need/want to use diferrentiation of OPER, we can use "UndefinedOperBin"
as values of OPERd1 and OPERd2.

Note : this macro uses the very old and low level C-macros of concatenation an stringification, 
i.e when use it with OPER=MyOper :
   *   c##OPER  will be expanded as cMyOper
   *   #OPER    will be expanded as "MyOper"
  

*/

#define MACRO_SD_DEFINE_STD_cBinaryF(NSPACE,OPER,OPERd1,OPERd2) \
template <class TypeElem> class c##OPER  : public cBinaryF<TypeElem>\
{\
      public :\
            static TypeElem Operation(const TypeElem & aV1,const TypeElem & aV2) {return NSPACE::OPER(aV1,aV2);}\
            using cBinaryF<TypeElem>::mF1;\
            using cBinaryF<TypeElem>::mF2;\
            using cBinaryF<TypeElem>::mDataF1;\
            using cBinaryF<TypeElem>::mDataF2;\
            using cImplemF<TypeElem>::mDataBuf;\
            inline c##OPER(cFormula<TypeElem> aF1,cFormula<TypeElem> aF2,const std::string & aName) :\
                   cBinaryF<TypeElem> (aF1,aF2,aName) \
            { }\
      private  :\
            const std::string &  NameOperator() const override {static std::string s(#OPER); return s;}\
            virtual std::string GenCodeShortExpr() const override {\
                return #NSPACE + std::string("::") + #OPER+std::string("(") + mF1->GenCodeFormName() + "," + mF2->GenCodeFormName() + ")";\
            }\
            virtual std::string GenCodeDef() const override {\
                return  #NSPACE + std::string("::") + #OPER+std::string("(") + mF1->GenCodeRef() + ","  + mF2->GenCodeRef() + ")";\
            }\
            void ComputeBuf(int aK0,int aK1) override\
            {\
                for (int aK=aK0 ; aK<aK1 ; aK++)\
                     mDataBuf[aK] =  NSPACE::OPER(mDataF1[aK],mDataF2[aK]);\
            }\
            cFormula<TypeElem> Derivate(int aK) const override \
            {\
                 return\
                   mF1->Derivate(aK) * OPERd1(mF1,mF2)\
                 + mF2->Derivate(aK) * OPERd2(mF1,mF2) ;\
            }\
};

/* This macro gives the standard definition of function declared in 
    MACRO_SD_DECLARE_STD_BINARY_FUNC_OP, it's quite basic and miror declaration,
    note the use of ## in definition on 2 formula, to generate name of class
*/

#define MACRO_SD_DEFINE_STD_BINARY_FUNC_OP(OPER)\
template <class TypeElem> cFormula <TypeElem>\
         OPER(const cFormula <TypeElem> & aF1 ,const cFormula <TypeElem> & aF2)\
{\
    return cGenOperatorBinaire<c##OPER<TypeElem> >::Generate(aF1,aF2,#OPER);\
}\
template <class TypeElem> \
inline cFormula<TypeElem> OPER(const TypeElem & aV1,const cFormula<TypeElem> & aF2)\
{\
  return OPER(aF2->CoordF()->CsteOfVal(aV1),aF2);\
}\
template <class TypeElem> \
inline cFormula<TypeElem> OPER(const cFormula<TypeElem> & aF1,const TypeElem & aV2)\
{\
    return OPER(aF1, aF1->CoordF()->CsteOfVal(aV2));\
}\


/* In my standard use in photogrammetry, I have one function and its two derivative,
and I want to generate an operator derivable.  This macro allow to do it in
one line.

Meaning of macro-parameter are the same as MACRO_SD_DEFINE_STD_cBinaryF

It makes the assumption that user don want to derivate the derivate, and use
"UndefinedOperBin" for that.
*/

#define MACRO_SD_DEFINE_STD_BINARY_FUNC_OP_DERIVABLE(NSPACE,OPER,OPERd1,OPERd2)\
MACRO_SD_DECLARE_STD_BINARY_FUNC_OP(OPER)\
MACRO_SD_DECLARE_STD_BINARY_FUNC_OP(OPERd1)\
MACRO_SD_DECLARE_STD_BINARY_FUNC_OP(OPERd2)\
MACRO_SD_DEFINE_STD_cBinaryF(NSPACE,OPER,OPERd1,OPERd2)\
MACRO_SD_DEFINE_STD_cBinaryF(NSPACE,OPERd1,UndefinedOperBin,UndefinedOperBin)\
MACRO_SD_DEFINE_STD_cBinaryF(NSPACE,OPERd2,UndefinedOperBin,UndefinedOperBin)\
MACRO_SD_DEFINE_STD_BINARY_FUNC_OP(OPER)\
MACRO_SD_DEFINE_STD_BINARY_FUNC_OP(OPERd1)\
MACRO_SD_DEFINE_STD_BINARY_FUNC_OP(OPERd2)




} //   NS_Symbolic_Derivative


#endif //  _SymbDer_MACRO_H_
