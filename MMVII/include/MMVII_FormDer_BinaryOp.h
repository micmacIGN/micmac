#ifndef _MMVII_FormDer_BinaryOp_H_
#define _MMVII_FormDer_BinaryOp_H_

/** \file MMVII_FormDer_BinaryOp.h
    \brief File for definition of binary operators inside formula

*/



namespace  NS_MMVII_FormalDerivative
{

/* *************************************************** */
/* *************************************************** */
/* *                                                 * */
/* *        BINARY  FORMULA                          * */
/* *                                                 * */
/* *************************************************** */
/* *************************************************** */

            /*  ----------------------------------------------------------
               Class implementing binary operation on formula
                  MOTHER CLASS : cBinaryF
                  DERIVED :   cSumF / cMulF / cSubF / cDivF / cPowF
             ----------------------------------------------------------------*/

template <class TypeElem> class cBinaryF : public cImplemF<TypeElem>
{
      public :
            typedef cImplemF<TypeElem>   tImplemF;
            typedef typename tImplemF::tCoordF   tCoordF;
            typedef typename tImplemF::tFormula   tFormula;
            typedef typename tImplemF::tBuf       tBuf;

            /// An operator must describe its name
            virtual std::string  NameOperator() const = 0;
            /// We can compute print of binary formula using operator name
            std::string  InfixPPrint() const override 
            {
               return "("+ NameOperator() + " "+ mF1->InfixPPrint() + " " + mF2->InfixPPrint() + ")";
            }
      protected  :
            std::vector<tFormula> Ref() const override{return std::vector<tFormula>{mF1,mF2};}
            inline cBinaryF(tFormula aF1,tFormula aF2,const std::string & aName):
                 tImplemF (aF1->CoordF(),aName),
                 mF1      (aF1),
                 mDataF1  (aF1->DataBuf()),
                 mF2      (aF2),
                 mDataF2  (aF2->DataBuf())
            {
                // It doesn't work to mix formula from different context
                if (mF1->CoordF()!=mF2->CoordF())
                   UserSError("Mix formula from different context");
            }

            tFormula  mF1;   ///< First argument of operator
            const TypeElem  *mDataF1; ///< Fast access to data of buf F1
            tFormula  mF2;   ///< Second argument of operator
            const TypeElem  *mDataF2; ///< Fast access to data of buf F2
};


template <class TypeElem> class cSumF : public cBinaryF <TypeElem>
{
      public :
            /// Required by constant reduction in  cGenOperatorBinaire
            static TypeElem Operation(const TypeElem & aV1,const TypeElem & aV2) {return aV1+aV2;}
            using cBinaryF<TypeElem>::mF1;
            using cBinaryF<TypeElem>::mF2;
            using cBinaryF<TypeElem>::mDataF1;
            using cBinaryF<TypeElem>::mDataF2;
            using cImplemF<TypeElem>::mDataBuf;

            inline cSumF(cFormula<TypeElem> aF1,cFormula<TypeElem> aF2,const std::string & aName) :
                   cBinaryF<TypeElem> (aF1,aF2,aName) 
            { }
      private  :
            std::string  NameOperator() const override {return "+";}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                     mDataBuf[aK] =  mDataF1[aK] + mDataF2[aK];
            }
            cFormula<TypeElem> Derivate(int aK) const override {return  mF1->Derivate(aK) + mF2->Derivate(aK);}
};

template <class TypeElem> class cMulF : public cBinaryF<TypeElem>
{
      public :
            /// Required by constant reduction in  cGenOperatorBinaire
            static TypeElem Operation(const TypeElem & aV1,const TypeElem & aV2) {return aV1*aV2;}
            using cBinaryF<TypeElem>::mF1;
            using cBinaryF<TypeElem>::mF2;
            using cBinaryF<TypeElem>::mDataF1;
            using cBinaryF<TypeElem>::mDataF2;
            using cImplemF<TypeElem>::mDataBuf;

            inline cMulF(cFormula<TypeElem> aF1,cFormula<TypeElem> aF2,const std::string & aName) :
                   cBinaryF<TypeElem> (aF1,aF2,aName) 
            { }
      private  :
            std::string  NameOperator() const override {return "*";}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                     mDataBuf[aK] =  mDataF1[aK] * mDataF2[aK];
            }
            /// Formula (FG)' = F'G + G'F
            cFormula<TypeElem> Derivate(int aK) const override 
            {
                return  mF2*mF1->Derivate(aK) + mF1*mF2->Derivate(aK);
            }
};

template <class TypeElem> class cSubF : public cBinaryF<TypeElem>
{
      public :
            /// Required by constant reduction in  cGenOperatorBinaire
            static TypeElem Operation(const TypeElem & aV1,const TypeElem & aV2) {return aV1-aV2;}
            using cBinaryF<TypeElem>::mF1;
            using cBinaryF<TypeElem>::mF2;
            using cBinaryF<TypeElem>::mDataF1;
            using cBinaryF<TypeElem>::mDataF2;
            using cImplemF<TypeElem>::mDataBuf;

            inline cSubF(cFormula<TypeElem> aF1,cFormula<TypeElem> aF2,const std::string & aName) :
                   cBinaryF<TypeElem> (aF1,aF2,aName) 
            { }
      private  :
            std::string  NameOperator() const override {return "-";}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                     mDataBuf[aK] =  mDataF1[aK] - mDataF2[aK];
            }
            /// Formula (F-G)' = F' - G'
            cFormula<TypeElem> Derivate(int aK) const override {return mF1->Derivate(aK) - mF2->Derivate(aK);}
};

template <class TypeElem> class cDivF : public cBinaryF<TypeElem>
{
      public :
            /// Required by constant reduction in  cGenOperatorBinaire
            static TypeElem Operation(const TypeElem & aV1,const TypeElem & aV2) {return aV1/aV2;}
            using cBinaryF<TypeElem>::mF1;
            using cBinaryF<TypeElem>::mF2;
            using cBinaryF<TypeElem>::mDataF1;
            using cBinaryF<TypeElem>::mDataF2;
            using cImplemF<TypeElem>::mDataBuf;

            inline cDivF(cFormula<TypeElem> aF1,cFormula<TypeElem> aF2,const std::string & aName) :
                   cBinaryF<TypeElem> (aF1,aF2,aName) 
            { }
      private  :
            std::string  NameOperator() const override {return "/";}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                     mDataBuf[aK] =  mDataF1[aK] / mDataF2[aK];
            }
            /// Formula (F/G)' = (F'G - G'F) / G ^ 2
            cFormula<TypeElem> Derivate(int aK) const override 
            {
               return (mF1->Derivate(aK)*mF2 - mF2->Derivate(aK)*mF1)/square(mF2);
            }
};

template <class TypeElem> class cPowF : public cBinaryF<TypeElem>
{
      public :
            /// Required by constant reduction in  cGenOperatorBinaire
            static TypeElem Operation(const TypeElem & aV1,const TypeElem & aV2) {return std::pow(aV1,aV2);}
            using cBinaryF<TypeElem>::mF1;
            using cBinaryF<TypeElem>::mF2;
            using cBinaryF<TypeElem>::mDataF1;
            using cBinaryF<TypeElem>::mDataF2;
            using cImplemF<TypeElem>::mDataBuf;

            inline cPowF(cFormula<TypeElem> aF1,cFormula<TypeElem> aF2,const std::string & aName) :
                   cBinaryF<TypeElem> (aF1,aF2,aName) 
            { }
      private  :
            std::string  NameOperator() const override {return "^";}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                     mDataBuf[aK] =  std::pow(mDataF1[aK],mDataF2[aK]);
            }
            /// Formula F ^ G = exp(G log(F))
            /// (F^G) ' = (F^G) (G F'/F + G'log(F))
            cFormula<TypeElem> Derivate(int aK) const override 
            {
               return   pow(mF1,mF2) * ( (mF1->Derivate(aK)/mF1)*mF2 + mF2->Derivate(aK)*log(mF1)) ;
            }
};


      /* ---------------------------------------*/
      /*           Global Functio on unary op   */
      /* ---------------------------------------*/
/**  A Helper class to avoid code duplication on the process , see detailed comment in  cGenOperatorUnaire (analogous) */

template <class TypeCompiled>  class cGenOperatorBinaire
{
    public :
         typedef typename TypeCompiled::tElem     tElem;
         typedef typename TypeCompiled::tCoordF  tCoordF;
         typedef typename TypeCompiled::tImplemF  tImplemF;
         typedef typename tImplemF::tFormula      tFormula;

         static tFormula   Generate(tFormula aF1,tFormula aF2,const std::string & aNameOp)
         {
             // Extract context (take F1 ou F2, does not matter, they must be the same)
             tCoordF * aPCont = aF1->CoordF();  
             std::string aNameForm =  aF1.NameFormulaBin(aNameOp,aF2);

             if (aPCont->ExistFunc(aNameForm))
               return aPCont->FuncOfName(aNameForm);

             // Maybe the two operand are constant ? Then we can reduce
             {
                 const tElem * aC1 = aF1->ValCste();
                 const tElem * aC2 = aF2->ValCste();
                 if (aC1 && aC2) 
                 {
                    tElem  aC12= TypeCompiled::Operation(*aC1,*aC2);
                    std::cout << "ZZZZZZ " << *aC1 << " " << *aC2 << " " << aC12<< "\n";
                 }
             }

             tFormula aResult (new TypeCompiled(aF1,aF2,aNameForm));
             aPCont->AddFormula(aResult);
             return aResult;
         }
};

       
template <class TypeElem>
cFormula<TypeElem> operator +
                              (
                                    const cFormula <TypeElem>& aF1,
                                    const cFormula<TypeElem> & aF2
                              ) 
{
     // Use the fact that 0 is neutral element to simplify
     if (aF1->IsCste0()) return aF2;
     if (aF2->IsCste0()) return aF1;

     // Use commutativity of + to have a unique representation
     if (aF1->Name() > aF2->Name()) 
        return aF2+aF1;

     return cGenOperatorBinaire<cSumF<TypeElem> >::Generate(aF1,aF2,"+");
}

template <class TypeElem>
cFormula<TypeElem> operator -
                              (
                                    const cFormula <TypeElem>& aF1,
                                    const cFormula<TypeElem> & aF2
                              ) 
{
     // Use the fact that 0 is neutral element to simplify
     if (aF1->IsCste0()) return -aF2;
     if (aF2->IsCste0()) return aF1;

     return cGenOperatorBinaire<cSubF<TypeElem> >::Generate(aF1,aF2,"-");
}

template <class TypeElem>
cFormula<TypeElem> operator *
                              (
                                    const cFormula <TypeElem>& aF1,
                                    const cFormula<TypeElem> & aF2
                              ) 
{
     // Use the fact that 1 is neutral element to simplify
     if (aF1->IsCste1()) return aF2;
     if (aF2->IsCste1()) return aF1;

     // Use the fact that 0 is absorbant element to simplify
     if (aF1->IsCste0()) return aF1;
     if (aF2->IsCste0()) return aF2;


     // Use commutativity of + to have a unique representation
     if (aF1->Name() > aF2->Name()) 
        return aF2 * aF1;

     return cGenOperatorBinaire<cMulF<TypeElem> >::Generate(aF1,aF2,"*");
}

template <class TypeElem>
cFormula<TypeElem> operator /
                              (
                                    const cFormula <TypeElem>& aF1,
                                    const cFormula<TypeElem> & aF2
                              ) 
{
     if (aF1->IsCste0()) return aF1;  // 0/F2 -> 0
     if (aF2->IsCste1()) return aF1;  // F1/1 -> F1

     return cGenOperatorBinaire<cDivF<TypeElem> >::Generate(aF1,aF2,"/");
}

template <class TypeElem>
cFormula<TypeElem>   pow
                              (
                                    const cFormula <TypeElem>& aF1,
                                    const cFormula<TypeElem> & aF2
                              ) 
{
     return cGenOperatorBinaire<cPowF<TypeElem> >::Generate(aF1,aF2,"^");
}

      /* ----------------------------------------------------------*/
      /*  Binary   Operator  between Formula and  constants        */
      /* ----------------------------------------------------------*/

        // ++++++++++++++++++++++++
template <class TypeElem> 
inline cFormula<TypeElem> operator +(const TypeElem & aV1,const cFormula<TypeElem> & aF2)
{
  return aF2->CoordF()->CsteOfVal(aV1) + aF2;
}
template <class TypeElem> 
inline cFormula<TypeElem> operator +(const cFormula<TypeElem> & aF1,const TypeElem & aV2)
{
    return aV2+aF1;
}
        // ************************
template <class TypeElem> 
inline cFormula<TypeElem> operator *(const TypeElem & aV1,const cFormula<TypeElem> & aF2)
{
  return aF2->CoordF()->CsteOfVal(aV1) * aF2;
}
template <class TypeElem> 
inline cFormula<TypeElem> operator *(const cFormula<TypeElem> & aF1,const TypeElem & aV2)
{
    return aV2*aF1;
}
        // --------------------------
template <class TypeElem> 
inline cFormula<TypeElem> operator -(const TypeElem & aV1,const cFormula<TypeElem> & aF2)
{
  return aF2->CoordF()->CsteOfVal(aV1) - aF2;
}
template <class TypeElem> 
inline cFormula<TypeElem> operator -(const cFormula<TypeElem> & aF1,const TypeElem & aV2)
{
  return aF1-aF1->CoordF()->CsteOfVal(aV2) ;
}
        // /////////////////////////
template <class TypeElem> 
inline cFormula<TypeElem> operator /(const TypeElem & aV1,const cFormula<TypeElem> & aF2)
{
  return aF2->CoordF()->CsteOfVal(aV1) / aF2;
}
template <class TypeElem> 
inline cFormula<TypeElem> operator /(const cFormula<TypeElem> & aF1,const TypeElem & aV2)
{
  return aF1/aF1->CoordF()->CsteOfVal(aV2) ;
}



}; //   NS_MMVII_FormalDerivative


#endif //  _MMVII_FormDer_BinaryOp_H_
