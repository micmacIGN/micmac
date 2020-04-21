#ifndef _MMVII_FormDer_UnaryOp_H_
#define _MMVII_FormDer_UnaryOp_H_

/** \file MMVII_FormDer_UnaryOp.h
    \brief File for unary operator on formal derivative
*/


namespace  NS_MMVII_FormalDerivative
{

/* *************************************************** */
/* *************************************************** */
/* *                                                 * */
/* *        UNARY FORMULA                            * */
/* *                                                 * */
/* *************************************************** */
/* *************************************************** */


            /*  ----------------------------------------------------------
               Class implementing unary operation on formula
                  MOTHER CLASS : cUnaryF
                  DERIVED :   cSquareF / cExpF / cMin1F / cLogF
             ----------------------------------------------------------------*/

template <class TypeElem> class cUnaryF : public cImplemF<TypeElem>
{
      public :
            typedef cImplemF<TypeElem> tImplemF;
            typedef typename tImplemF::tCoordF tCoordF;
            typedef typename tImplemF::tFormula tFormula;
            typedef typename tImplemF::tBuf     tBuf;

            virtual std::string  PostName() const {return "";}
            std::string  InfixPPrint() const override 
            {
               return this->NameOperator() + " "+  mF->InfixPPrint() + PostName() ;
            }

            /// In the cas an additional parameter is used, as "powc F30 3.14"
            TypeElem Extrac1Param (const std::string & aString)
            {
                std::string aBuf1,aBuf2;
                TypeElem aVal;

                std::stringstream aStream(aString);

                aStream >> aBuf1 >> aBuf2 >> aVal;
                return aVal;
            }

      protected  :
            std::vector<tFormula> Ref() const override{return std::vector<tFormula>{mF};}
            inline cUnaryF(tFormula aF,const std::string & aName) :
                 tImplemF (aF->CoordF(),aName),
                 mF       (aF),
                 mDataF   (mF->DataBuf())
            { }
            tFormula        mF;    ///< The formula argument of unary operator
            const TypeElem* mDataF; ///< Fast access to data of buf
};


/**  Classes for square */
template <class TypeElem> class cSquareF : public cUnaryF<TypeElem>
{
     public :
            using cUnaryF<TypeElem>::mF;
            using cUnaryF<TypeElem>::mDataF;
            using cImplemF<TypeElem>::mDataBuf;
     
            cSquareF (cFormula<TypeElem> aF,const std::string & aName) :
                cUnaryF <TypeElem> (aF,aName)
            { }
            static TypeElem Operation(const TypeElem & aV1) {return aV1 * aV1;}
      private :
            const std::string &  NameOperator() const override {static std::string s("square"); return s;}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                    mDataBuf[aK] =  mDataF[aK] * mDataF[aK];
            }
            ///  rule : (FF)' =   2 F' F
            cFormula<TypeElem> Derivate(int aK) const override 
            {
                return  2.0  * mF->Derivate(aK)  * mF;
            }
};

template <class TypeElem> class cCubeF : public cUnaryF<TypeElem>
{
     public :
            using cUnaryF<TypeElem>::mF;
            using cUnaryF<TypeElem>::mDataF;
            using cImplemF<TypeElem>::mDataBuf;
     
            cCubeF (cFormula<TypeElem> aF,const std::string & aName) :
                cUnaryF <TypeElem> (aF,aName)
            { }
            static TypeElem Operation(const TypeElem & aV1) {return aV1 * aV1 * aV1;}
      private :
            const std::string &  NameOperator() const override {static std::string s("cube"); return s;}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                    mDataBuf[aK] =  mDataF[aK] * mDataF[aK] * mDataF[aK];
            }
            ///  rule : (F^3)' =   3 F' F^2
            cFormula<TypeElem> Derivate(int aK) const override 
            {
                return  3.0  * mF->Derivate(aK)  * square(mF);
            }
};

template <class TypeElem> class cPow4 : public cUnaryF<TypeElem>
{
     public :
            using cUnaryF<TypeElem>::mF;
            using cUnaryF<TypeElem>::mDataF;
            using cImplemF<TypeElem>::mDataBuf;
     
            cPow4 (cFormula<TypeElem> aF,const std::string & aName) :
                cUnaryF <TypeElem> (aF,aName)
            { }
            static TypeElem Operation(const TypeElem & aV1) {return pow4(aV1);}
      private :
            const std::string &  NameOperator() const override {static std::string s("pow4"); return s;}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                    mDataBuf[aK] =  pow4(mDataF[aK]);
            }
            ///  rule : (F^4)' =   4 F' F^2
            cFormula<TypeElem> Derivate(int aK) const override 
            {
                return  4.0  * mF->Derivate(aK)  * cube(mF);
            }
};
template <class TypeElem> class cPow5 : public cUnaryF<TypeElem>
{
     public :
            using cUnaryF<TypeElem>::mF;
            using cUnaryF<TypeElem>::mDataF;
            using cImplemF<TypeElem>::mDataBuf;
     
            cPow5 (cFormula<TypeElem> aF,const std::string & aName) :
                cUnaryF <TypeElem> (aF,aName)
            { }
            static TypeElem Operation(const TypeElem & aV1) {return pow5(aV1);}
      private :
            const std::string &  NameOperator() const override {static std::string s("pow5"); return s;}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                    mDataBuf[aK] =  pow5(mDataF[aK]);
            }
            ///  rule : (F^5)' =   5 F' F^4
            cFormula<TypeElem> Derivate(int aK) const override 
            {
                return  5.0  * mF->Derivate(aK)  * pow4(mF);
            }
};
template <class TypeElem> class cPow6 : public cUnaryF<TypeElem>
{
     public :
            using cUnaryF<TypeElem>::mF;
            using cUnaryF<TypeElem>::mDataF;
            using cImplemF<TypeElem>::mDataBuf;
     
            cPow6 (cFormula<TypeElem> aF,const std::string & aName) :
                cUnaryF <TypeElem> (aF,aName)
            { }
            static TypeElem Operation(const TypeElem & aV1) {return pow6(aV1);}
      private :
            const std::string &  NameOperator() const override {static std::string s("pow6"); return s;}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                    mDataBuf[aK] =  pow6(mDataF[aK]);
            }
            ///  rule : (F^6)' =   6 F' F^5
            cFormula<TypeElem> Derivate(int aK) const override 
            {
                return  6.0  * mF->Derivate(aK)  * pow5(mF);
            }
};
template <class TypeElem> class cPow7 : public cUnaryF<TypeElem>
{
     public :
            using cUnaryF<TypeElem>::mF;
            using cUnaryF<TypeElem>::mDataF;
            using cImplemF<TypeElem>::mDataBuf;
     
            cPow7 (cFormula<TypeElem> aF,const std::string & aName) :
                cUnaryF <TypeElem> (aF,aName)
            { }
            static TypeElem Operation(const TypeElem & aV1) {return pow7(aV1);}
      private :
            const std::string &  NameOperator() const override {static std::string s("pow7"); return s;}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                    mDataBuf[aK] =  pow7(mDataF[aK]);
            }
            ///  rule : (F^7)' =   7 F' F^6
            cFormula<TypeElem> Derivate(int aK) const override 
            {
                return  7.0  * mF->Derivate(aK)  * pow6(mF);
            }
};


template <class TypeElem> class cExpF : public cUnaryF<TypeElem>
{
     public :
            using cUnaryF<TypeElem>::mF;
            using cUnaryF<TypeElem>::mDataF;
            using cImplemF<TypeElem>::mDataBuf;

            cExpF (cFormula<TypeElem> aF,const std::string & aName) :
                cUnaryF <TypeElem> (aF,aName)
            { }
            static TypeElem Operation(const TypeElem & aV1) {return std::exp(aV1);}
      private :
            const std::string &  NameOperator() const override {static std::string s("exp"); return s;}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                    mDataBuf[aK] = std::exp(mDataF[aK]);
            }
            ///  rule : (exp FF)' =   F' exp F
            cFormula<TypeElem> Derivate(int aK) const override 
            {
                return   mF->Derivate(aK)  * exp(mF);
            }
};

template <class TypeElem> class cMin1F : public cUnaryF<TypeElem>
{
     public :
            using cUnaryF<TypeElem>::mF;
            using cUnaryF<TypeElem>::mDataF;
            using cImplemF<TypeElem>::mDataBuf;

            cMin1F (cFormula<TypeElem> aF,const std::string & aName) :
                cUnaryF <TypeElem> (aF,aName)
            { }
            static TypeElem Operation(const TypeElem & aV1) {return - aV1;}
      private :
            const std::string &  NameOperator() const override {static std::string s("-"); return s;}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                    mDataBuf[aK] = - mDataF[aK];
            }
            /// rule : (-F)'  =  - (F')
            cFormula<TypeElem> Derivate(int aK) const override 
            {
                return   - mF->Derivate(aK)  ;
            }
};

template <class TypeElem> class cLogF : public cUnaryF<TypeElem>
{
     public :
            using cUnaryF<TypeElem>::mF;
            using cUnaryF<TypeElem>::mDataF;
            using cImplemF<TypeElem>::mDataBuf;

            cLogF (cFormula<TypeElem> aF,const std::string & aName) :
                cUnaryF <TypeElem> (aF,aName)
            { }
            static TypeElem Operation(const TypeElem & aV1) {return std::log(aV1);}
      private :
            const std::string &  NameOperator() const override {static std::string s("log"); return s;}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                    mDataBuf[aK] = std::log(mDataF[aK]);
            }
            /// rule : (log F)'  =  F' / F
            cFormula<TypeElem> Derivate(int aK) const override 
            {
                return   mF->Derivate(aK) / mF ;
            }
};


template <class TypeElem> class cPowCste : public cUnaryF<TypeElem>
{
     public :
            using cUnaryF<TypeElem>::mF;
            using cUnaryF<TypeElem>::mDataF;
            using cImplemF<TypeElem>::mDataBuf;

            cPowCste (cFormula<TypeElem> aF,const std::string & aName) :
                cUnaryF <TypeElem> (aF,aName),
                mExp  (cUnaryF<TypeElem>::Extrac1Param (aName))
            { 
            }
            // Cannot be static because of mExp
            TypeElem Operation(const TypeElem & aV1) {return std::pow(aV1,mExp);}
      private :
            const std::string &  NameOperator() const override {static std::string s("powc"); return s;}
            virtual std::string  PostName() const {return " " + std::to_string(mExp);}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                    mDataBuf[aK] = std::pow(mDataF[aK],mExp);
            }
            /// rule : (log F)'  =  F' / F
            cFormula<TypeElem> Derivate(int aK) const override 
            {
                return   (mExp*mF->Derivate(aK)) * pow(mF,mExp-1.0);
            }

            TypeElem mExp;
};


      /* ---------------------------------------*/
      /*           Global Functio on unary op   */
      /* ---------------------------------------*/

/**  A Helper class to avoid code duplication on the process :
       * compute name
       * test existence
       * eventualy create
     Template parameter is the class created
*/

template <class TypeCompiled>  class cGenOperatorUnaire
{
    public :
         typedef typename TypeCompiled::tElem     tElem;
         typedef typename TypeCompiled::tCoordF     tCoordF;
         typedef typename TypeCompiled::tImplemF     tImplemF;
         typedef typename tImplemF::tFormula  tFormula;

         static tFormula   Generate(tFormula aF,const std::string & aNameOp,const std::string & Aux="")
         {
             tCoordF* aPCont = aF->CoordF();  // Get the context from the formula
             std::string aNameForm = aF.NameFormulaUn(aNameOp,Aux);  // Compute the name formula should have

             if (aPCont->ExistFunc(aNameForm))  // If it already exist 
               return aPCont->FuncOfName(aNameForm);  // Then return formula whih this name

             auto opType = new TypeCompiled(aF,aNameForm);
             if (REDUCE_CSTE)
             {
                 const tElem * aC1 = aF->ValCste();
                 if (aC1)
                 {
                    SHOW_REDUCE(aNameOp + std::to_string(*aC1) + Aux);
                    tElem  aC2 = opType->Operation(*aC1);      // We have to use Operation() on a class instance because of cPowCste
                    delete opType;
                    return CreateCste(aC2,aF);
                 }
             }
             tFormula aResult(opType);
             aPCont->AddFormula(aResult); // indicate to the context to remember this new formula
             return aResult;              // return it
         }
};
    
template <class TypeElem> 
inline cFormula<TypeElem>  square(const cFormula<TypeElem> & aF)
{
    return cGenOperatorUnaire<cSquareF<TypeElem> >::Generate(aF,"square");
}

template <class TypeElem> 
inline cFormula<TypeElem>  cube(const cFormula<TypeElem> & aF)
{
    return cGenOperatorUnaire<cCubeF<TypeElem> >::Generate(aF,"cube");
}
template <class TypeElem> 
inline cFormula<TypeElem>  pow4(const cFormula<TypeElem> & aF)
{
    return cGenOperatorUnaire<cPow4<TypeElem> >::Generate(aF,"pow4");
}
template <class TypeElem> 
inline cFormula<TypeElem>  pow5(const cFormula<TypeElem> & aF)
{
    return cGenOperatorUnaire<cPow5<TypeElem> >::Generate(aF,"pow5");
}
template <class TypeElem> 
inline cFormula<TypeElem>  pow6(const cFormula<TypeElem> & aF)
{
    return cGenOperatorUnaire<cPow6<TypeElem> >::Generate(aF,"pow6");
}
template <class TypeElem> 
inline cFormula<TypeElem>  pow7(const cFormula<TypeElem> & aF)
{
    return cGenOperatorUnaire<cPow7<TypeElem> >::Generate(aF,"pow7");
}


template <class TypeElem> 
inline cFormula<TypeElem> exp(const cFormula<TypeElem> & aF)
{
    return cGenOperatorUnaire<cExpF<TypeElem> >::Generate(aF,"exp");
}
template <class TypeElem> 
inline cFormula<TypeElem>  operator - (const cFormula<TypeElem> & aF)
{
    // rule  - (- x) ==  x
    if (REDUCE_MM &&  (aF->NameOperator()=="-"))
    {
       if (aF->Ref().size()==1)
       {
          SHOW_REDUCE("-(-x)");
          return aF->Ref()[0];
       }
    }

    return cGenOperatorUnaire<cMin1F<TypeElem> >::Generate(aF,"-");
}

template <class TypeElem> 
inline cFormula<TypeElem>  log (const cFormula<TypeElem> & aF)
{
    return cGenOperatorUnaire<cLogF<TypeElem> >::Generate(aF,"log");
}

template <class TypeElem> 
inline cFormula<TypeElem>  pow (const cFormula<TypeElem> & aF,const TypeElem& aVal )
{
    if (aVal==TypeElem(2)) return square(aF);
    if (aVal==TypeElem(3)) return cube(aF);
    return cGenOperatorUnaire<cPowCste<TypeElem> >::Generate(aF,"powc",std::to_string(aVal));
}
template <class TypeElem> 
inline cFormula<TypeElem>  pow (const cFormula<TypeElem> & aF,const int & aVal )
{
   return pow(aF,TypeElem(aVal));
}

template <class TypeElem> inline cFormula<TypeElem>  pow8 (const cFormula<TypeElem> & aF){return pow(aF,8);}
template <class TypeElem> inline cFormula<TypeElem>  pow9 (const cFormula<TypeElem> & aF){return pow(aF,9);}
}; //   NS_MMVII_FormalDerivative



#endif // _MMVII_FormDer_UnaryOp_H_
