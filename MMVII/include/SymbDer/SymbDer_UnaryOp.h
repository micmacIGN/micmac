#ifndef _SymbDer_UnaryOp_H_
#define _SymbDer_UnaryOp_H_

/** \file SymbDer_UnaryOp.h
    \brief File for unary operator on symbolic derivative
*/


namespace  NS_SymbolicDerivative
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

      protected  :
            virtual std::string GenCodeShortExpr() const override
            {
                return this->NameOperator() + "(" + mF->GenCodeFormName() + ")";
            }

            virtual std::string GenCodeDef() const override
            {
                return this->NameOperator() + "(" + mF->GenCodeRef() + ")";
            }

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
            const std::string &  NameOperator() const override {static std::string s("std::exp"); return s;}
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
            const std::string &  NameOperator() const override {static std::string s("std::log"); return s;}
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

/**  Classe for square root */
template <class TypeElem> class cSqrtF : public cUnaryF<TypeElem>
{
     public :
            using cUnaryF<TypeElem>::mF;
            using cUnaryF<TypeElem>::mDataF;
            using cImplemF<TypeElem>::mDataBuf;

            cSqrtF (cFormula<TypeElem> aF,const std::string & aName) :
                cUnaryF <TypeElem> (aF,aName)
            { }
            static TypeElem Operation(const TypeElem & aV1) {return std::sqrt(aV1);}
      private :
            const std::string &  NameOperator() const override {static std::string s("std::sqrt"); return s;}
            void ComputeBuf(int aK0,int aK1) override
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                    mDataBuf[aK] =  std::sqrt(mDataF[aK]);
            }
            ///  rule : (sqrt(F))' =   F' / (2 * sqrt(F))
            cFormula<TypeElem> Derivate(int aK) const override
            {
                return  mF->Derivate(aK)  / (2.0 * sqrt(mF));
            }
};

template <class TypeElem>  cFormula<TypeElem> cos(const cFormula<TypeElem> & aF);
template <class TypeElem>  cFormula<TypeElem> sin(const cFormula<TypeElem> & aF);


/**  Classe for sin */
template <class TypeElem> class cSinF : public cUnaryF<TypeElem>
{
     public :
            using cUnaryF<TypeElem>::mF;
            using cUnaryF<TypeElem>::mDataF;
            using cImplemF<TypeElem>::mDataBuf;

            cSinF (cFormula<TypeElem> aF,const std::string & aName) :
                cUnaryF <TypeElem> (aF,aName)
            { }
            static TypeElem Operation(const TypeElem & aV1) {return std::sin(aV1);}
      private :
            const std::string &  NameOperator() const override {static std::string s("std::sin"); return s;}
            void ComputeBuf(int aK0,int aK1) override
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                    mDataBuf[aK] =  std::sin(mDataF[aK]);
            }
            ///  rule : (sin(F))' =   F' cos(F) 
            cFormula<TypeElem> Derivate(int aK) const override
            {
                return  mF->Derivate(aK)  * cos(mF);
            }
};

/**  Classe for cos */
template <class TypeElem> class cCosF : public cUnaryF<TypeElem>
{
     public :
            using cUnaryF<TypeElem>::mF;
            using cUnaryF<TypeElem>::mDataF;
            using cImplemF<TypeElem>::mDataBuf;

            cCosF (cFormula<TypeElem> aF,const std::string & aName) :
                cUnaryF <TypeElem> (aF,aName)
            { }
            static TypeElem Operation(const TypeElem & aV1) {return std::cos(aV1);}
      private :
            const std::string &  NameOperator() const override {static std::string s("std::cos"); return s;}
            void ComputeBuf(int aK0,int aK1) override
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                    mDataBuf[aK] =  std::cos(mDataF[aK]);
            }
            ///  rule : (sin(F))' =   F' cos(F) 
            cFormula<TypeElem> Derivate(int aK) const override
            {
                return  - mF->Derivate(aK)  * sin(mF);
            }
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

         static tFormula   Generate(tFormula aF,const std::string & aNameOp)
         {
             tCoordF* aPCont = aF->CoordF();  // Get the context from the formula
             std::string aNameForm = aF.NameFormulaUn(aNameOp);  // Compute the name formula should have

             if (aPCont->ExistFunc(aNameForm))  // If it already exist 
               return aPCont->FuncOfName(aNameForm);  // Then return formula whih this name

             if (REDUCE_CSTE)
             {
                 const tElem * aC1 = aF->ValCste();
                 if (aC1)
                 {
                    SHOW_REDUCE(aNameOp + std::to_string(*aC1));
                    tElem  aC2 = TypeCompiled::Operation(*aC1);
                    return CreateCste(aC2,aF);
                 }
             }
             tFormula aResult(new TypeCompiled(aF,aNameForm));
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
inline cFormula<TypeElem> sqrt(const cFormula<TypeElem> & aF)
{
    return cGenOperatorUnaire<cSqrtF<TypeElem> >::Generate(aF,"sqrt");
}

template <class TypeElem>
inline cFormula<TypeElem> cos(const cFormula<TypeElem> & aF)
{
    return cGenOperatorUnaire<cCosF<TypeElem> >::Generate(aF,"cos");
}

template <class TypeElem>
inline cFormula<TypeElem> sin(const cFormula<TypeElem> & aF)
{
    return cGenOperatorUnaire<cSinF<TypeElem> >::Generate(aF,"sin");
}


template <class TypeElem> inline cFormula<TypeElem>  pow8 (const cFormula<TypeElem> & aF){return pow(aF,8);}
template <class TypeElem> inline cFormula<TypeElem>  pow9 (const cFormula<TypeElem> & aF){return pow(aF,9);}
} //   NS_Symbolic_Derivative



#endif // _SymbDer_UnaryOp_H_
