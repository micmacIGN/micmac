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

            virtual std::string  NameOperator() const = 0;
            std::string  InfixPPrint() const override 
            {
               return NameOperator() + " "+  mF->InfixPPrint() ;
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


/*   Probably not more efficient than implementing Square as F*F, because derivation would give
     F'F + F'F  BUT would be reorder as F'F + F'F and unified ...
     By the way it was a test, if necessary replace Square by F*F */
template <class TypeElem> class cSquareF : public cUnaryF<TypeElem>
{
     public :
            using cUnaryF<TypeElem>::mF;
            using cUnaryF<TypeElem>::mDataF;
            using cImplemF<TypeElem>::mDataBuf;
     
            cSquareF (cFormula<TypeElem> aF,const std::string & aName) :
                cUnaryF <TypeElem> (aF,aName)
            { }
      private :
            std::string  NameOperator() const override {return "square";}
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


template <class TypeElem> class cExpF : public cUnaryF<TypeElem>
{
     public :
            using cUnaryF<TypeElem>::mF;
            using cUnaryF<TypeElem>::mDataF;
            using cImplemF<TypeElem>::mDataBuf;

            cExpF (cFormula<TypeElem> aF,const std::string & aName) :
                cUnaryF <TypeElem> (aF,aName)
            { }
      private :
            std::string  NameOperator() const override {return "exp";}
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
      private :
            std::string  NameOperator() const override {return "-";}
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
      private :
            std::string  NameOperator() const override {return "log";}
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
         typedef typename TypeCompiled::tCoordF     tCoordF;
         typedef typename TypeCompiled::tImplemF     tImplemF;
         typedef typename tImplemF::tFormula  tFormula;

         static tFormula   Generate(tFormula aF,const std::string & aNameOp)
         {
             tCoordF* aPCont = aF->CoordF();  // Get the context from the formula
             std::string aNameForm = aF.NameFormulaUn(aNameOp);  // Compute the name formula should have

             if (aPCont->ExistFunc(aNameForm))  // If it already exist 
               return aPCont->FuncOfName(aNameForm);  // Then return formula whih this name

             tFormula aResult (new TypeCompiled(aF,aNameForm)); // else create it
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
inline cFormula<TypeElem> exp(const cFormula<TypeElem> & aF)
{
    return cGenOperatorUnaire<cExpF<TypeElem> >::Generate(aF,"exp");
}
template <class TypeElem> 
inline cFormula<TypeElem>  operator - (const cFormula<TypeElem> & aF)
{
    return cGenOperatorUnaire<cMin1F<TypeElem> >::Generate(aF,"-");
}
template <class TypeElem> 
inline cFormula<TypeElem>  log (const cFormula<TypeElem> & aF)
{
    return cGenOperatorUnaire<cLogF<TypeElem> >::Generate(aF,"log");
}

}; //   NS_MMVII_FormalDerivative



#endif // _MMVII_FormDer_UnaryOp_H_
