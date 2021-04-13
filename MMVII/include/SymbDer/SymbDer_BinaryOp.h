#ifndef _SymbDer_BinaryOp_H_
#define _SymbDer_BinaryOp_H_

/** \file SymbDer_BinaryOp.h
    \brief File for definition of binary operators inside formula

*/



namespace  NS_SymbolicDerivative
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
                  DERIVED :   cSumF / cMulF / cSubF / cDivF / cPowF /
             ----------------------------------------------------------------*/

template <class TypeElem> class cBinaryF : public cImplemF<TypeElem>
{
      public :
            typedef cImplemF<TypeElem>   tImplemF;
            typedef typename tImplemF::tCoordF   tCoordF;
            typedef typename tImplemF::tFormula   tFormula;
            typedef typename tImplemF::tBuf       tBuf;

            /// An operator must describe its name
            /// We can compute print of binary formula using operator name
            std::string  InfixPPrint() const override 
            {
               return "("+ this->NameOperator() + " "+ mF1->InfixPPrint() + " " + mF2->InfixPPrint() + ")";
            }
            /// Is it an associative operator where order does not matters
            static bool IsAssociatif() {return false;}
            static bool IsDistribExt() {return false;} // like +,-

            /// We need a default value to compile in associative
            static tFormula FOperation(const tFormula & aV1,const tFormula & aV2) 
            {
                InternalError("No operation defined",aV1->CoordF()->Name());
                return aV1;
            }
      protected  :
            void AssocSortedVect(std::vector<tFormula> & aV);
            void EmpileAssoc (const cFormula <TypeElem>& aF, std::vector<tFormula > & aV);
            virtual std::string GenCodeShortExpr() const override {
                return "(" + mF1->GenCodeFormName() + " " + this->NameOperator() +  " " + mF2->GenCodeFormName() + ")";
            }

            virtual std::string GenCodeDef() const override {
                return "(" + mF1->GenCodeRef() + " " + this->NameOperator() +  " " + mF2->GenCodeRef() + ")";
            }
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
                   UserSError("Mix formula from different context",this->mCoordF->Name());
            }

            tFormula  mF1;   ///< First argument of operator
            const TypeElem  *mDataF1; ///< Fast access to data of buf F1
            tFormula  mF2;   ///< Second argument of operator
            const TypeElem  *mDataF2; ///< Fast access to data of buf F2
};

/// for example  if F = (A+B)+ (C+(D+E)) push on V "A B C D E"
template <class TypeElem> void cBinaryF<TypeElem>::EmpileAssoc
     (
        const cFormula <TypeElem>& aF,
        std::vector<tFormula> & aV
     )
{
    if (aF->NameOperator() != this->NameOperator())
    {
       aV.push_back(aF);
       return;
    }
    for (auto aSubF : aF->Ref())
       EmpileAssoc(aSubF,aV);
}

template <class TypeElem> void SortOnName(std::vector<cFormula<TypeElem> > & aV)
{
   std::sort
   (
      aV.begin(),aV.end(),
      [](const cFormula<TypeElem> & f1,const cFormula<TypeElem> & f2) {return f1->Name()<f2->Name();}
   );
}
template <class TypeElem> void cBinaryF<TypeElem>::AssocSortedVect(std::vector<tFormula> & aV)
{
   EmpileAssoc(mF1,aV);
   EmpileAssoc(mF2,aV);
   SortOnName(aV);
}


template <class TypeElem> class cSumF : public cBinaryF <TypeElem>
{
      public :
            using cBinaryF<TypeElem>::mF1;
            using cBinaryF<TypeElem>::mF2;
            using cBinaryF<TypeElem>::mDataF1;
            using cBinaryF<TypeElem>::mDataF2;
            using cImplemF<TypeElem>::mDataBuf;
            typedef typename cBinaryF <TypeElem>::tFormula   tFormula;

            inline cSumF(cFormula<TypeElem> aF1,cFormula<TypeElem> aF2,const std::string & aName) :
                   cBinaryF<TypeElem> (aF1,aF2,aName) 
            { }

                // ============== BEHAVIOUR FOR REDUCTION =============

            /// Required by constant reduction in  cGenOperatorBinaire
            static TypeElem Operation(const TypeElem & aV1,const TypeElem & aV2) {return aV1+aV2;};
            /// It is associative
            static bool IsAssociatif() {return true;}
            /// For assoc reduc
            /// For distributivity
            static bool IsDistribExt() {return true;} 
            static tFormula FOperation(const tFormula & aV1,const tFormula & aV2) {return aV1+aV2;}

            cImplemF<TypeElem> * ReducAssoc() override ;

      private  :
            const std::string &  NameOperator() const override {static std::string s("+"); return s;}

            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                     mDataBuf[aK] =  mDataF1[aK] + mDataF2[aK];
            }
            cFormula<TypeElem> Derivate(int aK) const override {return  mF1->Derivate(aK) + mF2->Derivate(aK);}
};

template <class TypeElem>  cImplemF<TypeElem> *  cSumF<TypeElem>::ReducAssoc()
{
    if (! REDUCE_ASSOCP)
       return this;

    static int aSzCum=0; ///< Stat to see num of + operation
    std::vector<tFormula> aVF;
    this->AssocSortedVect(aVF);
    aSzCum += aVF.size();
    
    // Now we try to make reduction between last reduced formula and  next one
    // using distributivity

    std::vector<tFormula> aVR; ///< Current vector of reduced formulas

    bool ReduceDone = true;

    while (ReduceDone)
    {
       SortOnName(aVF);
       aVR.clear();
       aVR.push_back(aVF[0]);
       ReduceDone = false;
       for (int aKIn=1 ; aKIn<int(aVF.size()) ; aKIn++)
       {
           // we analyse A+B
           tFormula aFA = aVR.back();
           tFormula aFB = aVF[aKIn];
           //  rule : A+A => 2*A
           if (aFA->Name() == aFB->Name())
           {
               ReduceDone = true;
               SHOW_REDUCE("ApA");
               aVR.back() = aFA * CreateCste(2.0,aFA);
           }
           else if (aFA->IsMult())  // A1 A2 +B
           {
              tFormula A1 = aFA->Ref().at(0);
              tFormula A2 = aFA->Ref().at(1);
              if (aFB->IsMult())  // A1 A2 + B1 B2
              {
                  tFormula B1 = aFB->Ref().at(0);
                  tFormula B2 = aFB->Ref().at(1);
                  if (A1->Name()== B1->Name())  // A1 A2 + A1 B2 => A1 (A2+B2)
                  {
                       ReduceDone = true;
                       SHOW_REDUCE("AB + AC");
                       aVR.back() = A1 * (A2+B2);
                  }
                  else if (A2->Name()== B2->Name())   // A1 A2 + B1 A2 => (A1+B1) * A2
                  {
                       ReduceDone = true;
                       SHOW_REDUCE("AB + CB");
                       aVR.back() = (A1+B1) *A2;
                  }
                  else if (A1->Name()== B2->Name())  // A1 A2 + B1 A1 => A1 * (A2+B1)
                  {
                       ReduceDone = true;
                       SHOW_REDUCE("AB + CA");
                       aVR.back() = A1 * (A2+B1);
                  }
                  else if (A2->Name()== B1->Name())  // A1 A2 + A2 B2 => A2 * (A1+B2)
                  {
                       ReduceDone = true;
                       SHOW_REDUCE("AB + BC");
                       aVR.back() = A2 * (A1+B2);
                  }
                  else
                     aVR.push_back(aFB);
              }
              else
              {
                  if (A1->Name()== aFB->Name())  // A1 A2 + A1 => A1 * (A2+1)
                  {
                       ReduceDone = true;
                       SHOW_REDUCE("BA+B");
                       aVR.back() = A1 * (A2+CreateCste(1.0,aFA));
                  }
                  else if (A2->Name()== aFB->Name())  // A1 A2 + A2 => A2 * (A1+1)
                  {
                       ReduceDone = true;
                       SHOW_REDUCE("AB+B");
                       aVR.back() = A2 * (A1+CreateCste(1.0,aFA));
                  }
                  else
                     aVR.push_back(aFB);
              }
           }
           else   if (aFB->IsMult())  // A1 A2 + B1 B2
           {
              tFormula B1 = aFB->Ref().at(0);
              tFormula B2 = aFB->Ref().at(1);
              if (aFA->Name()== B1->Name())  // B1 + B1 B2 => B2 * (B2+1)
              {
                   ReduceDone = true;
                   SHOW_REDUCE("A+AB");
                   aVR.back() = B1 * (B2+CreateCste(1.0,aFA));
              }
              if (aFA->Name()== B2->Name())  // B2 + B1 B2 => B2 * (B1+1)
              {
                   ReduceDone = true;
                   SHOW_REDUCE("A+BA");
                   aVR.back() = B2 * (B1+CreateCste(1.0,aFA));
              }
              else
                  aVR.push_back(aFB);
           }
           else
              aVR.push_back(aFB);
 
       }
       aVF = aVR;
    }
    tFormula aRes = aVR.back();
    for (int aK=aVR.size()-2 ; aK>=0 ; aK--)
        aRes = aVR[aK] + aRes;
    return aRes.RawPtr();
}

template <class TypeElem> class cMulF : public cBinaryF<TypeElem>
{
      public :
            using cBinaryF<TypeElem>::mF1;
            using cBinaryF<TypeElem>::mF2;
            using cBinaryF<TypeElem>::mDataF1;
            using cBinaryF<TypeElem>::mDataF2;
            using cImplemF<TypeElem>::mDataBuf;
            typedef typename cBinaryF <TypeElem>::tFormula   tFormula;

            inline cMulF(cFormula<TypeElem> aF1,cFormula<TypeElem> aF2,const std::string & aName) :
                   cBinaryF<TypeElem> (aF1,aF2,aName) 
            { }
                // ============== BEHAVIOUR FOR REDUCTION =============

            bool  IsMult() const {return true;}

            /// Required by constant reduction in  cGenOperatorBinaire
            static TypeElem Operation(const TypeElem & aV1,const TypeElem & aV2) {return aV1*aV2;}
            /// It is associative
            static bool IsAssociatif() {return true;}
            /// For assoc reduc
            static tFormula FOperation(const tFormula & aV1,const tFormula & aV2) {return aV1*aV2;}

            /// For distributivity
            virtual bool IsDistribInt() const override {return true;} 
            tFormula VOper2 (const tFormula & aV1,const tFormula & aV2) const override {return aV1*aV2;}
      private  :
            const std::string &  NameOperator() const override {static std::string s("*"); return s;}
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
            typedef typename cBinaryF <TypeElem>::tFormula   tFormula;

            inline cSubF(cFormula<TypeElem> aF1,cFormula<TypeElem> aF2,const std::string & aName) :
                   cBinaryF<TypeElem> (aF1,aF2,aName) 
            { }

                // ============== BEHAVIOUR FOR REDUCTION =============
            /// For distributivity
            static bool IsDistribExt() {return true;} 
            static tFormula FOperation(const tFormula & aV1,const tFormula & aV2) {return aV1-aV2;}
      private  :
            const std::string &  NameOperator() const override {static std::string s("-"); return s;}
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
            typedef typename cBinaryF <TypeElem>::tFormula   tFormula;

            inline cDivF(cFormula<TypeElem> aF1,cFormula<TypeElem> aF2,const std::string & aName) :
                   cBinaryF<TypeElem> (aF1,aF2,aName) 
            { }
            /// For distributivity
            virtual bool IsDistribInt() const override {return true;} 
            tFormula VOper2 (const tFormula & aV1,const tFormula & aV2) const override {return aV1/aV2;}
      private  :
            const std::string &  NameOperator() const override {static std::string s("/"); return s;}
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
            const std::string &  NameOperator() const override {static std::string s("^"); return s;}
            virtual std::string GenCodeShortExpr() const override {
                return "std::pow(" + mF1->GenCodeFormName() + "," + mF2->GenCodeFormName() + ")";
            }

            virtual std::string GenCodeDef() const override {
                return "std::pow(" + mF1->GenCodeRef() + ","  + mF2->GenCodeRef() + ")";
            }
            void ComputeBuf(int aK0,int aK1) override
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                     mDataBuf[aK] =  std::pow(mDataF1[aK],mDataF2[aK]);
            }
            /// Formula F ^ G = exp(G log(F))
            /// (F^G) ' = (F^G) (G F'/F + G'log(F))
            cFormula<TypeElem> Derivate(int aK) const override 
            {
                if (mF2->ValCste())
                    return mF2 * mF1->Derivate(aK) * pow(mF1,mF2 - 1.0);
                else
                    return pow(mF1,mF2) * ( (mF1->Derivate(aK)/mF1)*mF2 + mF2->Derivate(aK)*log(mF1)) ;
            }
};


      /* ---------------------------------------*/
      /*          Global Function on unary op   */
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
             if (REDUCE_CSTE)
             {
                 const tElem * aC1 = aF1->ValCste();
                 const tElem * aC2 = aF2->ValCste();
                 if (aC1 && aC2) 
                 {
                    SHOW_REDUCE("Cste x Cste");
                    tElem  aC12= TypeCompiled::Operation(*aC1,*aC2);
                    return CreateCste(aC12,aF1);
                 }
             }

             if (  
                      REDUCE_DISTRIB 
                   && TypeCompiled::IsDistribExt() 
                   && aF1->IsDistribInt() 
                   && aF2->IsDistribInt()
                   && (aF2->NameOperator() == aF2->NameOperator())
                 )
             {
                // Add IsMult, we dont want to reduce a/b+a/c
                if ((aF1->Ref()[0]->Name()==aF2->Ref()[0]->Name()) && aF1->IsMult())
                {
                  SHOW_REDUCE("(A$B)#(A$C) => A$(B#C)");
                  return aF1->VOper2(aF1->Ref()[0],TypeCompiled::FOperation(aF1->Ref()[1],aF2->Ref()[1]));
                }
                if (aF1->Ref()[1]->Name()==aF2->Ref()[1]->Name())
                {
                    SHOW_REDUCE("(A$B)#(C$B) => (A#C)$B");
                    return aF1->VOper2(TypeCompiled::FOperation(aF1->Ref()[0],aF2->Ref()[0]),aF1->Ref()[1]);
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
     if (aF1->IsCste(0)) return aF2;
     if (aF2->IsCste(0)) return aF1;

     // Use commutativity of + to have a unique representation
     if (aF1->Name() > aF2->Name()) 
        return aF2+aF1;

     // Use commutativity of + to have a unique representation
     if (false && REDUCE_ApA && (aF1->Name() == aF2->Name()) )
     {
         SHOW_REDUCE("ApA");
         return aF1 * CreateCste(2.0,aF1);
     }

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
     if (aF1->IsCste(0)) return -aF2;
     if (aF2->IsCste(0)) return aF1;

     //  A - (-B) = A + B
     if (REDUCE_MM && (aF2->NameOperator()=="-") && (aF2->Ref().size()==1))
     {
         SHOW_REDUCE("a-(-b))");
         return aF1 + aF2->Ref()[0];
     }

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
     if (aF1->IsCste(1)) return aF2;
     if (aF2->IsCste(1)) return aF1;

     // Use the fact that 0 is absorbant element to simplify
     if (aF1->IsCste(0)) return aF1;
     if (aF2->IsCste(0)) return aF2;

     // Remove a multiplication
     if (aF1->IsCste(-1)) return -aF2;
     if (aF2->IsCste(-1)) return -aF1;

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
     if (aF1->IsCste(0)) return aF1;  // 0/F2 -> 0
     if (aF2->IsCste(1)) return aF1;  // F1/1 -> F1

     if (aF2->IsCste(-1)) return -aF1; // F1/-1 -> -F1
     return cGenOperatorBinaire<cDivF<TypeElem> >::Generate(aF1,aF2,"/");
}

template <class TypeElem>
cFormula<TypeElem>   pow
                              (
                                    const cFormula <TypeElem>& aF1,
                                    const cFormula<TypeElem> & aF2
                              ) 
{
    if (aF2->IsCste(0)) return aF1->CoordF()->Cste1();
    if (aF2->IsCste(1)) return aF1;
    if (aF2->IsCste(2)) return square(aF1);
    if (aF2->IsCste(3)) return cube(aF1);
    if (aF2->IsCste(4)) return pow4(aF1);
    if (aF2->IsCste(5)) return pow5(aF1);
    if (aF2->IsCste(6)) return pow6(aF1);
    if (aF2->IsCste(7)) return pow7(aF1);
    // Don't use pow8 nor pow9: they are defined as pow(x,8/9) and then loop back here
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

        // powpowpowpowpowpowpow

template <class TypeElem> cFormula <TypeElem>  pow (const TypeElem & aV1,const cFormula <TypeElem> & aF2)
{
   return exp(log(aV1)*aF2);
}
template <class TypeElem>
inline cFormula<TypeElem>  pow (const cFormula<TypeElem> & aF1,const TypeElem& aV2 )
{
    return pow(aF1, aF1->CoordF()->CsteOfVal(aV2));
}

template <class TypeElem>
inline cFormula<TypeElem>  pow (const cFormula<TypeElem> & aF,const int & aVal )
{
   return pow(aF,TypeElem(aVal));
}

/*  *************************************************************** */
/*                                                                  */
/*          MACRO PART                                              */
/*                                                                  */
/*  *************************************************************** */

/**  This function is used when dont wish to define derivation, because with ad hoc
  function one may "obliged" to define a derivate even when dont use it */

template <class TypeElem> cFormula <TypeElem>
         UndefinedOperBin(const cFormula <TypeElem> & aF1 ,const cFormula <TypeElem> & aF2)
{
    UserSError("UndefinedOperBin",aF1->Name()+aF2->Name());
    return aF1; // must return something
}


} //   NS_Symbolic_Derivative


#endif //  _SymbDer_BinaryOp_H_
