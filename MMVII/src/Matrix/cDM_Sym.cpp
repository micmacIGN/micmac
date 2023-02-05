
#include "cMMVII_Appli.h"
#include "MMVII_Geom2D.h"


namespace MMVII
{

/*  Used typically for QR decompose,  once we have  M =QR (or RQ), the decomposition is not unique because we 
 *  can arbitrarily change signs, let S be any sign matrix (diag with +-1) we still have
 *
 *       M = Q (SS)R = (QS)(SR)
 *
 *  So we fix that by imposign that the R (= triang) has a positive diag. To understand the code, also remember
 *  that right-multiply by a sign-matrix is equivalent to change the signs of certain colum, as :
 *
 *   (a b)  (-1 0)  =  (-a b)
 *   (c d)  (0  1)     (-c d)
 */
template <class Type> void  NormalizeProdDiagPos(cDenseMatrix<Type> &aM1,cDenseMatrix<Type> & aM2 ,bool TestOn1)
{
    cMatrix<Type>::CheckSquare(aM1);
    cMatrix<Type>::CheckSquare(aM2);
    cMatrix<Type>::CheckSizeMul(aM1,aM2);

    const cDenseMatrix<Type> &aMTest = TestOn1 ? aM1 : aM2;

    size_t aSz = aM1.Sz().x();

    FakeUseIt(aSz);
    for (size_t aK=0;aK<aSz; aK++)
    {
       if (aMTest.GetElem(aK,aK)<0)
       {
          aM1.SelfColChSign(aK);
          aM2.SelfLineChSign(aK);
       }
    }
}

static std::vector<eLevelCheck> VEigenDoTestSucces{eLevelCheck::Error};

void PushErrorEigenErrorLevel(eLevelCheck aLevel)
{
    VEigenDoTestSucces.push_back(aLevel);
}
void PopErrorEigenErrorLevel()
{
    VEigenDoTestSucces.pop_back();
}

bool EigenDoTestSuccess() 
{
    return VEigenDoTestSucces.back() != eLevelCheck::NoCheck;
}
void OnEigenNoSucc(const  char * aMesg,int aLine,const char * aFile)
{
	/* MPD : prefer warning thar are repeated once during process execution  one at end
    StdOut() << "EIGEN operation didnot reached success : " << aMesg 
             << " at line "<< aLine << " of "<< aFile<<"\n";
	     */

    MMVII_DEV_WARNING(std::string("EIGEN operation didnot reached success : ") + aMesg + " at line " +  ToStr(aLine) +" of " +  aFile);

    if (VEigenDoTestSucces.back() == eLevelCheck::Error)
    {
        MMVII_INTERNAL_ERROR("Unhandled error in Eigen operation");
    }
}

/* ============================================= */
/*      cDenseMatrix<Type>                       */
/* ============================================= */


template <class Type> double  cDenseMatrix<Type>::Unitarity() const
{
     cDenseMatrix<Type> atMM = Transpose() * (*this);
     cDenseMatrix<Type> aId(atMM.Sz().x(),eModeInitImage::eMIA_MatrixId);
     return aId.DIm().L2Dist(atMM.DIm());

}


template <class Type> double cDenseMatrix<Type>::Diagonalicity() const
{
   cMatrix<Type>::CheckSquare(*this);
   int aNb = Sz().x();
   double aRes = 0;
   for (int aX=0 ; aX<aNb ; aX++)
   {
       for (int aY=0 ; aY<aNb ; aY++)
       {
            if (aX!=aY)
               aRes += Square(GetElem(aX,aY));
       }
   }
   return sqrt(aRes/std::max(1,aNb*aNb-aNb));
}


template <class Type> double cDenseMatrix<Type>::Symetricity() const
{
   cMatrix<Type>::CheckSquare(*this);
   int aNb = Sz().x();
   double aRes = 0;
   for (int aX=0 ; aX<aNb ; aX++)
   {
       for (int aY=aX+1 ; aY<aNb ; aY++)
       {
            aRes += Square(GetElem(aX,aY)-GetElem(aY,aX)) / 2.0;
       }
   }
   return sqrt(aRes/DIm().NbElem());
}

template <class Type> double cDenseMatrix<Type>::AntiSymetricity() const
{
   cMatrix<Type>::CheckSquare(*this);
   int aNb = Sz().x();
   double aRes = 0;
   for (int aX=0 ; aX<aNb ; aX++)
   {
       for (int aY=aX+1 ; aY<aNb ; aY++)
       {
            aRes += Square(GetElem(aX,aY)+GetElem(aY,aX)) / 2.0;
       }
   }
   for (int aX=0 ; aX<aNb ; aX++)
       aRes += Square(GetElem(aX,aX));
   return sqrt(aRes/DIm().NbElem());
}


template <class Type> void cDenseMatrix<Type>::SelfSymetrize()
{
   cMatrix<Type>::CheckSquare(*this);
   int aNb = Sz().x();
   for (int aX=0 ; aX<aNb ; aX++)
   {
       for (int aY=aX+1 ; aY<aNb ; aY++)
       {
            Type aV =  (GetElem(aX,aY)+GetElem(aY,aX)) / 2.0;
            SetElem(aX,aY,aV);
            SetElem(aY,aX,aV);
       }
   }
}

template <class Type> void cDenseMatrix<Type>::SelfSymetrizeBottom()
{
   cMatrix<Type>::CheckSquare(*this);
   int aNb = Sz().x();
   for (int aX=0 ; aX<aNb ; aX++)
   {
       for (int aY=aX+1 ; aY<aNb ; aY++)
       {
            SetElem(aX,aY,GetElem(aY,aX));
       }
   }
}




template <class Type> void cDenseMatrix<Type>::SelfAntiSymetrize()
{
   cMatrix<Type>::CheckSquare(*this);
   int aNb = Sz().x();
   for (int aX=0 ; aX<aNb ; aX++)
   {
       for (int aY=aX ; aY<aNb ; aY++)
       {
            Type aV =  (GetElem(aX,aY)-GetElem(aY,aX)) / 2.0;
            SetElem(aX,aY,aV);
            SetElem(aY,aX,-aV);
       }
   }
}


template <class Type> cDenseMatrix<Type>  cDenseMatrix<Type>::Symetrize() const 
{
   cDenseMatrix<Type> aRes = Dup();
   aRes.SelfSymetrize();
   return aRes;
}
template <class Type> cDenseMatrix<Type>  cDenseMatrix<Type>::AntiSymetrize() const 
{
   cDenseMatrix<Type> aRes = Dup();
   aRes.SelfAntiSymetrize();
   return aRes;
}



template <class Type>  void cDenseMatrix<Type>::TransposeIn(tDM & aM2) const
{

     MMVII_INTERNAL_ASSERT_medium(aM2.Sz() == PSymXY(Sz()) ,"Bad size for in place transposition")
     MMVII_INTERNAL_ASSERT_medium(&(aM2.DIm()) != &(this->DIm()) ,"Use TransposeIn with same matrix");

     // Rest du bug bizarre sur :RandomSquareRegMatrix
     // cPixBox<2> aBox = *this;
     // for (const auto & aP : aBox)
/*
     for (const auto & aP : *this)
     {

  StdOut() << "xxxxx "  << aP << Sz() << " " << DIm().Sz() << "\n";
          aM2.SetElem(aP.y(),aP.x(),GetElem(aP));
     }
*/

     //  for (const auto & aP : *this) => generate bug with -O0 compile option, why : ?!?
     // not satifyed with this correction, but the show must go on ...
     cPixBox<2> aBox = *this;
     for (const auto & aP : aBox)
          aM2.SetElem(aP.y(),aP.x(),GetElem(aP));
}

template <class Type>  void cDenseMatrix<Type>::SelfTransposeIn()
{
    cMatrix<Type>::CheckSquare(*this);
    for (int aX=0 ; aX<Sz().x() ; aX++)
    {
        for (int aY=aX+1 ; aY<Sz().x() ; aY++)
        {
            Type  aVxy = GetElem(aX,aY);
            SetElem(aX,aY,GetElem(aY,aX));
            SetElem(aY,aX,aVxy);
            // std::swap(GetElem(aX,aY),GetElem(aY,aX));
        }
    }
}

template <class Type>  cDenseMatrix<Type> cDenseMatrix<Type>::Transpose() const
{
   cDenseMatrix<Type> aRes(Sz().y(),Sz().x());
   TransposeIn(aRes);
   return aRes;
}

     // ========= line inversion ===== 

template <class Type>  void cDenseMatrix<Type>::SelfLineInverse() 
{
    for (int aY=0 ; aY< (Sz().y()/2) ; aY++)
    {
        int aYCompl = Sz().y()-1-aY;
        for (int aX=0 ; aX<Sz().x() ; aX++)
	{
             std::swap(GetReference_V(aX,aY),GetReference_V(aX,aYCompl));
	}
    }
}

template <class Type>  void cDenseMatrix<Type>::SelfLineChSign(int aNumL)
{
    for (int aX=0 ; aX<Sz().x() ; aX++)
    {
         GetReference_V(aX,aNumL) *= -1;
    }
}

template <class Type>  cDenseMatrix<Type> cDenseMatrix<Type>::LineInverse() const
{
    cDenseMatrix<Type> aRes = Dup();
    aRes.SelfLineInverse();

    return aRes;
}

template <class Type>  void cDenseMatrix<Type>::SelfColInverse() 
{
    for (int aX=0 ; aX<Sz().x()/2 ; aX++)
    {
        int aXCompl = Sz().x()-1-aX;
        for (int aY=0 ; aY< Sz().y() ; aY++)
	{
             std::swap(GetReference_V(aX,aY),GetReference_V(aXCompl,aY));
	}
    }
}

template <class Type>  void cDenseMatrix<Type>::SelfColChSign(int aNumC)
{
    for (int aY=0 ; aY<Sz().y() ; aY++)
    {
         GetReference_V(aNumC,aY) *= -1;
    }
}


// tDM  LineInverse() const;  ///< cont version of SelfLineInverse

        // void SelfColChSign(int aNumC);  ///<  chang signe of column aNumC
     

     // ========= Triangular =============

template <class Type>  void cDenseMatrix<Type>::SelfTriangSup()
{
     for (const auto & aP : *this)
     {
         if (aP.x() < aP.y())
         {
            SetElem(aP.x(),aP.y(),0.0);
         }
     }
}

template <class Type>  double cDenseMatrix<Type>::TriangSupicity() const   ///< How close to triangular sup
{
     double aNb=0;
     double aSom =0.0;
     for (const auto & aP : *this)
     {
         if (aP.x() < aP.y())
         {
            aNb++;
            aSom += Square(GetElem(aP.x(),aP.y()));
         }
     }
     aSom /= std::max(1.0,aNb);
     return std::sqrt(aSom);
}



/* ===================================================== */
/* =====              INSTANTIATION                ===== */
/* ===================================================== */


#define INSTANTIATE_SYM_DENSE_MATRICES(Type)\
template  class  cDenseMatrix<Type>;\
template void  NormalizeProdDiagPos(cDenseMatrix<Type> &aM1,cDenseMatrix<Type> & aM2 ,bool TestOn1);

INSTANTIATE_SYM_DENSE_MATRICES(tREAL4)
INSTANTIATE_SYM_DENSE_MATRICES(tREAL8)
INSTANTIATE_SYM_DENSE_MATRICES(tREAL16)


};
