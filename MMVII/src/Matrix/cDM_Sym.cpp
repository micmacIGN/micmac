
#include "cMMVII_Appli.h"
#include "MMVII_Geom2D.h"


namespace MMVII
{


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
    StdOut() << "EIGEN operation didnot reached success : " << aMesg 
             << " at line "<< aLine << " of "<< aFile<<"\n";
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

INSTANTIATE_SYM_DENSE_MATRICES(tREAL4)
INSTANTIATE_SYM_DENSE_MATRICES(tREAL8)
INSTANTIATE_SYM_DENSE_MATRICES(tREAL16)


};
