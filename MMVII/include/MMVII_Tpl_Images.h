#ifndef  _MMVII_Tpl_Images_H_
#define  _MMVII_Tpl_Images_H_
namespace MMVII
{


/** \file MMVII_Tpl_Images.h
    \brief  Implemantation of operators on images, matrix.  This is rather basic,
     more general/functionall are (will be) in  "ToCome.h" ;-)

*/

       //===========   Substraction ===========

template<class T1,class T2,class T3,int Dim>  
   void DiffImageInPlace(cDataTypedIm<T1,Dim> & aI1,const cDataTypedIm<T2,Dim> & aI2,const cDataTypedIm<T3,Dim> & aI3)
{
    aI1.AssertSameArea(aI2); 
    aI1.AssertSameArea(aI3);

    for (int aK=0 ; aK<aI1.NbElem() ; aK++)
        aI1.GetRDL(aK) = aI2.GetRDL(aK) - aI3.GetRDL(aK) ;
}

template<class T1,class T2,class T3>  
   cIm2D<T1> DiffImage(T1* /*Type specifier*/ ,const cIm2D<T2> & aI2,const cIm2D<T3> & aI3)
{
     cIm2D<T1>  aI1(aI2.DIm().P0(),aI2.DIm().P1());
     DiffImageInPlace(aI1.DIm(),aI2.DIm(),aI3.DIm());
     return aI1;
}

template<class T2,class T3>   cIm2D<T2> operator - (const cIm2D<T2> & aI2,const cIm2D<T3> & aI3)  
{
   return DiffImage((T2 *)nullptr,aI2,aI3);
}

template<class T2>   cDenseMatrix<T2> operator - (const cDenseMatrix<T2> & aI2,const cDenseMatrix<T2> & aI3)  
{
    return cDenseMatrix<T2>(aI2.Im()-aI3.Im());
}

       //===========   Addition ===========

template<class T1,class T2,class T3,int Dim>  
   void AddImageInPlace(cDataTypedIm<T1,Dim> & aI1,const cDataTypedIm<T2,Dim> & aI2,const cDataTypedIm<T3,Dim> & aI3)
{
    aI1.AssertSameArea(aI2); 
    aI1.AssertSameArea(aI3);

    for (int aK=0 ; aK<aI1.NbElem() ; aK++)
        aI1.GetRDL(aK) = aI2.GetRDL(aK) + aI3.GetRDL(aK) ;
}

template<class T1,class T2,class T3>  
   cIm2D<T1> AddImage(T1* /*Type specifier*/ ,const cIm2D<T2> & aI2,const cIm2D<T3> & aI3)
{
     cIm2D<T1>  aI1(aI2.DIm().P0(),aI2.DIm().P1());
     AddImageInPlace(aI1.DIm(),aI2.DIm(),aI3.DIm());
     return aI1;
}

template<class T2,class T3>   cIm2D<T2> operator + (const cIm2D<T2> & aI2,const cIm2D<T3> & aI3)  
{
   return AddImage((T2 *)nullptr,aI2,aI3);
}

template<class T2>   cDenseMatrix<T2> operator + (const cDenseMatrix<T2> & aI2,const cDenseMatrix<T2> & aI3)  
{
    return cDenseMatrix<T2>(aI2.Im()+aI3.Im());
}

       //===========   MulCste ===========

template<class T1,class T2,class T3,int Dim>  
   void MulImageCsteInPlace(cDataTypedIm<T1,Dim> & aI1,const cDataTypedIm<T2,Dim> & aI2,const T3 & aV3)
{
    aI1.AssertSameArea(aI2); 

    for (int aK=0 ; aK<aI1.NbElem() ; aK++)
        aI1.GetRDL(aK) = aI2.GetRDL(aK) * aV3;
}

template<class T1,class T2,class T3>  
   cIm2D<T1> MulImageCste(T1* /*Type specifier*/ ,const cIm2D<T2> & aI2,const  T3 & aV3)
{
     cIm2D<T1>  aI1(aI2.DIm().P0(),aI2.DIm().P1());
     MulImageCsteInPlace(aI1.DIm(),aI2.DIm(),aV3);
     return aI1;
}

template<class T2,class T3>   cIm2D<T2> operator * (const cIm2D<T2> & aI2,const  T3 & aV3)  
{
   return MulImageCste((T2 *)nullptr,aI2,aV3);
}

template<class T2,class T3>   cDenseMatrix<T2> operator * (const cDenseMatrix<T2> & aI2,const  T3 & aV3)  
{
    return cDenseMatrix<T2>(aI2.Im()*aV3);
}
template<class T2,class T3>   cDenseMatrix<T2> operator * (const  T3 & aV3,const cDenseMatrix<T2> & aI2)
{
    return cDenseMatrix<T2>(aI2.Im()*aV3);
}

       //===========   Copy, +=  ===========

template<class T1,class T2,int Dim>  
   void CopyIn(cDataTypedIm<T1,Dim> & aI1,const cDataTypedIm<T2,Dim> & aI2)
{
    aI1.AssertSameArea(aI2); 

    for (int aK=0 ; aK<aI1.NbElem() ; aK++)
        aI1.GetRDL(aK) = aI2.GetRDL(aK) ;
}

template<class T1,class T2,int Dim>  
   void AddIn(cDataTypedIm<T1,Dim> & aI1,const cDataTypedIm<T2,Dim> & aI2)
{
    aI1.AssertSameArea(aI2); 

    for (int aK=0 ; aK<aI1.NbElem() ; aK++)
        aI1.GetRDL(aK) += aI2.GetRDL(aK) ;
}

template<class T1,class T2,int Dim>  
   void WeightedAddIn(cDataTypedIm<T1,Dim> & aI1,const T2 & aV,const cDataTypedIm<T2,Dim> & aI2)
{
    aI1.AssertSameArea(aI2); 

    for (int aK=0 ; aK<aI1.NbElem() ; aK++)
        aI1.GetRDL(aK) += aV*aI2.GetRDL(aK) ;
}

 
template<class T1,class T2,int Dim>  
   void DivCsteIn(cDataTypedIm<T1,Dim> & aI1,const T2 & aV2)
{
    for (int aK=0 ; aK<aI1.NbElem() ; aK++)
        aI1.GetRDL(aK) /= aV2;
}

/*****************************************************/
/*                                                   */
/*          AGREGATION                               */
/*                                                   */
/*****************************************************/

template<class T1,class T2,int Dim>  
   double DotProduct(const cDataTypedIm<T1,Dim> & aI1,const cDataTypedIm<T2,Dim> & aI2)
{
    aI1.AssertSameArea(aI2); 
    double aSom = 0.0;
    for (int aK=0 ; aK<aI1.NbElem() ; aK++)
        aSom += aI1.GetRDL(aK) * aI2.GetRDL(aK) ;

    return aSom;
}



};

#endif  //  _MMVII_Tpl_Images_H_
