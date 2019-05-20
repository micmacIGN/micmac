#ifndef  _MMVII_Matrix_H_
#define  _MMVII_Matrix_H_
namespace MMVII
{


/** \file MMVII_Matrix.h
    \brief Classes for matrix manipulation, 
*/

/** Also algorithm will mainly use eigen,  storage will be 
    done by MMVII class
*/


/** Matrix can store their data in 4,8,16 ... byte however for having unique virtual communication
    we fixe  the interface with the "highest" precision
*/


class cEigenImplementDense;

typedef tREAL16 tMatrElem;

/** A dense vector is no more than a 1D Image, but with a different interface */

template <class Type> class  cDenseVect 
{
    public :
        friend class cEigenImplementDense;
        typedef cIm1D<Type>  tIM;
        typedef cDataIm1D<Type>      tDIM;

        cDenseVect(int aSz,Type * aDataLin=0);
        const Type & operator() (int aK) const {return DIm().GetV(aK);}
        Type & operator() (int aK) {return DIm().GetV(aK);}
        const int & Sz() const {return DIm().Sz();}

        // operator -= 
    private :
        void AssertSameSize();

        Type * RawData();
        tDIM & DIm(){return mIm.DIm();}
        const tDIM & DIm() const {return mIm.DIm();}

        tIM mIm;
};

/** a Interface class */
class cDataMatrix  : public cRect2
{
     public :
         const cPt2di & Sz() const {return cRect2::Sz();}
         tMatrElem operator() (int aX,int  aY) const {return GetElem(aX,aY);} ///< Syntactic sugar

         virtual tMatrElem GetElem(int aX,int  aY) const = 0;
         virtual void  SetElem(int  aX,int  aY,const tMatrElem &) = 0;

         virtual void  MulCol(cDenseVect<tREAL4> &,const cDenseVect<tREAL4> &) const;
         virtual tMatrElem MulCol(int  aY,const cDenseVect<tREAL4> &)const;
         virtual void  MulLine(cDenseVect<tREAL4> &,const cDenseVect<tREAL4> &) const;
         virtual tMatrElem MulLine(int  aX,const cDenseVect<tREAL4> &)const;

         virtual void  MulCol(cDenseVect<tREAL8> &,const cDenseVect<tREAL8> &) const;
         virtual tMatrElem MulCol(int  aY,const cDenseVect<tREAL8> &)const;
         virtual void  MulLine(cDenseVect<tREAL8> &,const cDenseVect<tREAL8> &) const;
         virtual tMatrElem MulLine(int  aX,const cDenseVect<tREAL8> &)const;

         virtual void  MulCol(cDenseVect<tREAL16> &,const cDenseVect<tREAL16> &) const;
         virtual tMatrElem MulCol(int  aY,const cDenseVect<tREAL16> &)const;
         virtual void  MulLine(cDenseVect<tREAL16> &,const cDenseVect<tREAL16> &) const;
         virtual tMatrElem MulLine(int  aX,const cDenseVect<tREAL16> &)const;


     protected :
         template <class Type> void TplMulCol(cDenseVect<Type> &,const cDenseVect<Type> &) const;
         template <class Type> tMatrElem TplMulCol(int aY,const cDenseVect<Type> &) const;

         template <class Type> void TplMulLine(cDenseVect<Type> &,const cDenseVect<Type> &) const;
         template <class Type> tMatrElem TplMulLine(int aX,const cDenseVect<Type> &) const;


         template <class Type> void TplCheckSizeCol(const cDenseVect<Type> &)const;
         template <class Type> void TplCheckSizeLine(const cDenseVect<Type> &)const;

         cDataMatrix(int aX,int aY);
};




/*

class cMatrix
{
    public :
         cDataMatrix & Mat()  {return *(mSPtr.get());}
         const cDataMatrix & Mat() const  {return *(mSPtr.get());}
    private :
         std::shared_ptr<tDIM> mSPtr;  ///< shared pointer to real image

};
*/

#if(0)
#endif


};

#endif  //  _MMVII_Matrix_H_
