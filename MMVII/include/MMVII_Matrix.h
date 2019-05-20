#ifndef  _MMVII_Matrix_H_
#define  _MMVII_Matrix_H_
namespace MMVII
{


/** \file MMVII_Matrix.h
    \brief Classes for matrix manipulation, 
*/

/**   \file MMVII_Matrix.h

    Also algorithm will mainly use eigen,  storage will be 
    done by MMVII class
*/


/** Matrix can store their data in 4,8,16 ... byte,  however for having unique virtual communication
    we fixe  the interface with the "highest" precision
*/
typedef tREAL16 tMatrElem;


template <class Type> class  cDenseVect ;
class cMatrix  ;
template <class Type> class cUnOptDenseMatrix ;
template <class Type> class cDenseMatrix ;



/** A dense vector is no more than a 1D Image, but with a different interface */

template <class Type> class  cDenseVect 
{
    public :
        typedef cIm1D<Type>  tIM;
        typedef cDataIm1D<Type>      tDIM;

        cDenseVect(int aSz, eModeInitImage=eModeInitImage::eMIA_NoInit);
        const Type & operator() (int aK) const {return DIm().GetV(aK);}
        Type & operator() (int aK) {return DIm().GetV(aK);}
        const int & Sz() const {return DIm().Sz();}

        double L1Dist(const cDenseVect<Type> & aV) const;
        double L2Dist(const cDenseVect<Type> & aV) const;

        Type * RawData();
        const Type * RawData() const;

        // No need to duplicate all services offered by Image Classe
        tDIM & DIm(){return mIm.DIm();}
        const tDIM & DIm() const {return mIm.DIm();}
        // operator -= 
    private :

        tIM mIm;
};

/* To come, sparse vector, will be vect<int> + vect<double> */

/** a Interface class , derived class will be :
       - dense matrix
       - various sparse matrix
*/
class cMatrix  : public cRect2
{
     public :

         virtual tMatrElem V_GetElem(int aX,int  aY) const = 0;
         virtual void  V_SetElem(int  aX,int  aY,const tMatrElem &) = 0;
         virtual eTyNums  TypeNum() const = 0;

         const cPt2di & Sz() const {return cRect2::Sz();}
         tMatrElem operator() (int aX,int  aY) const {return V_GetElem(aX,aY);} ///< Syntactic sugar

         // In place multiplication of Line/Col vector , furnish the result
         virtual void  MulColInPlace(cDenseVect<tREAL4> &,const cDenseVect<tREAL4> &) const;
         virtual tMatrElem MulColElem(int  aY,const cDenseVect<tREAL4> &)const;
         cDenseVect<tREAL4>  MulCol(const cDenseVect<tREAL4> &) const; ///< Create a new vector
         virtual void ReadColInPlace(int aX,cDenseVect<tREAL4>&) const;
         virtual void WriteCol(int aX,const cDenseVect<tREAL4>&) ;

         virtual void  MulLineInPlace(cDenseVect<tREAL4> &,const cDenseVect<tREAL4> &) const;
         virtual tMatrElem MulLineElem(int  aX,const cDenseVect<tREAL4> &)const;
         cDenseVect<tREAL4>  MulLine(const cDenseVect<tREAL4> &) const;
         virtual void ReadLineInPlace(int aY,cDenseVect<tREAL4>&) const;
         virtual void WriteLine(int aY,const cDenseVect<tREAL4>&) ;
                    
                     //  ============= tREAL8 ===============

         virtual void  MulColInPlace(cDenseVect<tREAL8> &,const cDenseVect<tREAL8> &) const;
         virtual tMatrElem MulColElem(int  aY,const cDenseVect<tREAL8> &)const;
         cDenseVect<tREAL8>  MulCol(const cDenseVect<tREAL8> &) const;
         virtual void ReadColInPlace(int aX,cDenseVect<tREAL8>&) const;
         virtual void WriteCol(int aX,const cDenseVect<tREAL8>&) ;

         virtual void  MulLineInPlace(cDenseVect<tREAL8> &,const cDenseVect<tREAL8> &) const;
         virtual tMatrElem MulLineElem(int  aX,const cDenseVect<tREAL8> &)const;
         cDenseVect<tREAL8>  MulLine(const cDenseVect<tREAL8> &) const;
         virtual void ReadLineInPlace(int aY,cDenseVect<tREAL8>&) const;
         virtual void WriteLine(int aY,const cDenseVect<tREAL8>&) ;

                     //  ============= tREAL16 ===============

         virtual void  MulColInPlace(cDenseVect<tREAL16> &,const cDenseVect<tREAL16> &) const;
         virtual tMatrElem MulColElem(int  aY,const cDenseVect<tREAL16> &)const;
         cDenseVect<tREAL16> MulLine(const cDenseVect<tREAL16> &) const;
         virtual void ReadColInPlace(int aX,cDenseVect<tREAL16>&) const;
         virtual void WriteCol(int aX,const cDenseVect<tREAL16>&) ;

         virtual void  MulLineInPlace(cDenseVect<tREAL16> &,const cDenseVect<tREAL16> &) const;
         virtual tMatrElem MulLineElem(int  aX,const cDenseVect<tREAL16> &)const;
         cDenseVect<tREAL16> MulCol(const cDenseVect<tREAL16> &) const;
         virtual void ReadLineInPlace(int aY,cDenseVect<tREAL16>&) const;
         virtual void WriteLine(int aY,const cDenseVect<tREAL16>&) ;



         //  Matrix multiplication
         void MatMulInPlace(const cMatrix & aM1,const cMatrix & aM2);
         // cMatrix  Mul(const cMatrix & aM2) const;


         // Multiplication of points, works for square matrix so that result is also a point,
         //  only for double now as the memory issue is not  crucial for points
/*
         virtual cPt2dr MulCol(const cPt2dr & aP) const;
         virtual cPt3dr MulCol(const cPt3dr & aP) const;
         virtual cPt2dr MulLine(const cPt2dr & aP) const;
         virtual cPt3dr MulLine(const cPt3dr & aP) const;
*/

              // Check size  inline so that they can be skipped in optimized mode
         /// Check that aVY * this  is valide
         template <class Type> void TplCheckSizeY(const cDenseVect<Type> & aV) const
         {
            MMVII_INTERNAL_ASSERT_medium(Sz().y()== aV.Sz(),"Bad size for vect column multiplication")
         }
         /// Check that this * aVx  is valide
         template <class Type> void TplCheckSizeX(const cDenseVect<Type> & aV) const
         {
            MMVII_INTERNAL_ASSERT_medium(Sz().x()== aV.Sz(),"Bad size for vect line multiplication")
         }
         /// Check that aVY * this * aVX  is valide, VY line vector of size SzY, VX Col vector
         template <class Type> void TplCheckSizeYandX(const cDenseVect<Type> & aVY,const cDenseVect<Type> & aVX) const
         {
              TplCheckSizeY(aVY);
              TplCheckSizeX(aVX);
         }

         ///  Check that aM1 * aM2 is valide
         static void CheckSizeMul(const cMatrix & aM1,const cMatrix & aM2)
         {
            MMVII_INTERNAL_ASSERT_medium(aM1.Sz().x()== aM2.Sz().y() ,"Bad size for mat multiplication")
         }
         ///  Check that this = aM1 * aM2 is valide
         void CheckSizeMulInPlace(const cMatrix & aM1,const cMatrix & aM2) const
         {
            CheckSizeMul(aM1,aM2);
            MMVII_INTERNAL_ASSERT_medium(aM1.Sz().y()== Sz().y() ,"Bad size for in place mat multiplication")
            MMVII_INTERNAL_ASSERT_medium(aM2.Sz().x()== Sz().x() ,"Bad size for in place mat multiplication")
         }
         static void CheckSquare(const cMatrix & aM1)
         {
            MMVII_INTERNAL_ASSERT_medium(aM1.Sz().x()== aM1.Sz().y() ,"Expected Square Matrix")
         }

      //  Constructor && destr
         virtual ~cMatrix();  ///< Public because called by shared ptr 
     protected :
         cMatrix(int aX,int aY);

     private :
};

template <class Type> cDenseVect<Type> operator * (const cDenseVect<Type> &,const cMatrix&);
template <class Type> cDenseVect<Type> operator * (const cMatrix&,const cDenseVect<Type> &);

/** Unopt dense Matrix are not very Usefull in "final" version
of  MMVII, but it's the opportunity to test the default method
implemanted in cMatrix,   It contains all the data of cDenseMatrix
 */


template <class Type> class cUnOptDenseMatrix : public cMatrix
{
    public :
        typedef cIm2D<Type> tIm;
        typedef cDataIm2D<Type> tDIm;
        const cPt2di & Sz() const {return cRect2::Sz();}

        cUnOptDenseMatrix(int aX,int aY,eModeInitImage=eModeInitImage::eMIA_NoInit);
        cUnOptDenseMatrix(tIm);

        /// Non virtual for inline without specifying scope cUnOptDenseMatrix::
        const Type & GetElem(int aX,int  aY) const { return DIm().GetV(cPt2di(aX,aY));}
        const Type & GetElem(const cPt2di & aP) const { return DIm().GetV(aP);}
        // Type & GetElem(int aX,int  aY) { return DIm().GetV(cPt2di(aX,aY));}
        // Type & GetElem(const cPt2di & aP) { return DIm().GetV(aP);}  => non implementee car pas de check val


        /// Non virtual for inline without specifying scope cUnOptDenseMatrix::
        void  SetElem(int  aX,int  aY,const Type & aV) { DIm().SetV(cPt2di(aX,aY),aV);}

        tMatrElem V_GetElem(int aX,int  aY) const override ;
        void  V_SetElem(int  aX,int  aY,const tMatrElem &) override;
        virtual eTyNums  TypeNum() const override;

       tDIm & DIm() {return mIm.DIm();}
       const tDIm & DIm() const {return mIm.DIm();}
       tIm & Im() {return mIm;}
       const tIm & Im() const {return mIm;}
   protected :

       tIm  mIm;  ///< Image that contains the data
};

template <class Type> class cResulSymEigenValue;
template <class Type> class cConst_EigenMatWrap;
template <class Type> class cNC_EigenMatWrap;

/**  Dense Matrix, probably one single class. 
     Targeted to be instantiated with 4-8-16 byte floating point
     It contains optimzed version of Mul Matrix*vect .  And all the
     algorithm specific to dense matrix decomposition
*/


template <class Type> class cDenseMatrix : public cUnOptDenseMatrix<Type>
{
   public :
        typedef cDataIm2D<Type> tDIm;
        typedef cIm2D<Type> tIm;

        typedef  cMatrix    tMat;
        typedef  cUnOptDenseMatrix<Type> tUO_DM;
        typedef  cDenseMatrix<Type>    tDM;

        typedef cConst_EigenMatWrap<Type> tConst_EW;
        typedef cNC_EigenMatWrap<Type> tNC_EW;


        const cPt2di & Sz() const {return cRect2::Sz();}
        cDenseMatrix(int aX,int aY,eModeInitImage=eModeInitImage::eMIA_NoInit);
        cDenseMatrix(int aX,eModeInitImage=eModeInitImage::eMIA_NoInit);
        cDenseMatrix(tIm);
        cDenseMatrix Dup() const;
        static cDenseMatrix Diag(const cDenseVect<Type> &);

        // To contourn borring new template scope ....
        const tDIm & DIm() const {return tUO_DM::DIm();}
        tDIm & DIm() {return tUO_DM::DIm();}
        tIm & Im() {return tUO_DM::Im();}
        const tIm & Im() const {return tUO_DM::Im();}
        const Type & GetElem(int aX,int  aY) const    { return tUO_DM::GetElem(cPt2di(aX,aY));}
        const Type & GetElem(const cPt2di & aP) const { return tUO_DM::GetElem(aP);}
        // Type & GetElem(int aX,int  aY)     { return tUO_DM::GetElem(cPt2di(aX,aY));}
        // Type & GetElem(const cPt2di & aP)  { return tUO_DM::GetElem(aP);}
        void  SetElem(int  aX,int  aY,const Type & aV) {  tUO_DM::SetElem(aX,aY,aV);}

        void Show() const;

        //  ====  Mul and inverse =========
              /** this does not override  cMatrix::MatMulInPlace as it is a specialization
                 to dense matrix, call eigen
              */
        void MatMulInPlace(const tDM & aM1,const tDM & aM2);

       
        tDM  Inverse() const;  ///< Basic inverse
        tDM  Inverse(double Eps,int aNbIter) const;  ///< N'amene rien, eigen fonctionne deja tres bien

        //  ====  Orthognal matrix

        double Unitarity() const; ///< test the fact that M is unatiry, basic : distance of Id to tM M
        cResulSymEigenValue<Type> SymEigenValue() const;

        //  ====  Symetricity/Transpose manipulation

        double Symetricity() const; ///< how much close to a symetrix matrix, square only , 
        void SelfSymetrize() ; ///< replace by closest  symetrix matrix, square only
        tDM    Symetrize() const ; ///< return closest  symetrix matrix, square only

        double AntiSymetricity() const; ///< how much close to a symetrix matrix, square only
        void SelfAntiSymetrize() ; ///< closest  symetrix matrix, square only
        tDM    AntiSymetrize() const ; ///< closest  symetrix matrix, square only

        void TransposeIn(tDM & M2) const;  ///< Put transposate in M2
        void SelfTransposeIn() ;  ///< transposate in this, square only
        tDM  Transpose() const;  ///< Put transposate in M2
         

        //  =====   Overridng of cMatrix classe  ==== 
        void  MulColInPlace(cDenseVect<tREAL4> &,const cDenseVect<tREAL4> &) const override;
        tMatrElem MulColElem(int  aY,const cDenseVect<tREAL4> &)const override;
        void  MulLineInPlace(cDenseVect<tREAL4> &,const cDenseVect<tREAL4> &) const override;
        tMatrElem MulLineElem(int  aX,const cDenseVect<tREAL4> &)const override;

        void  MulColInPlace(cDenseVect<tREAL8> &,const cDenseVect<tREAL8> &) const override;
        tMatrElem MulColElem(int  aY,const cDenseVect<tREAL8> &)const override;
        void  MulLineInPlace(cDenseVect<tREAL8> &,const cDenseVect<tREAL8> &) const override;
        tMatrElem MulLineElem(int  aX,const cDenseVect<tREAL8> &)const override;

        void  MulColInPlace(cDenseVect<tREAL16> &,const cDenseVect<tREAL16> &) const override;
        tMatrElem MulColElem(int  aY,const cDenseVect<tREAL16> &)const override;
        void  MulLineInPlace(cDenseVect<tREAL16> &,const cDenseVect<tREAL16> &) const override;
        tMatrElem MulLineElem(int  aX,const cDenseVect<tREAL16> &)const override;

};

template <class Type> class cResulSymEigenValue
{
    public :
        cResulSymEigenValue(int aNb);
        cDenseVect<Type>    mEVal;
        cDenseMatrix<Type>  mEVect;
};

template <class Type> cDenseMatrix<Type> operator * (const cDenseMatrix<Type> &,const cDenseMatrix<Type>&);
template <class T1,class T2> cDenseVect<T1> operator * (const cDenseVect<T1> &,const cDenseMatrix<T2>&);
template <class T1,class T2> cDenseVect<T1> operator * (const cDenseMatrix<T2>&,const cDenseVect<T1> &);


// Not usefull  as cUnOptDenseMatrix is not usefull either, but required in bench
template <class Type> cUnOptDenseMatrix<Type> operator * (const cUnOptDenseMatrix<Type> &,const cUnOptDenseMatrix<Type>&);
template <class T1,class T2> cDenseVect<T1> operator * (const cDenseVect<T1> &,const cUnOptDenseMatrix<T2>&);
template <class T1,class T2> cDenseVect<T1> operator * (const cUnOptDenseMatrix<T2>&,const cDenseVect<T1> &);

#if(0)
#endif


};

#endif  //  _MMVII_Matrix_H_
