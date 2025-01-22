#ifndef  _MMVII_Matrix_H_
#define  _MMVII_Matrix_H_

#include "MMVII_Image2D.h"

namespace MMVII
{

/** \file MMVII_Matrix.h
    \brief Classes for matrix manipulation, 
*/

/**   \file MMVII_Matrix.h

    Also algorithm will mainly use eigen,  storage will be 
    done by MMVII class
*/



template <class Type> class cSparseVect ;
template <class Type> class cDenseVect ;
template <class Type> class cMatrix  ;
template <class Type> class cUnOptDenseMatrix ;
template <class Type> class cDenseMatrix ;
template <class Type> class cLeasSqtAA ;

template <class Type> std::ostream & operator << (std::ostream & OS,const cDenseVect<Type> &aV);
template <class Type> std::ostream & operator << (std::ostream & OS,const cMatrix<Type> &aMat);

template <class Type> struct  cCplIV 
{
    public :
       cCplIV(const  int & aI,const Type & aV) : mInd(aI), mVal(aV) {}
       int   mInd;
       Type  mVal;
};

template <class Type> class  cSparseVect  : public cMemCheck
{
    public :

        typedef cSparseVect<Type>       tSV;
        typedef cCplIV<Type>            tCplIV;
        typedef std::vector<tCplIV>     tCont;
        typedef typename tCont::const_iterator   const_iterator;
        typedef typename tCont::iterator         iterator;

        const_iterator  end()   const { return  mIV.get()->end();}
        const_iterator  begin() const { return  mIV.get()->begin();}
        int  size() const {return mIV.get()->size();}
        const tCont & IV() const { return *(mIV.get());}
        tCont & IV() { return *(mIV.get());}

        void AddIV(const int & anInd,const Type & aV) ;   /// "Raw" add, dont check if ind exist
        void AddIV(const tCplIV & aCpl) ; /// "Raw" add, dont check if ind exist
        void CumulIV(const tCplIV & aCpl) ; /// Create only if not exist, else add in place

        double DotProduct(const cDenseVect<Type> &) const; //== scalar product
	/// Random sparse vector
        static cSparseVect<Type>  RanGenerate(int aNbVar,double aProba,tREAL8 aMinVal= 1e-2,int aMinSize=1);

        // generate NbVect of dimension NbVar with average density a Proba, assuring that in all vector we have
        //  dist(Uk, < aCosMax
        // static std::list<cSparseVect<Type> >  GenerateKVect(int aNbVar,int aNbVect,double aProba,tREAL8 aDMin);

        /// SzInit fill with arbitray value, only to reserve space
        // cSparseVect(int aSzReserve=-1,int aSzInit=-1) ;  
        cSparseVect(int aSzReserve=-1) ;  
        cSparseVect(const cDenseVect<Type> &);
	/// Check the vector can be used in a matrix,vect [0,Nb[, used in the assertions
        bool IsInside(int aNb) const;
	void Reset();

        const tCplIV  * Find(int anInd) const;  /// return the pair of a given index
        tCplIV  * Find(int anInd) ;  /// return the pair of a given index

        // Maximum index, aDef is used if empty, if aDef<=-2  & empty erreur
        int MaxIndex(int aDef=-1) const;

        void EraseIndex(int anInd);
        /// Create a real duplicata, as copy-constructor return the same shared ptr
        tSV  Dup() const;
    private :
	/*
         inline void MakeSort(){if (!mIsSorted) Sort();}
         void Sort();
	 bool                           mIsSorted;
	 */
         std::shared_ptr<tCont>         mIV;
};


/** A dense vector is no more than a 1D Image, but with a different interface */

template <class Type> class  cDenseVect 
{
    public :
        typedef cIm1D<Type>  tIM;
        typedef cDataIm1D<Type>      tDIM;
        typedef cSparseVect<Type> tSpV;
        typedef cDenseVect<Type> tDV;

        cDenseVect(int aSz, eModeInitImage=eModeInitImage::eMIA_NoInit);
        cDenseVect(tIM anIm);
        cDenseVect(const std::vector<Type> & aVect);
        //  Adapt size , set
        cDenseVect(const tSpV &,int aSz=-1);
        static cDenseVect<Type>  Cste(int aSz,const Type & aVal);
        cDenseVect<Type>  Dup() const;
        static cDenseVect<Type>  RanGenerate(int aNbVar);
	/// 
        void ResizeAndCropIn(const int & aX0,const int & aX1,const cDenseVect<Type> &);
        void Resize(const int & aSz);
	/// Create a sub vector in interval [K0,K1[, its a duplication
        cDenseVect  SubVect(int aK0,int aK1) const;

        const Type & operator() (int aK) const {return DIm().GetV(aK);}
        Type & operator() (int aK) {return DIm().GetV(aK);}
        const int & Sz() const {return DIm().Sz();}

        // For vector/matrix it's more standard than norm are a sum and not an average
        double L1Dist(const cDenseVect<Type> & aV,bool Avg=false) const;
        double L2Dist(const cDenseVect<Type> & aV,bool Avg=false) const;

        double L1Norm(bool Avg=false) const;   ///< Norm som abs
        double L2Norm(bool Avg=false) const;   ///< Norm square
        double SqL2Norm(bool Avg=false) const;   ///< Norm square
        double LInfNorm() const; ///< Nomr max

        tDV  VecUnit() const;  // return V/|V|

        Type * RawData();
        const Type * RawData() const;
        std::vector<Type> ToStdVect() const;  ///< duplication

        // No need to duplicate all services offered by Image Classe
        tDIM & DIm(){return mIm.DIm();}
        const tDIM & DIm() const {return mIm.DIm();}

        tIM & Im(){return mIm;}
        const tIM & Im() const {return mIm;}

        Type ProdElem() const; ///< Mul of all element, usefull for det computation
        Type SumElem() const; ///< Som of all element
        Type AvgElem() const; ///< Avereage of all elements
        void SetAvg(const Type & anAvg); ///< multiply by a cste to fix the average


        // operator -= 
        double DotProduct(const cDenseVect &) const; //== scalar product
        void TplCheck(const tSpV & aV)  const
        {
            MMVII_INTERNAL_ASSERT_medium(aV.IsInside(Sz()) ,"Sparse Vector out dense vect");
        }
        void  WeightedAddIn(Type aWeight,const tSpV & aColLine);
        void  WeightedAddIn(Type aWeight,const tDV & aColLine);

           /*  =========  Othognalization & projection stuff =========== */

                        //   ----------  Projection, Dist to space ------------------
        /// return orthognal projection on subspace defined by aVVect, use least square (slow ? At least good enough for bench )
        tDV    ProjOnSubspace(const std::vector<tDV>  & aVVect) const;
        /// return distance to subspace (i.e distance to proj)
        Type   DistToSubspace(const std::vector<tDV>  &) const;

	///  Theoretically max min distance, but before I find an exact algorithm, just the max on all vect , compute 2 way
	static  Type ApproxDistBetweenSubspace(const std::vector<tDV>  &,const std::vector<tDV>  &);


                       // --------------  Gram schmitd method for orthogonalization -------------------
        /**  Elementary step of Gram-Schmit orthogonalization method ;  return a vector orthogonal
             to all VV and that belong to the space "this+aVV", assumme aVV are already orthogonal */
        tDV  GramSchmidtCompletion(const std::vector<tDV> & aVV) const;
        /// full method of gram schmidt to orthogonalize
        static std::vector<tDV>  GramSchmidtOrthogonalization(const std::vector<tDV> & aVV) ;

                // ----------------  Base complementation in orthogonal subspace, slow but dont require initial base orthog --------
         /// Return a unitary vector not colinear to VV, by iteration then randomization, untill the Dist to subspace > DMin
        static tDV  VecComplem(const std::vector<tDV> & aVV,Type DMin=0.1) ;
        /// Complement the base with vector orthogonal to the base, and orthog between them (if WithInit contain initial vect + added)
        static std::vector<tDV>  BaseComplem(const std::vector<tDV> & aVV,bool WithInit=false,Type DMin=0.1) ;

    private :

	static  Type ASymApproxDistBetweenSubspace(const std::vector<tDV>  &,const std::vector<tDV>  &);
        tIM mIm;
};

/// return a vectot with avg=0 and som sq = 1
cDenseVect<tREAL8> NormalizeMoyVar(const cDenseVect<tREAL8> & aV0,tREAL8 aEpsilon = 1e-5);

///  fit the equation B +A Vx  = Y, return  [A,B]
std::pair<tREAL8,tREAL8> LstSq_Fit_AxPBEqY(const cDenseVect<tREAL8> & aVX,const cDenseVect<tREAL8> & aVY);

// return A X + B ....
cDenseVect<tREAL8> MulAXPB(const cDenseVect<tREAL8> & , tREAL8 A,tREAL8 B);


/* To come, sparse vector, will be vect<int> + vect<double> */

/** a Interface class , derived class will be :
       - dense matrix
       - various sparse matrix
*/
template <class Type> class cMatrix  : public cRect2
{
     public :
         typedef cSparseVect<Type> tSpV;
         typedef Type              tVal;
         typedef cDenseVect<Type>  tDV;
         typedef cMatrix<Type>     tMat;

         virtual Type V_GetElem(int aX,int  aY) const = 0;
         virtual void  V_SetElem(int  aX,int  aY,const Type &) = 0;
         virtual eTyNums  TypeNum() const = 0;

         const cPt2di & Sz() const {return cRect2::Sz();}
         Type operator() (int aX,int  aY) const {return V_GetElem(aX,aY);} ///< Syntactic sugar


         //==== Put here to make bench easier, but else not very usefull
         virtual double TriangSupicity() const ; ///< How much is  it triangular sup
         virtual void SelfSymetrizeBottom() ;    ///< Symetrize by setting copying up in Bottom

                     //  ============= tREAL4 ===============
         virtual void  Add_tAB(const tDV & aCol,const tDV & aLine) ;
         virtual void  Add_tAA(const tDV & aColLine,bool OnlySup=true) ;
         virtual void  Sub_tAA(const tDV & aColLine,bool OnlySup=true) ;
         virtual void  Weighted_Add_tAA(Type aWeight,const tDV & aColLine,bool OnlySup=true) ;

         // Column operation
         virtual void  MulColInPlace(tDV &,const tDV &) const;
         virtual Type MulColElem(int  aY,const tDV &)const;
         tDV  MulCol(const tDV &) const; ///< Create a new vector
         virtual void ReadColInPlace(int aX,tDV &) const;
         virtual tDV  ReadCol(int aX) const;
         virtual void WriteCol(int aX,const tDV &) ;

         // Line operation
         virtual void  MulLineInPlace(tDV &,const tDV &) const;
         virtual Type MulLineElem(int  aX,const tDV &)const;
         tDV  MulLine(const tDV &) const;
         virtual void ReadLineInPlace(int aY,tDV &) const;
         virtual tDV ReadLine(int aY) const;
         virtual void WriteLine(int aY,const tDV &) ;
         std::vector<tDV>  MakeLines() const;  // generate all the lines



         //  Matrix multiplication
         void MatMulInPlace(const tMat & aM1,const tMat & aM2);


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
         void TplCheckSizeY(const tDV & aV) const
         {
            MMVII_INTERNAL_ASSERT_medium(Sz().y()== aV.Sz(),"Bad size for vect column multiplication")
         }
         /// Check that this * aVx  is valide
         void TplCheckSizeX(const tDV & aV) const
         {
            MMVII_INTERNAL_ASSERT_medium(Sz().x()== aV.Sz(),"Bad size for vect line multiplication")
         }
         /// Check that aVY * this * aVX  is valide, VY line vector of size SzY, VX Col vector
         void TplCheckSizeYandX(const tDV & aVY,const tDV & aVX) const
         {
              TplCheckSizeY(aVY);
              TplCheckSizeX(aVX);
         }

        // void TplCheckX(const cSparseVect & V) {}; template <class Type> class  cSparseVect 
         ///  Check that aM1 * aM2 is valide
      
         void TplCheckX(const cSparseVect<Type> & aV)  const
         {
            MMVII_INTERNAL_ASSERT_medium(aV.IsInside(Sz().x()) ,"Sparse Vector X-out matrix");
         }
         void TplCheckY(const cSparseVect<Type> & aV)  const
         {
            MMVII_INTERNAL_ASSERT_medium(aV.IsInside(Sz().y()) ,"Sparse Vector Y-out matrix");
         }

         static void CheckSizeMul(const tMat & aM1,const tMat & aM2)
         {
            MMVII_INTERNAL_ASSERT_medium(aM1.Sz().x()== aM2.Sz().y() ,"Bad size for mat multiplication")
         }
         ///  Check that this = aM1 * aM2 is valide
         void CheckSizeMulInPlace(const tMat & aM1,const tMat & aM2) const
         {
            CheckSizeMul(aM1,aM2);
            MMVII_INTERNAL_ASSERT_medium(aM1.Sz().y()== Sz().y() ,"Bad size for in place mat multiplication")
            MMVII_INTERNAL_ASSERT_medium(aM2.Sz().x()== Sz().x() ,"Bad size for in place mat multiplication")
         }
         static void CheckSquare(const tMat & aM1)
         {
            MMVII_INTERNAL_ASSERT_medium(aM1.Sz().x()== aM1.Sz().y() ,"Expected Square Matrix")
         }


      //  Sparse vector
        virtual void  Weighted_Add_tAA(Type aWeight,const tSpV & aColLine,bool OnlySup=true);

      //  Constructor && destr
         virtual ~cMatrix() = default;  ///< Public because called by shared ptr 
     protected :
         cMatrix(int aX,int aY);

     private :
};



template <class Type> cDenseVect<Type> operator * (const cDenseVect<Type> &,const cMatrix<Type>&);
template <class Type> cDenseVect<Type> operator * (const cMatrix<Type>&,const cDenseVect<Type> &);

/** Unopt dense Matrix are not very Usefull in "final" version
of  MMVII, but it's the opportunity to test the default method
implemanted in cMatrix,   It contains all the data of cDenseMatrix
 */


template <class Type> class cUnOptDenseMatrix : public cMatrix<Type>
{
    public :
        typedef cIm2D<Type> tIm;
        typedef cDataIm2D<Type> tDIm;
        const cPt2di & Sz() const {return cRect2::Sz();}
        void Resize(const cPt2di &);
        void ResizeAndCropIn(const cPt2di & aP0,const cPt2di & aP1,const cUnOptDenseMatrix<Type> &);

        cUnOptDenseMatrix(int aX,int aY,eModeInitImage=eModeInitImage::eMIA_NoInit);
        cUnOptDenseMatrix(tIm);

        /// Non virtual for inline without specifying scope cUnOptDenseMatrix::
        const Type & GetElem(int aX,int  aY) const { return DIm().GetV(cPt2di(aX,aY));}
        const Type & GetElem(const cPt2di & aP) const { return DIm().GetV(aP);}
        // Type & GetElem(int aX,int  aY) { return DIm().GetV(cPt2di(aX,aY));}
        // Type & GetElem(const cPt2di & aP) { return DIm().GetV(aP);}  => non implementee car pas de check val


        /// Non virtual for inline without specifying scope cUnOptDenseMatrix::
        void  SetElem(int  aX,int  aY,const Type & aV) { DIm().SetV(cPt2di(aX,aY),aV);}
        void  AddElem(int  aX,int  aY,const Type & aV) { DIm().AddVal(cPt2di(aX,aY),aV);}

        Type V_GetElem(int aX,int  aY) const override ;
        void  V_SetElem(int  aX,int  aY,const Type &) override;
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
template <class Type> class cStrStat2;
template <class Type> class cNC_EigenMatWrap;
template <class Type> class cResulQR_Decomp;
template <class Type> class cResulRQ_Decomp;
template <class Type> class cResulSVDDecomp;
template <class Type> class cResulEigenDecomp;


/**  Dense Matrix, probably one single class. 
     Targeted to be instantiated with 4-8-16 byte floating point
     It contains optimzed version of Mul Matrix*vect .  And all the
     algorithm specific to dense matrix decomposition
*/


template <class Type> class cDenseMatrix : public cUnOptDenseMatrix<Type>
{
   public :
        typedef Type            tVal;
        typedef cSparseVect<Type> tSpV;
        typedef cDenseVect<Type> tDV;
        typedef cDataIm2D<Type> tDIm;
        typedef cIm2D<Type> tIm;
        typedef cResulSVDDecomp<Type> tRSVD;

        typedef  cMatrix<Type>           tMat;
        typedef  cUnOptDenseMatrix<Type> tUO_DM;
        typedef  cDenseMatrix<Type>      tDM;

        typedef cConst_EigenMatWrap<Type> tConst_EW;
        typedef cNC_EigenMatWrap<Type> tNC_EW;


	tDM  ExtendSquareMat     (int aNewSz,eModeInitImage); ///< Create a square matrix include smaller, mode of extension specified
	tDM  ExtendSquareMatId   (int aNewSz);   ///<  Extension with identity
	tDM  ExtendSquareMatNull (int aNewSz);   ///<  Extension with 0

        const cPt2di & Sz() const {return cRect2::Sz();}
        cDenseMatrix(int aX,int aY,eModeInitImage=eModeInitImage::eMIA_NoInit);
        cDenseMatrix(int aX,eModeInitImage=eModeInitImage::eMIA_NoInit);  ///< Square
        cDenseMatrix(tIm);
        cDenseMatrix Dup() const;
        static cDenseMatrix Identity(int aSz);  ///< return identity matrix
        static cDenseMatrix Diag(const tDV &);
        static cDenseMatrix FromLines(const std::vector<tDV> &);  // Create from set of "line vector"
        cDenseMatrix ClosestOrthog() const;  ///< return closest 

        tDM SubMatrix(const cPt2di & aSz) const;
        tDM SubMatrix(const cPt2di & aP0,const cPt2di & aP1) const;

        /**  Generate a random square matrix having "good" conditioning property , i.e with eigen value constraint,
            usefull for bench as when the random matrix is close to singular, it may instability that fail
            the numerical test.
        */
        static tDM RandomSquareRegMatrix(const cPt2di&aSz,bool IsSym,double aAmplAcc,double aCondMinAccept);
        static tRSVD RandomSquareRegSVD(const cPt2di&aSz,bool IsSym,double aAmplAcc,double aCondMinAccept);


        static tDM RandomOrthogMatrix(const int aSz);

        /* Generate a matrix rank deficient, where aSzK is the size of the kernel */
        static tRSVD RandomSquareRankDefSVD(const cPt2di & aSz,int aSzK);
        static tDM RandomSquareRankDefMatrix(const cPt2di & aSz,int aSzK);

        /** Compute the kernel of the matrix, due to numeric rounding, we dont have exactly M*K=0,
            so VP indicate how close we are from that, made using SVD */
        tDV Kernel(Type * aVP=nullptr) const;
        /// More general version of Kernel, apply to any eigen value
        tDV EigenVect(const Type & aVal,Type * aVP=nullptr) const;

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
        void  AddElem(int  aX,int  aY,const Type & aV) {  tUO_DM::AddElem(aX,aY,aV);}

	void PushByLine(std::vector<Type> &) const; /// write all the matrix, do it line-first
	void PushByCol(std::vector<Type> &) const; /// write all the matrix, do it Colum-first

        void Show() const;

        //  ====  Mul and inverse =========
              /** this does not override  cMatrix::MatMulInPlace as it is a specialization
                 to dense matrix, call eigen
              */
        void MatMulInPlace(const tDM & aM1,const tDM & aM2);

       
        // tDM  SymInverse() const;  ///< Inverse of symetric matrix
	void InverseInPlace(const tDM & aM);  ///< Put M-1 in this
        tDM  Inverse() const;  ///< Basic inverse
        tDM  Inverse(double Eps,int aNbIter) const;  ///< N'amene rien, eigen fonctionne deja tres bien en general 

        void  SolveIn(tDM& aRes,const tDM &,eTyEigenDec aType=eTyEigenDec::eTED_PHQR) const;
        tDM  Solve(const tDM &,eTyEigenDec aType=eTyEigenDec::eTED_PHQR) const;
        tDV  SolveColumn(const tDV &,eTyEigenDec aType=eTyEigenDec::eTED_PHQR) const;
        tDV  SolveLine(const tDV &,eTyEigenDec aType=eTyEigenDec::eTED_PHQR) const;


        //  ====  Orthognal matrix

        double Unitarity() const; ///< test the fact that M is unatiry, basic : distance of Id to tM M
        cResulSymEigenValue<Type> SymEigenValue() const;
        tRSVD  SVD() const;

        cResulQR_Decomp<Type>    QR_Decomposition() const;
        cResulRQ_Decomp<Type>    RQ_Decomposition() const;
        cResulEigenDecomp<Type>  Eigen_Decomposition() const;


        //  ====  Symetricity/Transpose/Triangularise manipulation

        double Symetricity() const; ///< how much close to a symetrix matrix, square only , 
        void SelfSymetrize() ; ///< replace by closest  symetrix matrix, square only
        tDM    Symetrize() const ; ///< return closest  symetrix matrix, square only

        double AntiSymetricity() const; ///< how much close to a symetrix matrix, square only
        void SelfAntiSymetrize() ; ///< closest  symetrix matrix, square only
        tDM    AntiSymetrize() const ; ///< closest  symetrix matrix, square only

        void   SelfTriangSup();         ///< Make the image triangular
        double TriangSupicity() const override;        ///< How much is  it triangular sup
        void SelfSymetrizeBottom() override ; ///< Symetrize by setting copying up in Bottom

        void TransposeIn(tDM & M2) const;  ///< Put transposate in M2
        void SelfTransposeIn() ;  ///< transposate in this, square only
        tDM  Transpose() const;  ///< Put transposate in M2

	  
        void SelfLineInverse() ;  ///< line inversion   (L1 L2 .. LN) =>   (LN ... L2 L1) ,  used for RQ decomposition (QR => RQ)
        tDM  LineInverse() const;  ///< cont version of SelfLineInverse
        void SelfColInverse() ;  ///< colum inversion   ,  used for RQ decomposition (QR => RQ)

	void SelfLineChSign(int aNumL); ///<  chang signe of line aNumL, used in QR/RQ to normalize with diag>0 of R
	void SelfColChSign(int aNumC);  ///<  chang signe of column aNumC, used in QR/RQ to normalize with diag>0 of R
         
        double Diagonalicity() const; ///< how much close to a diagonal matrix, square only , 
        Type   Det() const;  ///< compute the determinant, not sur optimise

        void ChangSign(); ///< Multiply by -1
        void SetDirectBySign(); ///< Multiply by -1 if indirect

         tDV  Random1LineCombination() const; // retunr a vector that is a random combination of lines
        //  =====   Overridng of cMatrix classe  ==== 
        void  MulColInPlace(tDV &,const tDV &) const override;
        Type MulColElem(int  aY,const tDV &)const override;
        void  MulLineInPlace(tDV &,const tDV &) const override;
        Type MulLineElem(int  aX,const tDV &)const override;
        void  Add_tAB(const tDV & aCol,const tDV & aLine) override;
        void  Add_tAA(const tDV & aColLine,bool OnlySup=true) override;
        void  Sub_tAA(const tDV & aColLine,bool OnlySup=true) override;

        void  Weighted_Add_tAA(Type aWeight,const tDV & aColLine,bool OnlySup=true) override;

        // ====  Sparse vector 
        void  Weighted_Add_tAA(Type aWeight,const tSpV & aColLine,bool OnlySup=true) override;

        // === method implemente with DIm
        Type L2Dist(const cDenseMatrix<Type> & aV,bool Avg=false) const;
        Type SqL2Dist(const cDenseMatrix<Type> & aV,bool Avg=false) const;
	//  void operator -= (const cDenseMatrix<Type> &) ;  => see  "include/MMVII_Tpl_Images.h"

   private :
        Type & GetReference_V(int aX,int  aY) { return DIm().GetReference_V(cPt2di(aX,aY));}  // dont check vals

};

template <class Type> class cResulEigenDecomp
{
      public :
         cResulEigenDecomp<Type>(int aN);
         cDenseMatrix<Type>  mEigenVec_R;  ///< real part
         cDenseMatrix<Type>  mEigenVec_I;  ///< imaginary part

         cDenseVect<Type>    mEigenVal_R;  ///<
         cDenseVect<Type>    mEigenVal_I;  ///<
};



template <class Type> class cResulSymEigenValue
{
    public :
        friend class cDenseMatrix<Type>;
        friend class cStrStat2<Type>;

        cDenseMatrix<Type>  OriMatr() const; ///< Check the avability to reconstruct original matrix
	  // =>  mEigenVectors * cDenseMatrix<Type>::Diag(mEigenValues) * mEigenVectors.Transpose();


        const cDenseVect<Type>   &  EigenValues() const ; ///< Eigen values, in growing order !!! != cResulSVDDecomp
        const cDenseMatrix<Type> &  EigenVectors()const ; ///< Eigen vector
        void  SetKthEigenValue(int aK,const Type & aVal) ;  ///< Eigen values
        Type  Cond(Type Def=Type(-1)) const ; ///< Conditioning, def value is when all 0, if all0 and Def<0 : Error

    private :
        cResulSymEigenValue(int aNb);
        cDenseVect<Type>    mEigenValues;  ///< Eigen values
        cDenseMatrix<Type>  mEigenVectors; ///< Eigen vector
};


template <class Type> class cResulSVDDecomp
{
    public :
        friend class cDenseMatrix<Type>;

        cDenseMatrix<Type>  OriMatr() const; ///< Check the avability to reconstruct original matrix   

        const cDenseVect<Type>   &  SingularValues() const ; ///< Eigen values, in decreasing order !!! != cResulSymEigenValue
        const cDenseMatrix<Type> &  MatU()const ; ///< Eigen vector
        const cDenseMatrix<Type> &  MatV()const ; ///< Eigen vector
        // void  SetKthEigenValue(int aK,const Type & aVal) ;  ///< Eigen values
        // Type  Cond(Type Def=Type(-1)) const ; ///< Conditioning, def value is when all 0, if all0 and Def<0 : Error

    private :
        cResulSVDDecomp(int aNb);
        cDenseVect<Type>    mSingularValues;  ///< Eigen values
        cDenseMatrix<Type>  mMatU; ///< Eigen vector
        cDenseMatrix<Type>  mMatV; ///< Eigen vector

};

template <class Type> class cResulQR_Decomp
{
    public :
        typedef cDenseMatrix<Type> tDM;
        // friend class tDM;
        friend tDM;

        cDenseMatrix<Type>  OriMatr() const;  // Q * R

        const tDM &  Q_Matrix() const; ///< Unitary
        const tDM &  R_Matrix() const; ///< Triang
        tDM &  Q_Matrix() ; ///< Unitary
        tDM &  R_Matrix() ; ///< Triang

    private :
        cResulQR_Decomp(int aSzX,int aSzY);
    protected :
        cResulQR_Decomp(const tDM&,const tDM &);

        tDM  mQ_Matrix; ///< Unitary Matrix
        tDM  mR_Matrix; ///< Triangular superior

};

template <class Type> class cResulRQ_Decomp : public cResulQR_Decomp<Type>
{
    public :
        typedef cDenseMatrix<Type> tDM;
        //friend class tDM;
        friend tDM;

        tDM  OriMatr() const;  // R * Q

    private :
        cResulRQ_Decomp(const tDM&,const tDM &);
};


/// Auxialry class for cDecSumSqLinear

template <class Type>  class cElemDecompQuad
{
     public :
         cElemDecompQuad(const Type& aW,const cDenseVect<Type> & aV,const Type & aCste);
         Type              mW;  // weight
         cDenseVect<Type>  mCoeff;
         Type              mCste;  // weight
};
/**  Class to decompose a positive quadratic form a sum of square of linear form :

     input A,B,X0  => out  (Wi Li  Ci), such that :
      t(X-AX0) A (X-X0)  - 2tB (X-X0) + tBB =  Sum ( Wi (Li X-Ci))^2 
      Li are norm 1 and orthogonal to each others

     Generally B will be equal to 0 , as the elements commes from minimization where the 
     value are stored relativelt to current solution, so B~0 at stability ...
*/

template <class Type>  class cDecSumSqLinear
{
     public :
         typedef cElemDecompQuad<Type>  tElem;
         void  Set(const cDenseVect<Type> & aX0,const cDenseMatrix<Type> & aMat,const cDenseVect<Type> & aVectB);
         cDecSumSqLinear();

         // recover initial system (check/Bench)
         cLeasSqtAA<Type>  OriSys() const;
         const std::vector<tElem> &   VElems() const;
     private :
         int                  mNbVar;
         std::vector<tElem>   mVElems;

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

/**  Class to calculate/manipulate 1-2 moment : aver/var/cov
*/
template <class Type> class cStrStat2
{
    public :
       cStrStat2(int aSz);
       /// Add a vectors to stats
       void Add(const cDenseVect<Type> & );
       /// Make average (instead of sums) and centered (for cov)
       void Normalise(bool CenteredAlso=true);
       ///  Compute eigen values
       const cResulSymEigenValue<Type> & DoEigen();
       /// Coordinate in the orthognal system were aver=0 and variables are uncorralted
       void ToNormalizedCoord(cDenseVect<Type>  & aV1,const cDenseVect<Type>  & aV2) const;
       /// Kth Coordinate of previous
       double KthNormalizedCoord(int,const cDenseVect<Type>  & aV2) const;
       // Accessors
       cDenseMatrix<Type>& Cov() ;
       double              Pds() const;
       const cDenseVect<Type>  & Moy() const;
       const cDenseMatrix<Type>& Cov() const;
    private :
       int                       mSz;  ///<  Size
       double                    mPds; ///< Sum of weight (later we will have possibility to add weight)
       cDenseVect<Type>          mMoy; ///< Som/average
       cDenseVect<Type>          mMoyMulVE; ///< Moy * EigenVect
       mutable cDenseVect<Type>  mTmp; ///< Use as temporary for some computation
       cDenseMatrix<Type>        mCov;  ///< Cov Matrix
       cResulSymEigenValue<Type> mEigen;  ///< Eigen/Value vectors stored here after DoEigen
};


/** More a less special case to  cStrStat2 for case 2 variable, very current and
probably much more efficient */

template <class Type> class cMatIner2Var
{
    public :
       cMatIner2Var ();
       cMatIner2Var(const cMatIner2Var<Type> &) = default;
       void Add(const double & aPds,const Type & aV1,const Type & aV2);
       void Add(const Type & aV1,const Type & aV2);
       const Type & S0()  const {return mS0;}
       const Type & S1()  const {return mS1;}
       const Type & S11() const {return mS11;}
       const Type & S2()  const {return mS2;}
       const Type & S12() const {return mS12;}
       const Type & S22() const {return mS22;}
       void Normalize();
       void Add(const cMatIner2Var&);
       void Add(const cMatIner2Var&,const Type & aMul) ;
       Type Correl(const Type &aEpsilon=1e-10) const;
       Type CorrelNotC(const Type &aEpsilon=1e-10) const; // Non centered correl
       Type StdDev1() const;
       Type StdDev2() const;

       /// [A B] as least-square solution of V1 = A V2 + B
       std::pair<Type,Type> FitLineDirect() const;  
    private :
        Type  mS0;   ///< Som of    W
        Type  mS1;   ///< Som of    W * V1
        Type  mS11;  ///< Som of    W * V1 * V1
        Type  mS2;   ///< Som of    W * V2
        Type  mS12;  ///< Som of    W * V1 * V2
        Type  mS22;  ///< Som of    W * V2 * V2
};

/** Class for averaging with weight */
template <class TypeWeight,class TypeVal=TypeWeight> class cWeightAv
{
     public :
        cWeightAv();
        cWeightAv(const std::vector<TypeVal> &);
	static TypeVal AvgCst(const std::vector<TypeVal> &);


        void Add(const TypeWeight & aWeight,const TypeVal & aVal);
        TypeVal Average() const;
        TypeVal Average(const TypeVal  & aDef) const;
        const TypeVal & SVW() const;  /// Accessor to sum weighted vals
        const TypeWeight & SW() const;  /// Accessor to sum weighted vals
        void  Reset();
    private :
        TypeWeight  mSW;   ///< Som of    W
        TypeVal     mSVW;   ///< Som of    VW
};

/** Class for making standard star on residuals */
class cStdStatRes
{
     public :
        cStdStatRes();

        void Add(tREAL8 aVal);

        tREAL8  Avg() const;
        tREAL8  QuadAvg() const;
        tREAL8  DevStd() const;
        tREAL8  ErrAtProp(tREAL8 aProp) const;
        tREAL8  Min() const;
        tREAL8  Max() const;
	int     NbMeasures() const;

     private :
        mutable std::vector<tREAL8>       mVRes;
        cWeightAv<tREAL8,tREAL8>  mAvgDist;
        cWeightAv<tREAL8,tREAL8>  mAvgDist2;
        cBoundVals<tREAL8>        mBounds;
};


/**  Class for robust average   avg weithgted by   sigma/(sigma+|aRes|) */
class cRobustAvg
{
      public :
         cRobustAvg(tREAL8 aSigma) ;
         void Add(tREAL8 aVal);
	 tREAL8 Average() const;
      public :
         tREAL8                    mSigma;
         tREAL8                    mS2;
	 cWeightAv<tREAL8,tREAL8>  mAvg;
};

/**  Class for robust average with a sigma computed  from proportion */
class cRobustAvgOfProp
{
      public :
         cRobustAvgOfProp();
         void Add(tREAL8 aVal);
	 tREAL8 Average(tREAL8 aProp) const;
      public :
	 std::vector<double>  mVals;
};


/// A function rather specific to bench, assimilate image to a distribution on var X,Y and compute it 0,1,2 moments
template <class Type> cMatIner2Var<double> StatFromImageDist(const cDataIm2D<Type> & aIm);

///  BUGED:  Class to compute non biased variance from a statisic 

/** Class to compute non biased variance from a statisic
    This generalise the standard formula
            EstimVar = N/(N-1) EmpirVar
    To the case where there is a weighting.

    It can be uses with N variable to factorize the computation on Weight

     The method id BUGED as the theoreticall formula is probably wrong, or maybe it is the bench itself ?
     However it's probably worth use it than nothing ...
 */

template <const int Dim> class cUB_ComputeStdDev
{
    public :
        typedef  double tTab[Dim];

        cUB_ComputeStdDev();

        void Add(const  double * aVal,const double & aPds);
        const double  *  ComputeUnBiasedVar() ;  // return a pointer because of Dim
        const double  *  ComputeBiasedVar() ;
        double  DeBiasFactor() const;

        bool OkForUnBiasedVar() const; ///< Test DeBiasFactor can be computed and is != 0

    private :
        double    mSomW; ///< Sum of Weight
        double    mSomWW; ///< Sum of Weight ^2
        tTab      mSomWV;  ///< Weighted som of vals
        tTab      mSomWVV; ///< Weighted som of vals ^2
        tTab      mVar;   ///< Buffer to compute the unbiased variance
        tTab      mBVar;   ///< Buffer to compute the empirical variance
};

template <class Type>  class cComputeStdDev
{
     public :
         cComputeStdDev();
         void  Add(const Type & aW,const Type & aV)
         {
             mSomW   += aW;
             mSomWV  += aW *aV;
             mSomWV2 += aW * Square(aV);
         }
         void  Add(const Type & aV)
         {
             mSomW   ++;
             mSomWV  += aV;
             mSomWV2 += Square(aV);
         }

         const Type & SomW()   const {return mSomW;  }
         const Type & SomWV()  const {return mSomWV; }
         const Type & SomWV2() const {return mSomWV2;}
         Type  NormalizedVal(const Type &) const;
         cComputeStdDev<Type>  Normalize(const Type & Epsilon = 0.0) const;
	 Type  StdDev(const Type & Epsilon = 0.0) const;
         void  SelfNormalize(const Type & Epsilon = 0.0);
     private :
         Type mSomW; 
         Type mSomWV; 
         Type mSomWV2; 
         Type mStdDev; 
};

template<class Type> class cSymMeasure
{
    public :
        cSymMeasure();
        void Add(const Type  & aV1,const Type  & aV2)
        {
            mDif += Square(aV1-aV2);
            mDev.Add(aV1);
            mDev.Add(aV2);
        }
        void Add(const Type  & aW,const Type  & aV1,const Type  & aV2)
        {
            mDif += aW * Square(aV1-aV2);
            mDev.Add(aW,aV1);
            mDev.Add(aW,aV2);
        }
        Type  Sym(const Type & Espilon=1) const;
        const cComputeStdDev<Type > & Dev() const ;
    private :
        Type                   mDif;
        cComputeStdDev<Type >  mDev;
};


/* ============================================== */
/*                                                */
/*         Point/Matrix                           */
/*                                                */
/* ============================================== */

// Operation with points,  as points have fixed size contrarily to matrix, the syntax is not
// as fluent as I would like

#define CHECK_SZMAT_COL(aMAT,aPT) MMVII_INTERNAL_ASSERT_tiny(aMAT.Sz().y()==aPT.TheDim,"Bad size in Col Pt")
#define CHECK_SZMAT_LINE(aMAT,aPT) MMVII_INTERNAL_ASSERT_tiny(aMAT.Sz().x()==aPT.TheDim,"Bad size in Line Pt")
#define CHECK_SZMAT_SQ(aMAT,aPT) MMVII_INTERNAL_ASSERT_tiny((aMAT.Sz().x()==aPT.TheDim) &&(aMAT.sz().y()==aPt.TheDim),"Bad size Sq Pt")

#define CHECK_SZPT_VECT(aMAT,aPT) MMVII_INTERNAL_ASSERT_tiny(aMAT.Sz()==aPT.TheDim,"Bad size in Vec/Pt")

template <class Type>  cDenseMatrix<Type> M3x3FromLines(const cPtxd<Type,3>&L1,const cPtxd<Type,3> &L2,const cPtxd<Type,3> &L3);
template <class Type>  cDenseMatrix<Type> M3x3FromCol(const cPtxd<Type,3>&C1,const cPtxd<Type,3> &C2,const cPtxd<Type,3> &C3);
template <class Type>  cDenseMatrix<Type> M2x2FromLines(const cPtxd<Type,2>&L1,const cPtxd<Type,2> &L2);
template <class Type>  cDenseMatrix<Type> M2x2FromCol(const cPtxd<Type,2>&L1,const cPtxd<Type,2> &L2);
template <class Type>  cDenseMatrix<Type> MatDiag(const cPtxd<Type,2>&aPt); /// return a 2x2 diagonal matrix
template <class Type>  cDenseMatrix<Type> MatrSim(const cPtxd<Type,2>& aIm_Ox);  /// return the 2x2 similitude given image of Ox
template <class Type>  cDenseMatrix<Type> MatrRot(const Type&);  /// return the 2x2  rotation



template <class Type,int Dim> void GetCol(cPtxd<Type,Dim> &,const cDenseMatrix<Type> &,int aCol);
// template <class Type,int Dim> cPtxd<Type,Dim> GetCol(const cDenseMatrix<Type> &,int aCol);
template <class Type,int Dim> void SetCol(cDenseMatrix<Type> &,int aCol,const cPtxd<Type,Dim> &);

template <class Type,int Dim> void GetLine(cPtxd<Type,Dim> &,int aLine,const cDenseMatrix<Type> &);
// template <class Type,int Dim> cPtxd<Type,Dim> GetLine(int aLine,const cDenseMatrix<Type> &);
template <class Type,int Dim> void SetLine(int aLine,cDenseMatrix<Type> &,const cPtxd<Type,Dim> &);

// Mutiplication by a square Matrix only, do we know size
template <class Type,const int Dim>  cPtxd<Type,Dim> operator * (const cPtxd<Type,Dim> &,const cDenseMatrix<Type> &);
template <class Type,const int Dim>  cPtxd<Type,Dim> operator * (const cDenseMatrix<Type> &,const cPtxd<Type,Dim> &);



template<class Type,const int DimOut,const int DimIn>void MulCol(cPtxd<Type,DimOut>&,const cDenseMatrix<Type>&,const cPtxd<Type,DimIn>&);
template<class Type,const int DimOut,const int DimIn>void MulLine(cPtxd<Type,DimOut>&,const cPtxd<Type,DimIn>&aLine,const cDenseMatrix<Type>&);

template<class Type,const int Dim> cPtxd<Type,Dim> SolveCol(const cDenseMatrix<Type>&,const cPtxd<Type,Dim>&);
template<class Type,const int Dim> cPtxd<Type,Dim> SolveLine(const cPtxd<Type,Dim>&,const cDenseMatrix<Type>&);

/// Considering matrix like a quadratic form, return the scalar product
template<class Type,const int DimOut,const int DimIn> Type 
     QScal(const cPtxd<Type,DimOut>&,const cDenseMatrix<Type>&,const cPtxd<Type,DimIn>&);


//  !!  =>  logically this class  should be defined in file images, but for there is now a tricky dependances
//      this clas requires DenseVect, so Matrix should appear before Image2D
//     but Matrix require Image2D, so Image2D should appear before Matrix ...
//  To break this, would require separate file for vector and matrix
//
/** Class for image of any dimension, relatively slow probably */

template <class Type> class cDataGenDimTypedIm : public cMemCheck
{
    public :
        typedef Type  tVal;
        typedef tNumTrait<Type> tTraits;
        typedef typename tTraits::tBase  tBase;
        typedef cDenseVect<int>          tIndex;
        typedef cDenseVect<tREAL4>       tRIndex;

        const Type &  GetV(const tIndex&) const;
        void SetV(const tIndex&,const tBase & aVal) ;
        void AddV(const tIndex&,const tBase & aVal) ;


        tREAL4   GetNLinearVal(const tRIndex&) const; // Get value by N-Linear interpolation
        void     AddNLinearVal(const tRIndex&,const double & aVal) ; // Get value by N-Linear interpolation

        tREAL4   RecGetNLinearVal(const tRIndex& aRIndex,tIndex& aIIndex,int aDim) const; // slow but easy version
        void RecAddNLinearVal(const tRIndex& aRIndex,const double &,tIndex& aIIndex,int aDim) ; // slow but easy version

        cDataGenDimTypedIm(const tIndex& aSz);
        cDataGenDimTypedIm();
        ~cDataGenDimTypedIm();
        cDataGenDimTypedIm(const cDataGenDimTypedIm<Type> &) = delete;
        void Resize(const tIndex &);

        Type *   RawDataLin() const; 
        int      NbElem() const; 
        int Adress(const tIndex&) const;
        const tIndex & Sz() const;
        void AddData(const cAuxAr2007 &);
        cIm2D<Type>  ToIm2D() const;  // If 2 d, return a standard image , sharing same date. If not=>ERROR ...
    protected  :
        void PrivateAssertOk(const tIndex&) const;
# if (The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_tiny )
        void AssertOk(const tIndex& anIndex) const {PrivateAssertOk(anIndex);}
#else
        void AssertOk(const tIndex&)  const {}
#endif

        int      mDim;
        int      mNbElem;
        tIndex   mSz;
        tIndex   mMulSz;
        Type *   mRawDataLin; ///< raw data containing pixel values
};

/**  This class is just a copy of Eigen indexe, this allow to use eigen sparse
      matrix w/o having to include eigen headers
*/

template <class Type> struct cEigenTriplet
{
     public :
        typedef long int StorageIndex;
        typedef Type   Scalar;

        const StorageIndex &  col () const {return mCol;}
        const StorageIndex &  row () const {return mRow;}
        const Scalar &  value () const     {return mValue;}

        cEigenTriplet(const StorageIndex & aRow,const StorageIndex & aCol,const Scalar & aValue) :
                mRow   (aRow),
                mCol   (aCol),
                mValue (aValue)
        {
        }
     private :

        StorageIndex  mRow;
        StorageIndex  mCol;
        Scalar        mValue;
};

// Return X sol of  "V3 X = aVec" usign Sparse Cholesky decomposition

template<class Type> cDenseVect<Type> EigenSolveCholeskyarseFromV3
                                      (
                                           const std::vector<cEigenTriplet<Type> > & aV3,
                                           const cDenseVect<Type> & aVec
                                      );

// Return least sqaure X sol of  "V3 X = aVec" usign Conjugate Gradient

template<class Type> cDenseVect<Type> EigenSolveLsqGC
                                      (
                                           const std::vector<cEigenTriplet<Type> > & aV3,
                                           const std::vector<Type> & aVec,
                                           int   aNbVar
                                      );

class cParamCtrNLsq
{
     public :
        /// Memorize a new error , and eventualy indicate statbility
        bool StabilityAfterNextError(double) ;
        cParamCtrNLsq();
     private : 
        double GainRel(int aK1,int aK2) const;  // ex GainRel(1,2) 
        inline double ValBack(int aK) const;  // ex GainRel(1,2) 
        std::vector<double> mVER;
};

class  cParamCtrlOpt
{
    public :
        cParamCtrlOpt(const cParamRansac &,const cParamCtrNLsq &);
        const cParamRansac  & ParamRS() const;
        const cParamCtrNLsq & ParamLSQ() const;
        static cParamCtrlOpt  Default();
    private :
        cParamRansac  mParamRS;
        cParamCtrNLsq mParamLSQ;
};





};

#endif  //  _MMVII_Matrix_H_
