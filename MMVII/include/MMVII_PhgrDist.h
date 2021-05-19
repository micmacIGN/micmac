#ifndef  _MMVII_PHGR_DIST_H_
#define  _MMVII_PHGR_DIST_H_


namespace MMVII
{
/**  Indicate the nature of each coefficient of the distorsion
*/
enum class eTypeFuncDist
           {
              eRad,   ///<  Coefficient for radial distorsion 
              eDecX,  ///< Coefficient for decentric distorsion x mean derived by Cx (it has X and Y components)
              eDecY,  ///< Coefficient for decentric distorsion y mean derived by Cy (it has X and Y components)
              eMonX,  ///< Coefficient for a monom in X, of the polynomial distorsion
              eMonY,  ///< Coefficient for a monom in Y, of the polynomial distorsion
              eNbVals ///< Tag for number of value
           };

/**  This class store a complete description of each parameters of the distorsion,
     it is used for computing the formula, the vector of name and (later) automatize
     conversion, print understandable values ....

     It's rather bad design with the same classe coding different things, and some fields
     used of not according to the others, but as it internal/final classes, quick and dirty
     exceptionnaly allowed ...
*/
class cDescOneFuncDist
{
    public :
      cDescOneFuncDist(eTypeFuncDist aType,const cPt2di aDeg);
      /// Majorarion of norms of jacobian , used in simulation
      double MajNormJacOfRho(double aRho) const;

      eTypeFuncDist mType;   ///< Type of dist (Rad, DecX ....)
      std::string   mName;   ///< Name : used as id for code gen & prety print
      cPt2di        mDegMon;  ///< Degree for a polynomial
      int           mNum;     ///< Order for radial and decentric
      int           mDegTot;  ///< Total degree of the polynome
};




//  used for bench now To put elsewhere later,
NS_SymbolicDerivative::cCalculator<double> * EqDist(const cPt3di & aDeg,bool WithDerive,int aSzBuf);

std::vector<cDescOneFuncDist>   DescDist(const cPt3di & aDeg);


};

#endif  //  _MMVII_PHGR_DIST_H_
