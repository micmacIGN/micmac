#include "MMVII_PoseTriplet.h"
#include "MMVII_Tpl_Images.h"


namespace MMVII
{

class cView; ///< class storing a view of a triplet
class cTriplet; ///< a class storing three views

typedef cIsometry3D<tREAL8>  tPose;

double DistBase(cPt3dr  aB1,cPt3dr  aB2)
{
      if (aB1.x()*aB2.x()+aB1.y()*aB2.y() < 0) aB2 = - aB2;
      double aD1 = Norm2(aB1);
      double aD2 = Norm2(aB2);

      if (aD1 > aD2)
         aB1 = aB1 * (aD2/aD1);
      else
         aB2 = aB2 * (aD1/aD2);

      return Norm2(aB1-aB2);
}

cSimilitud3D<double> ComputeSim3D(std::vector<tPose>& aVPosesA,
                                  std::vector<tPose>& aVPosesB)
{
    MMVII_INTERNAL_ASSERT_always( aVPosesA.size() == aVPosesB.size(), "ComputeSim3D vectors of different sizes" );


    ///
    /// \brief Calculte  rotation
    ///
    cDenseMatrix<double> aRotation(3,3,eModeInitImage::eMIA_Null);
    for (size_t aK=0; aK<aVPosesA.size(); aK++ )
    {
        cDenseMatrix<double> aRotCur = aVPosesA[aK].Rot().Mat()
                                     * aVPosesB[aK].Rot().Mat().Transpose();

        // add two matrices
        aRotation = aRotation + aRotCur;

    }
    aRotation = aRotation * (1.0/aVPosesA.size());
    aRotation = aRotation.ClosestOrthog();
    cRotation3D aRot3d(aRotation,false);

    ///
    ///  \brief Calculate scale and translation
    ///
    std::vector<cPt3dr> aVP1;
    std::vector<cPt3dr> aVP2;
    cPt3dr aCentroidA(0,0,0);
    cPt3dr aCentroidB(0,0,0);
    for (size_t aK=0; aK<aVPosesA.size(); aK++ )
    {
        cPt3dr anOrigin(0,0,0);
        cPt3dr aP1 = aVPosesA[aK].Tr() + aVPosesA[aK].Rot().Mat() * anOrigin;
        cPt3dr aP2 = aVPosesB[aK].Tr() + aVPosesB[aK].Rot().Mat() * anOrigin;

        aVP1.push_back(aP1);
        aVP2.push_back(aP2);

        aCentroidA += aP1;
        aCentroidB += aP2;
    }
    aCentroidA = aCentroidA * (1.0/aVPosesA.size());
    aCentroidB = aCentroidB * (1.0/aVPosesB.size());

    double aSumD1=0;
    double aSumD2=0;
    for (size_t aK=0; aK<aVPosesA.size(); aK++ )
    {
        double aDist1 = Norm2(aVP1[aK]-aCentroidA);
        double aDist2 = Norm2(aVP2[aK]-aCentroidB);

        aSumD1 += aDist1;
        aSumD2 += aDist2;
    }
    aSumD1 /= aVPosesA.size();
    aSumD2 /= aVPosesB.size();

    double aScale = aSumD1/aSumD2;

    cPt3dr aTr = aCentroidA - aCentroidB * aScale;

    return cSimilitud3D<double>(aScale,aTr,aRot3d);
}

   /* ********************************************************** */
   /*                                                            */
   /*                        cView                               */
   /*                                                            */
   /* ********************************************************** */

cView::cView(const tPose aPose,const std::string aName) :
    mName(aName),
    mPose(aPose)
{
};

cView::cView() :
    cView(tPose(),"")
{}

void cView::AddData(const cAuxAr2007 &anAuxInit)
{
    cAuxAr2007 anAux("View",anAuxInit);

    MMVII::AddData(cAuxAr2007("Name",anAux),mName);
    MMVII::AddData(cAuxAr2007("Ori",anAux),mPose);

}

void AddData(const  cAuxAr2007 &anAux,cView &aV)
{
    aV.AddData(anAux);
}

    /* ********************************************************** */
    /*                                                            */
    /*                        cTriplet                            */
    /*                                                            */
    /* ********************************************************** */
cTriplet::cTriplet() :
    mPoses(std::vector<cView>())
{
    mPoses.push_back(cView());
    mPoses.push_back(cView());
    mPoses.push_back(cView());
}

void cTriplet::AddData(const cAuxAr2007 &anAuxInit)
{
    cAuxAr2007 anAux("Triplet",anAuxInit);
    //
    // Save the relative poses
    // Pose1, Pose 21, Pose31
    //
    MMVII::AddData(cAuxAr2007("Pose1",anAux),mPoses[0]);
    MMVII::AddData(cAuxAr2007("Pose21",anAux),mPoses[1]);
    MMVII::AddData(cAuxAr2007("Pose31",anAux),mPoses[2]);
    MMVII::AddData(cAuxAr2007("BH",anAux),mBH);
    MMVII::AddData(cAuxAr2007("Residual",anAux),mResidual);

}

void AddData(const cAuxAr2007& anAux,cTriplet& aTri)
{
    aTri.AddData(anAux);
}
    /* ********************************************************** */
    /*                                                            */
    /*                        cTripletSet                         */
    /*                                                            */
    /* ********************************************************** */

cTripletSet::cTripletSet() :
    mName("v0")
{}

void cTripletSet::PushTriplet(cTriplet &aTri)
{
    mSet.push_back(aTri);
}

void cTripletSet::ToFile(const std::string &aName) const
{
    SaveInFile(this->mSet,aName);
}

cTripletSet * cTripletSet::FromFile(const std::string &aName)
{
    StdOut() << aName << std::endl;
    cTripletSet * aRes = new cTripletSet;
    StdOut() << "Before Readfromfile" << std::endl;
    ReadFromFile(aRes->Set(),aName);

    return aRes;
}

void cTripletSet::AddData(const  cAuxAr2007 & anAuxInit)
{
     cAuxAr2007 anAux("TripletSet",anAuxInit);
     // ...
     // Put the data in  tag "cTripletSet"

     // Add data for
     //    mName
     //    ...
     //
     //MMVII::AddData(cAuxAr2007("Name",anAux),mName);
     MMVII::AddData(cAuxAr2007("Sets",anAux),mSet);


}

void AddData(const  cAuxAr2007 & anAux,cTripletSet & aSet)
{
     aSet.AddData(anAux);
}

}; // MMVII




