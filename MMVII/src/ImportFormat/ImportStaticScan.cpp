#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_ReadFileStruct.h"
#include "MMVII_util_tpl.h"
#include "MMVII_Geom3D.h"


/**
   \file importStaticScan.cpp

   \brief import static scan into instrument geometry
*/


namespace MMVII
{
/* ********************************************************** */
/*                                                            */
/*                 cAppli_ImportStaticScan                    */
/*                                                            */
/* ********************************************************** */

class cAppli_ImportStaticScan : public cMMVII_Appli
{
public :
    cAppli_ImportStaticScan(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
    int Exe() override;
    cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
    cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

    std::vector<std::string>  Samples() const override;
private :


    // Mandatory Arg
    std::string              mNameFile;
    std::string              mStationName;
    std::string              mScanName;

    // Optional Arg
    tREAL8                   mAngTolerancy;
    std::string              mTransfoIJK;

};

cAppli_ImportStaticScan::cAppli_ImportStaticScan(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli    (aVArgs,aSpec),
    mAngTolerancy   (1e-6),
    mTransfoIJK         ("ijk")
{
}

cCollecSpecArg2007 & cAppli_ImportStaticScan::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
           <<  Arg2007(mNameFile ,"Name of Input File",{eTA2007::FileAny})
           <<  Arg2007(mStationName ,"Station name",{eTA2007::Topo}) // TODO: change type to future station
           <<  Arg2007(mScanName ,"Scan name",{eTA2007::Topo}) // TODO: change type to future scan
        ;
}

cCollecSpecArg2007 & cAppli_ImportStaticScan::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return    anArgOpt
           << AOpt2007(mAngTolerancy,"AngTol","Angle tolerancy",{eTA2007::HDV})
           << AOpt2007(mTransfoIJK,"Transfo","Transfo to have primariy rotation axis as Z and X as theta origin",{{eTA2007::HDV}})
        ;
}

cPt3dr cart2spher(const cPt3dr & aPtCart) // returns theta phi dist
{
    tREAL8 theta =  atan2(aPtCart.y(),aPtCart.x());
    tREAL8 dist = Norm2(aPtCart);
    tREAL8 distxy = sqrt(aPtCart.BigX2()+aPtCart.BigY2());
    tREAL8 phi =  atan2(aPtCart.z(),distxy);
    return {theta, phi, dist};
}

int cAppli_ImportStaticScan::Exe()
{
    tREAL8 aDistMinToExist = 0.01;
    cTriangulation3D<tREAL8> aTriangulation3DXYZ(mNameFile);

    StdOut() << "Got " <<aTriangulation3DXYZ.NbPts() <<" points.\n";
    if (aTriangulation3DXYZ.HasPtAttribute())
    {
        StdOut() << "Intensity found.\n";
    }

    StdOut() << "Sample:\n";
    for (size_t i=0; (i<10)&&(i<aTriangulation3DXYZ.VPts().size()); ++i)
    {
        StdOut() << aTriangulation3DXYZ.KthPts(i);
        if (aTriangulation3DXYZ.HasPtAttribute())
            StdOut() << " " << aTriangulation3DXYZ.KthPtsPtAttribute(i);
        StdOut() << "\n";
    }
    StdOut() << "...\n";

    cRotation3D<tREAL8>  aRotFrame = cRotation3D<tREAL8>::RotFromCanonicalAxes(mTransfoIJK);

    std::vector<cPt3dr> aVectPtsTPD(aTriangulation3DXYZ.NbPts()); // all points in theta-phi-dist
    size_t aNbPtsNul = 0;
    for (size_t i=0; i<aVectPtsTPD.size(); ++i)
    {
        aVectPtsTPD[i] = cart2spher(aRotFrame.Value(aTriangulation3DXYZ.KthPts(i)));
        if (aVectPtsTPD[i].z()<aDistMinToExist)
            aNbPtsNul++;
    }
    StdOut() << aNbPtsNul << " null points\n";

    // check theta-phi :
    StdOut() << "Spherical sample:\n";
    for (size_t i=0; (i<10)&&(i<aVectPtsTPD.size()); ++i)
    {
        StdOut() << aVectPtsTPD[i];
        StdOut() << "\n";
    }
    StdOut() << "...\n";

    cWhichMinMax<int, tREAL8> aMinMaxTheta;
    cWhichMinMax<int, tREAL8> aMinMaxPhi;
    for (const auto & aPtAng: aVectPtsTPD)
    {
        if (aPtAng.z()<aDistMinToExist) continue;
        aMinMaxTheta.Add(0,aPtAng.x());
        aMinMaxPhi.Add(0,aPtAng.y());
    }
    cBox2dr aBoxAng( {aMinMaxTheta.Min().ValExtre(), aMinMaxPhi.Min().ValExtre()},
                     {aMinMaxTheta.Max().ValExtre(), aMinMaxPhi.Max().ValExtre()});
    StdOut() << "Box: " << aBoxAng << "\n";

    // find phi min diff
    // successive phi diff is useful, but only if we are in the same column
    tREAL8 previousTheta = NAN;
    tREAL8 previousPhi = NAN;
    tREAL8 minDiffPhi = INFINITY; // signed value that is min in abs. For successive points on one column
    tREAL8 angularPrecisionInSteps = 1; // we suppose that theta changes slower than phi... Is it ok???
    for (const auto & aPtAng: aVectPtsTPD)
    {
        if (aPtAng.z()<aDistMinToExist) continue;
        auto aDiffPhi = aPtAng.y()-previousPhi;
        auto aDiffTheta = aPtAng.x()-previousTheta;
        if (fabs(aDiffTheta)<fabs(minDiffPhi)*angularPrecisionInSteps) // we are on the same column
        {
            if (fabs(aDiffPhi)<fabs(minDiffPhi))
            {
                //std::cout<<"with prev "<<previousTheta<< " "<< previousPhi<< "  curr "<<aPtAng<<":\n";
                //std::cout<<"up: "<<minDiffPhi <<" " <<aDiffPhi<<"\n";
                minDiffPhi = aDiffPhi;
            }
        }
        previousTheta = aPtAng.x();
        previousPhi = aPtAng.y();
    }


    auto phiStep = minDiffPhi;
    StdOut() << "phiStep " << phiStep << ",  " << (aBoxAng.P1().y()-aBoxAng.P0().y())/fabs(phiStep) << " steps\n";
    // find theta step
    tREAL8 aColChangeDetectorInPhistep = 100;
    previousTheta = NAN;
    previousPhi = NAN;
    std::vector<tREAL8> aVDiffColTheta;
    for (const auto & aPtAng: aVectPtsTPD)
    {
        if (aPtAng.z()<aDistMinToExist) continue;
        //StdOut() << "Pt "<<aPtAng<< " difphi " <<-(aPtAng.y()-previousPhi)/phiStep <<"\n";
        if (-(aPtAng.y()-previousPhi)/phiStep > aColChangeDetectorInPhistep)
        {
            aVDiffColTheta.push_back(aPtAng.x()-previousTheta);
        }
        previousTheta = aPtAng.x();
        previousPhi = aPtAng.y();
    }
    StdOut() << "Found "<<aVDiffColTheta.size()<<" cols\n";
    //StdOut() << "DiffColTheta: ";
    //for (auto & v: aVDiffColTheta)
    //    StdOut() << v <<" ";
    //StdOut() << "\n";

    // compute line and col for each point
    std::vector<size_t> aVectPtsLine(aTriangulation3DXYZ.NbPts());
    std::vector<size_t> aVectPtsCol(aTriangulation3DXYZ.NbPts());
    size_t aCurrentCol = 0;
    previousPhi = NAN;
    size_t aMaxLine = 0;
    for (size_t i=0; i<aVectPtsTPD.size(); ++i)
    {
        auto aPtAng = aVectPtsTPD[i];
        if (aPtAng.z()<aDistMinToExist)
        {
            aVectPtsLine[i] = 0;
            aVectPtsCol[i] = 0;
            continue;
        }
        size_t aLine = (aPtAng.y()-aBoxAng.P0().y())/fabs(phiStep);
        if (aLine>aMaxLine) aMaxLine = aLine;
        if (-(aPtAng.y()-previousPhi)/phiStep > aColChangeDetectorInPhistep)
            aCurrentCol++;
        aVectPtsLine[i] = aLine;
        aVectPtsCol[i] = aCurrentCol;
        previousPhi = aPtAng.y();
    }
    StdOut() << "Max col found: "<<aCurrentCol<<"\n";
    StdOut() << "Max line found: "<<aMaxLine<<"\n";

    StdOut() << "Image size: "<<(int)aVDiffColTheta.size()
             << " "<< (int)((aBoxAng.P1().y()-aBoxAng.P0().y())/fabs(phiStep))<<"\n";
    //fill raster
    cIm2D<tU_INT1> aRasterIntens(cPt2di(aCurrentCol+1, aMaxLine+1), 0, eModeInitImage::eMIA_Null);
    auto & aRasterIntensData = aRasterIntens.DIm();
    for (size_t i=0; i<aVectPtsTPD.size(); ++i)
    {
        auto aPtAng = aVectPtsTPD[i];
        if (aPtAng.z()<aDistMinToExist) continue;
        aRasterIntensData.SetV(cPt2di(aVectPtsCol[i], aMaxLine-aVectPtsLine[i]), aTriangulation3DXYZ.KthPtsPtAttribute(i)*255);
    }
    aRasterIntensData.ToFile("toto.png");

    return EXIT_SUCCESS;
}

std::vector<std::string>  cAppli_ImportStaticScan::Samples() const
{
    return
        {

        };
}


tMMVII_UnikPApli Alloc_ImportStaticScan(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
    return tMMVII_UnikPApli(new cAppli_ImportStaticScan(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportStaticScan
    (
        "ImportStaticScan",
        Alloc_ImportStaticScan,
        "Import static scan cloud point into instrument raster geometry",
        {eApF::Cloud},
        {eApDT::Ply},
        {eApDT::MMVIICloud},
        __FILE__
        );

}; // MMVII

